import os

import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import scipy
import torch
import torch.nn as nn
import wandb

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.example3d as example3d
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.train_utils as train_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def compute_lqr(system: example3d.Example3DDynamics):
    x_equilibrium = system.x_equilibrium.to(device)
    u_equilibrium = system.u_equilibrium.to(device)
    A_batch, B_batch = system.linearized_dynamics(
        x_equilibrium.unsqueeze(0), u_equilibrium.unsqueeze(0)
    )
    A = A_batch.squeeze(0).cpu().detach().numpy()
    B = B_batch.squeeze(0).cpu().detach().numpy()
    Q = np.eye(3)
    R = np.eye(1)
    # (A, B) is stabilizable (the uncontrollable mode x1 - x2 is stable),
    # so the continuous-time ARE has a stabilizing solution.
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = -np.linalg.solve(R, B.T @ S)
    return K, S


def approximate_lqr(
    system: example3d.Example3DDynamics,
    controller: controllers.NeuralNetworkController,
    lyapunov_nn: lyapunov.NeuralNetworkLyapunov,
    upper_limit: torch.Tensor,
    logger,
):
    K, S = compute_lqr(system)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    x = (torch.rand((100000, 3), dtype=dtype, device=device) - 0.5) * 2 * upper_limit
    V = torch.sum(x * (x @ S_torch), axis=1, keepdim=True)
    u = x @ K_torch.T

    def approximate(system, system_input, target, lr, max_iter):
        optimizer = torch.optim.Adam(system.parameters(), lr=lr)
        for i in range(max_iter):
            optimizer.zero_grad()
            output = torch.nn.MSELoss()(system.forward(system_input), target)
            logger.info(f"iter {i}, loss {output.item()}")
            output.backward()
            optimizer.step()

    approximate(controller, x, u, lr=0.01, max_iter=500)
    approximate(lyapunov_nn, x, V, lr=0.01, max_iter=1000)


@hydra.main(config_path="./config", config_name="example3d_state_training.yaml")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    train_utils.set_seed(cfg.seed)

    dt = cfg.model.dt
    system_continuous = example3d.Example3DDynamics()
    dynamics = dynamical_system.FirstOrderDiscreteTimeSystem(
        system_continuous,
        dt=dt,
        integration=dynamical_system.IntegrationMethod[cfg.model.integration],
    )

    controller = controllers.NeuralNetworkController(
        nlayer=cfg.model.controller_nlayer,
        in_dim=3,
        out_dim=1,
        hidden_dim=cfg.model.controller_hidden_dim,
        clip_output="clamp",
        u_lo=torch.tensor(cfg.model.u_lo),
        u_up=torch.tensor(cfg.model.u_up),
        x_equilibrium=system_continuous.x_equilibrium,
        u_equilibrium=system_continuous.u_equilibrium,
    )
    controller.eval()

    absolute_output = True
    if cfg.model.lyapunov.quadratic:
        _, S = compute_lqr(system_continuous)
        S_torch = torch.from_numpy(S).type(dtype).to(device)
        R = torch.linalg.cholesky(S_torch)
        lyapunov_nn = lyapunov.NeuralNetworkQuadraticLyapunov(
            goal_state=torch.zeros(3, dtype=dtype).to(device),
            x_dim=3,
            R_rows=3,
            eps=0.01,
            R=R,
        )
    else:
        lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
            goal_state=system_continuous.x_equilibrium,
            hidden_widths=cfg.model.lyapunov.hidden_widths,
            x_dim=3,
            R_rows=4,
            absolute_output=absolute_output,
            eps=0.01,
            activation=nn.LeakyReLU,
            V_psd_form=cfg.model.V_psd_form,
        )
    lyapunov_nn.eval()

    kappa = cfg.model.kappa
    derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=0,
        box_up=0,
        rho_multiplier=1,
        kappa=kappa,
        hard_max=cfg.train.hard_max,
    )

    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    grid_size = torch.tensor([20, 20, 20], device=device)
    logger = logging.getLogger(__name__)
    if cfg.approximate_lqr:
        limit = cfg.model.limit_scale[0] * torch.tensor(cfg.model.limit, device=device)
        approximate_lqr(system_continuous, controller, lyapunov_nn, limit, logger)
        torch.save(
            {"state_dict": derivative_lyaloss.state_dict()},
            os.path.join(os.getcwd(), "lyaloss_lqr.pth"),
        )

    if cfg.model.load_lyaloss is not None:
        load_lyaloss = os.path.join(
            os.path.dirname(__file__), "../", cfg.model.load_lyaloss
        )
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss)["state_dict"])

    if absolute_output:
        positivity_lyaloss = None
    else:
        positivity_lyaloss = lyapunov.LyapunovPositivityLoss(
            lyapunov_nn, 0.01 * torch.eye(3, device=device)
        )

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
        )
        # wandb.config.update(cfg)

    save_lyaloss = cfg.model.save_lyaloss
    V_decrease_within_roa = cfg.model.V_decrease_within_roa

    if cfg.train.derivative_x_buffer_path is not None:
        derivative_x_buffer = torch.load(cfg.train.derivative_x_buffer_path)
    else:
        derivative_x_buffer = None

    if cfg.train.train_lyaloss:
        for n in range(len(cfg.model.limit_scale)):
            limit_scale = cfg.model.limit_scale[n]
            limit = limit_scale * torch.tensor(cfg.model.limit, device=device)
            lower_limit = -limit
            upper_limit = limit

            derivative_lyaloss = lyapunov.LyapunovDerivativeLoss(
                dynamics,
                controller,
                lyapunov_nn,
                box_lo=lower_limit,
                box_up=upper_limit,
                rho_multiplier=cfg.model.rho_multiplier[n],
                kappa=kappa,
                hard_max=cfg.train.hard_max,
            )

            if save_lyaloss:
                save_lyaloss_path = os.path.join(
                    os.getcwd(), f"lyaloss_{limit_scale}.pth"
                )
            else:
                save_lyaloss_path = None

            candidate_roa_states = limit_scale * torch.tensor(
                cfg.loss.candidate_roa_states,
                device=device,
            )
            train_utils.train_lyapunov_with_buffer(
                derivative_lyaloss=derivative_lyaloss,
                positivity_lyaloss=positivity_lyaloss,
                observer_loss=None,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                grid_size=grid_size,
                learning_rate=cfg.train.learning_rate,
                weight_decay=0.0,
                max_iter=cfg.train.max_iter,
                enable_wandb=cfg.train.wandb.enabled,
                derivative_ibp_ratio=cfg.loss.ibp_ratio_derivative,
                derivative_sample_ratio=cfg.loss.sample_ratio_derivative,
                positivity_ibp_ratio=cfg.loss.ibp_ratio_positivity,
                positivity_sample_ratio=cfg.loss.sample_ratio_positivity,
                save_best_model=save_lyaloss_path,
                pgd_steps=cfg.train.pgd_steps,
                buffer_size=cfg.train.buffer_size,
                batch_size=cfg.train.batch_size,
                epochs=cfg.train.epochs,
                samples_per_iter=cfg.train.samples_per_iter,
                l1_reg=cfg.loss.l1_reg,
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                V_decrease_within_roa=V_decrease_within_roa,
                Vmin_x_boundary_weight=cfg.loss.Vmin_x_boundary_weight,
                Vmax_x_boundary_weight=cfg.loss.Vmax_x_boundary_weight,
                candidate_roa_states=candidate_roa_states,
                candidate_roa_states_weight=cfg.loss.candidate_roa_states_weight,
                derivative_x_buffer=derivative_x_buffer,
                logger=logger,
                always_candidate_roa_regulizer=cfg.loss.always_candidate_roa_regulizer,
            )

        torch.save(
            {
                "state_dict": lyapunov_nn.state_dict(),
                "rho": derivative_lyaloss.get_rho(),
            },
            os.path.join(os.getcwd(), "lyapunov_nn.pth"),
        )
    else:
        limit = cfg.model.limit_scale[-1] * torch.tensor(cfg.model.limit, device=device)
        lower_limit = -limit
        upper_limit = limit
        derivative_lyaloss.x_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
            lyapunov_nn,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            eps=limit,
            steps=100,
            direction="minimize",
        )

    derivative_lyaloss_check = lyapunov.LyapunovDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=cfg.model.rho_multiplier[-1],
        kappa=0.0,
        hard_max=True,
    )
    pgd_verifier_find_counterexamples = False
    counterexamples_check = torch.zeros((0, 3), device=device)
    for seed in range(100):
        train_utils.set_seed(seed)
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                eps=limit,
                steps=100,
                direction="minimize",
            )
            if derivative_lyaloss.x_boundary is not None:
                derivative_lyaloss_check.x_boundary = torch.cat(
                    (x_min_boundary, derivative_lyaloss.x_boundary), dim=0
                )
        x_check_start = (
            (
                torch.rand((50000, 3), device=device)
                - torch.full((3,), 0.5, device=device)
            )
            * limit
            * 2
        )
        adv_x = train_utils.pgd_attack(
            x_check_start,
            derivative_lyaloss_check,
            eps=limit,
            steps=cfg.pgd_verifier_steps,
            lower_boundary=lower_limit,
            upper_boundary=upper_limit,
            direction="minimize",
        ).detach()
        adv_lya = derivative_lyaloss_check(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        msg = f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        counterexamples_check = torch.cat(
            (counterexamples_check, adv_x[adv_output.squeeze(1) > 0]), dim=0
        )
        if max_adv_violation > 0:
            pgd_verifier_find_counterexamples = True
        logger.info(msg)

    logger.info(
        f"PGD verifier finds counter examples? {pgd_verifier_find_counterexamples}"
    )
    if counterexamples_check.shape[0] > 0:
        torch.save(
            counterexamples_check,
            os.path.join(os.getcwd(), "counterexamples_check.pth"),
        )

    x0 = (torch.rand((40, 3), device=device) - 0.5) * 2 * limit
    x_traj, V_traj = models.simulate(derivative_lyaloss, 500, x0)
    plt.plot(torch.stack(V_traj).cpu().detach().squeeze().numpy())
    plt.savefig(os.path.join(os.getcwd(), "Vtraj_roa.png"))

    rho = derivative_lyaloss.get_rho().item()
    print("rho = ", rho)
    labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
    for plot_idx in [[0, 1], [0, 2], [1, 2]]:
        fig = plt.figure()
        train_utils.plot_V_heatmap(
            fig,
            lyapunov_nn,
            rho,
            lower_limit,
            upper_limit,
            system_continuous.nx,
            x_boundary=derivative_lyaloss_check.x_boundary,
            plot_idx=plot_idx,
        )
        plt.xlabel(labels[plot_idx[0]])
        plt.ylabel(labels[plot_idx[1]])
        plt.savefig(os.path.join(os.getcwd(), f"V_roa_{str(plot_idx)}.png"))


if __name__ == "__main__":
    main()
