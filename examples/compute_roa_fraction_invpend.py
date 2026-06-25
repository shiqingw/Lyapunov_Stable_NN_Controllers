"""Compute the fraction of the state-space box covered by the certified
forward-invariant region {x in B : V(x) <= rho} for the inverted pendulum.

Run from the repo root, e.g.:
    python examples/compute_roa_fraction_invpend.py --rho 5.0

The state-space box B is [lower, upper] (defaults match the training/
verification configs: theta in [-3.14, 3.14], theta_dot in [-3.0, 3.0]). V is
the trained quadratic Lyapunov function V(x) = x^T (eps I + R^T R) x, so
{V <= rho} is an ellipse centered at the origin; the certified region is that
ellipse intersected with B.
"""
import argparse
import math

import torch

import neural_lyapunov_training.models as models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=5.0,
                        help="Verified sub-level set value (rho_l from bisection).")
    parser.add_argument("--model", type=str,
                        default="models/invpend_state_feedback.pth",
                        help="Path to the trained model (.pth).")
    parser.add_argument("--lower", type=float, nargs=2, default=[-3.14, -3.0])
    parser.add_argument("--upper", type=float, nargs=2, default=[3.14, 3.0])
    parser.add_argument("--samples", type=int, default=20_000_000,
                        help="Total Monte Carlo samples.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Rebuild the exact graph used for verification, then load the weights.
    # Must match verification/invpend_state_feedback_lyapunov_in_levelset.yaml.
    loss = models.create_invpend_model(
        m=1.0, l=1.0, beta=0.0,
        lyapunov_func="lyapunov.NeuralNetworkQuadraticLyapunov",
        lyapunov_parameters={"R_rows": 2, "eps": 0.01},
        controller_parameters={
            "nlayer": 4, "hidden_dim": 8, "clip_output": "clamp",
            "u_lo": torch.tensor([-15.0]), "u_up": torch.tensor([15.0]),
        },
        loss_parameters={"kappa": 0.001},
        loss_func="lyapunov.LyapunovDerivativeSimpleLossWithVBox",
    )
    ckpt = torch.load(args.model, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    loss.load_state_dict(state_dict)
    V = loss.lyapunov.eval()

    lower = torch.tensor(args.lower)
    upper = torch.tensor(args.upper)
    box_area = float(torch.prod(upper - lower))
    rho = args.rho

    # Q matrix and analytic (unclipped) ellipse area.
    Q = (V.eps * torch.eye(2) + V.R.t() @ V.R).detach()
    detQ = torch.det(Q).item()
    # Area of {x : x^T Q x <= rho} in 2-D is pi * rho / sqrt(det Q).
    area_ellipse = math.pi * rho / math.sqrt(detQ)
    half_widths = torch.sqrt(rho * torch.diag(torch.linalg.inv(Q)))  # ellipse bbox
    inside_box = bool(torch.all(half_widths <= (upper - lower) / 2))

    # Monte Carlo over the box.
    torch.manual_seed(args.seed)
    batch = 2_000_000
    hit, total = 0, 0
    while total < args.samples:
        n = min(batch, args.samples - total)
        x = lower + (upper - lower) * torch.rand(n, 2)
        v = torch.sum(x * (x @ Q), dim=1)
        hit += int((v <= rho).sum())
        total += n
    frac_mc = hit / total

    print(f"rho                         = {rho}")
    print(f"state-space box area        = {box_area:.4f}")
    print(f"ellipse bbox half-widths    = {half_widths.tolist()}")
    print(f"ellipse fully inside box?   = {inside_box}")
    print(f"ellipse area (unclipped)    = {area_ellipse:.4f}  "
          f"({100 * area_ellipse / box_area:.2f}% of box)")
    print(f"Monte Carlo covered fraction= {100 * frac_mc:.2f}%  "
          f"({hit}/{total})")


if __name__ == "__main__":
    main()
