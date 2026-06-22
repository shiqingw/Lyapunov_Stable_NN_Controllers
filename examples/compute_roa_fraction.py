"""Compute the fraction of the state-space box covered by the certified
forward-invariant region {x in B : V(x) <= rho} for the example3d system.

Run from the repo root, e.g.:
    python examples/compute_roa_fraction.py --rho 0.88125

The state-space box B is [lower, upper]^3 (defaults match the training/
verification configs: [-1, 1]^3). V is the trained quadratic Lyapunov
function V(x) = x^T (eps I + R^T R) x, so {V <= rho} is an ellipsoid centered
at the origin; the certified region is that ellipsoid intersected with B.
"""
import argparse
import math
import os

import torch

import neural_lyapunov_training.models as models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=0.88125,
                        help="Verified sub-level set value (rho_l from bisection).")
    parser.add_argument("--model", type=str,
                        default="models/example3d_state_feedback.pth",
                        help="Path to the trained model (.pth).")
    parser.add_argument("--lower", type=float, nargs=3, default=[-1.0, -1.0, -1.0])
    parser.add_argument("--upper", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--samples", type=int, default=20_000_000,
                        help="Total Monte Carlo samples.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Rebuild the exact graph used for verification, then load the weights.
    loss = models.create_example3d_model(
        lyapunov_func="lyapunov.NeuralNetworkQuadraticLyapunov",
        lyapunov_parameters={"R_rows": 3, "eps": 0.01},
        controller_parameters={
            "nlayer": 4, "hidden_dim": 8, "clip_output": "clamp",
            "u_lo": torch.tensor([-20.0]), "u_up": torch.tensor([20.0]),
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
    box_vol = float(torch.prod(upper - lower))
    rho = args.rho

    # Q matrix and analytic (unclipped) ellipsoid volume.
    Q = (V.eps * torch.eye(3) + V.R.t() @ V.R).detach()
    detQ = torch.det(Q).item()
    vol_ellipsoid = (4.0 / 3.0) * math.pi * rho ** 1.5 / math.sqrt(detQ)
    half_widths = torch.sqrt(rho * torch.diag(torch.linalg.inv(Q)))  # ellipsoid bbox
    inside_box = bool(torch.all(half_widths <= (upper - lower) / 2))

    # Monte Carlo over the box.
    torch.manual_seed(args.seed)
    batch = 2_000_000
    hit, total = 0, 0
    while total < args.samples:
        n = min(batch, args.samples - total)
        x = lower + (upper - lower) * torch.rand(n, 3)
        v = torch.sum(x * (x @ Q), dim=1)
        hit += int((v <= rho).sum())
        total += n
    frac_mc = hit / total

    print(f"rho                         = {rho}")
    print(f"state-space box volume      = {box_vol:.4f}")
    print(f"ellipsoid bbox half-widths  = {half_widths.tolist()}")
    print(f"ellipsoid fully inside box? = {inside_box}")
    print(f"ellipsoid volume (unclipped)= {vol_ellipsoid:.4f}  "
          f"({100 * vol_ellipsoid / box_vol:.2f}% of box)")
    print(f"Monte Carlo covered fraction= {100 * frac_mc:.2f}%  "
          f"({hit}/{total})")


if __name__ == "__main__":
    main()
