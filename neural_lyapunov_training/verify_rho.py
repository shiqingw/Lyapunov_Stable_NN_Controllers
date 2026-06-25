"""Verify that a *single*, fixed level set {x in B : V(x) <= rho} satisfies the
Lyapunov condition. This is exactly one `check_rho` from bisect.py, without the
bisection loop -- use it when you already know which rho you want to certify.

Example:
    python -m neural_lyapunov_training.verify_rho \
        --lower_limit -5 -5 -5 --upper_limit 5 5 5 \
        --hole_size 0.001 --rho 0.88125 \
        --config verification/example3d_state_feedback_lyapunov_in_levelset.yaml \
        --spec_prefix verification/specs/example3d_state_feedback \
        --output_folder ./output_single
"""
import argparse
import os
from contextlib import redirect_stdout, redirect_stderr
from complete_verifier.abcrown import ABCROWN


def check_rho(rho, args, additional_args):
    print(f"Generating specs with rho={rho}")
    output_gen_spec = os.path.join(args.output_folder, f"rho_{rho:.5f}_spec.txt")
    command = (
        "python -m neural_lyapunov_training.generate_vnnlib "
        f"--lower_limit {' '.join(map(str, args.lower_limit))} "
        f"--upper_limit {' '.join(map(str, args.upper_limit))} "
        f"--hole_size {args.hole_size} "
        f"--value_levelset {rho} "
    )
    if args.ignore_x_next:
        # Keep all 5 outputs declared (verifier shapes match the model), but
        # make the x_next-in-box disjuncts vacuous so only Y_0 (decrease) matters.
        command += "--ignore_x_next "
    command += f"{args.spec_prefix} >{output_gen_spec} 2>&1"
    os.system(command)

    print("Start verification")
    output_path = os.path.join(args.output_folder, f"rho_{rho:.5f}.txt")
    print("Output path:", output_path)
    with open(output_path, "w") as file:
        with redirect_stdout(file), redirect_stderr(file):
            verifier = ABCROWN(
                args=additional_args,
                csv_name=f"{args.spec_prefix}.csv",
                config=args.config,
                override_timeout=args.timeout,
                pgd_order=args.pgd_order,
                pgd_restarts=10000,
            )
            ret = verifier.main()
    print("Result:", ret)
    result = "safe"
    for k, v in ret.items():
        if "unsafe" in k:
            result = "unsafe"
    if result == "safe" and "unknown" in ret.keys():
        result = "unknown"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_prefix", type=str, default="specs/bisect")
    parser.add_argument("--output_folder", type=str, default="./output")
    parser.add_argument("-l", "--lower_limit", type=float, nargs="+", required=True)
    parser.add_argument("-u", "--upper_limit", type=float, nargs="+", required=True)
    parser.add_argument("-o", "--hole_size", type=float, default=0.001)
    parser.add_argument("--rho", type=float, required=True,
                        help="The single level set value to verify.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=200)
    parser.add_argument("--ignore_x_next", action="store_true",
                        help="Verify only the decrease condition (Y_0). Keeps the "
                             "x_next outputs declared but with vacuous +/-1e9 bounds, "
                             "so it stays compatible with this abcrown build. Sound "
                             "as a sublevel-set certificate only when {V<=rho} is "
                             "contained in the box B.")
    parser.add_argument("--pgd_order", type=str, default="before",
                        choices=["before", "after", "skip"],
                        help="PGD falsification order.")
    args, additional_args = parser.parse_known_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    result = check_rho(args.rho, args, additional_args)
    print(f"\nrho = {args.rho}: {result.upper()}")
    if result == "safe":
        print("The sub-level set {V(x) <= rho} satisfies the Lyapunov condition.")
    else:
        print("NOT certified at this rho (counterexample found or unknown).")
