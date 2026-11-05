"""``python -m verification``: compare mobslim with Java MATSim.

    uv run python -m verification run              # every case, then the report
    uv run python -m verification run -k equil-100 # cases whose name contains it (or a layer: l2)
    uv run python -m verification run --mobsim qsim --force-matsim
    uv run python -m verification report          # report from the saved results
    uv run python -m verification freeze          # MATSim references for pytest
    uv run python -m verification list
"""

from __future__ import annotations

import argparse
import sys

from verification import cases, report, runner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verification")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run cases and write the report")
    run.add_argument("-k", action="append", default=[], help="case name substring or layer")
    run.add_argument("--mobsim", action="append", choices=list(cases.MOBSIMS),
                     help="mobslim mobsims to run (default: all)")
    run.add_argument("--force-matsim", action="store_true", help="rerun MATSim even if cached")
    sub.add_parser("report", help="rewrite the report from saved results")
    fr = sub.add_parser("freeze", help="freeze MATSim references for pytest (tests/fixtures/matsim_reference)")
    fr.add_argument("-k", action="append", default=[], help="frozen case name substring")
    sub.add_parser("list", help="list cases")
    args = ap.parse_args(argv)

    if args.cmd == "freeze":
        from verification import freeze

        for path in freeze.freeze(args.k):
            print(f"Frozen: {path}")
        return 0
    if args.cmd == "list":
        for c in cases.CASES:
            print(f"{c.name:28s} {c.layer}  {c.description}")
        return 0
    if args.cmd == "run":
        selected = cases.select(args.k)
        if not selected:
            print("No case matches.", file=sys.stderr)
            return 1
        for c in selected:
            print(f"{c.name} ...", flush=True)
            if c.layer == "l1":
                runner.run_l1(c, force_matsim=args.force_matsim)
            elif c.layer == "l2":
                runner.run_l2(c, args.mobsim, force_matsim=args.force_matsim)
            elif c.layer == "l3":
                runner.run_l3(c, force_matsim=args.force_matsim)
    path = report.write(report.load_results(runner.OUTPUT))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
