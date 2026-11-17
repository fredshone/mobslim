"""``python -m showcase {scenario,correctness,perf,report,networks,weaving,weaving-report,opendd,tgsim,tgsim-i90,replay-report,all} [--force-matsim]``."""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m showcase", description=__doc__)
    ap.add_argument("what", choices=["scenario", "correctness", "perf", "report", "networks", "weaving",
                                     "weaving-report", "opendd", "tgsim", "tgsim-i90", "replay-report", "all"])
    ap.add_argument("--force-matsim", action="store_true", help="rerun MATSim even if cached")
    ap.add_argument("-k", action="append", default=[], help="perf, networks, weaving: only cases/runs matching")
    args = ap.parse_args()
    if args.what == "scenario":
        from showcase import corridor
        for variant in corridor.VARIANTS:
            print(corridor.write(variant))
    if args.what in ("correctness", "all"):
        from showcase import correctness
        correctness.main(args.force_matsim)
    if args.what in ("perf", "all"):
        from showcase import perf
        perf.main(args.k, args.force_matsim)
    if args.what in ("report", "perf", "all"):
        from showcase import perf_report
        perf_report.main()
    if args.what in ("networks", "all"):
        from showcase import networks
        networks.main(args.k)
    if args.what in ("weaving", "all"):
        from showcase.weaving import experiment
        experiment.main(args.k)
    if args.what in ("weaving", "weaving-report", "all"):
        from showcase.weaving import report
        report.main()
    if args.what in ("opendd", "all"):  # needs the openDD data in data/opendd (see showcase/opendd)
        from showcase.opendd import run
        run.main()
    if args.what in ("tgsim", "all"):  # needs the TGSIM I-395 data in data/tgsim (see showcase/tgsim)
        from showcase.tgsim import run
        run.main()
    if args.what in ("tgsim-i90", "all"):  # needs the TGSIM I-90/I-94 data in data/tgsim_i90 (see showcase/tgsim_i90)
        from showcase.tgsim_i90 import run
        run.main()
    if args.what == "replay-report":  # the Correctness table, from showcase/results alone
        from showcase import replay
        replay.headline()


if __name__ == "__main__":
    main()
