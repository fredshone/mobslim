"""The ``mobslim`` command.

    mobslim run config.yaml                  # run a mobslim YAML config
    mobslim run config.xml                   # run a MATSim config.xml
    mobslim run config.yaml -n 5 -o out/     # override iterations / output directory
    mobslim check config.yaml                # load and precheck against the inputs
    mobslim resolved config.yaml             # the complete config, with defaults
    mobslim reference                        # every config parameter
    mobslim from-matsim config.xml -o config.yaml
    mobslim convert plans.xml.gz plans_pq --network network.xml
    mobslim info plans_pq --network network.xml

Also available as ``python -m mobslim``.
"""

import argparse
import logging
import sys
from pathlib import Path

from mobslim.config import __main__ as config_cli
from mobslim.formats import __main__ as formats_cli


def _add_run(sub) -> None:
    run = sub.add_parser("run", help="run a mobslim YAML config or a MATSim config.xml")
    run.add_argument("config", help="*.yaml/*.yml (mobslim) or *.xml (MATSim)")
    run.add_argument("-n", "--iterations", type=int, help="number of iterations (overrides the config)")
    run.add_argument("-o", "--output-dir", help="output directory (overrides the config)")
    run.add_argument("--strict", action="store_true",
                     help="MATSim config: fail on params mobslim cannot honour")
    run.add_argument("-q", "--quiet", action="store_true", help="log warnings only")


def _run(args) -> int:
    from mobslim.config.mobslim import ConfigError
    from mobslim.controller import Controller

    path = Path(args.config)
    try:
        if path.suffix.lower() == ".xml" or path.name.lower().endswith(".xml.gz"):
            from mobslim.config.matsim import MatsimConfig

            cfg = MatsimConfig.load(path)
            if args.iterations is not None:
                first = int(cfg.get("controller", "firstIteration", "0"))
                cfg.set("controller", "lastIteration", first + args.iterations - 1)
            if args.output_dir is not None:
                cfg.set("controller", "outputDirectory", args.output_dir)
            controller = Controller.from_matsim_config(cfg, strict=args.strict)
        else:
            from mobslim.config.mobslim import MobslimConfig

            cfg = MobslimConfig.load(path)
            if args.iterations is not None:
                cfg.values["run"]["iterations"] = args.iterations
            if args.output_dir is not None:
                cfg.values["run"]["output_dir"] = str(Path(args.output_dir).resolve())
            controller = Controller.from_mobslim_config(cfg)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    result = controller.run()
    out = controller.params.output_dir
    last = result.metrics[-1]
    n = len(result.metrics)
    print(f"done: {n} iteration{'s' if n != 1 else ''}, final score "
          f"{last['score_executed']:.3f}" + (f", outputs in {out}" if out is not None else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mobslim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    config_cli.add_commands(sub)
    formats_cli.add_commands(sub)
    args = parser.parse_args(argv)
    quiet = getattr(args, "quiet", False)
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO, format="%(message)s")
    if args.command == "run":
        return _run(args)
    if args.command in config_cli.COMMANDS:
        return config_cli.run_command(args)
    return formats_cli.run_command(args)


if __name__ == "__main__":
    sys.exit(main())
