"""``python -m mobslim.config``: check a mobslim config, translate a MATSim
config, or print the parameter reference.

    python -m mobslim.config check config.yaml       # load + precheck against the inputs
    python -m mobslim.config from-matsim config.xml -o config.yaml
    python -m mobslim.config reference               # every parameter, as Markdown
    python -m mobslim.config resolved config.yaml    # the complete config with defaults
"""

import argparse
import logging
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mobslim.config", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="load a mobslim config and precheck it against its inputs")
    check.add_argument("config")
    resolved = sub.add_parser("resolved", help="print the complete config with every default")
    resolved.add_argument("config")
    conv = sub.add_parser("from-matsim", help="translate a MATSim config.xml")
    conv.add_argument("config")
    conv.add_argument("-o", "--output", help="YAML file to write (default: stdout)")
    conv.add_argument("--strict", action="store_true", help="fail on params mobslim cannot honour")
    sub.add_parser("reference", help="print the parameter reference (Markdown)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from mobslim.config.mobslim import ConfigError, MobslimConfig

    if args.command == "reference":
        from mobslim.config.schema import reference_markdown

        sys.stdout.write(reference_markdown())
        return 0
    if args.command == "from-matsim":
        from mobslim.config.convert import matsim_to_mobslim, to_yaml
        from mobslim.config.matsim import MatsimConfig, UnsupportedConfigError

        config, unsupported = matsim_to_mobslim(MatsimConfig.load(args.config))
        if unsupported and args.strict:
            raise UnsupportedConfigError(unsupported)
        text = to_yaml(config, unsupported, str(Path(args.config).resolve()))
        if args.output:
            Path(args.output).write_text(text)
        else:
            sys.stdout.write(text)
        return 0
    try:
        cfg = MobslimConfig.load(args.config)
        if args.command == "resolved":
            sys.stdout.write(cfg.to_yaml())
            return 0
        cfg.log_defaults()
        cfg.build_scenario()
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{args.config}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
