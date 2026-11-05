"""``python -m mobslim.formats``: convert and inspect population files.

    python -m mobslim.formats convert plans.xml.gz plans_pq --network network.xml
    python -m mobslim.formats convert plans_pq output_plans.xml.gz --network network.xml
    python -m mobslim.formats info plans_pq --network network.xml

Formats are chosen by path: ``*.xml``/``*.xml.gz`` is MATSim XML, anything
else a Parquet population directory (``--from``/``--to`` override).
Routes are kept unless ``--drop-routes``. Give ``--schedule`` (a MATSim
transit schedule) to keep pt rides in Parquet files.
"""

import argparse
import logging
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mobslim.formats", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert", help="convert a population between formats")
    conv.add_argument("input")
    conv.add_argument("output")
    info = sub.add_parser("info", help="summarise a population file")
    info.add_argument("input")
    for p in (conv, info):
        p.add_argument("--network", required=True, help="MATSim network XML")
        p.add_argument("--schedule", help="MATSim transit schedule XML (keeps pt rides)")
        p.add_argument("--attributes", help="extra person attributes (objectAttributes XML "
                                            "or a Parquet persons table)")
        p.add_argument("--from", dest="from_format", choices=("xml", "parquet"))
    conv.add_argument("--to", dest="to_format", choices=("xml", "parquet"))
    conv.add_argument("--drop-routes", action="store_true", help="write every leg unrouted")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from mobslim.formats import read_population, write_population
    from mobslim.network import read_network_xml

    network = read_network_xml(args.network)
    schedule = None
    if args.schedule:
        from mobslim.transit.io import read_transit_schedule_xml

        schedule = read_transit_schedule_xml(args.schedule, network)
    t0 = time.perf_counter()
    population = read_population(args.input, network, format=args.from_format,
                                 keep_routes=not getattr(args, "drop_routes", False),
                                 attributes_file=args.attributes, transit_schedule=schedule)
    t1 = time.perf_counter()
    print(f"read {args.input}: {_summary(population)} in {t1 - t0:.2f}s", file=sys.stderr)
    if args.command == "info":
        batch = population.live_batch()
        print(f"activity types: {', '.join(batch.act_types)}")
        print(f"modes: {', '.join(batch.modes)}")
        print(f"routed legs: {int(batch.leg_routed.sum())} of {batch.n_legs}")
        if len(population.attributes):
            print(f"attributes: {', '.join(population.attributes.names)}")
        return 0
    out = write_population(population, args.output, network, format=args.to_format,
                           transit_schedule=schedule)
    print(f"wrote {out} in {time.perf_counter() - t1:.2f}s", file=sys.stderr)
    return 0


def _summary(population) -> str:
    store = population.plans
    return (f"{len(population)} persons, {int(population.n_plans.sum())} plans, "
            f"{store.n_acts} activities, {store.n_legs} legs")


if __name__ == "__main__":
    sys.exit(main())
