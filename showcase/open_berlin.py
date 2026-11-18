"""The MATSim Open Berlin scenario (v6.4) as a performance showcase case.

Open Berlin (TU Berlin, VSP; https://github.com/matsim-scenarios/matsim-berlin)
is a calibrated MATSim model of Berlin and Brandenburg: a network of car,
truck, freight, ride and bike links plus the pt lines, the timetable, and
1%, 3%, 10% samples of residents, commercial traffic and freight. Its inputs
and outputs are licensed CC BY 4.0. They are large, so they are downloaded
here (into ``showcase/output/perf/open-berlin-v6.4/download/``), not
committed.

:func:`prepare` writes a case's inputs: each person's selected plan from
the calibrated plans (routed, so both MATSim and MobSlim simulate the same
routes), and a MATSim config for plain MATSim. Open Berlin's own run adds,
in Java, the scoring parameters of its 1,100 or so activity types
(``home_86400``: typical duration 86,400 s) and its replanning; the config
here carries the scoring parameters itself, and the showcase runs no
replanning. Further changes to Open Berlin's config:

* flow and storage capacity factors: the sample size (Open Berlin's own
  run uses the same factor for both);
* MATSim's default queue dynamics (``trafficDynamics=queue``, FIFO links),
  not kinematic waves and passing queues, as MobSlim's mobsims have those;
* modules of Open Berlin's code (simwrapper, annealing, counts) dropped.

The network routes car, truck, freight, ride and bike; only car, truck and
freight are simulated on the roads (``qsim.mainMode``), with their
vehicle types' sizes and speeds (``vehicleTypes.xml``); ride and bike are
teleported along their routes.
"""

from __future__ import annotations

import gzip
import re
import shutil
import urllib.request
from pathlib import Path

VERSION = "v6.4"
SVN = ("https://svn.vsp.tu-berlin.de/repos/public-svn/matsim/scenarios/countries/de/berlin/"
       f"berlin-{VERSION}/input/")
GITHUB = f"https://raw.githubusercontent.com/matsim-scenarios/matsim-berlin/main/input/{VERSION}/"
SHARED = {  # file -> URL, for every sample
    "network.xml.gz": SVN + f"berlin-{VERSION}-network-with-pt.xml.gz",
    "transitSchedule.xml.gz": SVN + f"berlin-{VERSION}-transitSchedule.xml.gz",
    "transitVehicles.xml.gz": SVN + f"berlin-{VERSION}-transitVehicles.xml.gz",
    "facilities.xml.gz": SVN + f"berlin-{VERSION}-facilities.xml.gz",
    "vehicleTypes.xml": GITHUB + f"berlin-{VERSION}-vehicleTypes.xml",
    "config.xml": GITHUB + f"berlin-{VERSION}.config.xml",
}
SAMPLES = {"1pct": 0.01, "3pct": 0.03, "10pct": 0.1}
# Open Berlin modules that plain MATSim does not know or the showcase does not use
DROP_MODULES = ("simwrapper", "replanningAnnealer", "counts", "informedModeChoice",
                "advancedScoring")
# qsim params left at MATSim's defaults (queue dynamics, FIFO links), as MobSlim's mobsims
DROP_QSIM = ("trafficDynamics", "linkDynamics", "insertingWaitingVehiclesBeforeDrivingVehicles",
             "numberOfThreads")


def download_dir(out: Path) -> Path:
    return out / f"open-berlin-{VERSION}" / "download"


def _fetch(url: str, path: Path) -> Path:
    """``url`` saved at ``path``, unless it is there already."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    part.rename(path)
    return path


def download(out: Path, sample: str) -> Path:
    """The shared inputs and ``sample``'s calibrated plans, downloaded once."""
    d = download_dir(out)
    for name, url in SHARED.items():
        _fetch(url, d / name)
    _fetch(SVN + f"berlin-{VERSION}-{sample}.plans.xml.gz", d / f"plans-{sample}.xml.gz")
    return d


def selected_plans(source: Path, target: Path) -> set[str]:
    """Each person of ``source`` with only their selected plan, written to
    ``target`` (streamed, line by line); returns the activity types used."""
    types: set[str] = set()
    plan_open = keep = False
    with gzip.open(source, "rt", encoding="utf-8") as src, \
            gzip.open(target.with_name(target.name + ".part"), "wt", encoding="utf-8",
                      compresslevel=3) as dst:
        for line in src:
            if not plan_open and "<plan " in line:
                plan_open, keep = True, 'selected="yes"' in line
            if not plan_open or keep:
                if keep and "<activity " in line:
                    types.add(re.search(r'type="([^"]+)"', line)[1])
                dst.write(line)
            if plan_open and "</plan>" in line:
                plan_open = False
    target.with_name(target.name + ".part").rename(target)
    return types


def _hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def activity_params(types: set[str]) -> str:
    """Scoring parameter sets for ``types`` (stage activities excluded), as
    Open Berlin's code makes them: the typical duration is the type's
    suffix in seconds (``leisure_5400``); types without one (commercial and
    freight stops) get an hour."""
    sets = []
    for t in sorted(types):
        if t.endswith(" interaction"):
            continue
        m = re.search(r"_(\d+)$", t)
        sets.append(f'\t\t\t<parameterset type="activityParams">\n'
                    f'\t\t\t\t<param name="activityType" value="{t}"/>\n'
                    f'\t\t\t\t<param name="typicalDuration" value="{_hms(int(m[1]) if m else 3600)}"/>\n'
                    f'\t\t\t</parameterset>')
    return "\n".join(sets)


def write_config(source: Path, target: Path, sample: float, plans: Path, files: Path,
                 types: set[str]) -> None:
    """Open Berlin's config made runnable by plain MATSim (see the module docs)."""
    from mobslim.config.matsim import MatsimConfig

    text = source.read_text(encoding="utf-8")
    # comments may hold markup (Open Berlin's config comments out whole modules)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    marker = '<parameterset type="scoringParameters">'
    if marker not in text:
        raise ValueError("Open Berlin config: no scoringParameters to add activity params to.")
    text = text.replace(marker, marker + "\n" + activity_params(types), 1)
    staged = target.with_name("config.source.xml")
    staged.write_text(text, encoding="utf-8")
    cfg = MatsimConfig.load(staged)
    for module in DROP_MODULES:
        cfg.drop_module(module)
    for param in DROP_QSIM:
        cfg.drop("qsim", param)
    cfg.set("network", "inputNetworkFile", str(files / "network.xml.gz"))
    cfg.set("plans", "inputPlansFile", str(plans))
    cfg.drop("plans", "removingUnnecessaryPlanAttributes")
    cfg.set("transit", "transitScheduleFile", str(files / "transitSchedule.xml.gz"))
    cfg.set("transit", "vehiclesFile", str(files / "transitVehicles.xml.gz"))
    cfg.set("vehicles", "vehiclesFile", str(files / "vehicleTypes.xml"))
    cfg.set("facilities", "inputFacilitiesFile", str(files / "facilities.xml.gz"))
    cfg.set("qsim", "flowCapacityFactor", sample)
    cfg.set("qsim", "storageCapacityFactor", sample)
    cfg.set("controller", "lastIteration", 0)
    cfg.set("controller", "outputDirectory", str(target.parent / "output"))
    cfg.set("controller", "overwriteFiles", "deleteDirectoryIfExists")
    cfg.set("vspExperimental", "vspDefaultsCheckingLevel", "ignore")
    cfg.write(target)
    staged.unlink()


def prepare(out: Path, sample: str, d: Path) -> Path:
    """Case inputs in ``d``: ``plans.xml.gz`` (selected plans) and ``config.xml``."""
    files = download(out, sample)
    d.mkdir(parents=True, exist_ok=True)
    plans = d / "plans.xml.gz"
    types = selected_plans(files / f"plans-{sample}.xml.gz", plans)
    cfg = d / "config.xml"
    write_config(files / "config.xml", cfg, SAMPLES[sample], plans, files, types)
    return cfg
