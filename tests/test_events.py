import gzip

from mobslim import Scenario
from mobslim.events import (
    ActivityEnd,
    EventHandler,
    EventLog,
    EventsManager,
    EventsXMLWriter,
    LinkEnter,
    LinkLeave,
)
from mobslim.mobsim import QSim
from tests.conftest import chain_network, commuters, routed, simulate


class LinkCounter(EventHandler):
    handles = (LinkEnter,)

    def __init__(self):
        self.n = 0

    def handle(self, event):
        self.n += 1


def test_dispatch_by_type():
    counter, log = LinkCounter(), EventLog()
    em = EventsManager([counter, log])
    em.process_many([LinkEnter(0, 0, 1), LinkLeave(1, 0, 1), ActivityEnd(2, 0, "h", 0)])
    assert counter.n == 1
    assert len(log.events) == 3
    assert log.of_type(LinkLeave) == [LinkLeave(1, 0, 1)]


def test_event_log_dataframe():
    log = EventLog()
    log.handle(LinkEnter(5.0, 3, 7))
    df = log.to_dataframe()
    assert df.iloc[0]["type"] == "entered link" and df.iloc[0]["link"] == 7


def test_events_xml_writer(tmp_path):
    net = chain_network([100, 100])
    scenario = routed(Scenario(net, commuters(2, 0, 2)))
    path = tmp_path / "events.xml.gz"
    writer = EventsXMLWriter(path, net, scenario.population)
    QSim().run(scenario, EventsManager([writer]))
    writer.finish()
    text = gzip.open(path, "rt").read()
    assert text.startswith("<?xml")
    assert 'type="entered link"' in text and 'link="0-1"' in text
    assert text.rstrip().endswith("</events>")


def test_mobsim_event_sequence():
    net = chain_network([100])
    events = simulate(routed(Scenario(net, commuters(1, 0, 1, dep=5))))
    assert [e.matsim_type for e in events] == [
        "actend", "departure", "entered link", "left link", "arrival", "actstart"
    ]


def test_custom_event_types_are_dispatched():
    from dataclasses import dataclass

    from mobslim.events import PersonEvent

    @dataclass(slots=True)
    class Boarding(PersonEvent):  # Event has only a time; PersonEvent adds the person
        stop: int

    @dataclass(slots=True)
    class LinkEnterVariant(LinkEnter):  # a subclass of a built-in type
        pass

    counter, log = LinkCounter(), EventLog()
    events = EventsManager([counter, log])
    events.process(Boarding(1.0, 0, 7))
    events.process(LinkEnterVariant(2.0, 0, 3))
    assert [type(e).__name__ for e in log.events] == ["Boarding", "LinkEnterVariant"]
    assert counter.n == 1  # handles=(LinkEnter,) also receives subclasses


def test_handlers_for_stays_live():
    events = EventsManager()
    fns = events.handlers_for(LinkEnter)
    counter = events.add_handler(LinkCounter())
    for fn in fns:
        fn(LinkEnter(0.0, 0, 0))
    assert counter.n == 1
