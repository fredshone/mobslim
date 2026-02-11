import xml.etree.ElementTree as ET

from fastquadtree import QuadTreeObjects

from mobslim.entities.vehicles import Vehicles
from mobslim.utils import string_to_seconds as tts


class Schedules:

    def __init__(self, vehicles: Vehicles):
        self._vehicles = vehicles
        self.stop_locations = {}
        self.stop_links = {}
        self.lines = {}

    def load_xml(self, path: str):
        """Load a network from an XML file.

        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE transitSchedule SYSTEM "http://www.matsim.org/files/dtd/transitSchedule_v1.dtd">
        <transitSchedule>
        <transitStops>
        <stopFacility id="192" x="05" y="0" linkRefId="1-2"/>
        <stopFacility id="293" x="15" y="0" linkRefId="2-3"/>
        <stopFacility id="495" x="2005" y="0" linkRefId="4-5"/>
        <stopFacility id="596" x="2015" y="0" linkRefId="5-6"/>
        </transitStops>
        <transitLine id="gelb">
        <transitRoute id="gelb_1">
            <transportMode>pt</transportMode>
            <routeProfile>
            <stop refId="192" departureOffset="00:00:00" arrivalOffset="00:00:00" awaitDeparture="true"/>
            <stop refId="293" departureOffset="00:00:30" arrivalOffset="00:00:30" awaitDeparture="true"/>
            <stop refId="495" departureOffset="00:30:00" arrivalOffset="00:30:00" awaitDeparture="true"/>
            <stop refId="596" departureOffset="00:30:30" arrivalOffset="00:30:30" awaitDeparture="true"/>
            </routeProfile>
            <route>
            <link refId="1-2"/>
            <link refId="2-3"/>
            <link refId="3-4"/>
            <link refId="4-5"/>
            <link refId="5-6"/>
            </route>
            <departures>
            <departure id="1000" departureTime="05:05:00" vehicleRefId="1000"/>
            </departures>
        </transitRoute>
        </transitLine>
        </transitSchedule>

        Args:
            path (str): The path to the XML file.
        """

        tree = ET.parse(path)
        root = tree.getroot()

        # Load stops
        for node in root.find("transitStops"):
            node_id = node.get("id")
            x = float(node.get("x"))
            y = float(node.get("y"))
            link_id = node.get("linkRefId")
            self.stop_locations[node_id] = (x, y)
            self.stop_links[node_id] = link_id

        # Load lines
        for line in root.findall("transitLine"):
            line_id = line.get("id")
            self.lines[line_id] = {}
            for route in line.findall("transitRoute"):
                route_id = route.get("id")
                network_mode = route.find("transportMode").text
                route_stops = []
                for stop in route.find("routeProfile").findall("stop"):
                    ref_id = stop.get("refId")
                    departure_offset = tts(stop.get("departureOffset"))
                    arrival_offset = tts(stop.get("arrivalOffset"))
                    await_departure = stop.get("awaitDeparture") == "true"
                    route_stops.append(
                        {
                            "ref_id": ref_id,
                            "departure_offset": departure_offset,
                            "arrival_offset": arrival_offset,
                            "await_departure": await_departure,
                        }
                    )
                route_links = [
                    link.get("refId")
                    for link in route.find("route").findall("link")
                ]
                departures = []
                for departure in route.find("departures").findall("departure"):
                    dep_id = departure.get("id")
                    departure_time = tts(departure.get("departureTime"))
                    vehicle_ref_id = departure.get("vehicleRefId")
                    departures.append(
                        {
                            "id": dep_id,
                            "departure_time": departure_time,
                            "vehicle_ref_id": vehicle_ref_id,
                        }
                    )
                self.lines[line_id][route_id] = {
                    "transport_mode": network_mode,
                    "route_stops": route_stops,
                    "route_links": route_links,
                    "departures": departures,
                }

    def stop_quad_tree(self) -> QuadTreeObjects:
        """Get a quad tree of the stop locations.

        Returns:
            QuadTree: A quad tree of the stop locations.
        """
        xs, ys = zip(*self.stop_locations.values())
        bb = (min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1)
        qt = QuadTreeObjects(bb, capacity=len(xs))
        locs = list(self.stop_locations.values())
        ids = list(self.stop_locations.keys())
        qt.insert_many(locs, ids)
        return qt
