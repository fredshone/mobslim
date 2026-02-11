import xml.etree.ElementTree as ET
from typing import Hashable, Optional


class Vehicles:
    def __init__(self):
        self._types = {}
        self._vehicles = {}

    def add_type(
        self,
        type_id: str,
        length: float,
        capacity: int,
        description: Optional[str] = None,
    ):
        self._types[type_id] = {
            "description": description,
            "length": length,
            "capacity": capacity,
        }

    def add_vehicle(self, vehicle_id: Hashable, type_id: Hashable):
        self._vehicles[vehicle_id] = type_id

    def get_type(self, type_id: Hashable) -> dict:
        return self._types.get(type_id)

    def get_vehicle(self, vehicle_id: Hashable) -> dict:
        type_id = self._vehicles.get(vehicle_id)
        if type_id is None:
            raise ValueError(f"Vehicle {vehicle_id} not found.")
        vehicle_type = self._types.get(type_id)
        if vehicle_type is None:
            raise ValueError(
                f"Vehicle type {type_id} not found for vehicle {vehicle_id}."
            )
        return vehicle_type

    def load_xml(self, path: str):
        """Load a vehicle schedule from an XML file.

        <?xml version="1.0" encoding="UTF-8"?>
        <vehicleDefinitions xmlns="http://www.matsim.org/files/dtd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.matsim.org/files/dtd http://www.matsim.org/files/dtd/vehicleDefinitions_v1.0.xsd">
        <vehicleType id="1">
        <description>pt</description>
        <capacity>
            <seats persons="50"/>
            <standingRoom persons="100"/>
        </capacity>
        <length meter="12"/>
        </vehicleType>
        <vehicle id="1000" type="1"/>
        </vehicleDefinitions>

        Args:
            path (str): The path to the XML file.
        """

        tree = ET.parse(path)
        root = tree.getroot()

        # Load vehicle types
        for veh_type in root.findall("vehicleType"):
            vehicle_id = veh_type.get("id")
            description = veh_type.find("description").text
            length = float(veh_type.find("length").get("meter"))
            capacity_elem = veh_type.find("capacity")
            seats = int(capacity_elem.find("seats").get("persons"))
            standing = int(capacity_elem.find("standingRoom").get("persons"))
            capacity = seats + standing
            self._types[vehicle_id] = {
                "description": description,
                "length": length,
                "capacity": capacity,
            }

        # Load vehicles
        for vehicle in root.findall("vehicle"):
            vehicle_id = vehicle.get("id")
            vehicle_type = vehicle.get("type")
            self._vehicles[vehicle_id] = vehicle_type
