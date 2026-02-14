"""Data models for scene generation, adapted from SAGE server/models.py."""

from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict
import json


@dataclass
class Point3D:
    """Represents a 3D coordinate point."""
    x: float
    y: float
    z: float


@dataclass
class Euler:
    """Represents a 3D rotation in Euler angles (x, y, z) in degrees."""
    x: float
    y: float
    z: float


@dataclass
class Dimensions:
    """Represents 3D dimensions."""
    width: float
    length: float
    height: float


@dataclass
class Wall:
    """Represents a wall in the room."""
    id: str
    start_point: Point3D
    end_point: Point3D
    height: float
    thickness: float = 0.1
    material: str = "drywall"


@dataclass
class Window:
    """Represents a window on a wall."""
    id: str
    wall_id: str
    position_on_wall: float  # 0-1, position along the wall
    width: float
    height: float
    sill_height: float  # height from floor to window sill
    window_type: str = "standard"
    window_material: str = "standard"


@dataclass
class Door:
    """Represents a door on a wall."""
    id: str
    wall_id: str
    position_on_wall: float  # 0-1, position along the wall
    width: float
    height: float
    door_type: str = "standard"
    opens_inward: bool = True
    opening: bool = False  # permanent opening without actual door
    door_material: str = "standard"


@dataclass
class Object:
    """Represents an object/furniture item in a room."""
    id: str
    room_id: str
    type: str
    description: str
    position: Point3D
    rotation: Euler
    dimensions: Dimensions
    source: str  # "objaverse", "generation", etc.
    source_id: str
    place_id: str  # wall_id, room_id, or another object_id
    place_location: str = "top"  # "top", "inside", or "both"
    place_guidance: str = "Standard placement for the object"
    placement_constraints: List[Dict] = None
    mass: float = 1.0

    def __post_init__(self):
        if self.placement_constraints is None:
            self.placement_constraints = []


@dataclass
class Room:
    """Represents a room in the layout."""
    id: str
    room_type: str
    position: Point3D
    dimensions: Dimensions
    walls: List[Wall]
    doors: List[Door]
    objects: List[Object]
    windows: List[Window]
    floor_material: str = "hardwood"
    ceiling_height: float = 2.7


@dataclass
class FloorPlan:
    """Represents the complete floor plan layout."""
    id: str
    rooms: List[Room]
    total_area: float
    building_style: str
    description: str
    created_from_text: str
    policy_analysis: Dict = None

    def __post_init__(self):
        if self.policy_analysis is None:
            self.policy_analysis = {}


def floor_plan_to_dict(fp: FloorPlan) -> dict:
    """Convert a FloorPlan to a JSON-serializable dict."""
    return asdict(fp)


def floor_plan_to_json(fp: FloorPlan) -> str:
    """Convert a FloorPlan to a JSON string."""
    return json.dumps(asdict(fp), indent=2)


def dict_to_point3d(d: dict) -> Point3D:
    return Point3D(x=d["x"], y=d["y"], z=d.get("z", 0.0))


def dict_to_euler(d: dict) -> Euler:
    return Euler(x=d.get("x", 0.0), y=d.get("y", 0.0), z=d.get("z", 0.0))


def dict_to_dimensions(d: dict) -> Dimensions:
    return Dimensions(width=d["width"], length=d["length"], height=d["height"])


def dict_to_wall(d: dict) -> Wall:
    return Wall(
        id=d["id"],
        start_point=dict_to_point3d(d["start_point"]),
        end_point=dict_to_point3d(d["end_point"]),
        height=d["height"],
        thickness=d.get("thickness", 0.1),
        material=d.get("material", "drywall"),
    )


def dict_to_window(d: dict) -> Window:
    return Window(
        id=d["id"],
        wall_id=d["wall_id"],
        position_on_wall=d["position_on_wall"],
        width=d["width"],
        height=d["height"],
        sill_height=d["sill_height"],
        window_type=d.get("window_type", "standard"),
        window_material=d.get("window_material", "standard"),
    )


def dict_to_door(d: dict) -> Door:
    return Door(
        id=d["id"],
        wall_id=d["wall_id"],
        position_on_wall=d["position_on_wall"],
        width=d["width"],
        height=d["height"],
        door_type=d.get("door_type", "standard"),
        opens_inward=d.get("opens_inward", True),
        opening=d.get("opening", False),
        door_material=d.get("door_material", "standard"),
    )


def dict_to_object(d: dict) -> Object:
    return Object(
        id=d["id"],
        room_id=d["room_id"],
        type=d["type"],
        description=d.get("description", ""),
        position=dict_to_point3d(d["position"]),
        rotation=dict_to_euler(d["rotation"]),
        dimensions=dict_to_dimensions(d["dimensions"]),
        source=d.get("source", "unknown"),
        source_id=d.get("source_id", ""),
        place_id=d.get("place_id", "floor"),
        place_location=d.get("place_location", "top"),
        place_guidance=d.get("place_guidance", "Standard placement for the object"),
        placement_constraints=d.get("placement_constraints", []),
        mass=d.get("mass", 1.0),
    )


def dict_to_room(d: dict) -> Room:
    return Room(
        id=d["id"],
        room_type=d["room_type"],
        position=dict_to_point3d(d["position"]),
        dimensions=dict_to_dimensions(d["dimensions"]),
        walls=[dict_to_wall(w) for w in d.get("walls", [])],
        doors=[dict_to_door(dr) for dr in d.get("doors", [])],
        objects=[dict_to_object(o) for o in d.get("objects", [])],
        windows=[dict_to_window(w) for w in d.get("windows", [])],
        floor_material=d.get("floor_material", "hardwood"),
        ceiling_height=d.get("ceiling_height", 2.7),
    )


def dict_to_floor_plan(d: dict) -> FloorPlan:
    return FloorPlan(
        id=d["id"],
        rooms=[dict_to_room(r) for r in d["rooms"]],
        total_area=d.get("total_area", 0.0),
        building_style=d.get("building_style", ""),
        description=d.get("description", ""),
        created_from_text=d.get("created_from_text", ""),
        policy_analysis=d.get("policy_analysis", {}),
    )
