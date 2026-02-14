"""Scene state management — holds FloorPlan layouts in memory."""

from typing import Dict, Optional
import uuid

from models import FloorPlan, Room, Object, dict_to_floor_plan, floor_plan_to_dict


class SceneState:
    """In-memory store for active floor plan layouts."""

    def __init__(self):
        self._layouts: Dict[str, FloorPlan] = {}

    def create_layout(self, floor_plan: FloorPlan) -> str:
        """Store a new layout and return its ID."""
        if not floor_plan.id:
            floor_plan.id = str(uuid.uuid4())[:8]
        self._layouts[floor_plan.id] = floor_plan
        return floor_plan.id

    def get_layout(self, layout_id: str) -> Optional[FloorPlan]:
        return self._layouts.get(layout_id)

    def get_layout_dict(self, layout_id: str) -> Optional[dict]:
        fp = self.get_layout(layout_id)
        if fp is None:
            return None
        return floor_plan_to_dict(fp)

    def get_room(self, layout_id: str, room_id: str) -> Optional[Room]:
        fp = self.get_layout(layout_id)
        if fp is None:
            return None
        for room in fp.rooms:
            if room.id == room_id:
                return room
        return None

    def update_room(self, layout_id: str, room: Room) -> bool:
        fp = self.get_layout(layout_id)
        if fp is None:
            return False
        for i, r in enumerate(fp.rooms):
            if r.id == room.id:
                fp.rooms[i] = room
                return True
        return False

    def add_objects_to_room(self, layout_id: str, room_id: str, objects: list[Object]) -> bool:
        room = self.get_room(layout_id, room_id)
        if room is None:
            return False
        room.objects.extend(objects)
        return True

    def remove_objects_from_room(self, layout_id: str, room_id: str, object_ids: list[str]) -> list[str]:
        """Remove objects by ID and return the IDs that were actually removed."""
        room = self.get_room(layout_id, room_id)
        if room is None:
            return []
        ids_set = set(object_ids)
        removed = [o.id for o in room.objects if o.id in ids_set]
        room.objects = [o for o in room.objects if o.id not in ids_set]
        return removed

    def list_layouts(self) -> list[str]:
        return list(self._layouts.keys())

    def delete_layout(self, layout_id: str) -> bool:
        if layout_id in self._layouts:
            del self._layouts[layout_id]
            return True
        return False
