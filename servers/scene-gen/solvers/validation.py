"""Layout and door/window validation.

Adapted from SAGE server/validation.py.
Validates room layouts for overlaps, connectivity, and door/window placement integrity.
"""

from typing import Dict, List, Any


def validate_room_only_layout(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate room layout for overlaps and connectivity (no doors/windows).

    Args:
        rooms_data: List of room dicts with "position" and "dimensions" keys.

    Returns:
        {"valid": bool, "issues": [...], "overlaps": [...], "detached_rooms": [...]}
    """
    issues = []
    overlaps = []
    detached_rooms = []

    for i, room1 in enumerate(rooms_data):
        pos1, dims1 = room1["position"], room1["dimensions"]
        x1_min, y1_min = pos1["x"], pos1["y"]
        x1_max = x1_min + dims1["width"]
        y1_max = y1_min + dims1["length"]

        for j, room2 in enumerate(rooms_data[i + 1 :], i + 1):
            pos2, dims2 = room2["position"], room2["dimensions"]
            x2_min, y2_min = pos2["x"], pos2["y"]
            x2_max = x2_min + dims2["width"]
            y2_max = y2_min + dims2["length"]

            if not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min):
                area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(
                    0, min(y1_max, y2_max) - max(y1_min, y2_min)
                )
                if area > 0.01:
                    overlaps.append({
                        "room1": room1["room_type"],
                        "room2": room2["room_type"],
                        "overlap_area": area,
                    })

    # Check connectivity (shared walls)
    if len(rooms_data) > 1:
        for i, room1 in enumerate(rooms_data):
            pos1, dims1 = room1["position"], room1["dimensions"]
            x1_min, y1_min = pos1["x"], pos1["y"]
            x1_max = x1_min + dims1["width"]
            y1_max = y1_min + dims1["length"]

            has_shared_wall = False
            for j, room2 in enumerate(rooms_data):
                if i == j:
                    continue
                pos2, dims2 = room2["position"], room2["dimensions"]
                x2_min, y2_min = pos2["x"], pos2["y"]
                x2_max = x2_min + dims2["width"]
                y2_max = y2_min + dims2["length"]
                tol = 0.1
                # Vertical shared wall
                if abs(x1_max - x2_min) < tol or abs(x2_max - x1_min) < tol:
                    if not (y1_max <= y2_min + tol or y2_max <= y1_min + tol):
                        has_shared_wall = True
                        break
                # Horizontal shared wall
                if abs(y1_max - y2_min) < tol or abs(y2_max - y1_min) < tol:
                    if not (x1_max <= x2_min + tol or x2_max <= x1_min + tol):
                        has_shared_wall = True
                        break
            if not has_shared_wall:
                detached_rooms.append(room1["room_type"])

    if overlaps:
        issues.append("Room overlaps detected")
    if detached_rooms:
        issues.append(f"Detached rooms (no shared walls): {detached_rooms}")

    return {"valid": len(issues) == 0, "issues": issues, "overlaps": overlaps, "detached_rooms": detached_rooms}


def validate_room_layout(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Full validation including doors and windows."""
    result = validate_room_only_layout(rooms_data)
    issues = list(result["issues"])
    door_issues = []
    window_issues = []

    for room in rooms_data:
        room_type = room["room_type"]
        doors = room.get("doors", [])
        windows = room.get("windows", [])

        # Door validation
        if not doors and len(rooms_data) > 1:
            if room_type.lower() not in ("utility room", "storage", "mechanical room"):
                door_issues.append(f"'{room_type}' has no doors — inaccessible room")
        for idx, door in enumerate(doors):
            ws = door.get("wall_side", "")
            if not ws:
                door_issues.append(f"'{room_type}' door {idx} has undefined wall_side")
            pos = door.get("position_on_wall", -1)
            if pos < 0 or pos > 1:
                door_issues.append(f"'{room_type}' door {idx} has invalid position_on_wall: {pos}")

        # Window validation
        for idx, window in enumerate(windows):
            ws = window.get("wall_side", "")
            if not ws:
                window_issues.append(f"'{room_type}' window {idx} has undefined wall_side")
            pos = window.get("position_on_wall", -1)
            if pos < 0 or pos > 1:
                window_issues.append(f"'{room_type}' window {idx} has invalid position_on_wall: {pos}")
            sill = window.get("sill_height", 0)
            if sill < 0.3 or sill > 1.8:
                window_issues.append(f"'{room_type}' window {idx} has unusual sill_height: {sill}m")

    if door_issues:
        issues.append(f"Door issues: {door_issues}")
    if window_issues:
        issues.append(f"Window issues: {window_issues}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "overlaps": result["overlaps"],
        "detached_rooms": result["detached_rooms"],
        "door_issues": door_issues,
        "window_issues": window_issues,
    }


def validate_llm_response_structure(llm_response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that an LLM-generated layout response has all required fields."""
    missing = []
    invalid = []

    if "building_style" not in llm_response:
        missing.append("building_style")
    if "rooms" not in llm_response:
        missing.append("rooms")
        return {"valid": False, "missing_fields": missing, "invalid_fields": invalid, "error": "'rooms' missing"}
    if not isinstance(llm_response["rooms"], list) or len(llm_response["rooms"]) == 0:
        invalid.append("rooms (must be non-empty list)")
        return {"valid": False, "missing_fields": missing, "invalid_fields": invalid, "error": "'rooms' invalid"}

    for i, room in enumerate(llm_response["rooms"]):
        prefix = f"room[{i}]"
        for field in ("room_type", "dimensions", "position"):
            if field not in room:
                missing.append(f"{prefix}.{field}")
        if "dimensions" in room and isinstance(room["dimensions"], dict):
            for df in ("width", "length", "height"):
                if df not in room["dimensions"]:
                    missing.append(f"{prefix}.dimensions.{df}")
        if "position" in room and isinstance(room["position"], dict):
            for pf in ("x", "y", "z"):
                if pf not in room["position"]:
                    missing.append(f"{prefix}.position.{pf}")

    return {"valid": not missing and not invalid, "missing_fields": missing, "invalid_fields": invalid, "error": None}
