"""Shared wall detection and connecting door logic for multi-room layouts.

Ported from SAGE server/utils.py (find_shared_walls, find_all_shared_walls,
calculate_room_wall_position_from_shared_wall) and server/llm_client.py
(MST-based connecting door selection).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rooms_to_dicts(rooms) -> List[Dict[str, Any]]:
    """Convert Room dataclasses to dicts with position/dimensions for wall math."""
    out = []
    for r in rooms:
        out.append({
            "id": r.id,
            "room_type": r.room_type,
            "position": {"x": r.position.x, "y": r.position.y, "z": r.position.z},
            "dimensions": {
                "width": r.dimensions.width,
                "length": r.dimensions.length,
                "height": r.dimensions.height,
            },
            "walls": [
                {"id": w.id, "start_point": {"x": w.start_point.x, "y": w.start_point.y, "z": w.start_point.z},
                 "end_point": {"x": w.end_point.x, "y": w.end_point.y, "z": w.end_point.z},
                 "height": w.height}
                for w in r.walls
            ],
        })
    return out


def _wall_side_to_wall_id(room, side: str) -> Optional[str]:
    """Map a cardinal side name ("north", "south", etc.) to the room's wall ID.

    Convention: wall IDs end with ``_n_wall``, ``_s_wall``, ``_e_wall``, ``_w_wall``.
    """
    suffix_map = {"north": "_n_wall", "south": "_s_wall", "east": "_e_wall", "west": "_w_wall"}
    suffix = suffix_map.get(side)
    if suffix is None:
        return None
    for w in room.walls:
        if w.id.endswith(suffix):
            return w.id
    return None


# ---------------------------------------------------------------------------
# Shared wall detection (ported from SAGE utils.py:543-631)
# ---------------------------------------------------------------------------

def find_shared_walls_between_rooms(
    r1: Dict[str, Any], r2: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check 4 opposing wall pairs (N/S, E/W) for overlap within tolerance."""
    pos1, dims1 = r1["position"], r1["dimensions"]
    pos2, dims2 = r2["position"], r2["dimensions"]

    b1 = {
        "x_min": pos1["x"], "x_max": pos1["x"] + dims1["width"],
        "y_min": pos1["y"], "y_max": pos1["y"] + dims1["length"],
    }
    b2 = {
        "x_min": pos2["x"], "x_max": pos2["x"] + dims2["width"],
        "y_min": pos2["y"], "y_max": pos2["y"] + dims2["length"],
    }

    tolerance = 0.01
    shared = []

    # North of r1 ↔ South of r2
    if abs(b1["y_max"] - b2["y_min"]) < tolerance:
        ox_min = max(b1["x_min"], b2["x_min"])
        ox_max = min(b1["x_max"], b2["x_max"])
        if ox_max > ox_min + tolerance:
            shared.append({"room1_wall": "north", "room2_wall": "south",
                           "overlap_start": ox_min, "overlap_end": ox_max,
                           "overlap_length": ox_max - ox_min})

    # South of r1 ↔ North of r2
    if abs(b1["y_min"] - b2["y_max"]) < tolerance:
        ox_min = max(b1["x_min"], b2["x_min"])
        ox_max = min(b1["x_max"], b2["x_max"])
        if ox_max > ox_min + tolerance:
            shared.append({"room1_wall": "south", "room2_wall": "north",
                           "overlap_start": ox_min, "overlap_end": ox_max,
                           "overlap_length": ox_max - ox_min})

    # East of r1 ↔ West of r2
    if abs(b1["x_max"] - b2["x_min"]) < tolerance:
        oy_min = max(b1["y_min"], b2["y_min"])
        oy_max = min(b1["y_max"], b2["y_max"])
        if oy_max > oy_min + tolerance:
            shared.append({"room1_wall": "east", "room2_wall": "west",
                           "overlap_start": oy_min, "overlap_end": oy_max,
                           "overlap_length": oy_max - oy_min})

    # West of r1 ↔ East of r2
    if abs(b1["x_min"] - b2["x_max"]) < tolerance:
        oy_min = max(b1["y_min"], b2["y_min"])
        oy_max = min(b1["y_max"], b2["y_max"])
        if oy_max > oy_min + tolerance:
            shared.append({"room1_wall": "west", "room2_wall": "east",
                           "overlap_start": oy_min, "overlap_end": oy_max,
                           "overlap_length": oy_max - oy_min})

    return shared


# ---------------------------------------------------------------------------
# All shared walls classification (ported from SAGE utils.py:2241-2439)
# ---------------------------------------------------------------------------

def find_all_shared_walls(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify every wall segment as room-room or room-exterior.

    Returns ``{"room_room_walls": [...], "room_exterior_walls": [...],
               "total_room_room_walls": int, "total_room_exterior_walls": int}``.
    """
    room_room_walls: List[Dict[str, Any]] = []

    for i in range(len(rooms_data)):
        for j in range(i + 1, len(rooms_data)):
            r1, r2 = rooms_data[i], rooms_data[j]
            for sw in find_shared_walls_between_rooms(r1, r2):
                info: Dict[str, Any] = {
                    "type": "room_room",
                    "room1": {"index": i, "id": r1.get("id", f"room_{i}"),
                              "room_type": r1["room_type"]},
                    "room2": {"index": j, "id": r2.get("id", f"room_{j}"),
                              "room_type": r2["room_type"]},
                    "room1_wall": sw["room1_wall"],
                    "room2_wall": sw["room2_wall"],
                    "direction": "x" if sw["room1_wall"] in ("north", "south") else "y",
                    "overlap_length": sw["overlap_length"],
                }
                if sw["room1_wall"] in ("north", "south"):
                    info["x_start"] = sw["overlap_start"]
                    info["x_end"] = sw["overlap_end"]
                    if sw["room1_wall"] == "north":
                        y = r1["position"]["y"] + r1["dimensions"]["length"]
                    else:
                        y = r1["position"]["y"]
                    info["y_start"] = y
                    info["y_end"] = y
                else:
                    info["y_start"] = sw["overlap_start"]
                    info["y_end"] = sw["overlap_end"]
                    if sw["room1_wall"] == "east":
                        x = r1["position"]["x"] + r1["dimensions"]["width"]
                    else:
                        x = r1["position"]["x"]
                    info["x_start"] = x
                    info["x_end"] = x
                room_room_walls.append(info)

    # --- Exterior wall segments (wall parts NOT shared with any room) ---
    room_exterior_walls: List[Dict[str, Any]] = []

    for i, room in enumerate(rooms_data):
        rp, rd = room["position"], room["dimensions"]
        walls_def = {
            "north": {"x_start": rp["x"], "x_end": rp["x"] + rd["width"],
                       "y_start": rp["y"] + rd["length"], "y_end": rp["y"] + rd["length"],
                       "direction": "x"},
            "south": {"x_start": rp["x"], "x_end": rp["x"] + rd["width"],
                       "y_start": rp["y"], "y_end": rp["y"],
                       "direction": "x"},
            "east":  {"x_start": rp["x"] + rd["width"], "x_end": rp["x"] + rd["width"],
                       "y_start": rp["y"], "y_end": rp["y"] + rd["length"],
                       "direction": "y"},
            "west":  {"x_start": rp["x"], "x_end": rp["x"],
                       "y_start": rp["y"], "y_end": rp["y"] + rd["length"],
                       "direction": "y"},
        }

        for side, wc in walls_def.items():
            # Collect shared segments on this wall
            segs = []
            for sw in room_room_walls:
                if (sw["room1"]["index"] == i and sw["room1_wall"] == side) or \
                   (sw["room2"]["index"] == i and sw["room2_wall"] == side):
                    if wc["direction"] == "x":
                        segs.append({"start": sw["x_start"], "end": sw["x_end"]})
                    else:
                        segs.append({"start": sw["y_start"], "end": sw["y_end"]})
            segs.sort(key=lambda s: s["start"])

            wall_start = wc["x_start"] if wc["direction"] == "x" else wc["y_start"]
            wall_end = wc["x_end"] if wc["direction"] == "x" else wc["y_end"]

            exterior = []
            cur = wall_start
            for seg in segs:
                if cur < seg["start"]:
                    exterior.append({"start": cur, "end": seg["start"]})
                cur = max(cur, seg["end"])
            if not segs:
                exterior.append({"start": wall_start, "end": wall_end})
            elif cur < wall_end:
                exterior.append({"start": cur, "end": wall_end})

            for ext in exterior:
                length = ext["end"] - ext["start"]
                if length <= 0.1:
                    continue
                ei: Dict[str, Any] = {
                    "type": "room_exterior",
                    "room": {"index": i, "id": room.get("id", f"room_{i}"),
                             "room_type": room["room_type"]},
                    "wall_side": side,
                    "direction": wc["direction"],
                    "overlap_length": length,
                }
                if wc["direction"] == "x":
                    ei["x_start"] = ext["start"]
                    ei["x_end"] = ext["end"]
                    ei["y_start"] = wc["y_start"]
                    ei["y_end"] = wc["y_end"]
                else:
                    ei["x_start"] = wc["x_start"]
                    ei["x_end"] = wc["x_end"]
                    ei["y_start"] = ext["start"]
                    ei["y_end"] = ext["end"]
                room_exterior_walls.append(ei)

    return {
        "room_room_walls": room_room_walls,
        "room_exterior_walls": room_exterior_walls,
        "total_room_room_walls": len(room_room_walls),
        "total_room_exterior_walls": len(room_exterior_walls),
    }


# ---------------------------------------------------------------------------
# Position mapping (ported from SAGE utils.py:2630-2679)
# ---------------------------------------------------------------------------

def calculate_room_wall_position_from_shared_wall(
    shared_wall_info: Dict[str, Any],
    shared_wall_position: float,
    room_data: Dict[str, Any],
    room_is_room1: bool,
) -> float:
    """Map a 0-1 position on a shared wall segment to 0-1 on the room's full wall."""
    if shared_wall_info["direction"] == "x":
        shared_start = shared_wall_info["x_start"]
        shared_end = shared_wall_info["x_end"]
    else:
        shared_start = shared_wall_info["y_start"]
        shared_end = shared_wall_info["y_end"]

    world_pos = shared_start + (shared_end - shared_start) * shared_wall_position
    wall_side = shared_wall_info["room1_wall"] if room_is_room1 else shared_wall_info["room2_wall"]

    rp, rd = room_data["position"], room_data["dimensions"]
    if wall_side in ("north", "south"):
        room_wall_pos = (world_pos - rp["x"]) / rd["width"]
    else:
        room_wall_pos = (world_pos - rp["y"]) / rd["length"]

    return max(0.0, min(1.0, room_wall_pos))


# ---------------------------------------------------------------------------
# MST-based connecting doors (ported from SAGE llm_client.py)
# ---------------------------------------------------------------------------

def compute_connecting_doors(
    rooms_data: List[Dict[str, Any]],
    shared_walls: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Use a minimum spanning tree to pick the minimum set of connecting doors.

    Returns a list of door specs, each with ``shared_wall_index``,
    ``center_position_on_shared_wall`` (0.5), default ``width``/``height``,
    and ``adjacent_room_types``.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed — returning one door per shared wall")
        return [
            {
                "shared_wall_index": i,
                "center_position_on_shared_wall": 0.5,
                "width": min(0.9, sw["overlap_length"] * 0.8),
                "height": 2.1,
                "door_type": "connecting",
                "opening": False,
                "adjacent_room_types": [sw["room1"]["room_type"], sw["room2"]["room_type"]],
            }
            for i, sw in enumerate(shared_walls["room_room_walls"])
        ]

    rr_walls = shared_walls["room_room_walls"]
    if not rr_walls:
        return []

    G = nx.Graph()
    for i in range(len(rooms_data)):
        G.add_node(i)

    for idx, sw in enumerate(rr_walls):
        r1, r2 = sw["room1"]["index"], sw["room2"]["index"]
        weight = 1.0 / max(sw["overlap_length"], 0.01)
        if not G.has_edge(r1, r2) or G[r1][r2]["weight"] > weight:
            G.add_edge(r1, r2, weight=weight, wall_index=idx)

    if nx.is_connected(G):
        mst = nx.minimum_spanning_tree(G, weight="weight")
    else:
        components = list(nx.connected_components(G))
        mst = nx.Graph()
        for comp in components:
            sub = G.subgraph(comp)
            if len(comp) > 1:
                mst = nx.union(mst, nx.minimum_spanning_tree(sub, weight="weight"))
            else:
                mst.add_node(list(comp)[0])
        comp_list = [list(c) for c in components]
        for ci in range(len(comp_list) - 1):
            best_d, best_e = float("inf"), None
            for a in comp_list[ci]:
                for b in comp_list[ci + 1]:
                    p1, p2 = rooms_data[a]["position"], rooms_data[b]["position"]
                    d = ((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_e = d, (a, b)
            if best_e:
                mst.add_edge(*best_e, weight=best_d, wall_index=-1)

    doors: List[Dict[str, Any]] = []
    for u, v, data in mst.edges(data=True):
        wi = data.get("wall_index", -1)
        if wi < 0 or wi >= len(rr_walls):
            continue
        sw = rr_walls[wi]
        doors.append({
            "shared_wall_index": wi,
            "center_position_on_shared_wall": 0.5,
            "width": min(0.9, sw["overlap_length"] * 0.8),
            "height": 2.1,
            "door_type": "connecting",
            "opening": False,
            "adjacent_room_types": [sw["room1"]["room_type"], sw["room2"]["room_type"]],
        })

    return doors
