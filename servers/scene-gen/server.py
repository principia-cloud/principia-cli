"""Scene Generation MCP Server for principia-cli.

Exposes tools for room creation, object placement, material generation,
vision analysis (Qwen3-VL), physics simulation (Isaac Sim), and USD export.
"""

import asyncio
import json
import os
import sys
import uuid
import logging
import tempfile
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Ensure package root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    FloorPlan, Room, Object, Wall, Door, Window,
    Point3D, Euler, Dimensions,
    dict_to_floor_plan, dict_to_room, dict_to_object, dict_to_door, dict_to_window,
    floor_plan_to_dict,
)
from state import SceneState
from solvers.validation import validate_room_only_layout, validate_room_layout, validate_llm_response_structure
from solvers.room_solver import RectangleContactSolver, RectangleContactRelaxationSolver, RectangleSpec
from solvers.placement_solver import (
    place_floor_objects_dfs, place_wall_objects, place_on_object_objects,
    parse_constraints_from_json,
)
from rendering.room_render import render_floor_plan, render_room
from services.qwen_vl import call_qwen_vl
from services.trellis import TrellisClient
from services.mesh_processing import process_generated_model, save_processed_mesh
from services.isaac_sim import IsaacConnection, create_scene, simulate_physics, export_usd
from services.materials import init_materials, generate_room_materials
from services.scene_export import export_room_scene, export_layout_scene
from services.shared_walls import (
    rooms_to_dicts, find_all_shared_walls, compute_connecting_doors,
    calculate_room_wall_position_from_shared_wall, _wall_side_to_wall_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("scene-gen")

RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(tempfile.gettempdir(), "scene-gen-results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Globals ---
state = SceneState()
mcp = FastMCP("scene-gen")

# Lazy-initialized external connections
_isaac: Optional[IsaacConnection] = None
_trellis: Optional[TrellisClient] = None


def _get_isaac() -> IsaacConnection:
    global _isaac
    if _isaac is None:
        _isaac = IsaacConnection()
    if not _isaac._sock:
        _isaac.connect()
    return _isaac


def _get_trellis() -> TrellisClient:
    global _trellis
    if _trellis is None:
        _trellis = TrellisClient()
    return _trellis


# ============================================================
# Scene Management Tools
# ============================================================

@mcp.tool()
def create_room(room_json: str) -> str:
    """Create a validated room layout from JSON produced by a subagent.

    Runs structural validation (overlaps, connectivity) and stores the layout.
    Input should be a JSON string with keys: building_style, description, rooms[].

    Returns a JSON string with layout_id and validation results.
    """
    try:
        data = json.loads(room_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    # Validate structure
    struct_check = validate_llm_response_structure(data)
    if not struct_check["valid"]:
        return json.dumps({"error": "Invalid structure", "details": struct_check})

    rooms_data = data["rooms"]

    # Validate room layout (overlaps, connectivity)
    layout_check = validate_room_only_layout(rooms_data)

    # Build walls for each room
    for room_data in rooms_data:
        if "walls" not in room_data or not room_data["walls"]:
            room_data["walls"] = _generate_walls(room_data)
        if "objects" not in room_data:
            room_data["objects"] = []
        if "doors" not in room_data:
            room_data["doors"] = []
        if "windows" not in room_data:
            room_data["windows"] = []

    layout_id = str(uuid.uuid4())[:8]
    fp = FloorPlan(
        id=layout_id,
        rooms=[dict_to_room(r) for r in rooms_data],
        total_area=sum(r["dimensions"]["width"] * r["dimensions"]["length"] for r in rooms_data),
        building_style=data.get("building_style", ""),
        description=data.get("description", ""),
        created_from_text=data.get("created_from_text", ""),
    )
    state.create_layout(fp)

    result = {
        "layout_id": layout_id,
        "validation": layout_check,
        "rooms": [
            {
                "id": r.id,
                "room_type": r.room_type,
                "wall_ids": [w.id for w in r.walls],
            }
            for r in fp.rooms
        ],
    }

    # Multi-room: include shared wall analysis
    if len(fp.rooms) > 1:
        rooms_dicts = rooms_to_dicts(fp.rooms)
        result["shared_walls"] = find_all_shared_walls(rooms_dicts)

    return json.dumps(result)


@mcp.tool()
def add_doors_windows(layout_id: str, placement_json: str) -> str:
    """Add doors and windows to an existing layout.

    placement_json should be a JSON string with:
    {
      "room_id": "...",
      "doors": [{"wall_id": "...", "position_on_wall": 0.5, "width": 0.9, "height": 2.1, ...}],
      "windows": [{"wall_id": "...", "position_on_wall": 0.5, "width": 1.2, "height": 1.0, "sill_height": 0.9, ...}]
    }

    Returns validation results.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})

    try:
        data = json.loads(placement_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    room_id = data.get("room_id")
    room = state.get_room(layout_id, room_id)
    if room is None:
        return json.dumps({"error": f"Room {room_id} not found in layout {layout_id}"})

    wall_ids = [w.id for w in room.walls]

    def _resolve_wall_id(raw_id: str) -> str:
        """Resolve wall_id by exact match or substring (SAGE-style)."""
        if raw_id in wall_ids:
            return raw_id
        # Substring match: "south" matches "room_01_s_wall", "s_wall" matches too
        for wid in wall_ids:
            if raw_id in wid:
                return wid
        return raw_id

    # Add doors
    invalid = []
    for d in data.get("doors", []):
        d.setdefault("id", f"door_{uuid.uuid4().hex[:6]}")
        d.setdefault("door_type", "standard")
        d.setdefault("opens_inward", True)
        d.setdefault("opening", False)
        d.setdefault("door_material", "standard")
        d["wall_id"] = _resolve_wall_id(d["wall_id"])
        if d["wall_id"] not in wall_ids:
            invalid.append(f"door wall_id '{d['wall_id']}'")
            continue
        room.doors.append(dict_to_door(d))

    # Add windows
    for w in data.get("windows", []):
        w.setdefault("id", f"window_{uuid.uuid4().hex[:6]}")
        w.setdefault("window_type", "standard")
        w.setdefault("window_material", "standard")
        w["wall_id"] = _resolve_wall_id(w["wall_id"])
        if w["wall_id"] not in wall_ids:
            invalid.append(f"window wall_id '{w['wall_id']}'")
            continue
        room.windows.append(dict_to_window(w))

    if invalid:
        return json.dumps({
            "error": f"Invalid wall_id(s): {invalid}. Available wall IDs: {wall_ids}",
        })

    # Validate the updated layout
    rooms_dicts = [_room_to_validation_dict(r) for r in fp.rooms]
    validation = validate_room_layout(rooms_dicts)

    return json.dumps({
        "layout_id": layout_id,
        "room_id": room_id,
        "doors_added": len(data.get("doors", [])),
        "windows_added": len(data.get("windows", [])),
        "validation": validation,
        "available_wall_ids": wall_ids,
    })


@mcp.tool()
def analyze_shared_walls(layout_id: str) -> str:
    """Analyze shared walls between rooms and suggest connecting doors.

    Returns shared wall classification (room-room vs room-exterior) and
    MST-based connecting door suggestions. Use the returned shared_wall_index
    values when calling add_connecting_doors.

    Only useful for layouts with 2+ rooms.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})

    if len(fp.rooms) < 2:
        return json.dumps({
            "layout_id": layout_id,
            "note": "Single-room layout — no shared walls.",
            "shared_walls": {"room_room_walls": [], "room_exterior_walls": [],
                             "total_room_room_walls": 0, "total_room_exterior_walls": 0},
            "suggested_connecting_doors": [],
        })

    rooms_dicts = rooms_to_dicts(fp.rooms)
    shared = find_all_shared_walls(rooms_dicts)
    doors = compute_connecting_doors(rooms_dicts, shared)

    return json.dumps({
        "layout_id": layout_id,
        "shared_walls": shared,
        "suggested_connecting_doors": doors,
    })


@mcp.tool()
def add_connecting_doors(layout_id: str, connecting_doors_json: str) -> str:
    """Add connecting doors between rooms on shared walls.

    connecting_doors_json is a JSON array of door specs, each with:
      - shared_wall_index: index into the room_room_walls array from analyze_shared_walls
      - center_position_on_shared_wall: 0-1 position along the shared segment (default 0.5)
      - width: door width in meters (will be clipped to 0.8 * overlap_length)
      - height: door height in meters (default 2.1)
      - door_type: "connecting" (default)
      - opening: true for archway / false for door (default false)

    Adds a Door to BOTH rooms that share the wall.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})

    try:
        door_specs = json.loads(connecting_doors_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    rooms_dicts = rooms_to_dicts(fp.rooms)
    shared = find_all_shared_walls(rooms_dicts)
    rr_walls = shared["room_room_walls"]

    added = []
    errors = []
    for spec in door_specs:
        sw_idx = spec.get("shared_wall_index")
        if sw_idx is None or sw_idx < 0 or sw_idx >= len(rr_walls):
            errors.append(f"Invalid shared_wall_index: {sw_idx}")
            continue

        sw = rr_walls[sw_idx]
        r1_idx, r2_idx = sw["room1"]["index"], sw["room2"]["index"]
        room1, room2 = fp.rooms[r1_idx], fp.rooms[r2_idx]
        r1_data, r2_data = rooms_dicts[r1_idx], rooms_dicts[r2_idx]

        # Clip width
        requested_w = spec.get("width", 0.9)
        max_w = sw["overlap_length"] * 0.8
        final_w = min(requested_w, max_w)

        height = spec.get("height", 2.1)
        shared_pos = spec.get("center_position_on_shared_wall", 0.5)

        # Clip position so door doesn't extend past shared segment
        half_ratio = (final_w / 2) / sw["overlap_length"] if sw["overlap_length"] > 0 else 0
        min_pos = max(0.1, half_ratio + 0.05)
        max_pos = min(0.9, 1.0 - half_ratio - 0.05)
        final_pos = max(min_pos, min(max_pos, shared_pos))

        # Map to each room's wall
        pos1 = calculate_room_wall_position_from_shared_wall(sw, final_pos, r1_data, True)
        pos2 = calculate_room_wall_position_from_shared_wall(sw, final_pos, r2_data, False)

        wall_id1 = _wall_side_to_wall_id(room1, sw["room1_wall"])
        wall_id2 = _wall_side_to_wall_id(room2, sw["room2_wall"])

        if wall_id1 is None or wall_id2 is None:
            errors.append(f"Cannot resolve wall IDs for shared wall {sw_idx}")
            continue

        door_type = spec.get("door_type", "connecting")
        is_opening = spec.get("opening", False)
        door_id = f"cdoor_{uuid.uuid4().hex[:6]}"

        door1 = Door(
            id=door_id, wall_id=wall_id1, position_on_wall=pos1,
            width=final_w, height=height, door_type=door_type,
            opens_inward=True, opening=is_opening,
        )
        door2 = Door(
            id=door_id, wall_id=wall_id2, position_on_wall=pos2,
            width=final_w, height=height, door_type=door_type,
            opens_inward=True, opening=is_opening,
        )

        room1.doors.append(door1)
        room2.doors.append(door2)
        added.append({
            "door_id": door_id,
            "shared_wall_index": sw_idx,
            "room1": {"id": room1.id, "wall_id": wall_id1, "position_on_wall": round(pos1, 4)},
            "room2": {"id": room2.id, "wall_id": wall_id2, "position_on_wall": round(pos2, 4)},
            "width": round(final_w, 3),
            "height": height,
        })

    result = {"layout_id": layout_id, "doors_added": len(added), "doors": added}
    if errors:
        result["errors"] = errors
    return json.dumps(result)


@mcp.tool()
def get_layout(layout_id: str) -> str:
    """Get the full FloorPlan JSON for a layout."""
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})
    return json.dumps(floor_plan_to_dict(fp))


@mcp.tool()
def get_room_info(layout_id: str, room_id: str) -> str:
    """Get details about a specific room, including a rendered visualization path.

    Returns room data JSON plus a path to a rendered PNG.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})
    room = state.get_room(layout_id, room_id)
    if room is None:
        return json.dumps({"error": f"Room {room_id} not found"})

    # Render the room
    render_dir = os.path.join(RESULTS_DIR, layout_id)
    os.makedirs(render_dir, exist_ok=True)
    render_path = os.path.join(render_dir, f"{room_id}_plan.png")
    try:
        render_room(fp, room_id, render_path)
    except Exception as e:
        logger.warning("Failed to render room %s: %s", room_id, e)
        render_path = None

    return json.dumps({
        "room": asdict(room),
        "render_path": render_path,
        "object_count": len(room.objects),
    })


# ============================================================
# Materials Tools
# ============================================================

@mcp.tool()
async def generate_materials(layout_id: str, material_descriptions: str) -> str:
    """Generate PBR material textures for room surfaces.

    material_descriptions is a JSON string:
    {"room_id": {"floor": "warm oak hardwood", "wall": "cream drywall"}}

    Returns paths to generated texture files.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})

    try:
        descs = json.loads(material_descriptions)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    output_dir = os.path.join(RESULTS_DIR, layout_id, "materials")

    # Run in thread to avoid blocking the MCP event loop during
    # model loading (~4GB) and inference (50 DDIM steps per texture).
    def _generate():
        results = {}
        for room_id, surfaces in descs.items():
            results[room_id] = generate_room_materials(room_id, surfaces, output_dir)
        return results

    results = await asyncio.to_thread(_generate)

    # Update room state so export can find the textures.
    # Matches SAGE: room.floor_material = "{room_id}_floor", wall.material = "{room_id}_wall"
    for room_id, surfaces in descs.items():
        room = state.get_room(layout_id, room_id)
        if room is None:
            continue
        if "floor" in surfaces:
            room.floor_material = f"{room_id}_floor_material"
        if "wall" in surfaces:
            for wall in room.walls:
                wall.material = f"{room_id}_wall_material"

    return json.dumps(results)


# ============================================================
# Object Tools
# ============================================================

@mcp.tool()
async def search_objects(object_specs: str) -> str:
    """Search ObjaThor (50k indoor 3D models) for objects matching specs.

    Uses combined CLIP (ViT-L-14) + SBERT (all-mpnet-base-v2) similarity search,
    matching SAGE's retrieval pipeline.

    object_specs is a JSON string:
    [{"name": "sofa", "description": "modern gray L-shaped sofa", "target_size": [200, 90, 80]}]

    Each spec:
      - name: object type (e.g. "sofa", "desk_lamp")
      - description: detailed description for search query
      - target_size: [width, length, height] in cm (optional, for size re-ranking)

    Query format follows SAGE: "A 3D model of {name}, {description}"
    Threshold: 31 (SAGE's similarity_threshold_floor)
    Max candidates: 3 per spec (matching SAGE get_object_candidates)

    Returns: [{name, query, matches: [{asset_id, score, dimensions: {x,y,z} in meters}]}]
    """
    try:
        specs = json.loads(object_specs)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    def _search():
        try:
            from services.objaverse_retrieval import get_retriever, get_bbox_dims
            retriever = get_retriever()
        except Exception as e:
            results = []
            for spec in specs:
                results.append({
                    "name": spec.get("name", "unknown"),
                    "description": spec.get("description", ""),
                    "target_size": spec.get("target_size", []),
                    "matches": [],
                    "note": f"ObjaThor retrieval unavailable: {e}. Use generate_3d_model instead.",
                })
            return results

        results = []
        for spec in specs:
            name = spec.get("name", "unknown")
            description = spec.get("description", "")
            target_size = spec.get("target_size", None)

            caption = f"A 3D model of {name}, {description}"

            candidates = retriever.retrieve(
                [caption],
                threshold=31,
                max_num_candidates=3,
            )

            if target_size and len(candidates) > 0:
                candidates = retriever.compute_size_difference(target_size, candidates)

            matches = []
            for asset_id, score in candidates:
                try:
                    bbox = get_bbox_dims(retriever.database[asset_id])
                    matches.append({
                        "asset_id": asset_id,
                        "score": round(score, 2),
                        "dimensions": {
                            "x": round(bbox["x"], 4),
                            "y": round(bbox["y"], 4),
                            "z": round(bbox["z"], 4),
                        },
                    })
                except Exception:
                    matches.append({
                        "asset_id": asset_id,
                        "score": round(score, 2),
                        "dimensions": None,
                    })

            results.append({
                "name": name,
                "query": caption,
                "matches": matches,
            })

        return results

    results = await asyncio.to_thread(_search)
    return json.dumps(results)


@mcp.tool()
async def load_objaverse_object(asset_id: str, layout_id: str) -> str:
    """Load an ObjaThor 3D asset by ID and export as OBJ + texture.

    The asset is loaded from ~/.objathor-assets/, transformed to Z-up
    coordinate system (matching SAGE), and exported as OBJ + MTL + texture.

    Args:
        asset_id: ObjaThor asset UID (from search_objects results).
        layout_id: Layout ID for organizing output files.

    Returns: {asset_id, obj_path, mtl_path, texture_path, dimensions, ...}
    """
    def _load():
        from services.objaverse_retrieval import get_retriever, export_objathor_as_obj
        retriever = get_retriever()
        output_dir = os.path.join(RESULTS_DIR, layout_id, "objaverse", asset_id)
        result = export_objathor_as_obj(asset_id, output_dir, retriever=retriever)
        result["status"] = "success"
        return result

    try:
        result = await asyncio.to_thread(_load)
        return json.dumps(result)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e), "asset_id": asset_id})
    except Exception as e:
        logger.error("Failed to load ObjaThor asset %s: %s", asset_id, e, exc_info=True)
        return json.dumps({"error": f"Load failed: {e}", "asset_id": asset_id})


@mcp.tool()
async def generate_3d_model(description: str, target_size: str) -> str:
    """Generate a 3D model from text description using TRELLIS.

    Runs the full pipeline: TRELLIS generation → GLB loading → vertex merging →
    coordinate transform (Y-up → Z-up) → VL-based front estimation →
    VL-based real-world size inference → mesh scaling.

    target_size is a JSON string: [width, length, height] in meters (used as a hint).

    Returns processed model paths and estimated real-world attributes.
    """
    try:
        size = json.loads(target_size)
    except json.JSONDecodeError:
        size = [1.0, 1.0, 1.0]

    def _generate():
        model_id = uuid.uuid4().hex[:8]
        output_dir = os.path.join(RESULTS_DIR, "generated_models", model_id)
        os.makedirs(output_dir, exist_ok=True)
        glb_path = os.path.join(output_dir, f"{model_id}.glb")

        trellis = _get_trellis()
        success = trellis.generate_model(
            description,
            seed=__import__("random").randint(0, 1_000_000),
            output_file=glb_path,
        )

        if not success:
            return {"status": "error", "message": "TRELLIS generation failed"}

        try:
            mesh_dict = process_generated_model(
                glb_path=glb_path,
                caption=description,
                reference_size=size,
                estimate_front_enabled=True,
                num_size_inferences=3,
            )
            file_paths = save_processed_mesh(mesh_dict, output_dir, model_id)
            attrs = mesh_dict.get("object_attributes", {})
            return {
                "status": "success",
                "model_id": model_id,
                "description": description,
                "target_size": size,
                "estimated_size": {
                    "width": attrs.get("width", 0),
                    "length": attrs.get("length", 0),
                    "height": attrs.get("height", 0),
                },
                "weight_kg": attrs.get("weight", 0),
                "pbr_parameters": attrs.get("pbr_parameters", {}),
                "files": file_paths,
                "raw_glb": glb_path,
            }
        except Exception as e:
            logger.error("Post-processing failed: %s", e, exc_info=True)
            return {
                "status": "partial",
                "message": f"TRELLIS succeeded but post-processing failed: {e}",
                "raw_glb": glb_path,
                "description": description,
                "target_size": size,
            }

    result = await asyncio.to_thread(_generate)
    return json.dumps(result)


@mcp.tool()
async def place_objects(layout_id: str, room_id: str, objects_json: str, constraints_json: str) -> str:
    """Place objects in a room using the DFS placement solver.

    objects_json: JSON array of object specs:
      [{"id": "bed_001", "type": "bed", "description": "...", "dimensions": {"width": 1.5, "length": 2.0, "height": 0.6}, "place_id": "floor", ...}]

    constraints_json: JSON mapping object_id to constraint list (dict or list-of-dicts format):
      {"bed_001": ["edge", "far, desk_001"], "desk_001": ["edge"]}

    Supported constraints:
      Global:    edge, middle
      Distance:  close to, near, far  (+ target object_id)
      Relative:  in front of, behind, left of, right of, side of, around  (+ target object_id)
      Direction: face to, face same as  (+ target object_id)
      Alignment: center aligned  (+ target object_id)

    Format: "constraint_name" for global, "constraint_name, target_id" for others.
    Fuzzy matching handles misspellings. "around" expands to close_to + face_to.

    Returns placed objects with positions.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})
    room = state.get_room(layout_id, room_id)
    if room is None:
        return json.dumps({"error": f"Room {room_id} not found"})

    try:
        objects_data = json.loads(objects_json)
        constraints = json.loads(constraints_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    def _place():
        objects_to_place = []
        for od in objects_data:
            od.setdefault("room_id", room_id)
            od.setdefault("position", {"x": 0, "y": 0, "z": 0})
            od.setdefault("rotation", {"x": 0, "y": 0, "z": 0})
            od.setdefault("source", "generation")
            od.setdefault("source_id", "")
            od.setdefault("place_id", "floor")
            od.setdefault("place_location", "top")
            objects_to_place.append(dict_to_object(od))

        new_ids = [o.id for o in objects_to_place]
        existing_ids = [o.id for o in room.objects]
        existing_id_mapping = {eid: f"existing-{eid}" for eid in existing_ids}
        structured_constraints = parse_constraints_from_json(
            constraints, new_ids, existing_ids, existing_id_mapping
        )

        floor_objs = [o for o in objects_to_place if o.place_id == "floor"]
        wall_objs = [o for o in objects_to_place if o.place_id == "wall"]
        on_objs = [o for o in objects_to_place if o.place_id not in ("floor", "wall")]

        placed = []

        if floor_objs:
            r = place_floor_objects_dfs(floor_objs, room, structured_constraints)
            placed.extend(r)

        if wall_objs:
            r = place_wall_objects(wall_objs, room, room.objects + placed)
            placed.extend(r)

        if on_objs:
            r = place_on_object_objects(on_objs, room.objects + placed, room)
            placed.extend(r)

        state.add_objects_to_room(layout_id, room_id, placed)

        unsourced = [o.id for o in placed if not o.source_id]

        result = {
            "placed": len(placed),
            "failed": len(objects_to_place) - len(placed),
            "objects": [asdict(o) for o in placed],
        }
        if unsourced:
            result["warning"] = (
                f"{len(unsourced)} object(s) have no source_id (no 3D mesh loaded): "
                f"{unsourced}. Call search_objects + load_objaverse_object (or "
                f"generate_3d_model) BEFORE placing objects, then pass source='objaverse' "
                f"and source_id=<asset_id> so meshes are available for export."
            )
        return result

    result = await asyncio.to_thread(_place)
    return json.dumps(result)


@mcp.tool()
def remove_objects(layout_id: str, room_id: str, object_ids: str) -> str:
    """Remove objects from a room by their IDs.

    object_ids is a JSON array of strings: ["obj_001", "obj_002"]
    """
    try:
        ids = json.loads(object_ids)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    removed = state.remove_objects_from_room(layout_id, room_id, ids)
    return json.dumps({"removed": removed, "requested": ids})


# ============================================================
# Vision Tools (Qwen3-VL)
# ============================================================

@mcp.tool()
async def analyze_floor_plan(layout_id: str, prompt: str) -> str:
    """Render the floor plan and analyze it with Qwen3-VL.

    Generates a PNG of the layout, sends it to the vision model
    with the given prompt, and returns the model's response.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})

    def _analyze():
        render_dir = os.path.join(RESULTS_DIR, layout_id)
        os.makedirs(render_dir, exist_ok=True)
        render_path = os.path.join(render_dir, "floor_plan.png")
        render_floor_plan(fp, render_path)
        response = call_qwen_vl(prompt, [render_path], max_tokens=8000)
        return {"response": response, "render_path": render_path}

    result = await asyncio.to_thread(_analyze)
    return json.dumps(result)


@mcp.tool()
async def run_semantic_critic(layout_id: str, room_id: str) -> str:
    """Run the semantic critic on a room using Qwen3-VL.

    Renders an annotated top-down view and asks the VL model to perform
    a 3-step analysis: room summary, object addition, object adjustment.

    Returns structured JSON with analysis_summary, object_addition_analysis,
    and object_existing_analysis.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})
    room = state.get_room(layout_id, room_id)
    if room is None:
        return json.dumps({"error": f"Room {room_id} not found"})

    # Render annotated top-down view
    render_dir = os.path.join(RESULTS_DIR, layout_id)
    os.makedirs(render_dir, exist_ok=True)
    render_path = os.path.join(render_dir, f"{room_id}_critic.png")
    render_room(fp, room_id, render_path)

    # Build rich room description
    room_description = _build_room_description(room)

    # Calculate floor occupancy ratio
    occupancy_hint = ""
    occupancy = _calc_floor_occupancy(room)
    if occupancy < 0.30:
        occupancy_hint = (
            f"\n[NOTE] Floor occupancy is only {occupancy:.0%}. The room may feel sparse. "
            "Consider recommending more floor objects to fill empty spaces.\n"
        )

    prompt = f"""You are an expert interior designer. Analyze this room design for semantic correctness and provide actionable improvement suggestions.

IMAGE PROVIDED:
Annotated top-down orthogonal view with color-coded object bounding boxes, facing direction arrows (yellow), object IDs, and coordinate axes.

ROOM INFORMATION:
{room_description}
{occupancy_hint}

Step 1:
ROOM Analysis Summary:
Based on the given image and room information, provide detailed reasoning on how to improve the room quality.
Analyze from: realism, functionality, layout, and completion.
Provide detailed problem causes and suggestions for each aspect.

Step 2:
Object Addition Analysis (At most 5-8 recommendations; Propose the most important):
Think about whether more objects are needed for completeness.
Consider:
- Background objects necessary for room type completeness
- Object combos (chair beside desk, lamp on table, etc.)
- [IMPORTANT] Each shelf must be full of small objects (>5 items)
- [IMPORTANT] Each supporter surface (table, desk, counter) needs small objects (2+ items)
- Small objects: books, lamps, picture frames, vases, clocks, plants, cups, decorative bowls, etc.
- Don't propose placing small objects directly on floor
- Balance floor/wall/surface objects
- Avoid making room too crowded — sufficient walking space is critical

OBJECT RESTRICTIONS:
DO NOT recommend: rugs, mats, carpets, windows, doors, curtains, ceiling-hanging objects
Placement locations allowed ONLY: "floor", "wall", or exact object id (e.g. table_001)

Step 3:
Object Adjustment Analysis (At most 1-2 recommendations; Propose the most important):
Check existing objects for:
- Incorrect orientations (chairs should face desks, sofas face TV, etc.)
- Collisions or objects blocking pathways
- Abnormal sizes or crowded floor areas
- Large empty spaces that need filling

OPERATIONS AVAILABLE:
- MOVE: Relocate and rotate existing object
- REMOVE: Remove problematic object
- REPLACE: Replace object with new one

Priority GUIDE (0-10):
- 9-10: Critical safety/functionality issues
- 7-8: Major usability issues, missing important items
- 5-6: Moderate aesthetic issues, suboptimal placement
- 3-4: Minor layout improvements
- 1-2: Optional nice-to-have
- 0: No issue

REQUIRED JSON RESPONSE FORMAT:
```json
{{
    "analysis_summary": {{
        "detailed_reasoning": "Detailed analysis of room quality and improvements needed.",
        "overall_room_rating": "excellent|good|fair|poor"
    }},
    "object_addition_analysis": {{
        "detailed_reasoning": "Whether more objects are needed and why.",
        "object_combos_analysis": [
            {{
                "object_id": "exact_object_id_in_the_list",
                "possible_object_combos": [
                    {{
                        "new_object_type": "object_type",
                        "new_object_quantity": "number",
                        "new_object_placement_location": "floor | wall | exact_object_id",
                        "new_object_placement_guidance": "placement guidance",
                        "new_object_priority": "priority_score (0-10)"
                    }}
                ],
                "priority": "priority_score (0-10)"
            }}
        ],
        "background_objects_analysis": [
            {{
                "new_background_object_type": "object_type",
                "new_background_object_quantity": "number",
                "new_background_object_placement_location": "floor | wall | exact_object_id",
                "new_background_object_placement_guidance": "placement guidance",
                "priority": "priority_score (0-10)"
            }}
        ]
    }},
    "object_existing_analysis": [
        {{
            "object_id": "exact_object_id",
            "object_type": "object_type",
            "issues_found": [
                {{
                    "criteria": "placement|orientation",
                    "score": "number (0-10)",
                    "issue_description": "Problem description",
                    "suggested_operation": {{
                        "type": "MOVE|REMOVE|REPLACE",
                        "target_object_id": "object_id_if_applicable",
                        "condition": "Positioning description",
                        "reasoning": "Why this fixes the issue"
                    }}
                }}
            ]
        }}
    ]
}}
```

Response Length Constraints:
At most 8 object addition recommendations.
At most 2 object adjustment recommendations.
"""

    def _critic():
        return call_qwen_vl(prompt, [render_path], max_tokens=12000)

    response = await asyncio.to_thread(_critic)
    return json.dumps({"critic_response": response, "render_path": render_path})


def _build_room_description(room: Room) -> str:
    """Build a rich text description of the room layout for the critic."""
    lines = [
        f"Room Type: {room.room_type}",
        f"Room Dimensions: {room.dimensions.width:.1f}m x {room.dimensions.length:.1f}m (height: {room.dimensions.height:.1f}m)",
        f"Total Objects: {len(room.objects)}",
    ]

    # Doors and windows
    if room.doors:
        for d in room.doors:
            lines.append(f"Door: wall={d.wall_id}, pos={d.position_on_wall:.2f}, width={d.width:.1f}m")
    if room.windows:
        for w in room.windows:
            lines.append(f"Window: wall={w.wall_id}, pos={w.position_on_wall:.2f}, width={w.width:.1f}m")

    # Objects with details
    if room.objects:
        lines.append("\nPlaced Objects:")
        for obj in room.objects:
            facing = {0: "+Y", 90: "-X", 180: "-Y", 270: "+X"}.get(
                round(obj.rotation.z / 90) * 90 % 360, f"{obj.rotation.z:.0f}deg"
            )
            lines.append(
                f"  - {obj.id} ({obj.type}): pos=({obj.position.x:.2f}, {obj.position.y:.2f}, {obj.position.z:.2f}), "
                f"size={obj.dimensions.width:.2f}x{obj.dimensions.length:.2f}x{obj.dimensions.height:.2f}m, "
                f"facing={facing}, place_id={obj.place_id}"
            )

    return "\n".join(lines)


def _calc_floor_occupancy(room: Room) -> float:
    """Calculate what fraction of the floor area is occupied by objects."""
    room_area = room.dimensions.width * room.dimensions.length
    if room_area <= 0:
        return 0.0

    occupied = 0.0
    for obj in room.objects:
        if obj.place_id == "floor":
            occupied += obj.dimensions.width * obj.dimensions.length

    return min(occupied / room_area, 1.0)


# ============================================================
# Isaac Sim Tools
# ============================================================

@mcp.tool()
async def build_scene(layout_id: str, room_id: str) -> str:
    """Build a room scene in Isaac Sim.

    Sends the room geometry and object meshes to the running Isaac Sim instance.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})
    room = state.get_room(layout_id, room_id)
    if room is None:
        return json.dumps({"error": f"Room {room_id} not found"})

    def _build():
        scene_dir = os.path.join(RESULTS_DIR, layout_id)
        os.makedirs(scene_dir, exist_ok=True)
        room_path = os.path.join(scene_dir, f"{room_id}.json")
        with open(room_path, "w") as f:
            json.dump(asdict(room), f)
        isaac = _get_isaac()
        return create_scene(isaac, scene_dir, room_path)

    try:
        result = await asyncio.to_thread(_build)
        return json.dumps({"status": "success", "result": result})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def simulate_physics_tool(layout_id: str) -> str:
    """Run physics simulation in Isaac Sim and report object stability.

    Returns lists of stable and unstable objects.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})

    def _simulate():
        isaac = _get_isaac()
        return simulate_physics(isaac)

    try:
        result = await asyncio.to_thread(_simulate)
        return json.dumps({
            "status": "success",
            "stable_objects": result.get("stable_objects", []),
            "unstable_objects": result.get("unstable_objects", []),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def export_usd_tool(layout_id: str, output_path: str) -> str:
    """Export the current Isaac Sim scene as a USD file."""
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})

    if not output_path:
        output_path = os.path.join(RESULTS_DIR, layout_id, "scene.usd")

    def _export():
        isaac = _get_isaac()
        return export_usd(isaac, output_path)

    try:
        result = await asyncio.to_thread(_export)
        return json.dumps({"status": "success", "usd_path": output_path, "result": result})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Standalone USD Export
# ============================================================

@mcp.tool()
async def export_scene(layout_id: str, room_id: str = "") -> str:
    """Export a scene as USDZ for visual validation (no Isaac Sim needed).

    If room_id is provided, exports a single room. If room_id is empty,
    exports the full layout (all rooms) with connecting-door deduplication.

    Returns: JSON with usdz_path, usd_path, mesh_count, meshes list.
    """
    fp = state.get_layout(layout_id)
    if fp is None:
        return json.dumps({"error": f"Layout {layout_id} not found"})

    # Warn about empty rooms — Stages 4-5 were likely skipped
    empty_rooms = [r.id for r in fp.rooms if len(r.objects) == 0]
    empty_warning = None
    if empty_rooms:
        empty_warning = (
            f"WARNING: {len(empty_rooms)} room(s) have 0 objects: {empty_rooms}. "
            "The exported scene will contain only walls and structural elements. "
            "You likely skipped Stages 4-5 (Object Recommendations + Placement). "
            "Go back and complete those stages before exporting."
        )
        logger.warning(empty_warning)

    output_dir = os.path.join(RESULTS_DIR, layout_id, "usd_export")

    # Run in thread — mesh booleans and USD writing are CPU-heavy.
    def _export():
        if not room_id:
            result = export_layout_scene(fp, layout_id, RESULTS_DIR, output_dir)
        else:
            room = state.get_room(layout_id, room_id)
            if room is None:
                return {"error": f"Room {room_id} not found"}
            result = export_room_scene(room, layout_id, RESULTS_DIR, output_dir)
        result["status"] = "success"
        if empty_warning:
            result["warning"] = empty_warning
        return result

    try:
        result = await asyncio.to_thread(_export)
        if "error" in result:
            return json.dumps(result)
        return json.dumps(result)
    except Exception as e:
        logger.error("USD export failed: %s", e, exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Helpers
# ============================================================

def _generate_walls(room_data: dict) -> list:
    """Generate 4 walls for a rectangular room from position + dimensions."""
    pos = room_data["position"]
    dims = room_data["dimensions"]
    x, y, z = pos["x"], pos["y"], pos.get("z", 0)
    w, l, h = dims["width"], dims["length"], dims.get("height", 2.7)
    rid = room_data.get("id", "room")

    return [
        {"id": f"{rid}_s_wall", "start_point": {"x": x, "y": y, "z": z},
         "end_point": {"x": x + w, "y": y, "z": z}, "height": h},
        {"id": f"{rid}_e_wall", "start_point": {"x": x + w, "y": y, "z": z},
         "end_point": {"x": x + w, "y": y + l, "z": z}, "height": h},
        {"id": f"{rid}_n_wall", "start_point": {"x": x + w, "y": y + l, "z": z},
         "end_point": {"x": x, "y": y + l, "z": z}, "height": h},
        {"id": f"{rid}_w_wall", "start_point": {"x": x, "y": y + l, "z": z},
         "end_point": {"x": x, "y": y, "z": z}, "height": h},
    ]


def _room_to_validation_dict(room: Room) -> dict:
    """Convert a Room dataclass to the dict format expected by validation functions."""
    return {
        "room_type": room.room_type,
        "position": {"x": room.position.x, "y": room.position.y, "z": room.position.z},
        "dimensions": {"width": room.dimensions.width, "length": room.dimensions.length, "height": room.dimensions.height},
        "doors": [
            {"wall_side": d.wall_id, "position_on_wall": d.position_on_wall,
             "width": d.width, "height": d.height}
            for d in room.doors
        ],
        "windows": [
            {"wall_side": w.wall_id, "position_on_wall": w.position_on_wall,
             "width": w.width, "height": w.height, "sill_height": w.sill_height}
            for w in room.windows
        ],
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    init_materials()
    logger.info("Scene-gen MCP server starting...")
    mcp.run()
