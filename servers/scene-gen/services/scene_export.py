"""Standalone USD/USDZ scene exporter.

Builds room geometry (floor, walls, ceiling with door/window cutouts),
loads placed object meshes, and packages everything as a USDZ file
viewable in macOS Quick Look / Preview — no Isaac Sim required.

Ported from SAGE scene/utils.py and usd_utils.py, stripped of all physics APIs.
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
import trimesh.transformations

logger = logging.getLogger(__name__)

# Add parent dir so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import FloorPlan, Room, Wall, Door, Window, Object

# ---------------------------------------------------------------------------
# Room geometry builders (ported from SAGE scene/utils.py)
# ---------------------------------------------------------------------------


def create_floor_mesh(room: Room) -> trimesh.Trimesh:
    """Create a floor mesh for a room."""
    pos = room.position
    dims = room.dimensions
    floor_thickness = 0.1
    return trimesh.creation.box(
        extents=[dims.width, dims.length, floor_thickness],
        transform=trimesh.transformations.translation_matrix([
            pos.x + dims.width / 2,
            pos.y + dims.length / 2,
            pos.z - floor_thickness / 2,
        ]),
    )


def create_ceiling_mesh(room: Room) -> trimesh.Trimesh:
    """Create a ceiling mesh for a room."""
    pos = room.position
    dims = room.dimensions
    ceiling_thickness = 0.1
    return trimesh.creation.box(
        extents=[dims.width, dims.length, ceiling_thickness],
        transform=trimesh.transformations.translation_matrix([
            pos.x + dims.width / 2,
            pos.y + dims.length / 2,
            pos.z + dims.height + ceiling_thickness / 2,
        ]),
    )


def create_wall_mesh(wall: Wall, room: Room) -> trimesh.Trimesh:
    """Create a wall mesh from wall definition with inward-facing half-thickness."""
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])

    wall_vector = end - start
    wall_length = np.linalg.norm(wall_vector)
    wall_direction = wall_vector / wall_length

    room_center = np.array([
        room.position.x + room.dimensions.width / 2,
        room.position.y + room.dimensions.length / 2,
        room.position.z,
    ])

    wall_center = (start + end) / 2

    normal1 = np.array([wall_direction[1], -wall_direction[0], 0])
    normal2 = np.array([-wall_direction[1], wall_direction[0], 0])

    wall_to_room = room_center - wall_center
    inward_normal = normal1 if np.dot(normal1, wall_to_room) > 0 else normal2

    half_thickness = wall.thickness / 2
    wall_center[2] = wall.start_point.z + wall.height / 2
    wall_center_offset = wall_center + inward_normal * (half_thickness / 2)

    wall_box = trimesh.creation.box(extents=[wall_length, half_thickness, wall.height])

    if abs(wall_direction[0]) < 0.001:  # Y-aligned wall
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    else:
        rotation_matrix = np.eye(4)

    transform = trimesh.transformations.translation_matrix(wall_center_offset) @ rotation_matrix
    wall_box.apply_transform(transform)
    return wall_box


def create_door_mesh(
    wall: Wall,
    door: Door,
    size_scale: float = 1.0,
    thickness_scale: float = 1.0,
    door_size_offset: float = 0.0,
) -> trimesh.Trimesh:
    """Create a door mesh positioned on the wall."""
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start

    door_position_3d = start + wall_vector * door.position_on_wall
    door_position_3d[2] = wall.start_point.z + door.height / 2

    door_box = trimesh.creation.box(
        extents=[
            door.width * size_scale + door_size_offset,
            wall.thickness * thickness_scale,
            door.height * size_scale + door_size_offset,
        ]
    )

    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    if abs(wall_direction[0]) < 0.001:
        door_box.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        )

    door_box.apply_translation(door_position_3d)
    return door_box


def create_window_mesh(wall: Wall, window: Window) -> trimesh.Trimesh:
    """Create a window mesh positioned on the wall."""
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start

    window_position_3d = start + wall_vector * window.position_on_wall
    window_position_3d[2] = wall.start_point.z + window.sill_height + window.height / 2

    window_box = trimesh.creation.box(
        extents=[window.width, wall.thickness, window.height]
    )

    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    if abs(wall_direction[0]) < 0.001:
        window_box.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        )

    window_box.apply_translation(window_position_3d)
    return window_box


def _get_door_unique_id(room: Room, door: Door) -> str:
    if door.door_type == "connecting":
        # Both rooms share the same door.id for connecting doors, so dedup by id
        return f"connecting_door_{door.id}"
    return f"door_{room.id}_{door.id}"


def _get_window_unique_id(room: Room, window: Window) -> str:
    if window.window_type == "connecting":
        return f"connecting_window_{window.wall_id}_{window.position_on_wall:.3f}"
    return f"window_{room.id}_{window.id}"


def create_room_meshes_with_openings(
    room: Room,
    processed_doors: Optional[set] = None,
    processed_windows: Optional[set] = None,
) -> Tuple[list, list, list, list, list, list]:
    """Create wall meshes with door and window openings cut out via boolean ops.

    Args:
        room: The room to process.
        processed_doors: Optional shared set for cross-room door dedup.
            When ``None`` a local set is created (single-room backward compat).
        processed_windows: Optional shared set for cross-room window dedup.

    Returns (wall_meshes, door_meshes, window_meshes,
             wall_ids,    door_ids,    window_ids)
    """
    wall_meshes, door_meshes, window_meshes = [], [], []
    wall_ids, door_ids, window_ids = [], [], []
    if processed_doors is None:
        processed_doors = set()
    if processed_windows is None:
        processed_windows = set()

    for wall in room.walls:
        wall_mesh = create_wall_mesh(wall, room)
        wall_ids.append(wall.id)

        # Cut door openings
        for door in room.doors:
            if door.wall_id != wall.id:
                continue
            door_uid = _get_door_unique_id(room, door)
            if door_uid in processed_doors:
                continue
            door_mesh = create_door_mesh(wall, door, door_size_offset=0.11)
            if not door.opening:
                door_meshes.append(door_mesh)
            processed_doors.add(door_uid)
            door_ids.append(door_uid)
            try:
                wall_mesh = wall_mesh.difference(door_mesh, engine="manifold")
            except Exception:
                try:
                    wall_mesh = wall_mesh.difference(
                        create_door_mesh(wall, door), engine="manifold"
                    )
                except Exception:
                    logger.warning("Boolean op failed for door %s on wall %s", door.id, wall.id)

        # Cut window openings
        for window in room.windows:
            if window.wall_id != wall.id:
                continue
            window_uid = _get_window_unique_id(room, window)
            if window_uid in processed_windows:
                continue
            window_mesh = create_window_mesh(wall, window)
            window_meshes.append(window_mesh)
            processed_windows.add(window_uid)
            window_ids.append(window.id)
            try:
                wall_mesh = wall_mesh.difference(window_mesh, engine="manifold")
            except Exception:
                try:
                    wall_mesh = wall_mesh.difference(
                        create_window_mesh(wall, window), engine="manifold"
                    )
                except Exception:
                    logger.warning("Boolean op failed for window %s on wall %s", window.id, wall.id)

        wall_meshes.append(wall_mesh)

    return wall_meshes, door_meshes, window_meshes, wall_ids, door_ids, window_ids


def apply_object_transform(mesh: trimesh.Trimesh, obj: Object) -> trimesh.Trimesh:
    """Apply position and rotation transforms (Z*Y*X Euler) to an object mesh."""
    transformed = mesh.copy()

    rx = np.radians(obj.rotation.x)
    ry = np.radians(obj.rotation.y)
    rz = np.radians(obj.rotation.z)

    rot_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
    rot_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
    rot_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])

    combined = rot_z @ rot_y @ rot_x
    translation = trimesh.transformations.translation_matrix([
        obj.position.x, obj.position.y, obj.position.z
    ])

    transformed.apply_transform(translation @ combined)
    return transformed


# ---------------------------------------------------------------------------
# Texture-coordinate generation
# ---------------------------------------------------------------------------

def _simple_planar_uv_mapping(mesh: trimesh.Trimesh) -> dict:
    """Fallback XY planar projection for UV mapping."""
    bounds = mesh.bounds
    mn, mx = bounds[0], bounds[1]
    verts = mesh.vertices
    span = mx - mn
    # Avoid division by zero
    span[span < 1e-8] = 1.0
    u = np.clip((verts[:, 0] - mn[0]) / span[0], 0, 1)
    v = np.clip((verts[:, 1] - mn[1]) / span[1], 0, 1)
    return {"vts": np.column_stack([u, v]).astype(np.float32), "fts": mesh.faces.copy()}


def create_floor_mesh_tex_coords(floor_mesh: trimesh.Trimesh) -> dict:
    """Generate UV coords for a floor mesh via xatlas (with planar fallback)."""
    try:
        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(
            floor_mesh.vertices.astype(np.float32),
            floor_mesh.faces.astype(np.uint32),
        )
        atlas.generate()
        _vmapping, indices, uvs = atlas.get_mesh(0)
        return {"vts": uvs, "fts": indices}
    except Exception as e:
        logger.debug("xatlas fallback for floor: %s", e)
        return _simple_planar_uv_mapping(floor_mesh)


def create_wall_mesh_tex_coords(wall_mesh: trimesh.Trimesh) -> dict:
    """Generate UV coords for a wall mesh via xatlas (with planar fallback)."""
    try:
        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(
            wall_mesh.vertices.astype(np.float32),
            wall_mesh.faces.astype(np.uint32),
        )
        atlas.generate()
        _vmapping, indices, uvs = atlas.get_mesh(0)
        return {"vts": uvs, "fts": indices}
    except Exception as e:
        logger.debug("xatlas fallback for wall: %s", e)
        return _simple_planar_uv_mapping(wall_mesh)


# ---------------------------------------------------------------------------
# Scene assembly — build_room_mesh_dict
# ---------------------------------------------------------------------------

def _load_object_mesh(source: str, source_id: str, results_dir: str, layout_id: str):
    """Load an object mesh + texture from the results directory.

    Returns (trimesh.Trimesh, texture_info_dict | None) or (None, None).
    """
    obj_dir = os.path.join(results_dir, layout_id, source, source_id)

    # OBJ file written by export_objathor_as_obj / save_processed_mesh
    obj_path = os.path.join(obj_dir, f"{source_id}.obj")
    if not os.path.exists(obj_path):
        # Try model.obj as alternate convention
        alt = os.path.join(obj_dir, "model.obj")
        if os.path.exists(alt):
            obj_path = alt
        else:
            return None, None

    try:
        mesh = trimesh.load(obj_path, force="mesh")
    except Exception as e:
        logger.warning("Failed to load mesh %s: %s", obj_path, e)
        return None, None

    # Try to find texture + UV coords
    texture_path = os.path.join(obj_dir, f"{source_id}_texture.png")
    if not os.path.exists(texture_path):
        texture_path = os.path.join(obj_dir, "texture.png")

    tex_info = None
    if os.path.exists(texture_path):
        # Extract UVs from the loaded mesh if available
        vts, fts = None, None
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            vts = mesh.visual.uv.astype(np.float32)
            fts = mesh.faces.copy()
        else:
            # Generate planar UVs as fallback
            uv_data = _simple_planar_uv_mapping(mesh)
            vts, fts = uv_data["vts"], uv_data["fts"]

        tex_info = {
            "vts": vts,
            "fts": fts,
            "texture_map_path": texture_path,
        }

    return mesh, tex_info


def build_room_mesh_dict(
    room: Room,
    layout_id: str,
    results_dir: str,
) -> Dict[str, dict]:
    """Build a mesh_info_dict for every element in the room.

    Each value is ``{"mesh": trimesh.Trimesh, "texture": {...} | None}``.
    """
    mesh_info_dict: Dict[str, dict] = {}

    # --- Floor ---
    floor_mesh = create_floor_mesh(room)
    floor_tex = create_floor_mesh_tex_coords(floor_mesh)
    floor_texture_path = os.path.join(
        results_dir, layout_id, "materials", f"{room.floor_material}.png"
    )
    floor_tex_info = {
        "vts": floor_tex["vts"],
        "fts": floor_tex["fts"],
        "texture_map_path": floor_texture_path,
    } if os.path.exists(floor_texture_path) else None
    mesh_info_dict[f"floor_{room.id}"] = {"mesh": floor_mesh, "texture": floor_tex_info}

    # --- Walls (with door/window cutouts) ---
    (wall_meshes, door_meshes, _window_meshes,
     wall_ids, door_ids, _window_ids) = create_room_meshes_with_openings(room)

    wall_material = room.walls[0].material if room.walls else "drywall"
    wall_texture_path = os.path.join(
        results_dir, layout_id, "materials", f"{wall_material}.png"
    )

    for wid, wmesh in zip(wall_ids, wall_meshes):
        wtex = create_wall_mesh_tex_coords(wmesh)
        tex_info = {
            "vts": wtex["vts"],
            "fts": wtex["fts"],
            "texture_map_path": wall_texture_path,
        } if os.path.exists(wall_texture_path) else None
        mesh_info_dict[wid] = {"mesh": wmesh, "texture": tex_info}

    # --- Ceiling ---
    ceiling_mesh = create_ceiling_mesh(room)
    ceiling_tex = create_wall_mesh_tex_coords(ceiling_mesh)
    ceiling_tex_info = {
        "vts": ceiling_tex["vts"],
        "fts": ceiling_tex["fts"],
        "texture_map_path": wall_texture_path,
    } if os.path.exists(wall_texture_path) else None
    mesh_info_dict[f"ceiling_{room.id}"] = {"mesh": ceiling_mesh, "texture": ceiling_tex_info}

    # --- Objects ---
    for obj in room.objects:
        obj_mesh, tex_info = _load_object_mesh(
            obj.source, obj.source_id, results_dir, layout_id
        )
        if obj_mesh is not None:
            transformed = apply_object_transform(obj_mesh, obj)
            mesh_info_dict[obj.id] = {"mesh": transformed, "texture": tex_info}
        else:
            logger.warning("Skipping object %s — mesh not found", obj.id)

    # --- Doors (static closed position, non-opening only) ---
    wall_map = {w.id: w for w in room.walls}
    door_center_list = []
    for door in room.doors:
        if door.opening:
            continue
        wall = wall_map.get(door.wall_id)
        if wall is None:
            continue
        # De-duplicate by position
        sp, ep = wall.start_point, wall.end_point
        cx = sp.x + (ep.x - sp.x) * door.position_on_wall
        cy = sp.y + (ep.y - sp.y) * door.position_on_wall
        dup = any(abs(cx - px) < 0.01 and abs(cy - py) < 0.01 for px, py in door_center_list)
        if dup:
            continue
        door_center_list.append((cx, cy))

        door_mesh = create_door_mesh(wall, door, size_scale=0.95, thickness_scale=0.95)
        mesh_info_dict[door.id] = {"mesh": door_mesh, "texture": None}

    return mesh_info_dict


def build_layout_mesh_dict(
    floor_plan: FloorPlan,
    layout_id: str,
    results_dir: str,
) -> Dict[str, dict]:
    """Build a mesh_info_dict for ALL rooms in the layout.

    Uses shared ``processed_doors``, ``processed_windows``, and
    ``door_center_list`` sets across rooms so that connecting doors
    appear exactly once in the output.
    """
    mesh_info_dict: Dict[str, dict] = {}

    # Shared dedup state across rooms
    processed_doors: set = set()
    processed_windows: set = set()
    door_center_list: list = []

    for room in floor_plan.rooms:
        # --- Floor ---
        floor_mesh = create_floor_mesh(room)
        floor_tex = create_floor_mesh_tex_coords(floor_mesh)
        floor_texture_path = os.path.join(
            results_dir, layout_id, "materials", f"{room.floor_material}.png"
        )
        floor_tex_info = {
            "vts": floor_tex["vts"],
            "fts": floor_tex["fts"],
            "texture_map_path": floor_texture_path,
        } if os.path.exists(floor_texture_path) else None
        mesh_info_dict[f"floor_{room.id}"] = {"mesh": floor_mesh, "texture": floor_tex_info}

        # --- Walls (with door/window cutouts, shared dedup) ---
        (wall_meshes, door_meshes, _window_meshes,
         wall_ids, door_ids, _window_ids) = create_room_meshes_with_openings(
            room, processed_doors=processed_doors, processed_windows=processed_windows
        )

        wall_material = room.walls[0].material if room.walls else "drywall"
        wall_texture_path = os.path.join(
            results_dir, layout_id, "materials", f"{wall_material}.png"
        )

        for wid, wmesh in zip(wall_ids, wall_meshes):
            wtex = create_wall_mesh_tex_coords(wmesh)
            tex_info = {
                "vts": wtex["vts"],
                "fts": wtex["fts"],
                "texture_map_path": wall_texture_path,
            } if os.path.exists(wall_texture_path) else None
            mesh_info_dict[wid] = {"mesh": wmesh, "texture": tex_info}

        # --- Ceiling ---
        ceiling_mesh = create_ceiling_mesh(room)
        ceiling_tex = create_wall_mesh_tex_coords(ceiling_mesh)
        ceiling_tex_info = {
            "vts": ceiling_tex["vts"],
            "fts": ceiling_tex["fts"],
            "texture_map_path": wall_texture_path,
        } if os.path.exists(wall_texture_path) else None
        mesh_info_dict[f"ceiling_{room.id}"] = {"mesh": ceiling_mesh, "texture": ceiling_tex_info}

        # --- Objects ---
        for obj in room.objects:
            obj_mesh, tex_info = _load_object_mesh(
                obj.source, obj.source_id, results_dir, layout_id
            )
            if obj_mesh is not None:
                transformed = apply_object_transform(obj_mesh, obj)
                mesh_info_dict[obj.id] = {"mesh": transformed, "texture": tex_info}
            else:
                logger.warning("Skipping object %s — mesh not found", obj.id)

        # --- Doors (dedup by position across rooms) ---
        wall_map = {w.id: w for w in room.walls}
        for door in room.doors:
            if door.opening:
                continue
            wall = wall_map.get(door.wall_id)
            if wall is None:
                continue
            sp, ep = wall.start_point, wall.end_point
            cx = sp.x + (ep.x - sp.x) * door.position_on_wall
            cy = sp.y + (ep.y - sp.y) * door.position_on_wall
            dup = any(abs(cx - px) < 0.01 and abs(cy - py) < 0.01 for px, py in door_center_list)
            if dup:
                continue
            door_center_list.append((cx, cy))

            door_mesh = create_door_mesh(wall, door, size_scale=0.95, thickness_scale=0.95)
            mesh_info_dict[door.id] = {"mesh": door_mesh, "texture": None}

    return mesh_info_dict


def export_layout_scene(
    floor_plan: FloorPlan,
    layout_id: str,
    results_dir: str,
    output_dir: str,
) -> dict:
    """Export the full multi-room layout as USDZ.

    Steps:
      1. ``build_layout_mesh_dict`` — all rooms with dedup
      2. ``mesh_dict_to_usd``       — write USD file
      3. ``export_usdz``            — package as .usdz

    Returns dict with ``usdz_path``, ``usd_path``, ``mesh_count``,
    ``room_count``, and ``meshes``.
    """
    os.makedirs(output_dir, exist_ok=True)

    mesh_info_dict = build_layout_mesh_dict(floor_plan, layout_id, results_dir)

    usd_path = os.path.join(output_dir, f"{layout_id}_layout.usda")
    mesh_dict_to_usd(mesh_info_dict, usd_path)

    usdz_path = export_usdz(usd_path)

    return {
        "usdz_path": usdz_path,
        "usd_path": usd_path,
        "mesh_count": len(mesh_info_dict),
        "room_count": len(floor_plan.rooms),
        "meshes": list(mesh_info_dict.keys()),
    }


# ---------------------------------------------------------------------------
# USD writer (standalone — no physics APIs)
# ---------------------------------------------------------------------------

def _sanitize_usd_name(name: str) -> str:
    """Make a name safe for use as a USD prim path component."""
    # USD prim names must start with a letter or underscore, and contain
    # only alphanumerics and underscores.
    sanitized = ""
    for ch in name:
        if ch.isalnum() or ch == "_":
            sanitized += ch
        else:
            sanitized += "_"
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_unnamed"


def mesh_dict_to_usd(mesh_info_dict: Dict[str, dict], output_path: str) -> str:
    """Write a USD file from the mesh_info_dict.

    Args:
        mesh_info_dict: ``{mesh_id: {"mesh": Trimesh, "texture": {...} | None}}``
        output_path: Path ending in ``.usda`` or ``.usdc``.

    Returns:
        The *output_path* written.
    """
    from pxr import Gf, Usd, UsdGeom, Vt, Sdf, UsdShade  # noqa: E402

    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create /World scope
    UsdGeom.Xform.Define(stage, "/World")

    for mesh_id, info in mesh_info_dict.items():
        mesh: trimesh.Trimesh = info["mesh"]
        texture = info.get("texture")

        safe_id = _sanitize_usd_name(mesh_id)
        prim_path = f"/World/{safe_id}"

        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32).flatten()
        vertex_counts = np.full(len(mesh.faces), 3, dtype=np.int32)

        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
        usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
        usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
        usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))

        # Compute tight extent
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        usd_mesh.CreateExtentAttr([
            Gf.Vec3f(*bbox_min.tolist()),
            Gf.Vec3f(*bbox_max.tolist()),
        ])

        if texture is not None and os.path.exists(texture.get("texture_map_path", "")):
            _bind_texture(stage, usd_mesh, prim_path, texture)
        else:
            # Set a neutral display colour
            _set_display_color(usd_mesh, mesh_id)

    stage.GetRootLayer().Save()
    return output_path


def _bind_texture(stage, usd_mesh, prim_path: str, texture: dict):
    """Create UsdPreviewSurface material with a diffuse texture and bind it."""
    from pxr import Sdf, UsdShade, UsdGeom, Vt  # noqa: E402

    vts = np.asarray(texture["vts"], dtype=np.float32)
    fts = np.asarray(texture["fts"], dtype=np.int32)
    texture_map_path = os.path.abspath(texture["texture_map_path"])

    # Face-varying tex coords: flatten vts[fts] to per-face-vertex
    tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

    pv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    pv.Set(Vt.Vec2fArray.FromNumpy(tex_coords))

    mat_path = prim_path + "_mat"
    material = UsdShade.Material.Define(stage, mat_path)
    st_input = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
    st_input.Set("st")

    # PBR shader
    pbr = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
    pbr.CreateIdAttr("UsdPreviewSurface")
    roughness = texture.get("pbr_parameters", {}).get("roughness", 1.0) if isinstance(texture, dict) else 1.0
    metallic = texture.get("pbr_parameters", {}).get("metallic", 0.0) if isinstance(texture, dict) else 0.0
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

    # ST reader
    st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(st_input)

    # Diffuse texture sampler
    sampler = UsdShade.Shader.Define(stage, f"{mat_path}/diffuseTexture")
    sampler.CreateIdAttr("UsdUVTexture")
    sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_map_path)
    sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        sampler.ConnectableAPI(), "rgb"
    )

    # Bind
    usd_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(usd_mesh).Bind(material)


def _set_display_color(usd_mesh, mesh_id: str):
    """Set a fallback display colour based on mesh category."""
    from pxr import Gf, Vt  # noqa: E402

    mid = mesh_id.lower()
    if "floor" in mid or "ceiling" in mid:
        color = Gf.Vec3f(0.75, 0.75, 0.75)
    elif "wall" in mid:
        color = Gf.Vec3f(0.85, 0.85, 0.80)
    elif "door" in mid:
        color = Gf.Vec3f(0.55, 0.35, 0.20)
    else:
        color = Gf.Vec3f(0.9, 0.9, 0.9)

    usd_mesh.CreateDisplayColorAttr(Vt.Vec3fArray([color]))


# ---------------------------------------------------------------------------
# USDZ packaging
# ---------------------------------------------------------------------------

def export_usdz(usd_path: str) -> str:
    """Package a .usda/.usdc into a .usdz (single-file archive with textures).

    Returns the path to the .usdz file.
    """
    from pxr import UsdUtils, Sdf  # noqa: E402

    usdz_path = os.path.splitext(usd_path)[0] + ".usdz"
    success = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(usd_path), usdz_path)
    if not success:
        raise RuntimeError(f"Failed to create USDZ package from {usd_path}")
    return usdz_path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def export_room_scene(
    room: Room,
    layout_id: str,
    results_dir: str,
    output_dir: str,
) -> dict:
    """Export a room scene as USDZ for visual validation.

    Steps:
      1. ``build_room_mesh_dict`` — room geometry + object meshes
      2. ``mesh_dict_to_usd``     — write USD file
      3. ``export_usdz``          — package as .usdz

    Returns dict with ``usdz_path``, ``usd_path``, ``mesh_count``.
    """
    os.makedirs(output_dir, exist_ok=True)

    mesh_info_dict = build_room_mesh_dict(room, layout_id, results_dir)

    usd_path = os.path.join(output_dir, f"{room.id}_scene.usda")
    mesh_dict_to_usd(mesh_info_dict, usd_path)

    usdz_path = export_usdz(usd_path)

    return {
        "usdz_path": usdz_path,
        "usd_path": usd_path,
        "mesh_count": len(mesh_info_dict),
        "meshes": list(mesh_info_dict.keys()),
    }
