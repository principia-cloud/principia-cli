"""GLB post-processing pipeline for TRELLIS-generated 3D models.

Adapted from SAGE server/objects/object_generation.py and server/objects/load_glb.py.

Pipeline: load GLB → merge vertices → coordinate transform (Y-up → Z-up) →
          VL-based front estimation → VL-based attribute inference → scale to real-world size.

Rendering auto-detects GPU: uses nvdiffrast (CUDA) when available,
falls back to pyrender (OpenGL/OSMesa), then trimesh.
"""

import json
import logging
import os
import random
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import KDTree

logger = logging.getLogger("scene-gen.mesh_processing")

# ---------------------------------------------------------------------------
# GPU / renderer detection (run once at import time)
# ---------------------------------------------------------------------------

_RENDERER: str = "trimesh"  # default fallback

def _detect_renderer() -> str:
    """Detect best available renderer: nvdiffrast > pyrender > trimesh."""
    # Try nvdiffrast + CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            import nvdiffrast.torch as dr  # noqa: F401
            logger.info("Renderer: nvdiffrast (CUDA GPU detected)")
            return "nvdiffrast"
    except ImportError:
        pass

    # Try pyrender
    try:
        import pyrender  # noqa: F401
        logger.info("Renderer: pyrender (CPU/OpenGL)")
        return "pyrender"
    except ImportError:
        pass

    logger.info("Renderer: trimesh (basic fallback)")
    return "trimesh"

_RENDERER = _detect_renderer()


# ---------------------------------------------------------------------------
# 1. GLB Loading
# ---------------------------------------------------------------------------

def load_glb(glb_path: str) -> dict:
    """Load a GLB file and extract mesh, UV coordinates, and texture.

    Returns a mesh_dict with keys:
        mesh: trimesh.Trimesh
        tex_coords: {vts: ndarray (N,2), fts: ndarray (M,3)}
        texture: ndarray (H,W,3|4) float32 in [0,1]
    """
    scene_or_mesh = trimesh.load(glb_path)

    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = None
        for geometry in scene_or_mesh.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                mesh = geometry
                break
        if mesh is None:
            raise ValueError("No valid mesh found in the GLB scene")
    else:
        mesh = scene_or_mesh

    vertices = mesh.vertices
    faces = mesh.faces

    # Extract UV coordinates
    vts = None
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        vts = mesh.visual.uv

    if vts is None:
        # Try from scene geometries
        if isinstance(scene_or_mesh, trimesh.Scene):
            for geometry in scene_or_mesh.geometry.values():
                if hasattr(geometry.visual, "uv") and geometry.visual.uv is not None:
                    vts = geometry.visual.uv
                    break

    if vts is not None:
        fts = faces.copy()
        # Handle UV/vertex count mismatch
        if len(vts) < len(vertices):
            padding = np.zeros((len(vertices) - len(vts), 2))
            vts = np.vstack([vts, padding])
        elif len(vts) > len(vertices):
            vts = vts[: len(vertices)]
    else:
        logger.warning("No UV coordinates found in GLB, creating dummy coords")
        vts = np.zeros((len(vertices), 2))
        fts = faces.copy()

    # Extract texture
    texture = None
    if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
        material = mesh.visual.material
        if hasattr(material, "image") and material.image is not None:
            tex_img = material.image
            if tex_img.mode not in ("RGB", "RGBA"):
                tex_img = tex_img.convert("RGB")
            texture = np.array(tex_img).astype(np.float32) / 255.0
        elif hasattr(material, "baseColorTexture") and material.baseColorTexture is not None:
            texture = np.array(material.baseColorTexture).astype(np.float32) / 255.0

    if texture is None:
        texture = np.ones((256, 256, 3), dtype=np.float32)

    return {
        "mesh": mesh,
        "tex_coords": {"vts": vts, "fts": fts},
        "texture": texture,
    }


# ---------------------------------------------------------------------------
# 2. Vertex Merging
# ---------------------------------------------------------------------------

def merge_vertices(mesh_dict: dict) -> dict:
    """Merge duplicate vertices while preserving UV face indices.

    Uses KDTree to map merged faces back to original texture coordinate indices.
    Adapted from SAGE server/objects/object_generation.py merge_vertices().
    """
    mesh = mesh_dict["mesh"]
    tex_coords = mesh_dict["tex_coords"]
    texture = mesh_dict["texture"]

    original_vertices = mesh.vertices.copy()
    original_faces = mesh.faces.copy()
    original_fts = tex_coords["fts"].copy()

    merged_mesh = mesh.copy()
    merged_mesh.merge_vertices(digits_vertex=6, merge_tex=True)

    merged_vertices = merged_mesh.vertices
    merged_faces = merged_mesh.faces

    # Map merged vertices → original vertices via KDTree
    original_tree = KDTree(original_vertices)
    _, _ = original_tree.query(merged_vertices)

    # Map merged faces → original faces via triangle matching
    merged_triangles = merged_vertices[merged_faces.reshape(-1)].reshape(-1, 3, 3)
    original_triangles = original_vertices[original_faces.reshape(-1)].reshape(-1, 3, 3)

    def _all_orderings(triangles: np.ndarray):
        orderings = np.array(
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        )
        all_ordered = triangles[:, None, orderings].reshape(-1, 3, 3)
        tri_indices = np.repeat(np.arange(len(triangles)), 6)
        return all_ordered, tri_indices

    original_all, orig_tri_idx = _all_orderings(original_triangles)
    original_flat = original_all.reshape(-1, 9)
    merged_flat = merged_triangles.reshape(-1, 9)

    tri_tree = KDTree(original_flat)
    _, tri_map_idx = tri_tree.query(merged_flat)
    face_mapping = orig_tri_idx[tri_map_idx]

    return {
        "mesh": merged_mesh,
        "tex_coords": {"vts": tex_coords["vts"].copy(), "fts": original_fts[face_mapping]},
        "texture": texture,
    }


# ---------------------------------------------------------------------------
# 3. Coordinate System Transform
# ---------------------------------------------------------------------------

def transform_trellis_to_scene(mesh_dict: dict) -> dict:
    """Convert TRELLIS output (Y-up) to scene coordinate system (Z-up).

    Operations:
    - Swap Y and Z axes
    - Recenter mesh at origin (XY) with Z min at 0
    - Flip winding order (and matching texture face indices)
    """
    mesh = mesh_dict["mesh"]

    # Swap Y ↔ Z
    verts = mesh.vertices.copy()
    verts[:, 1], verts[:, 2] = mesh.vertices[:, 2].copy(), mesh.vertices[:, 1].copy()
    mesh.vertices = verts

    # Recenter XY, set Z min to 0
    mesh.vertices[:, 0] -= 0.5 * (mesh.vertices[:, 0].max() + mesh.vertices[:, 0].min())
    mesh.vertices[:, 1] -= 0.5 * (mesh.vertices[:, 1].max() + mesh.vertices[:, 1].min())
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()

    # Flip winding order
    mesh.faces = mesh.faces[:, [0, 2, 1]].copy()
    mesh_dict["tex_coords"]["fts"] = mesh_dict["tex_coords"]["fts"][:, [0, 2, 1]].copy()

    mesh_dict["mesh"] = mesh
    return mesh_dict


# ---------------------------------------------------------------------------
# 4. Mesh Rendering (for VL model input)
#    Auto-selects: nvdiffrast (GPU) > pyrender (CPU/OpenGL) > trimesh
# ---------------------------------------------------------------------------

def _compute_camera_params(
    vertices: np.ndarray,
    theta_deg: float,
    phi_deg: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute camera eye position, look-at center, and bounding scale."""
    center = 0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
    scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    radius = scale * 1.5  # match SAGE's 1.5x radius

    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    eye = center + radius * np.array([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ])
    return eye, center, scale


# -- nvdiffrast (GPU) renderer -----------------------------------------------

def _build_camera_matrix_torch(eye: np.ndarray, at: np.ndarray, up: np.ndarray):
    """Build camera-to-world 4x4 matrix. Ported from SAGE camera.py."""
    import torch

    eye_t = torch.from_numpy(eye).float()
    at_t = torch.from_numpy(at).float()
    up_t = torch.from_numpy(up).float()

    z = at_t - eye_t
    z = z / z.norm()
    x = torch.cross(-up_t, z)
    x = x / x.norm()
    y = torch.cross(z, x)
    y = y / y.norm()

    mat = torch.zeros(4, 4)
    mat[0, :3] = x
    mat[1, :3] = y
    mat[2, :3] = z
    mat[3, :3] = eye_t
    mat[0, 3] = 0.0
    mat[1, 3] = 0.0
    mat[2, 3] = 0.0
    mat[3, 3] = 1.0
    return mat


def _get_projection_matrix(fov: float, H: int, W: int, near: float, far: float):
    """Build perspective projection matrix. Ported from SAGE camera.py."""
    import torch

    fov_rad = fov * np.pi / 180.0
    fx = 0.5 * H / np.tan(fov_rad / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    proj = torch.zeros(4, 4)
    proj[0, 0] = 2.0 * fx / W
    proj[1, 1] = 2.0 * fy / H
    proj[0, 2] = 2.0 * (cx / W - 0.5)
    proj[1, 2] = 2.0 * (cy / H - 0.5)
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = 1.0
    return proj


def _render_textured(rast_out, mesh_gpu: dict, H: int, W: int):
    """Software texture lookup from rasterization output.

    Ported from SAGE render.py render_textured_mesh().
    """
    import torch

    vt = mesh_gpu["vt"]            # (V, 2)
    ft = mesh_gpu["ft"]            # (F, 3) int
    tex_map = mesh_gpu["texture_map"]  # (Ht, Wt, 3)

    valid = (rast_out[0, :, :, 3] > 0)  # (H, W)
    triangle_id = (rast_out[0, :, :, 3] - 1).long()  # (H, W)
    bary_uv = rast_out[0, :, :, :2]  # (H, W, 2)
    bary_w = 1.0 - bary_uv[..., 0] - bary_uv[..., 1]
    bary = torch.stack([bary_uv[..., 0], bary_uv[..., 1], bary_w], dim=-1)  # (H, W, 3)

    rgb = torch.ones(H, W, 3, device=vt.device)  # white background

    valid_mask = valid.reshape(-1)
    tri_ids = triangle_id.reshape(-1)[valid_mask]
    bary_flat = bary.reshape(-1, 3)[valid_mask]

    # Look up UV coords for each triangle vertex
    uv_idx = ft[tri_ids]  # (P, 3) indices into vt
    uv0 = vt[uv_idx[:, 0]]  # (P, 2)
    uv1 = vt[uv_idx[:, 1]]
    uv2 = vt[uv_idx[:, 2]]

    # Interpolate UV
    interp_uv = (
        bary_flat[:, 0:1] * uv0
        + bary_flat[:, 1:2] * uv1
        + bary_flat[:, 2:3] * uv2
    )  # (P, 2)

    Ht, Wt = tex_map.shape[0], tex_map.shape[1]
    u = (interp_uv[:, 0] * (Wt - 1)).clamp(0, Wt - 1)
    v = (interp_uv[:, 1] * (Ht - 1)).clamp(0, Ht - 1)

    # Bilinear interpolation
    u0 = u.long().clamp(0, Wt - 2)
    v0 = v.long().clamp(0, Ht - 2)
    u1 = u0 + 1
    v1 = v0 + 1
    wu = u - u0.float()
    wv = v - v0.float()

    c00 = tex_map[v0, u0]
    c01 = tex_map[v0, u1]
    c10 = tex_map[v1, u0]
    c11 = tex_map[v1, u1]

    color = (
        (1 - wu).unsqueeze(-1) * (1 - wv).unsqueeze(-1) * c00
        + wu.unsqueeze(-1) * (1 - wv).unsqueeze(-1) * c01
        + (1 - wu).unsqueeze(-1) * wv.unsqueeze(-1) * c10
        + wu.unsqueeze(-1) * wv.unsqueeze(-1) * c11
    )

    rgb_flat = rgb.reshape(-1, 3)
    rgb_flat[valid_mask] = color
    return rgb_flat.reshape(H, W, 3)


def _render_with_nvdiffrast(
    mesh_dict: dict,
    eye: np.ndarray,
    center: np.ndarray,
    resolution: Tuple[int, int],
) -> np.ndarray:
    """GPU-accelerated rendering via nvdiffrast. Ported from SAGE."""
    import torch
    import torch.nn.functional as F
    import nvdiffrast.torch as dr

    H, W = resolution
    device = "cuda"

    vertices = mesh_dict["mesh"].vertices
    faces = mesh_dict["mesh"].faces
    vts = mesh_dict["tex_coords"]["vts"]
    fts = mesh_dict["tex_coords"]["fts"]
    texture = mesh_dict["texture"]

    # Build GPU mesh dict (matches SAGE build_mesh_dict)
    verts_t = torch.from_numpy(vertices.astype(np.float32)).to(device)
    verts_t = F.pad(verts_t, pad=(0, 1), value=1.0)  # (N, 4)
    faces_t = torch.from_numpy(faces.astype(np.int32)).contiguous().to(device)
    vts_t = torch.from_numpy(vts.astype(np.float32)).contiguous().to(device)
    vts_t[:, 1] = 1.0 - vts_t[:, 1]  # flip V
    fts_t = torch.from_numpy(fts.astype(np.int32)).contiguous().to(device)
    tex_t = torch.from_numpy(texture[:, :, :3].astype(np.float32)).contiguous().to(device)

    mesh_gpu = {
        "vertices": verts_t,
        "pos_idx": faces_t,
        "vt": vts_t,
        "ft": fts_t,
        "texture_map": tex_t,
    }

    # Camera + projection
    up = np.array([0.0, 0.0, 1.0])
    camera_matrix = _build_camera_matrix_torch(eye, center, up)
    proj_matrix = _get_projection_matrix(60.0, H, W, 0.001, 10.0)
    mvp = (proj_matrix @ camera_matrix.inverse()).to(device)

    # Rasterize
    glctx = dr.RasterizeGLContext(output_db=False)
    clip_verts = (verts_t @ mvp.T).unsqueeze(0).contiguous()
    rast_out, _ = dr.rasterize(glctx, clip_verts, faces_t, resolution=(H, W))

    # Texture lookup
    rgb = _render_textured(rast_out, mesh_gpu, H, W)

    return (rgb.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)


# -- pyrender (CPU/OpenGL) renderer ------------------------------------------

def _render_with_pyrender(
    mesh_dict: dict,
    eye: np.ndarray,
    center: np.ndarray,
    resolution: Tuple[int, int],
) -> np.ndarray:
    """Render using pyrender (supports headless via OSMesa/EGL)."""
    import pyrender

    if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    mesh = mesh_dict["mesh"]
    texture_data = mesh_dict["texture"]
    vts = mesh_dict["tex_coords"]["vts"]

    tex_image = Image.fromarray((texture_data * 255).astype(np.uint8)[:, :, :3])
    visual = trimesh.visual.TextureVisuals(uv=vts, image=tex_image)
    render_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        visual=visual,
        process=False,
    )

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    py_mesh = pyrender.Mesh.from_trimesh(render_mesh)
    scene.add(py_mesh)

    up = np.array([0.0, 0.0, 1.0])
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = true_up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = eye

    camera = pyrender.PerspectiveCamera(yfov=np.radians(60))
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(*resolution)
    color, _ = renderer.render(scene)
    renderer.delete()
    return color


# -- trimesh fallback ---------------------------------------------------------

def _render_with_trimesh(
    mesh_dict: dict,
    eye: np.ndarray,
    center: np.ndarray,
    resolution: Tuple[int, int],
) -> np.ndarray:
    """Lowest-quality fallback using trimesh scene export."""
    mesh = mesh_dict["mesh"]
    texture_data = mesh_dict["texture"]
    vts = mesh_dict["tex_coords"]["vts"]

    tex_image = Image.fromarray((texture_data * 255).astype(np.uint8)[:, :, :3])
    visual = trimesh.visual.TextureVisuals(uv=vts, image=tex_image)
    render_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        visual=visual,
        process=False,
    )

    scene = trimesh.Scene(render_mesh)
    try:
        data = scene.save_image(resolution=resolution)
        img = Image.open(trimesh.util.wrap_as_stream(data))
        return np.array(img.convert("RGB"))
    except Exception:
        logger.warning("trimesh scene rendering failed, returning blank image")
        return np.ones((*resolution, 3), dtype=np.uint8) * 255


# -- Unified entry point ------------------------------------------------------

def render_mesh_view(
    mesh_dict: dict,
    theta_deg: float = 135.0,
    phi_deg: float = 60.0,
    resolution: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """Render a single view of the mesh.

    Auto-selects the best available renderer:
      1. nvdiffrast (CUDA GPU) — highest quality, fastest
      2. pyrender (CPU via OSMesa/EGL) — good quality
      3. trimesh scene export — basic fallback

    Args:
        mesh_dict: Standard mesh_dict with mesh, tex_coords, texture.
        theta_deg: Azimuth angle in degrees.
        phi_deg: Elevation angle in degrees.
        resolution: (width, height) of the output image.

    Returns:
        RGB image as uint8 ndarray (H, W, 3).
    """
    eye, center, _ = _compute_camera_params(
        mesh_dict["mesh"].vertices, theta_deg, phi_deg
    )

    if _RENDERER == "nvdiffrast":
        try:
            return _render_with_nvdiffrast(mesh_dict, eye, center, resolution)
        except Exception as e:
            logger.warning("nvdiffrast render failed (%s), falling back to pyrender", e)

    if _RENDERER in ("nvdiffrast", "pyrender"):
        try:
            return _render_with_pyrender(mesh_dict, eye, center, resolution)
        except Exception as e:
            logger.warning("pyrender render failed (%s), falling back to trimesh", e)

    return _render_with_trimesh(mesh_dict, eye, center, resolution)


def render_mesh_views(
    mesh_dict: dict,
    directions: List[float] = None,
    resolution: Tuple[int, int] = (1024, 1024),
) -> List[str]:
    """Render the mesh from multiple angles and save as temp PNGs.

    Args:
        mesh_dict: Standard mesh_dict.
        directions: Azimuth angles in degrees. Defaults to [90, 180, 270, 0].
        resolution: Image resolution.

    Returns:
        List of temp file paths (PNG).
    """
    if directions is None:
        directions = [90.0, 180.0, 270.0, 0.0]

    paths = []
    for theta in directions:
        img = render_mesh_view(mesh_dict, theta_deg=theta, phi_deg=60.0, resolution=resolution)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        Image.fromarray(img).save(tmp.name)
        paths.append(tmp.name)

    return paths


def render_front_right_up_view(
    mesh_dict: dict,
    resolution: Tuple[int, int] = (1024, 1024),
) -> str:
    """Render the canonical front-right-up view and save as temp PNG.

    Returns the temp file path.
    """
    img = render_mesh_view(mesh_dict, theta_deg=135.0, phi_deg=60.0, resolution=resolution)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(img).save(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# 5. VL-based Front Estimation
# ---------------------------------------------------------------------------

def estimate_front(mesh_dict: dict, category: str = "object") -> dict:
    """Use Qwen3-VL to identify the front-facing direction and rotate the mesh.

    Renders 4 views at 90° intervals → VL model picks the front → rotates mesh.
    Adapted from SAGE server/objects/object_attribute_inference.py estimate_front_from_mesh().
    """
    from services.qwen_vl import estimate_front_direction

    view_paths = render_mesh_views(mesh_dict, directions=[90.0, 180.0, 270.0, 0.0])

    try:
        result = estimate_front_direction(view_paths, category=category)
        front_idx = int(result.get("front_index", 0))
        logger.info("VL identified front at index %d: %s", front_idx, result.get("reasoning", ""))
    except Exception as e:
        logger.warning("Front estimation failed: %s, defaulting to index 0", e)
        front_idx = 0
    finally:
        for p in view_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    # Rotate mesh to align front with canonical direction
    rotate_angles = [0.0, -90.0, -180.0, -270.0]
    if 0 <= front_idx < len(rotate_angles) and rotate_angles[front_idx] != 0.0:
        angle_rad = np.radians(rotate_angles[front_idx])
        rot_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        mesh_dict["mesh"].apply_transform(rot_matrix)

    return mesh_dict


# ---------------------------------------------------------------------------
# 6. VL-based Attribute Inference
# ---------------------------------------------------------------------------

def infer_real_world_size(
    mesh_dict: dict,
    caption: str,
    num_inferences: int = 3,
) -> dict:
    """Use Qwen3-VL to estimate real-world dimensions and scale the mesh.

    Renders the front-right-up view, sends to VL model N times, averages
    the estimated height/width/length, and scales the mesh accordingly.

    Adapted from SAGE server/objects/object_generation.py generate_model_from_text()
    and server/objects/object_attribute_inference.py infer_attributes_from_claude().

    Returns the mesh_dict with scaled vertices and added object_attributes.
    """
    from services.qwen_vl import infer_object_attributes

    heights, widths, lengths = [], [], []

    for _ in range(num_inferences):
        view_path = render_front_right_up_view(mesh_dict)
        try:
            attrs = infer_object_attributes(view_path, caption, mesh_dict["mesh"].vertices)
            heights.append(attrs.get("height", 1.0))
            widths.append(attrs.get("width", 1.0))
            lengths.append(attrs.get("length", 1.0))
        except Exception as e:
            logger.warning("Attribute inference failed: %s", e)
        finally:
            try:
                os.remove(view_path)
            except OSError:
                pass

    if not heights:
        logger.error("All attribute inferences failed, using default 1m cube")
        heights, widths, lengths = [1.0], [1.0], [1.0]

    h_mean = float(np.mean(heights))
    w_mean = float(np.mean(widths))
    l_mean = float(np.mean(lengths))

    object_attributes = {
        "height": h_mean,
        "width": w_mean,
        "length": l_mean,
        "caption": caption,
    }

    # Merge last successful attributes (for PBR params etc.)
    if heights:
        try:
            view_path = render_front_right_up_view(mesh_dict)
            full_attrs = infer_object_attributes(view_path, caption, mesh_dict["mesh"].vertices)
            full_attrs["height"] = h_mean
            full_attrs["width"] = w_mean
            full_attrs["length"] = l_mean
            object_attributes = full_attrs
            os.remove(view_path)
        except Exception:
            pass

    mesh_dict["object_attributes"] = object_attributes

    # Scale mesh to real-world dimensions
    verts = mesh_dict["mesh"].vertices
    current_h = float(verts[:, 2].max() - verts[:, 2].min())
    current_xy_max = max(
        float(verts[:, 0].max() - verts[:, 0].min()),
        float(verts[:, 1].max() - verts[:, 1].min()),
    )

    if current_h > 1e-6 and current_xy_max > 1e-6:
        scale_h = h_mean / current_h
        scale_xy = max(w_mean, l_mean) / current_xy_max
        scale_factor = np.mean([scale_h, scale_xy])
    else:
        scale_factor = 1.0

    mesh_dict["mesh"].vertices *= scale_factor

    # Lift slightly off floor to avoid z-fighting
    mesh_dict["mesh"].vertices[:, 2] += 0.001

    return mesh_dict


# ---------------------------------------------------------------------------
# 7. Full Pipeline
# ---------------------------------------------------------------------------

def process_generated_model(
    glb_path: str,
    caption: str,
    reference_size: Optional[List[float]] = None,
    estimate_front_enabled: bool = True,
    num_size_inferences: int = 3,
) -> dict:
    """Run the full post-processing pipeline on a TRELLIS-generated GLB.

    Steps:
    1. Load GLB → mesh_dict
    2. Merge duplicate vertices
    3. Transform coordinate system (Y-up → Z-up)
    4. Estimate front direction via VL model (optional)
    5. Infer real-world dimensions via VL model and scale mesh

    Args:
        glb_path: Path to the .glb file from TRELLIS.
        caption: Text description of the object (used for VL inference).
        reference_size: Optional [width, length, height] in meters hint.
        estimate_front_enabled: Whether to run front estimation.
        num_size_inferences: Number of VL inference rounds for size averaging.

    Returns:
        mesh_dict with processed mesh, tex_coords, texture, and object_attributes.
    """
    logger.info("Processing GLB: %s", glb_path)

    # Step 1: Load
    mesh_dict = load_glb(glb_path)
    logger.info(
        "Loaded: %d vertices, %d faces",
        mesh_dict["mesh"].vertices.shape[0],
        mesh_dict["mesh"].faces.shape[0],
    )

    # Step 2: Merge vertices
    mesh_dict = merge_vertices(mesh_dict)
    logger.info(
        "After merge: %d vertices, %d faces",
        mesh_dict["mesh"].vertices.shape[0],
        mesh_dict["mesh"].faces.shape[0],
    )

    # Step 3: Coordinate transform
    mesh_dict = transform_trellis_to_scene(mesh_dict)

    # Step 4: Front estimation
    if estimate_front_enabled:
        category = caption.split()[-1] if caption else "object"
        mesh_dict = estimate_front(mesh_dict, category=category)

    # Step 5: Size inference and scaling
    mesh_dict = infer_real_world_size(mesh_dict, caption, num_inferences=num_size_inferences)

    attrs = mesh_dict.get("object_attributes", {})
    logger.info(
        "Final dimensions: %.3fm W x %.3fm L x %.3fm H",
        attrs.get("width", 0),
        attrs.get("length", 0),
        attrs.get("height", 0),
    )

    return mesh_dict


def save_processed_mesh(mesh_dict: dict, output_dir: str, name: str) -> dict:
    """Save the processed mesh as OBJ + MTL + texture + metadata.

    Returns dict with file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    obj_path = os.path.join(output_dir, f"{name}.obj")
    mtl_path = os.path.join(output_dir, f"{name}.mtl")
    tex_path = os.path.join(output_dir, f"{name}_texture.png")
    meta_path = os.path.join(output_dir, f"{name}_attributes.json")

    mesh = mesh_dict["mesh"]
    tex_coords = mesh_dict["tex_coords"]
    texture = mesh_dict["texture"]

    # Save texture
    if texture is not None and texture.shape[0] > 1:
        tex_img = Image.fromarray((texture[:, :, :3] * 255).astype(np.uint8))
        tex_img.save(tex_path)

    # Save MTL
    with open(mtl_path, "w") as f:
        f.write("newmtl material0\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 0.8 0.8 0.8\n")
        f.write("Ks 0.5 0.5 0.5\n")
        f.write("Ns 96.0\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {os.path.basename(tex_path)}\n")

    # Save OBJ
    vts = tex_coords["vts"]
    fts = tex_coords["fts"]
    with open(obj_path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for vt in vts:
            f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
        f.write("\nusemtl material0\n")
        for i, face in enumerate(mesh.faces):
            ft = fts[i] if i < len(fts) else face
            f.write(f"f {face[0]+1}/{ft[0]+1} {face[1]+1}/{ft[1]+1} {face[2]+1}/{ft[2]+1}\n")

    # Save attributes
    attrs = mesh_dict.get("object_attributes", {})
    with open(meta_path, "w") as f:
        json.dump(attrs, f, indent=2)

    return {
        "obj_path": obj_path,
        "mtl_path": mtl_path,
        "texture_path": tex_path,
        "attributes_path": meta_path,
    }
