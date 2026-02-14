"""ObjaThor 3D object retrieval via CLIP + SBERT similarity.

Ported from SAGE server/objects/objaverse_retrieval.py.

Provides:
- ObjathorRetriever: searches 50k+ indoor objects by text query, returns
  mesh_dict compatible with the SAGE pipeline
- load_objathor_object: convenience wrapper that exports OBJ + texture files
"""

import gzip
import logging
import os
import pickle
import shutil
import tempfile
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import compress_json
import compress_pickle
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger("scene-gen.retrieval")

# ---------------------------------------------------------------------------
# Asset paths  (match SAGE objaverse_retrieval.py lines 49-59)
# ---------------------------------------------------------------------------

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")
OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser("~/.principia/data/scene-gen/objathor")
)
OBJATHOR_VERSIONED_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION)
OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(OBJATHOR_VERSIONED_DIR, "annotations.json.gz")
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")

# ---------------------------------------------------------------------------
# S3 on-demand asset download
# ---------------------------------------------------------------------------

_S3_ASSET_BASE_URL = os.environ.get(
    "OBJATHOR_S3_URL",
    "https://principia-scene-gen-assets.s3.us-east-1.amazonaws.com/objathor/2023_09_23/assets",
)

# Common texture filenames that ObjaThor assets may include
_TEXTURE_FILENAMES = [
    "albedo.jpg", "albedo.png",
    "emission.jpg", "emission.png",
    "normal.jpg", "normal.png",
    "metallic_Smoothness.jpg", "metallic_Smoothness.png",
]


def _download_file(url: str, dest: str, retries: int = 2) -> bool:
    """Download a file from *url* to *dest*. Returns True on success."""
    for attempt in range(retries + 1):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as exc:
            if attempt < retries:
                logger.debug("Download retry %d for %s: %s", attempt + 1, url, exc)
            else:
                logger.debug("Download failed for %s: %s", url, exc)
    return False


def _ensure_asset_available(asset_id: str) -> None:
    """If the asset is not present locally, download it from S3."""
    asset_dir = os.path.join(OBJATHOR_ASSETS_DIR, asset_id)
    pkl_path = os.path.join(asset_dir, f"{asset_id}.pkl.gz")

    if os.path.exists(pkl_path):
        return

    logger.info("Downloading asset %s from S3...", asset_id)
    tmp_dir = tempfile.mkdtemp(prefix=f"objathor_{asset_id}_")
    try:
        pkl_url = f"{_S3_ASSET_BASE_URL}/{asset_id}/{asset_id}.pkl.gz"
        tmp_pkl = os.path.join(tmp_dir, f"{asset_id}.pkl.gz")
        if not _download_file(pkl_url, tmp_pkl):
            logger.warning("Failed to download asset %s from S3", asset_id)
            return

        # Best-effort texture downloads
        for tex_name in _TEXTURE_FILENAMES:
            tex_url = f"{_S3_ASSET_BASE_URL}/{asset_id}/{tex_name}"
            tex_dest = os.path.join(tmp_dir, tex_name)
            _download_file(tex_url, tex_dest, retries=0)

        # Atomic move to final location
        os.makedirs(OBJATHOR_ASSETS_DIR, exist_ok=True)
        final_dir = os.path.join(OBJATHOR_ASSETS_DIR, asset_id)
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(tmp_dir, final_dir)
        logger.info("Asset %s downloaded successfully", asset_id)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Helpers  (match SAGE objaverse_retrieval.py lines 32-85)
# ---------------------------------------------------------------------------

def load_pkl_gz(file_path: str):
    """Load a .pkl.gz file."""
    with gzip.open(file_path, "rb") as f:
        return pickle.load(f)


def extract_vertices(vertices_data):
    """Extract vertices into a NumPy array from the given data format."""
    return np.array([[v["x"], v["y"], v["z"]] for v in vertices_data])


def extract_vts(vts_data):
    """Extract UV coordinates into a NumPy array from the given data format."""
    return np.array([[v["x"], v["y"]] for v in vts_data])


def create_faces(triangles_data):
    """Create faces (triangles) from the given indices."""
    return triangles_data.reshape(-1, 3)


def get_asset_metadata(obj_data: Dict[str, Any]):
    """Extract assetMetadata from annotation data. Matches SAGE exactly."""
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")


def get_bbox_dims(obj_data: Dict[str, Any]):
    """Extract bounding box {x, y, z} in meters. Matches SAGE exactly."""
    am = get_asset_metadata(obj_data)
    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info
    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]
    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}


# ---------------------------------------------------------------------------
# Retriever  (match SAGE objaverse_retrieval.py lines 88-202)
# ---------------------------------------------------------------------------

class ObjathorRetriever:
    """Text-based 3D object retrieval using combined CLIP + SBERT similarity.

    Lazy-loads CLIP (ViT-L-14, laion2b_s32b_b82k) and SBERT (all-mpnet-base-v2)
    on first query.  Pre-computed image features (CLIP) and text features (SBERT)
    are loaded from ``~/.objathor-assets/<version>/features/``.
    """

    def __init__(self, retrieval_threshold: float = 28.0):
        self.retrieval_threshold = retrieval_threshold
        self.use_text = True  # match SAGE line 137

        # Lazy-loaded models
        self._clip_model = None
        self._clip_tokenizer = None
        self._sbert_model = None

        # Load annotations  (SAGE line 98-99)
        logger.info("Loading ObjaThor annotations from %s", OBJATHOR_ANNOTATIONS_PATH)
        objathor_annotations = compress_json.load(OBJATHOR_ANNOTATIONS_PATH)
        self.database = {**objathor_annotations}
        logger.info("Loaded %d object annotations", len(self.database))

        # Load pre-computed features  (SAGE lines 101-117)
        clip_feat_path = os.path.join(OBJATHOR_FEATURES_DIR, "clip_features.pkl")
        sbert_feat_path = os.path.join(OBJATHOR_FEATURES_DIR, "sbert_features.pkl")

        objathor_clip_features_dict = compress_pickle.load(clip_feat_path)
        objathor_sbert_features_dict = compress_pickle.load(sbert_feat_path)
        assert (
            objathor_clip_features_dict["uids"] == objathor_sbert_features_dict["uids"]
        )

        objathor_uids = objathor_clip_features_dict["uids"]
        objathor_clip_features = objathor_clip_features_dict["img_features"].astype(np.float32)
        objathor_sbert_features = objathor_sbert_features_dict["text_features"].astype(np.float32)

        import torch
        import torch.nn.functional as F

        self._torch = torch
        self._F = F

        # CLIP image features: (N, views, dim) — normalize  (SAGE lines 119-122)
        self.clip_features = torch.from_numpy(objathor_clip_features)
        self.clip_features = F.normalize(self.clip_features, p=2, dim=-1)

        # SBERT text features: (N, dim)  (SAGE lines 124-126)
        self.sbert_features = torch.from_numpy(objathor_sbert_features)

        self.asset_ids = objathor_uids

        logger.info(
            "Loaded features for %d objects (CLIP %s, SBERT %s)",
            len(self.asset_ids),
            tuple(self.clip_features.shape),
            tuple(self.sbert_features.shape),
        )

    # -- Model loading (lazy, replaces SAGE's eager init_clip / init_sbert) --

    def _ensure_clip(self):
        if self._clip_model is None:
            import open_clip
            logger.info("Loading CLIP model ViT-L-14 (laion2b_s32b_b82k)...")
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k", device="cpu"
            )
            model.eval()
            self._clip_model = model
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
            logger.info("CLIP model loaded")

    def _ensure_sbert(self):
        if self._sbert_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SBERT model all-mpnet-base-v2...")
            self._sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
            logger.info("SBERT model loaded")

    # -- retrieve()  (SAGE lines 139-172) --

    def retrieve(self, queries, threshold=28, max_num_candidates=10):
        """Search for objects matching text queries.

        Args:
            queries: List of text descriptions (e.g. ["A 3D model of sofa, modern gray"]).
            threshold: CLIP similarity threshold. Default 28 matches SAGE.
            max_num_candidates: Max results to return.

        Returns:
            List of (asset_id, combined_score) tuples sorted by descending score.
        """
        torch = self._torch
        F = self._F

        self._ensure_clip()
        self._ensure_sbert()

        # CLIP text encoding  (SAGE lines 140-145)
        with torch.no_grad():
            query_feature_clip = self._clip_model.encode_text(
                self._clip_tokenizer(queries)
            )
            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)

        # CLIP similarity: (queries, objects, views) → max over views  (SAGE lines 147-150)
        clip_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk", query_feature_clip, self.clip_features
        )
        clip_similarities = torch.max(clip_similarities, dim=-1).values

        # SBERT similarity  (SAGE lines 152-155)
        query_feature_sbert = self._sbert_model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        )
        sbert_similarities = query_feature_sbert @ self.sbert_features.T

        # Combined score  (SAGE lines 157-160)
        if self.use_text:
            similarities = clip_similarities + sbert_similarities
        else:
            similarities = clip_similarities

        # Filter by CLIP threshold  (SAGE lines 162-172)
        threshold_indices = torch.where(clip_similarities > threshold)

        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = similarities[query_index, asset_index].item()
            unsorted_results.append((self.asset_ids[asset_index], score))

        # Sorting the results in descending order by score
        results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        return results[:max_num_candidates]

    # -- compute_size_difference()  (SAGE lines 174-202) --

    def compute_size_difference(self, target_size, candidates):
        """Re-rank candidates by penalizing size mismatch.

        Args:
            target_size: (w, l, h) in cm.
            candidates: [(asset_id, score), ...].

        Returns:
            Re-ranked [(asset_id, adjusted_score), ...].
        """
        torch = self._torch

        candidate_sizes = []
        for uid, _ in candidates:
            size = get_bbox_dims(self.database[uid])
            size_list = [size["x"] * 100, size["y"] * 100, size["z"] * 100]
            size_list.sort()
            candidate_sizes.append(size_list)

        candidate_sizes = torch.tensor(candidate_sizes)

        target_size_list = list(target_size)
        target_size_list.sort()
        target_size = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size).mean(axis=1) / 100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append(
                (uid, score - size_difference[i] * 10)
            )

        # sort the candidates by score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference

    # -- load_object()  (SAGE lines 205-266) --

    def load_object(self, asset_id, infer_attributes=True, caption=None):
        """Load an ObjaThor asset and return a mesh_dict.

        Returns the same structure as SAGE: {mesh, tex_coords, texture}
        plus optionally object_attributes if infer_attributes=True.

        Args:
            asset_id: ObjaThor UID.
            infer_attributes: If True, run VL-based size inference and scale mesh.
            caption: Text description for attribute inference.

        Returns:
            mesh_dict with keys: mesh (trimesh.Trimesh), tex_coords ({vts, fts}),
            texture (ndarray float32 [0,1]), and optionally object_attributes.
        """
        _ensure_asset_available(asset_id)
        asset_path = os.path.join(OBJATHOR_ASSETS_DIR, asset_id, asset_id + ".pkl.gz")
        data = load_pkl_gz(asset_path)

        # Extracting vertices and triangles (faces) from the data  (SAGE lines 211-217)
        vertices = np.array(data["vertices"])
        triangles = np.array(data["triangles"])
        vts = np.array(data["uvs"])

        vertices = extract_vertices(vertices)
        triangles = create_faces(triangles)

        vts = extract_vts(vts)
        fts = triangles  # face texture indices = face vertex indices (SAGE line 220)

        # Load albedo texture  (SAGE lines 222-225)
        albedo_name = os.path.basename(data["albedoTexturePath"])
        input_file_dir = os.path.dirname(asset_path)
        albedo_path = os.path.join(input_file_dir, albedo_name)
        if os.path.exists(albedo_path):
            albedo = np.array(Image.open(albedo_path)).astype(np.float32) / 255.0
        else:
            albedo = np.ones((256, 256, 3), dtype=np.float32)

        # Apply yRotOffset  (SAGE lines 227-228)
        rotation_matrix = R.from_euler("y", data["yRotOffset"], degrees=True).as_matrix()[:3, :3]
        vertices = vertices @ rotation_matrix.T

        # Coordinate transform  (SAGE lines 230-237)
        transformed_vertices = vertices.copy()
        transformed_vertices[:, 1] = vertices[:, 2]    # new y = old z
        transformed_vertices[:, 0] = -vertices[:, 0]   # new -x = old x
        transformed_vertices[:, 2] = vertices[:, 1]    # new z = old y

        transformed_vertices[:, 0] = transformed_vertices[:, 0] - 0.5 * (transformed_vertices[:, 0].max() + transformed_vertices[:, 0].min())
        transformed_vertices[:, 1] = transformed_vertices[:, 1] - 0.5 * (transformed_vertices[:, 1].max() + transformed_vertices[:, 1].min())
        transformed_vertices[:, 2] = transformed_vertices[:, 2] - transformed_vertices[:, 2].min()
        # Note: SAGE line 238 (+0.001) is commented out — offset applied at line 264 instead

        # Create and return trimesh object  (SAGE lines 242-251)
        mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=triangles)

        mesh_dict = {
            "mesh": mesh,
            "tex_coords": {
                "vts": vts,
                "fts": fts.copy(),
            },
            "texture": albedo,
        }

        # Attribute inference + scaling  (SAGE lines 253-263)
        if infer_attributes:
            try:
                from services.mesh_processing import infer_real_world_size
                mesh_dict = infer_real_world_size(mesh_dict, caption or "object")
                # infer_real_world_size already adds +0.001 z offset
                return mesh_dict
            except Exception as e:
                logger.warning("Attribute inference failed: %s, returning unscaled mesh", e)

        # Unconditional z offset  (SAGE line 264)
        mesh_dict["mesh"].vertices[:, 2] = mesh_dict["mesh"].vertices[:, 2] + 0.001

        return mesh_dict

    def get_object_metadata(self, asset_id: str) -> dict:
        """Return annotation metadata for an asset."""
        obj_data = self.database[asset_id]
        bbox = get_bbox_dims(obj_data)
        return {
            "asset_id": asset_id,
            "width": round(bbox["x"], 4),
            "height": round(bbox["y"], 4),
            "length": round(bbox["z"], 4),
            "annotation": obj_data,
        }


# ---------------------------------------------------------------------------
# OBJ export convenience function
# ---------------------------------------------------------------------------

def export_objathor_as_obj(asset_id: str, output_dir: str, retriever: Optional[ObjathorRetriever] = None) -> dict:
    """Load an ObjaThor asset and export as OBJ + MTL + texture files.

    This is a convenience wrapper around ObjathorRetriever.load_object()
    that writes out OBJ files for external consumption.

    Args:
        asset_id: The ObjaThor UID.
        output_dir: Directory to write OBJ + texture files.
        retriever: Optional retriever instance (for mesh_dict reuse).

    Returns:
        Dict with obj_path, mtl_path, texture_path, dimensions, etc.
    """
    # Get mesh_dict (no attribute inference — just raw geometry)
    if retriever is not None:
        mesh_dict = retriever.load_object(asset_id, infer_attributes=False)
    else:
        # Standalone loading without retriever instance
        _ensure_asset_available(asset_id)
        asset_path = os.path.join(OBJATHOR_ASSETS_DIR, asset_id, asset_id + ".pkl.gz")
        if not os.path.exists(asset_path):
            raise FileNotFoundError(f"Asset not found: {asset_path}")
        data = load_pkl_gz(asset_path)

        vertices = extract_vertices(np.array(data["vertices"]))
        triangles = create_faces(np.array(data["triangles"]))
        vts = extract_vts(np.array(data["uvs"]))

        rotation_matrix = R.from_euler("y", data["yRotOffset"], degrees=True).as_matrix()[:3, :3]
        vertices = vertices @ rotation_matrix.T

        transformed = vertices.copy()
        transformed[:, 1] = vertices[:, 2]
        transformed[:, 0] = -vertices[:, 0]
        transformed[:, 2] = vertices[:, 1]
        transformed[:, 0] -= 0.5 * (transformed[:, 0].max() + transformed[:, 0].min())
        transformed[:, 1] -= 0.5 * (transformed[:, 1].max() + transformed[:, 1].min())
        transformed[:, 2] -= transformed[:, 2].min()
        transformed[:, 2] += 0.001

        albedo_name = os.path.basename(data["albedoTexturePath"])
        albedo_path = os.path.join(OBJATHOR_ASSETS_DIR, asset_id, albedo_name)
        if os.path.exists(albedo_path):
            albedo = np.array(Image.open(albedo_path)).astype(np.float32) / 255.0
        else:
            albedo = np.ones((256, 256, 3), dtype=np.float32)

        mesh = trimesh.Trimesh(vertices=transformed, faces=triangles)
        mesh_dict = {
            "mesh": mesh,
            "tex_coords": {"vts": vts, "fts": triangles.copy()},
            "texture": albedo,
        }

    mesh = mesh_dict["mesh"]
    vts = mesh_dict["tex_coords"]["vts"]
    fts = mesh_dict["tex_coords"]["fts"]
    texture = mesh_dict["texture"]

    os.makedirs(output_dir, exist_ok=True)

    # Save texture
    tex_out_path = os.path.join(output_dir, f"{asset_id}_texture.png")
    if texture is not None and texture.shape[0] > 1:
        tex_img = Image.fromarray((texture[:, :, :3] * 255).astype(np.uint8))
        tex_img.save(tex_out_path)
    else:
        Image.new("RGB", (256, 256), (200, 200, 200)).save(tex_out_path)

    # Write MTL
    mtl_path = os.path.join(output_dir, f"{asset_id}.mtl")
    with open(mtl_path, "w") as f:
        f.write("newmtl material0\n")
        f.write("Ka 1.0 1.0 1.0\nKd 0.8 0.8 0.8\nKs 0.5 0.5 0.5\n")
        f.write("Ns 96.0\nd 1.0\nillum 2\n")
        f.write(f"map_Kd {os.path.basename(tex_out_path)}\n")

    # Write OBJ
    obj_path = os.path.join(output_dir, f"{asset_id}.obj")
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

    # Compute dimensions (meters)
    verts = mesh.vertices
    width = float(verts[:, 0].max() - verts[:, 0].min())
    length = float(verts[:, 1].max() - verts[:, 1].min())
    height = float(verts[:, 2].max() - verts[:, 2].min())

    return {
        "asset_id": asset_id,
        "obj_path": obj_path,
        "mtl_path": mtl_path,
        "texture_path": tex_out_path,
        "dimensions": {
            "width": round(width, 4),
            "length": round(length, 4),
            "height": round(height, 4),
        },
        "vertices_count": len(verts),
        "faces_count": len(mesh.faces),
    }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_retriever: Optional[ObjathorRetriever] = None


def get_retriever() -> ObjathorRetriever:
    """Get or create the singleton retriever instance."""
    global _retriever
    if _retriever is None:
        if not os.path.exists(OBJATHOR_ANNOTATIONS_PATH):
            raise RuntimeError(
                f"ObjaThor annotations not found at {OBJATHOR_ANNOTATIONS_PATH}. "
                "Run: python -c \"from objathor.dataset import download_assets; download_assets()\""
            )
        if not os.path.exists(os.path.join(OBJATHOR_FEATURES_DIR, "clip_features.pkl")):
            raise RuntimeError(
                f"CLIP features not found at {OBJATHOR_FEATURES_DIR}/clip_features.pkl. "
                "Run: python -c \"from objathor.dataset import download_features; download_features()\""
            )
        _retriever = ObjathorRetriever()
    return _retriever
