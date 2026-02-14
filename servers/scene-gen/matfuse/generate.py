"""MatFuse texture map generation from text prompts.

Adapted from sage/matfuse-sd/src/generate.py — hardcoded paths removed.
"""

from PIL import Image
import sys
import os
import numpy as np
import uuid
import shutil
import random
from typing import Optional

# Resolve paths relative to this file
_MATFUSE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure the matfuse package root is on sys.path so ldm/ and utils/ are importable
if _MATFUSE_DIR not in sys.path:
    sys.path.insert(0, _MATFUSE_DIR)

# Config and checkpoint paths (can be overridden via environment variables)
_CONFIG_PATH = os.path.join(_MATFUSE_DIR, "configs", "diffusion", "matfuse-ldm-vq_f8.yaml")
_CKPT_PATH = os.environ.get(
    "MATFUSE_CKPT",
    os.path.join(os.path.expanduser("~/.principia/data/scene-gen/matfuse"), "matfuse-full.ckpt"),
)

# Lazy-loaded model singleton
_matfuse_model = None


def _get_model():
    """Load the MatFuse model lazily on first use."""
    global _matfuse_model
    if _matfuse_model is not None:
        return _matfuse_model

    from utils.inference_helpers import get_model
    _matfuse_model = get_model(_CONFIG_PATH, _CKPT_PATH)
    return _matfuse_model


def pseudo_render_texture_map(albedo, roughness, normal_map, light_dir=np.array([0, 0, 1])):
    """Simple Lambertian rendering of SVBRDF maps for preview."""
    albedo = albedo / 255.0
    roughness = roughness / 255.0
    normal_map = (normal_map / 255.0) * 2 - 1

    roughness = np.mean(roughness, axis=-1)
    light_dir = light_dir / np.linalg.norm(light_dir)

    dot_product = np.einsum('ijk,k->ij', normal_map, light_dir)
    dot_product = np.clip(dot_product, 0, 1)
    diffuse = dot_product * (1 - roughness) + roughness
    rendered_map = albedo * diffuse[..., np.newaxis]
    rendered_map = np.clip(rendered_map * 255, 0, 255).astype(np.uint8)
    return rendered_map


def generate_texture_map_from_prompt(prompt: str) -> Image.Image:
    """Generate an albedo texture map from a text prompt.

    Args:
        prompt: Material description (e.g. "warm oak hardwood planks").

    Returns:
        PIL Image of the albedo map (512x512).
    """
    from utils.inference_helpers import run_generation

    model = _get_model()

    save_dir = os.path.join("/tmp/matfuse_texture_map", str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    try:
        result_tex = run_generation(
            model,
            input_image_emb=None,
            input_image_palette=None,
            sketch=None,
            prompt=prompt,
            num_samples=1,
            image_resolution=512,
            ddim_steps=50,
            seed=random.randint(0, 1_000_000),
            eta=0.0,
            guidance_scale=10.0,
            save_dir=save_dir,
        )[-1]

        H, W = result_tex.shape[0] // 2, result_tex.shape[1] // 2
        albedo = result_tex[:H, :W]
        return Image.fromarray(albedo)
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
