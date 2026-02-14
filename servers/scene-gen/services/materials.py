"""Material texture generation via MatFuse and Flux.

Adapted from SAGE server/floor_plan_materials/ and server/layout.py.
Both generators use lazy-loaded singletons — no external server needed.
Fallback chain: MatFuse → Flux (in-process) → placeholder.
Post-processing: repeat_texture() for seamless tiling (ported from SAGE).
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger("scene-gen.materials")

# Lazy-loaded generator functions
_matfuse_generate = None
_flux_pipeline = None
_matfuse_attempted = False
_flux_attempted = False


def _try_load_matfuse():
    """Try to load MatFuse generate function from local matfuse package."""
    global _matfuse_generate, _matfuse_attempted
    if _matfuse_attempted:
        return
    _matfuse_attempted = True
    try:
        from matfuse.generate import generate_texture_map_from_prompt
        _matfuse_generate = generate_texture_map_from_prompt
        logger.info("MatFuse loaded from local matfuse package")
    except ImportError as e:
        logger.info("MatFuse not available (missing deps): %s", e)
    except Exception as e:
        logger.warning("Failed to load MatFuse: %s", e)


def _try_load_flux():
    """Try to load Flux pipeline directly via diffusers (no server needed)."""
    global _flux_pipeline, _flux_attempted
    if _flux_attempted:
        return
    _flux_attempted = True
    try:
        import torch
        from diffusers import FluxPipeline

        model_id = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-Krea-dev")
        dtype = torch.bfloat16 if torch.cuda.is_available() or hasattr(torch.backends, "mps") else torch.float32
        logger.info("Loading Flux pipeline from %s (dtype=%s)...", model_id, dtype)
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.enable_model_cpu_offload()
        _flux_pipeline = pipe
        logger.info("Flux pipeline loaded successfully")
    except ImportError as e:
        logger.info("Flux not available (missing diffusers/torch): %s", e)
    except Exception as e:
        logger.warning("Failed to load Flux pipeline: %s", e)


def _flux_generate(prompt: str):
    """Generate an image from a text prompt using the Flux pipeline.

    Matches SAGE's Flux parameters: 512x512, guidance_scale=4.5.

    Returns:
        PIL Image (512x512) or None on failure.
    """
    if _flux_pipeline is None:
        return None
    try:
        import torch
        seed = int.from_bytes(os.urandom(4), "big") % (2**32)
        generator = torch.Generator().manual_seed(seed)
        result = _flux_pipeline(
            prompt,
            height=512,
            width=512,
            guidance_scale=4.5,
            num_inference_steps=28,
            generator=generator,
        )
        return result.images[0]
    except Exception as e:
        logger.error("Flux generation failed for '%s': %s", prompt, e)
        return None


def repeat_texture(texture_pil, times=2):
    """Crop center 80% and tile 2x2, repeated `times` times.

    Ported from SAGE server/layout.py:635-642.
    Creates seamless tiling by avoiding edge artifacts.
    512x512 → (after 2 iterations) ~1636x1636.
    """
    for _ in range(times):
        tex_np = np.array(texture_pil).astype(np.float32) / 255.0
        h, w = tex_np.shape[:2]
        # Crop center 80%
        tex_np = tex_np[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]
        # Tile 2x2
        tex_np = np.concatenate([tex_np, tex_np], axis=0)
        tex_np = np.concatenate([tex_np, tex_np], axis=1)
        from PIL import Image
        texture_pil = Image.fromarray(np.clip(tex_np * 255, 0, 255).astype(np.uint8))
    return texture_pil


def init_materials():
    """Initialize material generation backends. Call once at server startup.

    Eagerly loads the model weights so the first generate_materials call
    doesn't block for minutes while the 4.3GB checkpoint loads.
    """
    _try_load_matfuse()
    if _matfuse_generate is not None:
        # Warm up: force model load now instead of on first inference call
        try:
            logger.info("Warming up MatFuse model (loading checkpoint)...")
            from matfuse.generate import _get_model
            _get_model()
            logger.info("MatFuse model ready")
        except Exception as e:
            logger.warning("MatFuse warmup failed: %s", e)
    else:
        _try_load_flux()


def generate_material_texture(prompt: str, output_path: str) -> str:
    """Generate a PBR material texture from a text description.

    Fallback chain: MatFuse → Flux (in-process) → placeholder.
    All generated textures are post-processed with repeat_texture() for
    seamless tiling, matching SAGE's original behavior.

    Args:
        prompt: Text description of the material (e.g., "warm oak hardwood floor").
        output_path: Where to save the generated texture image.

    Returns:
        The path to the saved texture, or a placeholder path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Try MatFuse first (512x512 PBR albedo)
    if _matfuse_generate is not None:
        try:
            texture_pil = _matfuse_generate(prompt)
            texture_pil = repeat_texture(texture_pil, times=2)
            texture_pil.save(output_path)
            logger.info("Material generated (MatFuse): %s -> %s", prompt, output_path)
            return output_path
        except Exception as e:
            logger.error("MatFuse generation failed for '%s': %s", prompt, e)

    # Try Flux in-process — prompt matches SAGE: "A uniform, flat UV texture image of ..."
    if _flux_pipeline is None and not _flux_attempted:
        _try_load_flux()
    if _flux_pipeline is not None:
        texture_pil = _flux_generate(f"A uniform, flat UV texture image of {prompt}")
        if texture_pil is not None:
            texture_pil = repeat_texture(texture_pil, times=2)
            texture_pil.save(output_path)
            logger.info("Material generated (Flux): %s -> %s", prompt, output_path)
            return output_path

    # Placeholder: simple colored image
    try:
        from PIL import Image
        img = Image.new("RGB", (512, 512), (180, 160, 140))
        img.save(output_path)
        logger.info("Placeholder material saved: %s", output_path)
        return output_path
    except ImportError:
        logger.warning("Pillow not available, returning path without creating file")
        return output_path


def generate_room_materials(
    room_id: str,
    material_descriptions: Dict[str, str],
    output_dir: str,
) -> Dict[str, str]:
    """Generate material textures for a room's surfaces.

    Args:
        room_id: The room identifier.
        material_descriptions: Mapping of surface name to description,
            e.g. {"floor": "warm oak hardwood", "wall": "cream painted drywall"}.
        output_dir: Directory to save generated textures.

    Returns:
        Mapping of surface name to texture file path.
    """
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    for surface, desc in material_descriptions.items():
        filename = f"{room_id}_{surface}_material.png"
        out_path = os.path.join(output_dir, filename)
        results[surface] = generate_material_texture(desc, out_path)
    return results
