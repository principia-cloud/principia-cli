"""HTTP client to remote Qwen3-VL-30B for vision-language tasks.

Adapted from SAGE server/objects/object_attribute_inference.py and server/vlm.py.
Uses the OpenAI-compatible API served by vLLM.
"""

import base64
import os
import logging
from typing import Optional, List, Dict, Any

from openai import OpenAI

logger = logging.getLogger("scene-gen.qwen_vl")


def _get_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("QWEN_VL_URL", "http://localhost:8080/v1"),
        api_key=os.environ.get("QWEN_VL_API_KEY", "token-placeholder"),
    )


def _get_model() -> str:
    return os.environ.get("QWEN_VL_MODEL", "Qwen3-VL-30B-A3B-Instruct")


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_qwen_vl(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> str:
    """Send a text+image prompt to Qwen3-VL and return the response text.

    Args:
        prompt: The text prompt.
        image_paths: Optional list of local image file paths to include.
        max_tokens: Maximum response tokens.
        temperature: Sampling temperature.

    Returns:
        The model's text response.
    """
    client = _get_client()
    model = _get_model()

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    for path in (image_paths or []):
        b64 = encode_image_to_base64(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def estimate_front_direction(image_paths: List[str], category: str = "object") -> Dict[str, Any]:
    """Use Qwen3-VL to estimate the front-facing direction of a 3D object.

    Sends 4 rendered views and asks the model to identify which is the front.

    Args:
        image_paths: Exactly 4 PNG paths (views at 90, 180, 270, 0 degrees).
        category: Object category name for context.

    Returns:
        Dict with 'front_index' (0-3) and 'reasoning'.
    """
    import json as _json

    prompt = (
        f"You have been given 4 images of a {category}, each taken from a different angle. "
        "Identify the image that shows the 'front view' — the perspective where the object's "
        "main face or most important features are most clearly visible.\n\n"
        "Guidelines:\n"
        "1. The front view shows the most significant or visible face.\n"
        "2. For cabinets, front = doors/drawers visible. For chairs, front = seat and backrest.\n"
        "3. The front is where the object faces the camera directly.\n"
        "4. Indices start from 0.\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"front_index": <0-3>, "reasoning": "<brief explanation>"}'
    )

    response_text = call_qwen_vl(prompt, image_paths)
    try:
        return _json.loads(response_text)
    except _json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return _json.loads(match.group())
        return {"front_index": 0, "reasoning": "Failed to parse response, defaulting to index 0"}


def analyze_image(prompt: str, image_paths: List[str]) -> str:
    """General-purpose image analysis using Qwen3-VL."""
    return call_qwen_vl(prompt, image_paths)


def infer_object_attributes(
    image_path: str,
    caption: str,
    vertices: "np.ndarray",
) -> Dict[str, Any]:
    """Use Qwen3-VL to estimate real-world dimensions and PBR parameters.

    Adapted from SAGE server/objects/object_attribute_inference.py infer_attributes_from_claude().

    Args:
        image_path: Path to a rendered front-right-up view PNG.
        caption: Text description of the object.
        vertices: Mesh vertices (N,3) for computing relative dimension ordering.

    Returns:
        Dict with height, width, length (meters), weight (kg), pbr_parameters, etc.
    """
    import json as _json
    import re
    import numpy as np

    gen_w = float(vertices[:, 0].max() - vertices[:, 0].min())
    gen_l = float(vertices[:, 1].max() - vertices[:, 1].min())
    gen_h = float(vertices[:, 2].max() - vertices[:, 2].min())

    dims = sorted(
        [("width", gen_w), ("length", gen_l), ("height", gen_h)],
        key=lambda x: x[1],
        reverse=True,
    )
    ordering = " > ".join(d[0] for d in dims)

    prompt = f"""You are a professional 3D artist and object analyst.

GIVEN CAPTION: {caption}

TASK:
Analyze the 3D object in the image (rendered from the upper right view) and estimate its real-world dimensions.

HEIGHT DEFINITION:
- Height = distance from lowest to highest point of the object.

RELATIVE SIZE INFORMATION:
- The relative ordering of the object's dimensions (larger to smaller) is: {ordering}
- Your estimated width, length, and height should follow the same ordering.

Respond with ONLY a JSON object:
{{
    "name": "<specific object name>",
    "semantic_alignment": true,
    "explanation": "<reasoning for dimension estimates>",
    "width": <width in meters, number only>,
    "length": <length in meters, number only>,
    "height": <height in meters, number only>,
    "weight": <weight in kg, number only>,
    "pbr_parameters": {{
        "metallic": <0.0-1.0>,
        "roughness": <0.0-1.0>
    }}
}}"""

    response_text = call_qwen_vl(prompt, [image_path], max_tokens=8000, temperature=0.5)

    # Extract JSON from response
    try:
        result = _json.loads(response_text)
    except _json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            result = _json.loads(match.group())
        else:
            logger.warning("Failed to parse attribute response, using defaults")
            return {"height": 1.0, "width": 1.0, "length": 1.0, "weight": 1.0,
                    "pbr_parameters": {"metallic": 0.0, "roughness": 0.5}}

    # Validate numeric fields
    for field in ("height", "width", "length", "weight"):
        try:
            result[field] = float(result[field])
        except (ValueError, TypeError, KeyError):
            result[field] = 1.0

    # Validate PBR
    pbr = result.get("pbr_parameters", {})
    if not isinstance(pbr, dict):
        pbr = {}
    try:
        pbr["metallic"] = max(0.0, min(1.0, float(pbr.get("metallic", 0.0))))
    except (ValueError, TypeError):
        pbr["metallic"] = 0.0
    try:
        pbr["roughness"] = max(0.0, min(1.0, float(pbr.get("roughness", 0.5))))
    except (ValueError, TypeError):
        pbr["roughness"] = 0.5
    result["pbr_parameters"] = pbr

    return result
