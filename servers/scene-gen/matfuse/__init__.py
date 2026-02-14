"""MatFuse material texture generation.

Adapted from https://github.com/gvecchio/MatFuse — MIT License.
Generates SVBRDF texture maps (albedo, roughness, normal) from text prompts.
"""

import os
import sys
import logging

logger = logging.getLogger("scene-gen.matfuse")

# Ensure matfuse package root and sub-packages are importable
_matfuse_dir = os.path.dirname(os.path.abspath(__file__))
if _matfuse_dir not in sys.path:
    sys.path.insert(0, _matfuse_dir)

# Try to make taming-transformers importable (pip-installed or on PYTHONPATH)
try:
    import taming  # noqa: F401
except ImportError:
    logger.debug("taming-transformers not found; MatFuse model loading will fail")

from matfuse.generate import generate_texture_map_from_prompt  # noqa: F401
