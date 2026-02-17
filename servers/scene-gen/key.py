"""Environment-variable based configuration adapter.

Replaces SAGE's key.json with env vars provided via MCP config.
Text reasoning goes through principia subagent (no Anthropic key needed).
VLM calls go through Qwen3-VL directly.
"""
import os
import sys
import hashlib
import time

# --- VLM (Qwen3-VL) ---
API_TOKEN = os.environ.get("QWEN_VL_API_KEY", "")
API_URL_QWEN = os.environ.get("QWEN_VL_URL", "https://dashscope-us.aliyuncs.com/compatible-mode/v1")

# Map all vlm_types to the Qwen model (or env override)
_qwen_model = os.environ.get("QWEN_VL_MODEL", "qwen3-vl-30b-a3b-instruct")
MODEL_DICT = {
    "qwen": _qwen_model,
    "openai": _qwen_model,
    "claude": _qwen_model,  # claude type rerouted to subagent in vlm.py
    "glmv": _qwen_model,
}

API_URL_DICT = {
    "qwen": API_URL_QWEN,
    "openai": API_URL_QWEN,
    "glmv": API_URL_QWEN,
}

# --- External services ---
SERVER_URL = os.environ.get("TRELLIS_URL", "")
FLUX_SERVER_URL = os.environ.get("FLUX_SERVER_URL", "")

# --- Anthropic (unused — text calls go through principia subagent) ---
ANTHROPIC_API_KEY = ""

if SERVER_URL:
    print(f"TRELLIS SERVER_URL: {SERVER_URL}", file=sys.stderr)


def slurm_job_id_to_port(job_id, port_start=8080, port_end=40000):
    """Hash-based mapping function to convert SLURM job ID to a port number."""
    job_id_str = str(job_id)
    hash_obj = hashlib.md5(job_id_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    port_range = port_end - port_start + 1
    return port_start + (hash_int % port_range)


def is_client_valid(api_service: str) -> bool:
    """Stub — always valid since we use env-var tokens."""
    return True


def get_client_api_key(api_service: str) -> str:
    """Stub — returns the Qwen API token."""
    return API_TOKEN


def setup_oai_client():
    """Stub — returns None; not used in principia-cli."""
    return None
