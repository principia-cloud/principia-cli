# Scene-Gen Architecture

## Overview

The scene-gen MCP server generates simulation-ready 3D indoor scenes from natural language. It exposes **9 composite MCP tools** that orchestrate LLM reasoning, object retrieval, placement solving, rendering, and quality critics internally.

## System Diagram

```
principia-cli (orchestrator agent)
    │
    │ stdio (MCP protocol)
    ▼
server.py ──► layout_wo_robot.py (FastMCP, 9 tools)
                 │
                 ├── subagent.py ────► principia CLI subprocess (text LLM reasoning)
                 ├── vlm.py ─────────► Qwen3-VL (vision-language, OpenAI-compatible API)
                 ├── llm_client.py ──► Room generation, door/window generation via VLM
                 ├── room_solver.py ─► DFS constraint-based placement solver
                 ├── objects/ ───────► Object selection, retrieval (ObjaThor), generation (TRELLIS)
                 ├── floor_plan_materials/ ► Material generation (MatFuse diffusion model)
                 ├── nvdiffrast_rendering/ ► GPU-accelerated mesh rendering
                 ├── isaacsim/ ──────► Isaac Sim socket bridge (physics critic)
                 └── matfuse/ ───────► SVBRDF texture generation model
```

## LLM Call Strategy

All LLM calls flow through `vlm.py:call_vlm()`:

| Call type | Route | Used for |
|-----------|-------|----------|
| Text reasoning (`vlm_type="claude"`) | `subagent.py` → `principia` subprocess | Room generation, object recommendations, corrections, placement analysis |
| Vision+Language (`vlm_type="qwen"`) | Direct Qwen3-VL API | Semantic critic, object attribute inference, front estimation |
| OpenAI-compatible | Qwen3-VL API | Fallback VLM calls |

**No Anthropic API key is needed.** Text reasoning spawns a `principia` CLI subprocess that uses the user's existing Claude credentials. Image-based tasks go directly to Qwen3-VL.

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | Entry point — thin wrapper importing `mcp` from `layout_wo_robot` |
| `layout_wo_robot.py` | Main MCP server (~3800 lines). Defines all 9 `@mcp.tool()` functions |
| `key.py` | Environment variable adapter for API keys and endpoints |
| `subagent.py` | Spawns `principia` CLI for text LLM reasoning. Returns `SubagentResponse` compatible with anthropic Message format |
| `vlm.py` | Central VLM routing — `call_vlm()` dispatches to subagent or Qwen3-VL |
| `llm_client.py` | Room/door/window generation prompts via `call_vlm()` |
| `models.py` | Pydantic data models: `FloorPlan`, `Room`, `Object`, `Point3D`, etc. |
| `constants.py` | Path constants (`SERVER_ROOT_DIR`, `RESULTS_DIR`, `MATFUSE_ROOT_DIR`) |
| `validation.py` | Room layout validation (wall connectivity, dimensions) |
| `correction.py` | LLM-driven layout correction |
| `layout_parser.py` | Parse LLM-generated room layouts into model objects |
| `room_solver.py` | DFS constraint solver for object placement |
| `utils.py` | Shared utilities (JSON extraction, room priorities, etc.) |
| `visualizer.py` | Top-down room visualization |
| `room_render.py` | 3D room rendering with textures |
| `tex_utils.py` / `tex_utils_local.py` | UV mapping and texture coordinate generation |
| `foundation_models.py` | CLIP and SBERT model loading for object retrieval |
| `glb_utils.py` | GLB/glTF file processing |

### `objects/` — Object Pipeline

| File | Purpose |
|------|---------|
| `object_selection_planner.py` | Plan which objects to add, retrieve/generate them |
| `object_placement_planner.py` | Generate placement constraints from LLM |
| `object_addition_planner.py` | Plan object additions with critic feedback |
| `object_movement_planner.py` | Plan object movement operations |
| `get_objects.py` | Retrieve or generate 3D objects |
| `objaverse_retrieval.py` | Search ObjaThor (50k models) via CLIP+SBERT embeddings |
| `object_generation.py` | Generate 3D models via TRELLIS |
| `object_attribute_inference.py` | Infer object dimensions, materials via VLM rendering |
| `object_on_top_placement.py` | Specialized placement for objects on surfaces |
| `load_glb.py` | Load and process GLB meshes |

### `floor_plan_materials/` — Texture Generation

| File | Purpose |
|------|---------|
| `material_generator.py` | MatFuse wrapper for texture generation |
| `room_material.py` | Floor and wall material generation |
| `door_material.py` | Door texture generation |
| `window_material.py` | Window texture generation |
| `flux_generator.py` | Flux image generation (optional) |

### `nvdiffrast_rendering/` — GPU Rendering

| File | Purpose |
|------|---------|
| `render.py` | Rasterize textured meshes via nvdiffrast |
| `mesh.py` | Build mesh dictionaries for rendering |
| `camera.py` | Camera matrices and projections |
| `context.py` | OpenGL context management |

## The 9 MCP Tools

| Tool | Purpose |
|------|---------|
| `generate_room_layout(input_text)` | Generate room structure + materials + doors/windows from description |
| `get_current_layout()` | Return the full current layout data structure |
| `get_room_details(room_id)` | Get room details including object list |
| `list_rooms()` | List all rooms in current layout |
| `get_layout_from_json(json_file_path)` | Load a previously saved layout from JSON |
| `place_objects_in_room(room_id, placement_conditions)` | Add/remove/replace objects with full pipeline (selection → retrieval → placement → critics) |
| `get_room_information(room_id)` | Get room info with top-down visualization |
| `move_one_object_with_condition_in_room(room_id, condition)` | Move/reposition a single object |
| `get_layout_save_dir()` | Get the layout save directory path |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `QWEN_VL_URL` | Yes | Qwen3-VL API endpoint (OpenAI-compatible) |
| `QWEN_VL_MODEL` | Yes | Model name (e.g. `qwen3-vl-30b-a3b-instruct`) |
| `QWEN_VL_API_KEY` | Yes | API key for VLM endpoint |
| `RESULTS_DIR` | No | Output directory (default: `./results`) |
| `MATFUSE_CKPT` | No | MatFuse checkpoint path |
| `OBJATHOR_ASSETS_BASE_DIR` | No | ObjaThor assets base directory |
| `TRELLIS_URL` | No | TRELLIS 3D generation endpoint |
| `FLUX_SERVER_URL` | No | Flux image generation endpoint |
| `PHYSICS_CRITIC_ENABLED` | No | Enable physics critic (default: `true`) |
| `SEMANTIC_CRITIC_ENABLED` | No | Enable semantic critic (default: `true`) |
| `ISAAC_SIM_HOST` | No | Isaac Sim MCP extension host |

## Data Flow for `place_objects_in_room`

This is the most complex tool. A single call:

1. **LLM recommends objects** — via `subagent.py` → principia subprocess
2. **Retrieves 3D models** — searches ObjaThor (CLIP+SBERT), falls back to TRELLIS generation
3. **Infers attributes** — renders objects via nvdiffrast, sends images to Qwen3-VL for dimension/material estimation
4. **Generates placement constraints** — LLM analyzes room state and produces spatial constraints
5. **Solves placement** — DFS constraint solver finds valid positions
6. **Runs semantic critic** — Qwen3-VL evaluates room quality from rendered image
7. **Runs physics critic** — Isaac Sim checks physical stability (if enabled)
8. **Returns results** — object list, critic feedback, updated room state

## Installation

```bash
# Automated (from principia-cli root):
./scripts/setup-scene-gen.sh

# Manual:
pip install -r requirements.txt
# GPU deps (need CUDA toolkit):
sudo apt install nvidia-cuda-toolkit
CUDA_HOME=/usr pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
CUDA_HOME=/usr pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
```
