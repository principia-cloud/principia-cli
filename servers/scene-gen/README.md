# Scene Generation MCP Server

Generate simulation-ready 3D indoor scenes from natural language descriptions. Uses 9 composite MCP tools that internally orchestrate LLM reasoning, 3D object retrieval, constraint-based placement, texture generation, and quality critics.

## Quick Start

```bash
# Full setup (from principia-cli root)
./scripts/setup-scene-gen.sh

# Or manual
pip install -r requirements.txt
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## How It Works

```
User: "Generate a cozy 4m x 5m bedroom with warm modern style"
    │
    │  principia CLI orchestrates via SKILL.md workflow
    ▼
┌─────────────────────────────────────────────────────┐
│  MCP Server (9 tools, stdio transport)              │
│                                                     │
│  1. generate_room_layout(input_text)                │
│     └─ LLM generates walls → validates → adds       │
│        materials, doors, windows                     │
│                                                     │
│  2. place_objects_in_room(room_id, conditions) x4    │
│     └─ LLM recommends objects → retrieves 3D models │
│        → solves placement → runs critics             │
│                                                     │
│  3. get_room_information(room_id)                    │
│     └─ Returns state + top-down visualization        │
│                                                     │
│  4. move_one_object_with_condition_in_room(...)      │
│     └─ Fine-tune individual object positions         │
└─────────────────────────────────────────────────────┘
    │
    ▼
Output: USD scene, JSON layout, rendered images
```

## Configuration

Environment variables (set in `~/.principia/data/settings/principia_mcp_settings.json`):

| Variable | Required | Description |
|----------|----------|-------------|
| `QWEN_VL_URL` | Yes | Qwen3-VL API endpoint (OpenAI-compatible) |
| `QWEN_VL_MODEL` | Yes | Model name (e.g. `qwen3-vl-30b-a3b-instruct`) |
| `QWEN_VL_API_KEY` | Yes | API key for VLM |
| `RESULTS_DIR` | No | Output directory (default: `./results`) |
| `TRELLIS_URL` | No | TRELLIS 3D model generation endpoint |
| `FLUX_SERVER_URL` | No | Flux image generation endpoint |
| `MATFUSE_CKPT` | No | MatFuse checkpoint path |
| `OBJATHOR_ASSETS_BASE_DIR` | No | ObjaThor 3D asset database |
| `PHYSICS_CRITIC_ENABLED` | No | Enable Isaac Sim physics critic (`true`/`false`) |
| `SEMANTIC_CRITIC_ENABLED` | No | Enable VLM semantic critic (`true`/`false`) |
| `ISAAC_SIM_HOST` | No | Isaac Sim host for physics simulation |

Example config: [`mcp_settings_example.json`](mcp_settings_example.json)

## MCP Tools Reference

| Tool | Arguments | Description |
|------|-----------|-------------|
| `generate_room_layout` | `input_text` | Generate room structure + materials + doors/windows |
| `get_current_layout` | — | Get full layout data structure |
| `get_room_details` | `room_id` | Get room details with object list |
| `list_rooms` | — | List all rooms |
| `get_layout_from_json` | `json_file_path` | Load saved layout |
| `place_objects_in_room` | `room_id, placement_conditions` | Full object pipeline: select → retrieve → place → critique |
| `get_room_information` | `room_id` | Room state + visualization |
| `move_one_object_with_condition_in_room` | `room_id, condition` | Reposition single object |
| `get_layout_save_dir` | — | Get output directory path |

## Agent Workflow

The principia agent follows the workflow in [`skill/SKILL.md`](skill/SKILL.md):

1. **Stage 1**: `generate_room_layout` — create room from description
2. **Stage 2**: `get_room_information` — review layout
3. **Stage 3**: `place_objects_in_room` x4 — furnish iteratively (anchors → combos → surfaces → decor)
4. **Stage 4**: Review + `move_one_object_with_condition_in_room` — adjust
5. **Stage 5**: `get_layout_save_dir` — export

## Directory Structure

```
servers/scene-gen/
├── server.py                    # Entry point (thin wrapper)
├── layout_wo_robot.py           # Main MCP server (9 tools)
├── key.py                       # Env-var configuration adapter
├── subagent.py                  # principia subprocess for text LLM
├── vlm.py                       # Central VLM routing (subagent / Qwen3-VL)
├── llm_client.py                # Room generation prompts
├── models.py                    # Pydantic data models
├── constants.py                 # Path constants
├── validation.py                # Layout validation
├── correction.py                # LLM-driven corrections
├── layout_parser.py             # Parse LLM room output
├── room_solver.py               # DFS constraint placement solver
├── utils.py                     # Shared utilities
├── visualizer.py                # Top-down visualization
├── room_render.py               # 3D room rendering
├── tex_utils.py                 # UV mapping
├── tex_utils_local.py           # Local texture utilities
├── foundation_models.py         # CLIP + SBERT loading
├── glb_utils.py                 # GLB file processing
├── objects/                     # Object pipeline (10 files)
├── floor_plan_materials/        # Texture generation (5 files)
├── nvdiffrast_rendering/        # GPU rendering (4 files)
├── isaacsim/                    # Isaac Sim bridge
├── matfuse/                     # SVBRDF texture model
├── skill/SKILL.md               # Agent workflow instructions
├── requirements.txt             # Python dependencies
├── mcp_settings_example.json    # Example MCP config
├── ARCHITECTURE.md              # Detailed architecture docs
└── CLAUDE.md                    # Agent development guide
```

## GPU Requirements

The server requires CUDA GPU for:
- **nvdiffrast**: Differentiable mesh rendering (object attribute inference)
- **pytorch3d**: 3D model I/O
- **MatFuse**: SVBRDF texture generation
- **CLIP**: Object embedding computation

Install GPU deps (requires CUDA toolkit / `nvcc`):
```bash
sudo apt install nvidia-cuda-toolkit
CUDA_HOME=/usr pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
CUDA_HOME=/usr pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
```
