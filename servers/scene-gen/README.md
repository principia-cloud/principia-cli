# Scene Generation MCP Server

A Python MCP server for principia-cli that generates simulation-ready 3D indoor environments from natural language descriptions. Inspired by NVIDIA's SAGE framework.

## Architecture

```
principia-cli (coordinator + subagents)
    │
    │ stdio (MCP)
    ▼
MCP Server (Python, local)
    ├── HTTP ──────► Qwen3-VL (remote A4000, port 8080)
    ├── HTTP ──────► TRELLIS (Docker, port 8080)
    ├── TCP ───────► Isaac Sim (localhost)
    ├── in-process ► Flux/MatFuse (GPU or CPU offload)
    ├── in-process ► CLIP + SBERT (CPU)
    └── local disk ► Objaverse assets
```

## Setup

```bash
cd servers/scene-gen
pip install -r requirements.txt
```

## Configuration

Add to `~/.principia/data/settings/principia_mcp_settings.json`:

```json
{
  "scene-gen": {
    "command": "python",
    "args": ["servers/scene-gen/server.py"],
    "cwd": "<principia-cli-root>",
    "env": {
      "QWEN_VL_URL": "http://<a4000-host>:8080/v1",
      "QWEN_VL_MODEL": "Qwen3-VL-30B-A3B-Instruct",
      "TRELLIS_URL": "http://<trellis-host>:8080",
      "ISAAC_SIM_HOST": "localhost",
      "ISAAC_SIM_PORT": "8080",
      "RESULTS_DIR": "./results"
    }
  }
}
```

## Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `QWEN_VL_URL` | `http://192.168.1.50:8080/v1` | Qwen3-VL endpoint (OpenAI-compatible) |
| `QWEN_VL_MODEL` | `Qwen3-VL-30B-A3B-Instruct` | Model name for API calls |
| `QWEN_VL_API_KEY` | `token-abc123` | API key (often a placeholder for vLLM) |
| `TRELLIS_URL` | `http://localhost:8080` | TRELLIS 3D generation endpoint |
| `ISAAC_SIM_HOST` | `localhost` | Isaac Sim extension host |
| `ISAAC_SIM_PORT` | `8080` | Isaac Sim extension port |
| `OBJAVERSE_DIR` | `/data/objaverse` | Path to Objaverse assets |
| `MATFUSE_DIR` | `/path/to/matfuse-sd/src` | MatFuse installation (optional) |
| `RESULTS_DIR` | `./results` | Where generated scenes are saved |

## MCP Tools

### Scene Management
- `create_room(room_json)` — Create validated room layout
- `add_doors_windows(layout_id, placement_json)` — Add doors/windows
- `get_layout(layout_id)` — Get full FloorPlan JSON
- `get_room_info(layout_id, room_id)` — Get room details + visualization

### Materials
- `generate_materials(layout_id, material_descriptions)` — Generate PBR textures

### Objects
- `search_objects(object_specs)` — Search Objaverse models
- `generate_3d_model(description, target_size)` — Generate via TRELLIS
- `place_objects(layout_id, room_id, objects_json, constraints_json)` — DFS placement
- `remove_objects(layout_id, room_id, object_ids)` — Remove objects

### Vision (Qwen3-VL)
- `analyze_floor_plan(layout_id, prompt)` — Analyze floor plan image
- `run_semantic_critic(layout_id, room_id)` — Evaluate room quality

### Isaac Sim
- `build_scene(layout_id, room_id)` — Build scene in Isaac Sim
- `simulate_physics(layout_id)` — Run physics simulation
- `export_usd(layout_id, output_path)` — Export as USD

## Testing

```bash
# Standalone server test
python server.py

# With principia-cli
principia "Generate a cozy bedroom" --mcp scene-gen
```
