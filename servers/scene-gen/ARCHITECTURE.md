# Scene Generation System — Full Architecture

This document describes the complete data flow from server installation to CLI invocation to the full generation loop and final outputs.

---

## 1. System Overview

```
┌───────────────────────────────────────────────────────┐
│  User: principia "generate a cozy bedroom"            │
└───────────────────┬───────────────────────────────────┘
                    ▼
┌───────────────────────────────────────────────────────┐
│  principia-cli (TypeScript, Commander.js + Ink)       │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Controller → Task → API (Claude/GPT) loop       │  │
│  │   ├── System prompt (tools + skills + MCP list) │  │
│  │   ├── Tool router (file, cmd, mcp, skill, …)   │  │
│  │   └── Message state + history                   │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  Skills:  ~/.principia/skills/scene-gen/SKILL.md      │
│  MCP cfg: ~/.principia/data/settings/principia_mcp_settings.json │
└───────────────────┬───────────────────────────────────┘
                    │ stdio (JSON-RPC over stdin/stdout)
                    ▼
┌───────────────────────────────────────────────────────┐
│  scene-gen MCP Server (Python, FastMCP)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐  │
│  │ Solvers   │ │ Rendering│ │ Services             │  │
│  │ DFS place │ │ matplotlib│ │ Qwen-VL  (HTTP)     │  │
│  │ room solve│ │ PNG render│ │ TRELLIS  (HTTP)     │  │
│  │ validation│ │           │ │ Isaac Sim (TCP)     │  │
│  │           │ │           │ │ MatFuse  (in-proc)  │  │
│  └──────────┘ └──────────┘ └──────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

The system has three layers:

1. **principia-cli** — TypeScript CLI that hosts the LLM conversation loop and routes tool calls
2. **SKILL.md** — Markdown instructions injected into the LLM's context, defining the 8-stage workflow
3. **scene-gen MCP server** — Python process that does the actual computation (geometry, rendering, physics)

---

## 2. Installation & Setup

### 2.1 Install principia-cli

```bash
# Clone and build
git clone <repo> principia-cli
cd principia-cli
npm install && npm run build

# The CLI binary is symlinked at ~/.principia/bin/principia
# → points to ~/.principia/source/cli/dist/cli.mjs
```

The CLI is a Commander.js + Ink (React-for-terminal) application. Entry point: `cli/src/index.ts`.

### 2.2 Install the MCP server

```bash
cd servers/scene-gen
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Key dependencies:**
- `mcp[cli]>=1.9.4` — Model Context Protocol SDK (FastMCP server framework)
- `shapely>=2.0` — 2D polygon geometry for placement solver
- `rtree>=1.0` — R-tree spatial indexing for grid point removal
- `scipy` — Interpolation for distance constraint curves
- `matplotlib` — 2D floor plan rendering
- `openai>=1.0` — HTTP client for Qwen3-VL (OpenAI-compatible API)
- `torch`, `trimesh`, `pyrender` — 3D model processing

### 2.3 Register the MCP server

Edit `~/.principia/data/settings/principia_mcp_settings.json`:

```json
{
  "mcpServers": {
    "scene-gen": {
      "type": "stdio",
      "command": "/path/to/servers/scene-gen/.venv/bin/python",
      "args": ["servers/scene-gen/server.py"],
      "cwd": "/path/to/principia-cli",
      "env": {
        "QWEN_VL_URL": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "QWEN_VL_MODEL": "qwen3-vl-30b-a3b-instruct",
        "QWEN_VL_API_KEY": "sk-...",
        "TRELLIS_URL": "",
        "ISAAC_SIM_HOST": "",
        "ISAAC_SIM_PORT": "",
        "MATFUSE_CKPT": "",
        "RESULTS_DIR": "./results"
      },
      "timeout": 60,
      "autoApprove": [
        "create_room", "add_doors_windows", "get_layout", "get_room_info",
        "generate_materials", "search_objects", "generate_3d_model",
        "place_objects", "remove_objects", "analyze_floor_plan",
        "run_semantic_critic", "build_scene", "simulate_physics_tool",
        "export_usd_tool"
      ],
      "disabled": false
    }
  }
}
```

Key fields:
- **`type: "stdio"`** — The CLI spawns the Python process and communicates via stdin/stdout JSON-RPC
- **`command` + `args`** — The exact Python binary and script to run
- **`cwd`** — Working directory for the subprocess
- **`env`** — Environment variables injected into the subprocess
- **`autoApprove`** — Tool names that don't require user confirmation
- **`timeout`** — Max seconds per tool call (60s)

### 2.4 Install the skill

Place `SKILL.md` at:

```
~/.principia/skills/scene-gen/SKILL.md
```

The file has YAML frontmatter:

```yaml
---
name: scene-gen
description: Generate simulation-ready 3D indoor scenes from natural language descriptions using the SAGE workflow
---
```

The rest is the full 8-stage workflow instructions that get injected into the LLM's context.

---

## 3. What Happens When You Run `principia "generate a cozy bedroom"`

### 3.1 CLI Startup

**File:** `cli/src/index.ts`

1. **Parse arguments** via Commander.js
   - Prompt: `"generate a cozy bedroom"`
   - Flags: `--json` (JSON streaming), `-y` (auto-approve), `--model`, `--timeout`, etc.
2. **Detect input mode**:
   - TTY → Interactive Ink UI (React components in terminal)
   - Piped → Plain text mode (JSON lines or final-result-only)
3. **Initialize CLI context** (`initializeCli()`):
   - Create `StateManager` (reads `~/.principia/data/globalState.json`)
   - Create `McpHub` (reads MCP settings, spawns servers)
   - Create `Controller` (main orchestrator)
   - Set up `HostProvider` with CLI-specific services

### 3.2 MCP Server Spawn

**File:** `src/services/mcp/McpHub.ts`

When `McpHub` initializes, it reads `principia_mcp_settings.json` and for each server:

1. **Parse config** — extract `command`, `args`, `cwd`, `env`
2. **Expand env vars** — `${VAR_NAME}` patterns resolved from host environment
3. **Create transport** — `new StdioClientTransport({command, args, cwd, env, stderr: "pipe"})`
4. **Spawn subprocess** — The SDK forks the Python process: `.venv/bin/python servers/scene-gen/server.py`
5. **MCP handshake** — JSON-RPC `initialize` call, server responds with capabilities
6. **Discover tools** — `listTools()` returns all 14 tool definitions with input schemas
7. **Store connection** — Tools available for the LLM to call

The Python server starts via `mcp.run()` (FastMCP), which:
- Reads JSON-RPC messages from stdin
- Dispatches to `@mcp.tool()` decorated functions
- Writes JSON-RPC responses to stdout
- Logs to stderr (captured by the CLI)

### 3.3 Skill Discovery

**File:** `src/core/context/instructions/user-instructions/skills.ts`

On startup, the CLI scans for skills:

1. **Global:** `~/.principia/skills/*/SKILL.md`
2. **Project:** `.principia/skills/*/SKILL.md`, `.principiarules/skills/*/SKILL.md`
3. **Parse frontmatter** — extract `name` and `description`
4. **Check toggles** — skills can be enabled/disabled per-workspace
5. **Return metadata** — name + description (content loaded lazily on use)

### 3.4 System Prompt Construction

Before the first API call, the CLI builds a system prompt that includes:

**MCP servers section** (`src/core/prompts/system-prompt/components/mcp.ts`):
```
MCP SERVERS

The Model Context Protocol (MCP) enables communication between the system
and locally running MCP servers that provide additional tools...

## scene-gen (.venv/bin/python servers/scene-gen/server.py)

### Available Tools
- create_room: Create a validated room layout from JSON...
    Input Schema: {"type": "object", "properties": {"room_json": ...}}
- place_objects: Place objects in a room using the DFS placement solver...
    Input Schema: {"type": "object", "properties": {"layout_id": ..., ...}}
[... all 14 tools with full JSON schemas ...]
```

**Skills section** (`src/core/prompts/system-prompt/components/skills.ts`):
```
SKILLS

The following skills provide specialized instructions for specific tasks.
When a user's request matches a skill description, use the use_skill tool.

Available skills:
  - "scene-gen": Generate simulation-ready 3D indoor scenes...

To use a skill:
1. Match the user's request to a skill based on its description
2. Call use_skill with the skill_name parameter
3. Follow the instructions returned by the tool
```

**Tool definitions** — The LLM also gets XML tool specs for `use_mcp_tool` and `use_skill`:
```xml
<use_mcp_tool>
  <server_name>scene-gen</server_name>
  <tool_name>place_objects</tool_name>
  <arguments>{"layout_id": "...", ...}</arguments>
</use_mcp_tool>
```

### 3.5 LLM Conversation Loop

**File:** `src/core/task/` (Task, ToolExecutor, handlers)

The conversation loop is:

```
User message: "generate a cozy bedroom"
    ↓
API call #1: System prompt + user message → Claude
    ↓
Claude response: "I see this matches the scene-gen skill. Let me load it."
    → Tool call: use_skill(skill_name="scene-gen")
    ↓
Tool handler loads SKILL.md content → returns full instructions to Claude
    ↓
API call #2: Previous context + skill instructions → Claude
    ↓
Claude response: "Following Stage 1. I'll create a room layout."
    → Tool call: execute_command("principia \"Generate a room layout...\" --json -y")
    ↓
Subagent spawns, returns JSON → Claude gets the room JSON
    ↓
Claude response: "Now creating the room via MCP."
    → Tool call: use_mcp_tool(server_name="scene-gen", tool_name="create_room", arguments={...})
    ↓
MCP handler routes to Python server → server creates room → returns layout_id
    ↓
API call #N: Continue through Stages 2-8...
    ↓
Claude response: attempt_completion(result="Scene generated successfully...")
    ↓
Task complete → output to user
```

Each loop iteration:
1. **Stream API response** — Claude's text + tool calls arrive via SSE
2. **Parse tool calls** — Extract tool name + parameters from response
3. **Route to handler**:
   - `use_mcp_tool` → `UseMcpToolHandler` → `McpHub.callTool()` → Python server
   - `use_skill` → `UseSkillToolHandler` → load SKILL.md → return instructions
   - `execute_command` → shell execution (for subagent spawning)
   - `write_to_file`, `read_file`, etc. → file operation handlers
4. **Approval check**:
   - If tool is in `autoApprove` list → execute immediately
   - If `--yolo` flag → auto-approve everything
   - Otherwise → prompt user for approval
5. **Return result** — Tool output sent back to Claude in next API call
6. **Repeat** until `attempt_completion` or error

### 3.6 MCP Tool Call Routing (Detail)

**File:** `src/core/task/tools/handlers/UseMcpToolHandler.ts`

When Claude emits a `use_mcp_tool` call:

```
Claude → { server_name: "scene-gen", tool_name: "place_objects", arguments: "{...}" }
```

1. **Validate** — Check server_name and tool_name are present
2. **Parse arguments** — `JSON.parse(arguments)` into object
3. **Check auto-approve**:
   - Is `"place_objects"` in the server's `autoApprove` list? → Yes → skip user prompt
   - Or is `--yolo` mode active? → skip user prompt
   - Otherwise → show approval dialog
4. **Execute** — `mcpHub.callTool("scene-gen", "place_objects", parsedArguments)`
5. **Transport** — JSON-RPC message written to Python process stdin:
   ```json
   {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "place_objects", "arguments": {...}}}
   ```
6. **Python server** — FastMCP dispatches to `@mcp.tool() def place_objects(...)`, runs DFS solver
7. **Response** — JSON-RPC response on stdout:
   ```json
   {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "{\"placed\": 5, ...}"}]}}
   ```
8. **Process result** — Extract text content, truncate if >400KB, return to Claude
9. **Images** — If result contains `type: "image"` items, they're passed to multimodal models

---

## 4. The 8-Stage Workflow (What Claude Does)

Once the skill is loaded, Claude follows these stages. Each stage alternates between **subagent reasoning** (spawn a new principia process for JSON generation) and **MCP tool calls** (computation on the Python server).

### Stage 1: Room Structure

```
Subagent → "Generate room layout JSON for: a cozy bedroom"
         → Returns: { rooms: [{ id: "room_01", room_type: "bedroom", dimensions: {width: 4, length: 5, height: 2.7} }] }

MCP Tool → create_room(room_json)
         → Server: validates structure, generates 4 walls, stores FloorPlan in memory
         → Returns: { layout_id: "a1b2c3d4", rooms: [...] }
```

### Stage 2: Materials

```
Subagent → "Describe materials for a bedroom, modern style"
         → Returns: { "room_01": { "floor": "warm oak hardwood", "wall": "cream matte paint" } }

MCP Tool → generate_materials(layout_id, descriptions)
         → Server: If MatFuse available → generates 512×512 PBR maps (albedo, roughness, normal)
                   If no GPU → generates solid-color placeholder PNGs
         → Returns: { "room_01": { "floor": "/path/to/floor_albedo.png", ... } }
```

### Stage 3: Doors & Windows

```
MCP Tool → analyze_floor_plan(layout_id, "Which walls for door/windows?")
         → Server: renders floor plan PNG, sends to Qwen3-VL with prompt
         → Returns: VL model's analysis ("door on south wall, windows on east")

Subagent → "Based on analysis, generate door/window JSON"
         → Returns: { room_id: "room_01", doors: [...], windows: [...] }

MCP Tool → add_doors_windows(layout_id, placement_json)
         → Server: adds Door/Window objects to room, validates
         → Returns: validation results
```

### Stage 4: Object Recommendations

```
Subagent → "List all objects for a 4x5m bedroom, modern style"
         → Returns: [{ name: "bed", description: "queen platform bed", target_size: [1.6, 2.0, 0.5], priority: 1, place_on: "floor" }, ...]
         → Typically 30-60+ objects across 4 priority levels

MCP Tool → search_objects(specs) / generate_3d_model(description, size)
         → Server: searches Objaverse or generates via TRELLIS
         → Returns: model IDs and metadata
```

### Stage 5: Object Placement (Batched, 4 rounds)

This is where the DFS solver does the heavy lifting. Objects are placed in priority order:

**Batch 1 — Anchor furniture** (bed, wardrobe, desk):
```
Subagent → "Place anchor furniture with constraints"
         → Returns: {
             objects: [{ id: "bed_001", type: "bed", dimensions: {...}, place_id: "floor" }, ...],
             constraints: {
               "bed_001": ["edge", "far, door_01"],
               "wardrobe_001": ["edge", "far, bed_001"],
               "desk_001": ["edge", "far, bed_001"]
             }
           }

MCP Tool → place_objects(layout_id, room_id, objects_json, constraints_json)
         → Server:
           1. parse_constraints_from_json() — fuzzy match, expand "around" → close_to + face_to
           2. Separate floor/wall/on-object items
           3. For floor objects:
              a. Convert room to cm, create Shapely polygon
              b. Compute door swing + window strip obstacles
              c. Build 20cm grid via Shapely Point containment
              d. Remove obstacle points via R-tree
              e. DFS search: for each object, enumerate grid_point × 4 rotations
              f. Score each placement against constraints (edge, distance, relative, direction, alignment)
              g. Softmax sampling (T=0.4) over top-15 branches
              h. Backtrack and retry until solution found or time limit
              i. Convert solution: cm → meters, grid → world coordinates
           4. For wall objects: geometric scoring (55-65% wall height, alignment with floor objects)
           5. For on-object items: 50 random surface positions, collision check, 2cm margin
         → Returns: { placed: 5, failed: 0, objects: [{id, position, rotation, ...}, ...] }
```

**Batch 2 — Functional combos** (chair, nightstand, coffee table)
**Batch 3 — Surface objects** (lamps, books, vases — placed ON furniture)
**Batch 4 — Decorative & wall** (paintings, mirrors, clocks)

### Stage 6: Semantic Critic (up to 3 iterations)

```
MCP Tool → run_semantic_critic(layout_id, room_id)
         → Server:
           1. Render annotated top-down PNG (color-coded boxes, ID labels, facing arrows, door zones)
           2. Build rich room description text (dimensions, objects with positions/sizes/facing)
           3. Calculate floor occupancy ratio
           4. Send image + structured prompt to Qwen3-VL
           5. 3-step analysis:
              Step 1: Room quality assessment (realism, functionality, layout, completion)
              Step 2: Object addition recommendations (combos, background, shelf fullness)
              Step 3: Object adjustment recommendations (MOVE / REMOVE / REPLACE)
         → Returns: {
             analysis_summary: { overall_room_rating: "good", detailed_reasoning: "..." },
             object_addition_analysis: { object_combos_analysis: [...], background_objects_analysis: [...] },
             object_existing_analysis: [{ object_id: "...", issues_found: [...] }]
           }
```

Claude then:
- If rating is "excellent"/"good" and no high-priority issues → proceed
- If rating is "fair"/"poor" → address critic feedback, loop back to Stage 4-5
- Max 3 iterations

### Stage 7: Physics Validation (if Isaac Sim available)

```
MCP Tool → build_scene(layout_id, room_id)
         → Server: pushes room geometry + objects to Isaac Sim via TCP socket

MCP Tool → simulate_physics_tool(layout_id)
         → Server: runs physics simulation, returns stable/unstable object lists
         → If unstable objects: remove_objects() → re-simulate (max 2 iterations)
```

### Stage 8: Export

```
MCP Tool → export_usd_tool(layout_id, output_path)
         → Server: exports scene as Universal Scene Description file via Isaac Sim

Claude → attempt_completion(result="
  Layout ID: a1b2c3d4
  Room: bedroom 4.0m × 5.0m
  Total objects: 42
  Semantic critic: good
  USD: ./results/a1b2c3d4.usd
")
```

---

## 5. Subagent Execution

When SKILL.md says:
```bash
principia "Generate a room layout JSON for..." --json -y
```

Claude runs this via `execute_command`. Here's what happens:

1. **New CLI process** spawns with the prompt
2. **`--json`** → Output as JSON lines (structured, parseable)
3. **`-y` (yolo)** → Auto-approve all actions (no user prompts)
4. The subagent calls the same LLM API but with a focused prompt
5. The subagent typically just generates JSON without tool calls
6. Output is captured by the parent process and returned to the coordinator Claude
7. Claude parses the JSON and passes it to MCP tools

This creates a **two-tier agent architecture**:
- **Coordinator** (outer Claude) — follows SKILL.md, orchestrates stages, calls MCP tools
- **Subagents** (inner principia processes) — focused domain reasoning, produce JSON

---

## 6. File Locations Summary

### Configuration
| File | Purpose |
|------|---------|
| `~/.principia/data/settings/principia_mcp_settings.json` | MCP server registrations |
| `~/.principia/data/globalState.json` | CLI settings (model, mode, auto-approval) |
| `~/.principia/skills/scene-gen/SKILL.md` | Workflow instructions |
| `~/.principia/data/secrets.json` | API keys and OAuth tokens |

### MCP Server Code
| File | Purpose |
|------|---------|
| `servers/scene-gen/server.py` | FastMCP server, 14 tool handlers |
| `servers/scene-gen/models.py` | Dataclasses: FloorPlan, Room, Object, Wall, Door, Window |
| `servers/scene-gen/state.py` | In-memory scene state (SceneState) |
| `servers/scene-gen/solvers/placement_solver.py` | DFS grid solver, constraint parsing, wall/surface placement |
| `servers/scene-gen/solvers/room_solver.py` | Rectangle contact graph solver |
| `servers/scene-gen/solvers/validation.py` | Room structure validation |
| `servers/scene-gen/rendering/room_render.py` | Matplotlib 2D floor plan rendering |
| `servers/scene-gen/services/qwen_vl.py` | Qwen3-VL HTTP client |
| `servers/scene-gen/services/materials.py` | MatFuse PBR texture generation |
| `servers/scene-gen/services/trellis.py` | TRELLIS 3D model generation client |
| `servers/scene-gen/services/isaac_sim.py` | Isaac Sim TCP bridge |
| `servers/scene-gen/services/mesh_processing.py` | GLB post-processing pipeline |
| `servers/scene-gen/matfuse/` | MatFuse diffusion model (local copy) |

### Outputs
| Location | Content |
|----------|---------|
| `./results/<layout_id>/` | Per-layout directory |
| `./results/<layout_id>/floor_plan.png` | Floor plan render |
| `./results/<layout_id>/<room_id>_plan.png` | Room render with annotations |
| `./results/<layout_id>/<room_id>_critic.png` | Critic render |
| `./results/<layout_id>/materials/` | Generated PBR textures |
| `./results/<layout_id>/scene.usd` | Final USD export |
| `./results/generated_models/<model_id>/` | TRELLIS-generated 3D models |
| `~/.principia/data/tasks/<task_id>/` | Conversation history + metadata |

---

## 7. DFS Placement Solver — Technical Detail

The solver is the core algorithm. Here's how it places 5 objects in a 4×5m room in ~12 seconds:

### Grid Creation
```
Room: 4.0m × 5.0m → 400cm × 500cm (minus wall thickness padding)
Grid: 20cm spacing → 20 × 25 = 500 grid points
Each point: (x_cm, y_cm)
```

### Obstacle Removal
```
Door swing: 90cm × 90cm square → ~20 grid points removed
Window strip: thin rectangle → ~5 grid points removed
R-tree indexed for O(log n) point queries
```

### DFS Search
```
For each object (in constraint-priority order):
  1. Enumerate all valid placements: grid_point × 4_rotations = ~2000 candidates
  2. Filter: room containment (Shapely), collision (polygon intersection), facing wall distance
  3. Score each against constraints:
     - edge: bonus if back-center touches wall boundary
     - close_to: interp1d curve, peak at target distance
     - left_of: 3-tier scoring (strict angle check, loose, looser)
     - face_to: ray from object center in facing direction, check intersection with target polygon
     - center_aligned: center-to-center offset within half-grid tolerance
  4. Sort by weighted score, take top-15 candidates
  5. Softmax sample (T=0.4) to pick branch
  6. Recurse to next object
  7. If dead end: backtrack, try next branch
  8. Time limit: 60s default (scales with object count)
```

### Constraint Weights
```
global (edge, middle):     2.0×
relative (left_of, etc.):  1.0×
distance (close_to, etc.): 1.0×
direction (face_to, etc.): 1.0×
alignment:                 1.0×
```

---

## 8. External Services

| Service | Protocol | Required? | Purpose |
|---------|----------|-----------|---------|
| **Qwen3-VL** | HTTP (OpenAI-compat) | Yes (for critic + analysis) | Vision-language model for semantic evaluation |
| **TRELLIS** | HTTP (REST) | Optional | Text-to-3D model generation |
| **Isaac Sim** | TCP socket | Optional | Physics simulation + USD export |
| **MatFuse** | In-process (PyTorch) | Optional | PBR material texture generation |
| **Objaverse** | Local disk | Optional | 3D model search database |

When optional services are unavailable:
- TRELLIS missing → `search_objects` returns empty, `generate_3d_model` fails gracefully
- Isaac Sim missing → Stages 7-8 skipped, SKILL.md handles this
- MatFuse missing → Solid-color placeholder textures generated
- Objaverse missing → `search_objects` returns "not configured" note
