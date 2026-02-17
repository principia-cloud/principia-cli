# Scene-Gen Development Guide

This file is for AI agents (Claude Code, etc.) operating in the `servers/scene-gen/` directory.

## What This Is

A Python MCP server that generates 3D indoor scenes. It runs as a subprocess of `principia-cli` and communicates via stdio MCP protocol. The server exposes 9 composite tools — each tool internally orchestrates multiple steps (LLM calls, 3D retrieval, placement solving, critics).

## Critical Patterns

### LLM Calls — Never Use `anthropic` Directly

**All LLM/VLM calls go through `vlm.py:call_vlm()`**. There are two backends:

1. **Text reasoning** (`vlm_type="claude"`) → `subagent.py` → spawns `principia` CLI subprocess
2. **Vision+Language** (`vlm_type="qwen"`) → Direct Qwen3-VL API via OpenAI-compatible endpoint

Do NOT add `import anthropic` anywhere. If you need text LLM reasoning, call `call_vlm(vlm_type="claude", ...)` or use `call_llm_via_subagent()` from `subagent.py`.

### Configuration — Environment Variables Only

All config comes from env vars (see `key.py`). No hardcoded API keys, no config files read at runtime. The env vars are set in `~/.principia/data/settings/principia_mcp_settings.json` which the CLI passes to the MCP server process.

### File Hardlinks

Files in `servers/scene-gen/` are hardlinked to `~/.principia/source/servers/scene-gen/`. Editing one edits the other. This means:
- Don't create new files and forget to link them
- The installer (`scripts/setup-scene-gen.sh`) creates the link via `ln -sfn`

### The Main File is `layout_wo_robot.py`

This is the core — ~3800 lines, contains all 9 `@mcp.tool()` definitions and the placement pipeline. Changes here affect all tools.

### `models.py` Defines All Data Structures

`FloorPlan`, `Room`, `Object`, `Point3D`, `Dimensions`, `Euler`, `Wall`, `Door`, `Window` etc. These Pydantic models are the source of truth for scene state.

## Where Things Are

| Need to... | Look at |
|------------|---------|
| Add/modify an MCP tool | `layout_wo_robot.py` |
| Change how LLM text calls work | `subagent.py`, `vlm.py` |
| Change VLM (image) calls | `vlm.py` |
| Change room generation prompts | `llm_client.py` |
| Change object retrieval | `objects/objaverse_retrieval.py`, `objects/get_objects.py` |
| Change placement solving | `room_solver.py`, `objects/object_placement_planner.py` |
| Change texture generation | `floor_plan_materials/material_generator.py`, `matfuse/generate.py` |
| Change object attribute inference | `objects/object_attribute_inference.py` |
| Change critic behavior | Search for `semantic_critic` / `physics_critic` in `layout_wo_robot.py` |
| Change the agent workflow | `skill/SKILL.md` |
| Change env var config | `key.py`, `constants.py` |
| Change install/setup | `scripts/setup-scene-gen.sh` |

## Common Tasks

### Adding a New MCP Tool

1. Add `@mcp.tool()` function in `layout_wo_robot.py`
2. Add tool name to `autoApprove` in `mcp_settings_example.json`
3. Update `scripts/setup-scene-gen.sh` autoApprove list
4. Update `skill/SKILL.md` if the agent should use it
5. Update live settings: `~/.principia/data/settings/principia_mcp_settings.json`

### Adding a Python Dependency

1. Add to `requirements.txt`
2. If it needs CUDA compilation (like nvdiffrast, pytorch3d), add to `_ensure_gpu_pip_packages()` in `scripts/setup-scene-gen.sh`
3. If it needs an apt package, add to `_ensure_apt_packages()` in `scripts/setup-scene-gen.sh`

### Debugging Import Errors

The import chain is deep. Common issues:
- GPU packages (torch, nvdiffrast, pytorch3d) must be installed with CUDA toolkit
- `matfuse/` is loaded via `importlib` from `MATFUSE_ROOT_DIR` path — check `constants.py`
- `objects/` directory is added to `sys.path` in `layout_wo_robot.py` — relative imports inside use `objects.` prefix

### Testing

```bash
# Check imports (needs GPU env with all deps)
python -c "from layout_wo_robot import mcp; print(mcp.name)"

# List tools
python -c "from layout_wo_robot import mcp; print([t.name for t in mcp._tool_manager.list_tools()])"

# Run server standalone (stdio)
python server.py
```

## Don't

- Don't import `anthropic` — use `call_vlm()` or `call_llm_via_subagent()`
- Don't hardcode paths — use env vars from `key.py` / `constants.py`
- Don't add `ANTHROPIC_API_KEY` anywhere — text reasoning uses `principia` subprocess
- Don't modify `matfuse/ldm/` or `matfuse/utils/` — those are upstream MatFuse model code
- Don't create files outside `servers/scene-gen/` for this server (except `scripts/setup-scene-gen.sh`)
- Don't break the 9-tool contract — the agent workflow in `SKILL.md` depends on exactly these tools
