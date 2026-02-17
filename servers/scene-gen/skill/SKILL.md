---
name: scene-gen
description: Generate simulation-ready 3D indoor scenes from natural language descriptions using the SAGE composite tools
---

# Scene Generation Workflow

You are operating the SAGE scene generation pipeline. This pipeline uses **10 composite MCP tools** that handle internal orchestration (LLM reasoning, object retrieval, placement solving, rendering, critics). Follow the stages below **in order**.

> **CRITICAL: You MUST use the scene-gen MCP tools listed below to generate scenes. DO NOT use `isaacExec`, `execute_python`, or write USD/Python code directly to create or place objects. The scene-gen tools produce simulation-ready scenes with real 3D assets, generated textures, and quality critics. Direct scripting produces only untextured primitives and bypasses the entire pipeline.**

> **Each tool is self-contained.** `generate_room_layout` handles room generation + validation + materials + doors/windows internally. `place_objects_in_room` handles object selection + retrieval + placement + critics internally. You orchestrate the high-level flow; the tools handle the details.

> **MANDATORY: You MUST call `place_objects_in_room` exactly 4 times per room (anchor furniture → secondary furniture → surface objects → decorative items). You MUST call `load_scene_in_isaac_sim` for every room at the end. No exceptions. Do NOT collapse iterations or skip the final load. A room with only 1 placement call is incomplete.**

## Required task_progress Checklist

Use exactly this checklist format in every `task_progress` field, updating as you go:

```
- [ ] Stage 1: generate_room_layout
- [ ] Stage 2: get_room_information (review layout)
- [ ] Stage 3 iteration 1/4: place_objects_in_room (anchor furniture)
- [ ] Stage 3 iteration 2/4: place_objects_in_room (secondary furniture)
- [ ] Stage 3 iteration 3/4: place_objects_in_room (surface objects)
- [ ] Stage 3 iteration 4/4: place_objects_in_room (decorative items)
- [ ] Stage 4: get_room_information (review final state)
- [ ] Stage 5: get_layout_save_dir
- [ ] Stage 5: load_scene_in_isaac_sim
```

For multi-room layouts, repeat Stage 3 (all 4 iterations) and Stage 5 load for each room.

---

## Stage 1: Generate Room Layout

**Tool:** `generate_room_layout(input_text)`

Pass the user's description directly. The tool internally:
1. Generates room structure (walls, dimensions) via LLM reasoning
2. Validates and corrects the layout
3. Selects and generates materials (floor + wall textures)
4. Adds doors and windows with integrity checks

The tool returns a JSON summary with room IDs, layout ID, and scene recommendations.

**Store the returned room IDs** — you'll need them for subsequent tools.

```
generate_room_layout("A cozy 4m x 5m bedroom with warm modern style, queen bed, desk area, and reading corner")
```

---

## Stage 2: Review Layout

**Tool:** `get_room_information(room_id)`

Call this to inspect the generated layout. Returns room dimensions, walls, doors, windows, and a top-down visualization.

Use this **before and after** placing objects to understand the room state.

---

## Stage 3: Place Objects (4 iterations per room)

**Tool:** `place_objects_in_room(room_id, placement_conditions)`

Call this tool **4 times per room** to iteratively furnish it. Each call internally:
1. Uses LLM to recommend objects based on the conditions and existing room state
2. Retrieves 3D models from ObjaThor (CLIP+SBERT search) or generates via TRELLIS
3. Places objects using the constraint solver
4. Runs semantic and physics critics
5. Returns results with critic feedback

**Iteration strategy:**

| Iteration | placement_conditions focus |
|-----------|--------------------------|
| 1 | "Add essential anchor furniture: bed, wardrobe, desk, chair. Place large items first." |
| 2 | "Add functional combos and secondary furniture: nightstands, bookshelf, side tables, lamps." |
| 3 | "Add surface objects: books on shelves (5+ per shelf), items on desk and tables (2+ per surface), decorative objects." |
| 4 | "Add final decorative and wall items: wall art, mirrors, plants, remaining small objects for completeness." |

**For multi-room layouts:** Run 4 iterations for each room, starting with the main/largest room.

**Key rules for `placement_conditions`:**
- Be specific about object types, quantities, and relationships
- Reference existing object IDs when specifying placement (e.g., "add lamp on desk_001")
- Preferred spatial relationships: near/far, in front/side/left/right/around, center aligned, face to/face same as, on top of, above
- Include object style, color, material to match the room's aesthetic

---

## Stage 4: Review & Adjust

**Tool:** `get_room_information(room_id)` — Check the room state after placement.

**Tool:** `move_one_object_with_condition_in_room(room_id, condition)` — Fine-tune individual objects.

Condition format:
```
Move [object_type] (object_id: [object_id]) to [floor|wall|on top of object_id], [spatial relationships]
```

Examples:
- `Move chair (object_id: chair_001) to floor, facing and aligned with the desk.`
- `Move lamp (object_id: lamp_003) to on top of desk_002.`
- `Move picture (object_id: picture_004) to wall, above the sofa.`

---

## Stage 5: Export & Load into Isaac Sim

**Tool:** `get_layout_save_dir()` — Get the output directory path.

**Tool:** `load_scene_in_isaac_sim(room_id)` — Load the final scene into Isaac Sim with lighting and physics.

After all 4 placement iterations are complete for every room:

1. Call `get_layout_save_dir()` to get the output path.
2. Call `load_scene_in_isaac_sim(room_id)` for **every room**. This is mandatory — it creates the USD scene with proper lighting and physics. Do NOT skip this step.

Report to the user:
- Layout ID and room type(s)
- Total objects placed per room
- Final critic rating
- Output directory path

---

## Quick Reference: All 10 Tools

| Tool | Purpose |
|------|---------|
| `generate_room_layout(input_text)` | Generate complete room layout with materials, doors, windows |
| `get_current_layout()` | Get the full current layout data structure |
| `get_room_details(room_id)` | Get room details including object list |
| `list_rooms()` | List all rooms in current layout |
| `get_layout_from_json(json_file_path)` | Load a previously saved layout from JSON |
| `place_objects_in_room(room_id, placement_conditions)` | Add/remove/replace objects with full pipeline |
| `get_room_information(room_id)` | Get room info with visualization |
| `move_one_object_with_condition_in_room(room_id, condition)` | Move/reposition a single object |
| `get_layout_save_dir()` | Get the layout save directory path |
| `load_scene_in_isaac_sim(room_id)` | Load scene into Isaac Sim with lighting and physics (mandatory for every room) |

---

## Quality Rules

- **4 iterations per room** for `place_objects_in_room` — anchor furniture first, then combos, then surfaces, then decor
- Every shelf must have **5+ items**; every table/desk/counter needs **2+ items**
- FORBIDDEN objects: rugs, mats, carpets, curtains, blankets, ceiling-hanging objects
- All objects must match the building style
- The semantic critic runs automatically inside `place_objects_in_room` — check the returned feedback and address high-priority issues in the next iteration
- After all iterations, use `get_room_information` to verify the final state

## Error Handling

- If `generate_room_layout` fails, simplify the description and retry
- If `place_objects_in_room` returns errors, adjust conditions and retry
- If a tool is unavailable, note it and continue with available tools
- Max 2 retries per tool call before moving on
