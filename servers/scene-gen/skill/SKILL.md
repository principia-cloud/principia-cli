---
name: scene-gen
description: Generate simulation-ready 3D indoor scenes from natural language descriptions using the SAGE workflow
---

# Scene Generation Workflow

You are now operating the SAGE scene generation pipeline. Follow the 8 stages below **in order**. Track `layout_id` and `room_id` throughout — every MCP tool call after Stage 1 needs them.

Use **subagents** (via `principia "prompt" --json -y`) for domain reasoning that produces structured JSON. Use **MCP tools** (via the scene-gen server) for computation, rendering, solving, and external services.

> **State lives in the MCP server.** The server holds the FloorPlan, rooms, and objects in memory. You don't need to track object positions yourself — just call the tools and use the returned IDs.

### MANDATORY STAGE COMPLETION — READ THIS FIRST

**You MUST execute ALL 8 stages in order. Do NOT skip any stage.** A scene with just walls and no furniture is not a valid result. The minimum viable output is a furnished, critic-reviewed, exported scene.

| Stage | Required? | Skip condition |
|-------|-----------|----------------|
| 1. Room Structure | **ALWAYS** | — |
| 2. Materials | **ALWAYS** | — |
| 3. Doors & Windows | **ALWAYS** | — |
| 4. Object Recommendations + Sourcing | **ALWAYS** | — |
| 5. Object Placement | **ALWAYS** | — |
| 6. Semantic Critic | **ALWAYS** (at least 1 pass per room) | — |
| 7. Physics Validation | **ATTEMPT** — skip only if Isaac Sim is unavailable (connection error) | — |
| 8. Export | **ALWAYS** | — |

**Stages 4-6 are the core of the pipeline.** An empty room with no objects is never acceptable. You must source 3D models, place objects, and run the critic before exporting. If any service is unavailable (e.g., ObjaThor, TRELLIS), note it and continue with what is available — but never skip the stage entirely.

---

## Stage 1: Room Structure

**Goal:** Generate a validated room layout from the user's description.

**Step 1a — Subagent: Room JSON**

Decide whether the description implies a single room or multiple rooms. Keywords like "apartment", "house", "suite", "floor plan", or explicit multi-room requests → multi-room. A simple "bedroom" or "kitchen" → single room.

**Single-room prompt:**

```bash
principia "Generate a room layout JSON for: <USER_DESCRIPTION>.

You must generate ONLY a single room. Explicitly mention 'a single room' without mentioning any other rooms or spaces.

Return a JSON object with this exact structure:
{
  \"building_style\": \"<style matching the description>\",
  \"description\": \"<1-2 sentence summary>\",
  \"created_from_text\": \"<original user description>\",
  \"rooms\": [
    {
      \"id\": \"room_01\",
      \"room_type\": \"<bedroom|living_room|kitchen|bathroom|office|dining_room|etc>\",
      \"position\": {\"x\": 0, \"y\": 0, \"z\": 0},
      \"dimensions\": {
        \"width\": <meters, realistic for room type>,
        \"length\": <meters, realistic for room type>,
        \"height\": 2.7
      }
    }
  ]
}

Use realistic dimensions: bedrooms 3-5m wide, kitchens 3-4m, living rooms 4-7m, bathrooms 2-3m.
Return ONLY valid JSON, no markdown fences." --json -y
```

**Multi-room prompt:**

```bash
principia "Generate a multi-room layout JSON for: <USER_DESCRIPTION>.

Return a JSON object with this exact structure:
{
  \"building_style\": \"<style matching the description>\",
  \"description\": \"<1-2 sentence summary>\",
  \"created_from_text\": \"<original user description>\",
  \"rooms\": [
    {
      \"id\": \"room_01\",
      \"room_type\": \"living_room\",
      \"position\": {\"x\": 0, \"y\": 0, \"z\": 0},
      \"dimensions\": {\"width\": 5.0, \"length\": 4.5, \"height\": 2.7}
    },
    {
      \"id\": \"room_02\",
      \"room_type\": \"bedroom\",
      \"position\": {\"x\": 5.0, \"y\": 0, \"z\": 0},
      \"dimensions\": {\"width\": 4.0, \"length\": 4.5, \"height\": 2.7}
    }
  ]
}

CRITICAL POSITIONING RULES:
- Rooms must share walls (be adjacent). Place room_02.x = room_01.x + room_01.width to tile them east.
- Or tile north: room_02.y = room_01.y + room_01.length.
- No gaps and no overlaps between rooms.
- Use realistic dimensions: bedrooms 3-5m, kitchens 3-4m, living rooms 4-7m, bathrooms 2-3m.
Return ONLY valid JSON, no markdown fences." --json -y
```

**Step 1b — MCP Tool: Create Room**

Call `create_room` with the JSON from the subagent. Store the returned `layout_id` and all `room_id` values.

For multi-room layouts, the response includes a `shared_walls` field with room-room and room-exterior wall classification. Note these for Stage 3.

If validation fails, tell the user what went wrong and ask the subagent to fix the dimensions.

---

## Stage 2: Materials

**Goal:** Generate PBR material textures for floor and wall surfaces.

**Step 2a — Subagent: Material Descriptions**

```bash
principia "Describe materials for a <ROOM_TYPE> with style: <BUILDING_STYLE>.
Description: <USER_DESCRIPTION>

Return a JSON object mapping room_id to surface descriptions:
{
  \"<ROOM_ID>\": {
    \"floor\": \"<detailed material description, e.g. 'warm honey oak hardwood planks with subtle grain'>\",
    \"wall\": \"<detailed material description, e.g. 'soft cream matte paint with slight eggshell texture'>\"
  }
}

Match the style and mood of the room description. Be specific about color, texture, and finish.
Return ONLY valid JSON." --json -y
```

For multi-room layouts, include ALL room IDs in the prompt so each room gets appropriate materials.

**Step 2b — MCP Tool: Generate Materials**

Call `generate_materials(layout_id, material_descriptions_json)`.

---

## Stage 3: Doors & Windows

**Goal:** Analyze the room layout and place appropriate doors and windows.

### If layout has 2+ rooms (multi-room branch)

**Step 3a-multi — MCP Tool: Analyze Shared Walls**

Call `analyze_shared_walls(layout_id)`. This returns:
- `shared_walls`: room-room walls and room-exterior walls with overlap lengths
- `suggested_connecting_doors`: MST-based minimum set of doors connecting all rooms

**Step 3b-multi — Subagent: Connecting Door Specs**

Using the shared wall analysis, spawn a subagent to decide on connecting doors:

```bash
principia "Based on this shared wall analysis: <ANALYZE_SHARED_WALLS_RESPONSE>

The suggested connecting doors are: <SUGGESTED_CONNECTING_DOORS>

For each connecting door, confirm or adjust:
- shared_wall_index: which shared wall (from the analysis)
- center_position_on_shared_wall: 0.0-1.0 (default 0.5 = centered)
- width: door width in meters (0.8-0.9 typical, will be clipped to fit)
- height: 2.1
- door_type: 'connecting'
- opening: false (door) or true (archway, good for living room ↔ kitchen)

Return a JSON array of connecting door specs.
Return ONLY valid JSON." --json -y
```

**Step 3b-multi-exec — MCP Tool: Add Connecting Doors**

Call `add_connecting_doors(layout_id, connecting_doors_json)`.

**Step 3c — Per room: Entry Doors & Windows on Exterior Walls**

For each room, proceed with the standard door/window flow below, but:
- Place the **entry door** on an **exterior wall** of the entry room only (typically the largest or first room)
- Place **windows** only on **exterior walls** (use `room_exterior_walls` from the shared wall analysis)
- Do NOT place doors/windows on room-room shared walls — those already have connecting doors

### Standard door/window flow (single room, or per-room exterior in multi-room)

**Step 3a — MCP Tool: Analyze Floor Plan**

Call `analyze_floor_plan(layout_id, prompt)` with a prompt that **includes the actual wall IDs** from `create_room`'s response (the `wall_ids` field in each room):
> "Analyze this room layout. The room has these walls: <WALL_IDS_LIST>. Identify the best wall_id for the main entrance door and which wall_id(s) should have windows based on room dimensions and typical residential design. Consider traffic flow and natural light. You MUST use the exact wall_id strings provided (e.g. 'room_01_s_wall'), NOT indices. Respond with: which wall_id for the door, which wall_id(s) for windows, and recommended positions (0.0-1.0 along each wall)."

**Step 3b — Subagent: Door/Window Specs**

Using the VL model's analysis, spawn a subagent. **You MUST substitute the actual wall IDs** from `create_room`'s response into the prompt (e.g. `room_01_s_wall`, `room_01_e_wall`, `room_01_n_wall`, `room_01_w_wall`):

```bash
principia "Based on this room analysis: <VL_RESPONSE>

The room has these walls (use these EXACT IDs): <WALL_IDS_FROM_CREATE_ROOM_RESPONSE>

Generate door and window placement JSON:
{
  \"room_id\": \"<ROOM_ID>\",
  \"doors\": [
    {
      \"wall_id\": \"<wall_id>\",
      \"position_on_wall\": <0.0-1.0>,
      \"width\": 0.9,
      \"height\": 2.1,
      \"door_type\": \"standard\",
      \"opens_inward\": true
    }
  ],
  \"windows\": [
    {
      \"wall_id\": \"<wall_id>\",
      \"position_on_wall\": <0.0-1.0>,
      \"width\": 1.2,
      \"height\": 1.0,
      \"sill_height\": 0.9,
      \"window_type\": \"standard\"
    }
  ]
}

Every room needs at least 1 door. Most rooms need 1-2 windows.
Return ONLY valid JSON." --json -y
```

**Step 3c — MCP Tool: Add Doors & Windows**

Call `add_doors_windows(layout_id, placement_json)`.

---

## Stage 4: Object Recommendations

**Goal:** Determine what objects to place, then source 3D models for them.

**For multi-room layouts:** Run Stages 4-5 for each room. Process the main/largest room first, then secondary rooms.

**Step 4a — Subagent: Object List**

```bash
principia "You are furnishing a <ROOM_TYPE> (<WIDTH>m x <LENGTH>m, style: <BUILDING_STYLE>).
<IF_CRITIC_FEEDBACK>The semantic critic identified these issues: <CRITIC_ISSUES>. Address them.</IF_CRITIC_FEEDBACK>

List ALL objects needed, grouped by placement priority:

PRIORITY 1 - ANCHOR FURNITURE (place first):
Large foundational pieces. For a bedroom: bed, wardrobe, desk. For a kitchen: counters, cabinets, refrigerator. For a living room: sofa, TV stand, bookshelf.

PRIORITY 2 - FUNCTIONAL COMBOS:
Objects that pair with anchor furniture. Chairs with desks/tables, nightstands with beds, coffee table with sofa.

PRIORITY 3 - SURFACE OBJECTS:
Items that go ON TOP of furniture. Every shelf needs 5+ items. Every table/desk/counter needs 2+ items.
Examples: books, lamps, picture frames, vases, clocks, plants, kitchen utensils, decorative bowls.

PRIORITY 4 - DECORATIVE & WALL:
Wall art, mirrors, clocks, additional decor for completeness.

FORBIDDEN objects (never include): rugs, mats, curtains, blankets, ceiling-hanging objects.

For each object provide:
{
  \"name\": \"<object type>\",
  \"description\": \"<style-matched description with color/material>\",
  \"target_size\": [<width>, <length>, <height>] in meters,
  \"priority\": <1-4>,
  \"place_on\": \"floor\" | \"wall\" | \"<parent_object_name>\"
}

Return as a JSON array. Be comprehensive — a complete room needs 30-60+ objects.
Return ONLY valid JSON." --json -y
```

**Step 4b — Source 3D Models (MANDATORY — do NOT skip)**

Every object **must** have a 3D mesh loaded before it can be placed. Objects without a sourced mesh will be invisible in the exported scene.

For each object from the subagent:

1. Call `search_objects` with a batch of specs (up to 10 at a time)
2. For each match, call `load_objaverse_object(asset_id, layout_id)` to download and export the mesh as OBJ + texture
3. For any object with no good match, call `generate_3d_model(description, target_size_json)` via TRELLIS

**You MUST track the `asset_id` (from search) or `model_id` (from generation) for each object.** In Stage 5, pass `source="objaverse"` and `source_id=<asset_id>` (or `source="generation"` and `source_id=<model_id>`) so the placement solver links the object to its mesh file.

**DO NOT proceed to Stage 5 until every object has a sourced model.**

---

## Stage 5: Object Placement (Batched)

**Goal:** Place all objects using the DFS placement solver, in priority batches.

Place objects in **4 batches** matching the priority groups. For each batch:

**Step 5x-a — Subagent: Placement Constraints**

```bash
principia "You are an expert interior designer placing objects in a <ROOM_TYPE> (<WIDTH>m x <LENGTH>m).

Objects to place in this batch:
<OBJECT_LIST_WITH_IDS_AND_DIMENSIONS>

Already placed objects (reference these by their IDs in constraints):
<LIST_OF_ALREADY_PLACED_OBJECTS_WITH_POSITIONS>

DESIGN STRATEGY:
1. Think about the functional relationships between objects first
2. Rank objects by spatial importance (anchor first, dependent objects after)
3. Consider traffic flow — leave clear paths between door and key areas
4. Group functionally related objects (desk + chair, bed + nightstand)
5. Balance visual weight across the room

For each object, determine:
1. place_id: where the object goes
   - 'floor' → heavy/large furniture (beds, sofas, tables, desks, chairs, wardrobes, bookshelves)
   - 'wall' → wall-mounted items (paintings, wall shelves, mounted TVs, mirrors, clocks)
   - '<existing_object_id>' → small items that belong on/in specific furniture (lamp on desk, books on shelf)
   Key: 'beside the table' = floor; 'on the table' = table's object_id; desk lamp = desk's object_id
2. place_location (only when place_id is an object_id):
   - 'top' → object sits ON the surface (lamp on desk, vase on table, plate on counter)
   - 'inside' → object is contained WITHIN (books inside bookshelf, clothes inside wardrobe, dishes inside cabinet)
   - 'both' → either works (items on/in shelf, supplies on/in organizer)
3. Spatial constraints from the vocabulary below (use 4-6 per object when applicable)

CONSTRAINT VOCABULARY:
Global (no target needed):
  - 'edge'   → place against a wall
  - 'middle' → place in center/open area of room

Distance (requires target object_id):
  - 'close to, <object_id>' → within ~0.5m of target
  - 'near, <object_id>'     → within ~1.5m of target
  - 'far, <object_id>'      → at least ~2.5m away from target

Relative Position (relative to target's facing direction):
  - 'in front of, <object_id>' → in front of target (also aligns center)
  - 'behind, <object_id>'      → behind target
  - 'left of, <object_id>'     → to the left of target
  - 'right of, <object_id>'    → to the right of target
  - 'side of, <object_id>'     → either side of target

Direction:
  - 'face to, <object_id>'     → orient front toward target
  - 'face same as, <object_id>' → same orientation as target

Alignment:
  - 'center aligned, <object_id>' → align center axis with target

CONSTRAINT COMBINATION EXAMPLES:
Chair at desk:
  [\"edge\", \"close to, desk_001\", \"in front of, desk_001\", \"face to, desk_001\"]

Nightstand beside bed:
  [\"edge\", \"left of, bed_001\", \"close to, bed_001\", \"face same as, bed_001\"]

TV stand facing sofa:
  [\"edge\", \"far, sofa_001\", \"in front of, sofa_001\", \"center aligned, sofa_001\", \"face to, sofa_001\"]

Coffee table in front of sofa:
  [\"middle\", \"close to, sofa_001\", \"in front of, sofa_001\", \"center aligned, sofa_001\"]

Bookshelf against wall:
  [\"edge\"]

Wardrobe away from bed:
  [\"edge\", \"far, bed_001\"]

Pair of matching chairs:
  First: [\"middle\", \"near, table_001\", \"face to, table_001\"]
  Second: [\"middle\", \"near, table_001\", \"face to, table_001\", \"side of, chair_001\"]

DISTANCE INFERENCE RULES:
- Objects on different walls / opposite corners → 'far'
- Objects in same area / adjacent → 'near' or 'close to'
- Paired objects (chair+desk, nightstand+bed) → 'close to'

Return JSON with keys 'objects', 'constraints', 'design_strategy', and 'importance_ranking':

{
  \"design_strategy\": \"<brief description of layout approach>\",
  \"importance_ranking\": [\"<object_id in placement order>\"],
  \"objects\": [
    {
      \"id\": \"<unique_id>\",
      \"type\": \"<object type>\",
      \"description\": \"<style description>\",
      \"dimensions\": {\"width\": <m>, \"length\": <m>, \"height\": <m>},
      \"place_id\": \"<floor|wall|object_id>\",
      \"place_location\": \"<top|inside|both>\"
    }
  ],
  \"constraints\": {
    \"<object_id>\": [\"edge\", \"far, <other_id>\", \"face to, <other_id>\"],
    \"<object_id>\": [\"edge\", \"left of, <other_id>\", \"close to, <other_id>\"]
  }
}

Return ONLY valid JSON." --json -y
```

**Step 5x-b — MCP Tool: Place Objects**

**Before calling**, merge the sourced model info from Stage 4b into the objects array. Each object MUST include `"source"` and `"source_id"` fields referencing the loaded mesh:
- For ObjaThor objects: `"source": "objaverse", "source_id": "<asset_id>"`
- For generated objects: `"source": "generation", "source_id": "<model_id>"`

Then call `place_objects(layout_id, room_id, objects_json, constraints_json)`.

If the response contains a `"warning"` about unsourced objects, go back to Stage 4b and source them before continuing.

**Per-call limits:**
- Maximum **35-40 objects** per single placement call
- Maximum **10-12 object types** per single placement call
- If you have more, split into multiple calls within the same batch

**After each batch**, check the result. If objects failed to place, retry once with adjusted constraints or smaller alternatives.

---

## Stage 6: Semantic Critic (max 3 iterations)

**Goal:** Evaluate scene quality with the VL model and iterate if needed.

**For multi-room layouts:** Run the critic for each room separately. Max 3 iterations **per room**.

Call `run_semantic_critic(layout_id, room_id)`.

The critic returns a structured JSON response with `analysis_summary`, `object_addition_analysis`, and `object_existing_analysis`.

### Parsing the Critic Response

1. **Extract `analysis_summary.overall_room_rating`** — one of: excellent, good, fair, poor

2. **Process `object_existing_analysis`** (adjustments to current objects):
   - For each issue with `score >= 8`: execute the `suggested_operation`:
     - `REMOVE`: Call `remove_objects` for the object
     - `MOVE`: Call `remove_objects` then re-place with corrected constraints
     - `REPLACE`: Call `remove_objects`, source a new model, then place it
   - Ignore issues with `score < 6`

3. **Process `object_addition_analysis`** (new objects to add):
   - Collect recommendations from `object_combos_analysis` and `background_objects_analysis`
   - Sort by `priority` descending
   - Add all objects with `priority >= 6` via Stages 4-5 (source model → place)
   - Pay special attention to shelf/surface completeness recommendations

### Decision Logic

- **Rating "excellent" or "good" AND no issues with score >= 7** → proceed to Stage 7
- **Rating "fair" or "poor" OR any issue with score >= 7** → address issues, then re-run critic
- **Maximum 3 critic iterations** — after 3, proceed regardless

On each iteration, focus on the critic's specific suggestions: add missing objects, fix placements, remove problematic items.

---

## Stage 7: Physics Validation (max 2 iterations)

**Goal:** Ensure objects are physically stable in simulation.

**You MUST attempt this stage.** If Isaac Sim is not available (connection error), log the error and proceed to Stage 8 — but you must try first.

**Step 7a — Build Scene**

Call `build_scene(layout_id, room_id)` to push the scene to Isaac Sim.

**Step 7b — Simulate Physics**

Call `simulate_physics_tool(layout_id)`.

**Decision logic:**
- If **connection error** → note "Physics validation skipped — Isaac Sim unavailable", proceed to Stage 8
- If **no unstable objects** → proceed to Stage 8
- If **unstable objects found** → call `remove_objects(layout_id, room_id, unstable_ids_json)`, then re-simulate
- **Maximum 2 iterations** — after 2, proceed with remaining stable objects

---

## Stage 8: Export

**Goal:** Export the final scene as a USDZ file for visual validation.

**PRE-CHECK:** Before exporting, verify that every room has placed objects. If any room has 0 objects, you skipped Stages 4-5 — go back and complete them. The export tool will warn you if rooms are empty.

**Single-room:** Call `export_scene(layout_id, room_id)`.

**Multi-room:** Call `export_scene(layout_id)` with **no room_id** to export the full layout. This produces a single USDZ containing all rooms with connecting-door deduplication (each connecting door appears once, not twice).

This standalone exporter builds room geometry (floor, walls with door/window cutouts, ceiling), loads placed object meshes, and packages everything as a `.usdz` file viewable in macOS Quick Look — no Isaac Sim required.

If Isaac Sim is available and you want physics-accurate USD instead, use `export_usd_tool(layout_id, "./results/<LAYOUT_ID>.usd")`.

**Report to the user:**
- Layout ID: `<layout_id>`
- Room type(s) and dimensions
- Total objects placed (per room and total)
- Semantic critic rating (final, per room)
- USDZ file path (can be opened with Quick Look on macOS)

---

## Quality Rules

These rules apply throughout the workflow:

### Placement Priority Order
Always place objects in this order: anchor furniture → functional combos → surface objects → decorative items. Never place small items before the furniture they sit on exists.

### Per-Call Limits
- Max **35-40 objects** per `place_objects` call
- Max **10-12 object types** per `place_objects` call
- Place multiple object types in a single call for efficiency

### Forbidden Objects
NEVER add: rugs, mats, carpets, curtains, blankets, ceiling-hanging objects (light fixtures, chandeliers). These are pre-installed or handled separately.

### Completeness Requirements
- Every shelf must have **5+ items** (books, jars, decorative objects, etc.)
- Every supporter surface (tables, desks, counters, shelves) must have **2+ items** on it
- The room must feel **finished and lived-in**, not sparse

### Completion Checklist
Do NOT stop until ALL 8 conditions are met:
1. All necessary large AND small items are present with rich details
2. Every shelf is full of objects (5+ items per shelf)
3. Every supporter surface has small objects (2+ items per surface)
4. All scene requirements from the original layout generation are satisfied
5. All semantic critic recommendations with priority >= 6 have been addressed
6. No large empty floor spaces remain (target 30-40% floor occupancy)
7. Objects are style-consistent with the building style
8. All orientations are correct (chairs face tables, sofas face TV, desks face away from wall, etc.)

### Style Consistency
All placed objects must match the building style. When describing objects for search or generation, always include the style (e.g., "modern minimalist white desk lamp" not just "desk lamp").

### Placement Location Rules

**Choosing `place_id`** — decide based on object size, weight, and typical usage:

| Rule | place_id | Examples |
|------|----------|----------|
| Heavy/large furniture that sits on the ground | `"floor"` | bed, sofa, table, desk, wardrobe, bookshelf, chair |
| Items mounted on the wall above floor level | `"wall"` | paintings, wall shelves, mounted TVs, mirrors, clocks |
| Small items that belong on/in a specific piece of furniture | `"<object_id>"` | lamp on desk, books on shelf, plate on table |

**Key distinctions:**
- "beside the table" → `"floor"` (next to it, on the ground)
- "on the table" → `"<table_object_id>"` (on top of it)
- "desk lamp" → `"<desk_object_id>"` (functional pairing — lamp goes on desk, not floor)
- If the intended parent object doesn't exist yet, place the parent first (earlier batch), then place the child in a later batch

**Top vs Inside placement** — when `place_id` is an object_id, also set `"place_location"`:

| place_location | When to use | Examples |
|----------------|-------------|----------|
| `"top"` | Object sits ON the surface of the parent | lamp on desk, vase on table, pillow on bed, remote on coffee table, plate on dining table |
| `"inside"` | Object is contained WITHIN the parent | books inside bookshelf, clothes inside wardrobe, dishes inside dishwasher, files inside cabinet |
| `"both"` | Either is acceptable | storage containers on/in shelf, decorative items on/in display cabinet, supplies on/in desk organizer |

Default to `"top"` if unsure. Use `"inside"` for storage furniture (bookshelves, wardrobes, cabinets, fridges, drawers).

### Retry Logic
- If placement fails, retry ONCE with same or similar objects
- If still failing (space constraints), try smaller alternatives
- Do NOT give up after one failure
- NEVER use "replace all objects"

---

## Handling User Interaction Mid-Workflow

The user may interrupt or modify the scene at any point. Handle these patterns:

| User says | Action |
|-----------|--------|
| "Add a red lamp" / "I want a bookshelf" | Resume at **Stage 4** — source the object, then place it |
| "Make it more modern" / "Change the style to industrial" | Resume at **Stage 2** — regenerate materials, then re-evaluate objects |
| "Remove the chairs" / "Take out the plant" | Call `remove_objects` directly, then continue or run critic |
| "Start over" / "New room" | Begin a fresh workflow from **Stage 1** |
| "How does it look?" / "Show me the room" | Call `get_room_info(layout_id, room_id)` to get a render |
| "What's in the room?" | Call `get_layout(layout_id)` and summarize the objects |

After any mid-workflow change, consider running `run_semantic_critic` to verify the scene still looks good.

---

## Error Handling

- **MCP tool returns error**: Report the error to the user, attempt to fix (e.g., regenerate JSON), retry once
- **Subagent returns invalid JSON**: Re-prompt the subagent with the error details
- **TRELLIS generation fails**: Fall back to `search_objects` or skip the object and note it
- **Isaac Sim connection fails**: Skip Stages 7-8 and export what you have, noting physics wasn't validated
- **Placement solver can't fit objects**: Try smaller alternatives, remove constraints, or accept partial placement

---

## Quick Reference: MCP Tools

| Tool | Parameters | Returns |
|------|-----------|---------|
| `create_room` | `room_json` (string) | `layout_id`, validation, `shared_walls` (if multi-room) |
| `generate_materials` | `layout_id`, `material_descriptions` (JSON string) | texture file paths |
| `add_doors_windows` | `layout_id`, `placement_json` (string) | validation |
| `analyze_shared_walls` | `layout_id` | shared wall classification + suggested connecting doors |
| `add_connecting_doors` | `layout_id`, `connecting_doors_json` (JSON string) | doors added to both rooms |
| `search_objects` | `object_specs` (JSON string) | matched models |
| `generate_3d_model` | `description`, `target_size` (JSON string) | model paths, estimated size |
| `place_objects` | `layout_id`, `room_id`, `objects_json`, `constraints_json` | placed/failed counts |
| `remove_objects` | `layout_id`, `room_id`, `object_ids` (JSON string) | removed IDs |
| `analyze_floor_plan` | `layout_id`, `prompt` | VL model response |
| `run_semantic_critic` | `layout_id`, `room_id` | structured critic analysis |
| `build_scene` | `layout_id`, `room_id` | status |
| `simulate_physics_tool` | `layout_id` | stable/unstable objects |
| `export_scene` | `layout_id`, `room_id` (optional) | USDZ file path (no room_id = full layout) |
| `export_usd_tool` | `layout_id`, `output_path` | USD file path (requires Isaac Sim) |
| `get_layout` | `layout_id` | full FloorPlan JSON |
| `get_room_info` | `layout_id`, `room_id` | room data + render path |

## Quick Reference: Constraint Vocabulary

| Category | Constraint | Target? | Effect |
|----------|-----------|---------|--------|
| Global | `edge` | No | Against a wall |
| Global | `middle` | No | Center/open area |
| Distance | `close to` | Yes | Within ~0.5m |
| Distance | `near` | Yes | Within ~1.5m |
| Distance | `far` | Yes | At least ~2.5m away |
| Relative | `in front of` | Yes | In front (+ center aligned) |
| Relative | `behind` | Yes | Behind target |
| Relative | `left of` | Yes | Left side |
| Relative | `right of` | Yes | Right side |
| Relative | `side of` | Yes | Either side |
| Direction | `face to` | Yes | Orient front toward target |
| Direction | `face same as` | Yes | Match target rotation |
| Alignment | `center aligned` | Yes | Align center axis |
