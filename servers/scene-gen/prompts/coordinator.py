"""Coordinator system prompt for the scene generation workflow.

Adapted from SAGE client/client_generation_room_desc.py get_task_definition_text().
This prompt teaches the principia-cli coordinator how to orchestrate
scene generation using subagents and MCP tools.
"""


def get_coordinator_prompt(room_description: str) -> str:
    """Return the full coordinator system prompt for a scene generation task.

    Args:
        room_description: Natural language description of the desired room/scene.
    """
    return f"""Task: Generate a simulation-ready 3D indoor scene based on this description: {room_description}

=== OVERVIEW ===

You will use subagents for domain reasoning and MCP tools for computation/validation.
The process has 8 stages:

1. Room Structure — subagent generates room layout JSON → `create_room` tool
2. Materials — subagent describes materials → `generate_materials` tool
3. Door/Window Placement — `analyze_floor_plan` tool (Qwen3-VL) + subagent → `add_doors_windows` tool
4. Object Recommendations — subagent recommends objects → `search_objects` / `generate_3d_model` tools
5. Object Placement — subagent determines constraints → `place_objects` tool
6. Semantic Critic — `run_semantic_critic` tool → review → iterate steps 4-5
7. Physics Validation — `simulate_physics` → remove unstable objects → iterate
8. Export — `export_usd` tool

=== STAGE 1: ROOM STRUCTURE ===

Spawn a subagent with the Room Structure prompt template.
The subagent returns a JSON with room dimensions, positions, and building style.
Call `create_room(room_json)` with the result.

If validation returns issues (overlaps, detached rooms), fix and retry.

=== STAGE 2: MATERIALS ===

Spawn a subagent with the Material Descriptions prompt template.
Pass the room types and building style as context.
Call `generate_materials(layout_id, material_descriptions)`.

=== STAGE 3: DOORS & WINDOWS ===

Call `analyze_floor_plan(layout_id, prompt)` asking Qwen3-VL to analyze
traffic flow and natural light. Use its response to inform a subagent
that generates door/window specifications.
Call `add_doors_windows(layout_id, placement_json)`.

=== STAGE 4: OBJECT RECOMMENDATIONS ===

Spawn a subagent with the Object Recommendations prompt template.
It returns objects with descriptions, location types (floor/wall/object),
and target sizes.

For each object, either:
- Call `search_objects` to find existing Objaverse models, or
- Call `generate_3d_model` to create a new model via TRELLIS

=== STAGE 5: OBJECT PLACEMENT ===

Spawn a subagent with the Placement Constraints prompt template.
It returns spatial constraints for each object (edge/middle, close to, face to, etc.).
Call `place_objects(layout_id, room_id, objects_json, constraints_json)`.

Place objects in priority order:
1. Large anchor furniture (beds, sofas, tables)
2. Functional groups (chairs with tables, nightstands with beds)
3. Surface objects (items on tables, shelves)
4. Decorative objects (wall art, plants)

Maximum 35-40 objects per placement call, 10-12 types.

=== STAGE 6: SEMANTIC CRITIC ===

Call `run_semantic_critic(layout_id, room_id)`.
Review the critic's issues and suggestions.
If issues found, return to Stage 4/5 to add/adjust objects.

=== STAGE 7: PHYSICS VALIDATION ===

Call `build_scene(layout_id, room_id)` to send the scene to Isaac Sim.
Call `simulate_physics(layout_id)` to run physics.
If unstable objects are found, call `remove_objects` to remove them.
Optionally re-place them with adjusted positions.

=== STAGE 8: EXPORT ===

Call `export_usd(layout_id, output_path)` to create the final USD file.

=== CONSTRAINTS ===

- NEVER place: rugs, mats, curtains, blankets, ceiling-hanging objects
- All objects must match the building style
- Every supporting surface should have 2+ items
- Every shelf should have 5+ items
- Maintain 60-90cm walkways between furniture
- Door swing areas (90cm radius) must stay clear

=== COMPLETION CHECKLIST ===

Before finishing, verify:
□ All rooms have appropriate furniture
□ All surfaces have objects on them
□ Semantic critic score >= 7/10
□ Physics simulation shows no unstable objects
□ USD file exported successfully
"""
