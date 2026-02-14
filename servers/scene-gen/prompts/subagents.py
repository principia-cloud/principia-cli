"""Subagent prompt templates for domain-specific reasoning.

Each function returns a prompt string that the coordinator fills in
and passes to a principia-cli subagent via `principia "prompt" --json -y`.

Adapted from SAGE client/llm_client.py prompt templates and
server/objects/object_placement_planner.py constraint prompts.
"""


def room_structure_prompt(description: str, num_rooms: str = "single") -> str:
    """Generate room layout JSON from a natural language description.

    The subagent should return valid JSON with room positions, dimensions, and style.
    """
    return f"""You are an expert architectural designer. Generate a room layout as JSON.

DESCRIPTION: {description}
ROOM COUNT: {num_rooms} room(s)

Return a JSON object with this exact structure:
{{
  "building_style": "<style name, e.g. modern, rustic, minimalist>",
  "description": "<brief description of the layout>",
  "created_from_text": "{description}",
  "rooms": [
    {{
      "id": "<unique_room_id, e.g. bedroom_001>",
      "room_type": "<e.g. bedroom, living room, kitchen>",
      "position": {{"x": <float meters>, "y": <float meters>, "z": 0.0}},
      "dimensions": {{"width": <float meters>, "length": <float meters>, "height": <float meters, typically 2.7>}},
      "floor_material": "<e.g. hardwood, tile, carpet>",
      "ceiling_height": <float meters>
    }}
  ]
}}

RULES:
- All dimensions in meters
- Rooms must not overlap
- Multi-room layouts: rooms must share at least one wall (adjacent)
- Typical room sizes: bedroom 3x4m, living room 4x5m, kitchen 3x3.5m, bathroom 2x2.5m
- Position (0,0) is the bottom-left corner of the floor plan
- For single rooms, position at (0,0)

Return ONLY the JSON object, no extra text."""


def material_descriptions_prompt(rooms_info: str, building_style: str) -> str:
    """Describe floor and wall materials for each room."""
    return f"""You are an interior designer selecting materials for a {building_style} home.

ROOMS:
{rooms_info}

For each room, provide material descriptions for floor and walls.
The descriptions should be specific enough to generate PBR textures.

Return JSON:
{{
  "<room_id>": {{
    "floor": "<detailed material description, e.g. 'warm honey-toned oak hardwood planks with subtle grain patterns'>",
    "wall": "<detailed material description, e.g. 'smooth cream-white painted drywall with slight eggshell texture'>"
  }}
}}

RULES:
- Match the {building_style} aesthetic
- Be specific about color, texture, and finish
- Consider how materials complement each other across rooms
- Kitchens/bathrooms typically use tile or stone for floors

Return ONLY the JSON object."""


def object_recommendations_prompt(room_info: str, building_style: str) -> str:
    """Recommend objects/furniture for a room."""
    return f"""You are an experienced interior designer furnishing a room.

ROOM INFORMATION:
{room_info}

STYLE: {building_style}

Recommend objects to place in this room. For each object, specify:
- name: descriptive name
- description: detailed description matching the style
- location: "floor", "wall", or an existing object ID (for items placed on surfaces)
- size: [width, length, height] in centimeters
- quantity: how many to place
- priority: "essential", "important", or "decorative"

Return JSON:
{{
  "objects": [
    {{
      "name": "<e.g. queen_bed>",
      "description": "<e.g. modern platform queen bed with upholstered headboard in charcoal fabric>",
      "location": "floor",
      "size": [160, 200, 60],
      "quantity": 1,
      "priority": "essential"
    }}
  ]
}}

RULES:
- Start with essential furniture, then important, then decorative
- Include both large anchor pieces and small detail items
- Every supporting surface (desk, table, shelf) should have items on it
- Include wall-mounted items (art, mirrors, shelves)
- All items must match the {building_style} style
- Sizes in centimeters, realistic for each object type
- NEVER include: rugs, mats, curtains, blankets, ceiling fixtures

Return ONLY the JSON object."""


def placement_constraints_prompt(room_info: str, objects_info: str) -> str:
    """Determine spatial placement constraints for objects.

    Adapted from SAGE object_placement_planner.py constraint prompt.
    """
    return f"""You are an experienced interior designer creating a furniture layout.

ROOM:
{room_info}

OBJECTS TO PLACE:
{objects_info}

Assign placement constraints to each object. Available constraint types:

GLOBAL (required, pick one):
- "edge": Place near walls
- "middle": Place in central area

DISTANCE (relative to another object):
- "close to, <object_id>": Within ~30cm
- "near, <object_id>": ~50-150cm distance
- "far, <object_id>": >150cm distance

POSITION (relative to target's facing direction):
- "in front of, <object_id>": In front of target
- "around, <object_id>": Around target (chairs around table)
- "left of, <object_id>": To the left of target
- "right of, <object_id>": To the right of target
- "side of, <object_id>": To either side of target

ALIGNMENT:
- "center aligned, <object_id>": Align centers

ROTATION:
- "face to, <object_id>": Face toward target
- "face same as, <object_id>": Same orientation as target

Return JSON:
{{
  "reasoning": {{
    "design_strategy": "<overall approach>",
    "importance_ranking": "<priority order of objects>",
    "functional_relationships": "<how objects relate>"
  }},
  "constraints": {{
    "<object_id>": ["<constraint1>", "<constraint2>", ...],
    ...
  }}
}}

RULES:
- Objects can only reference previously listed objects in constraints
- Place anchor furniture first (beds, sofas, tables) with just ["edge"] or ["middle"]
- Secondary objects get 3-5 constraints each for precise placement
- Chairs near tables: ["middle", "close to, table_id", "in front of, table_id", "face to, table_id"]
- Nightstands next to beds: ["edge", "close to, bed_id", "left of, bed_id", "face same as, bed_id"]
- Objects against different walls should use "far" constraints
- Provide as many constraints as applicable (4-5 per object) for best results

Return ONLY the JSON object."""


def door_window_specs_prompt(room_info: str, vl_analysis: str) -> str:
    """Specify door and window placements based on VL model analysis."""
    return f"""You are an architect specifying door and window placements.

ROOM INFORMATION:
{room_info}

VISION MODEL ANALYSIS (traffic flow and natural light):
{vl_analysis}

Specify doors and windows for each room.

Return JSON:
{{
  "room_id": "<room_id>",
  "doors": [
    {{
      "wall_id": "<room_id>_<n/s/e/w>_wall",
      "position_on_wall": <0.0-1.0, position along wall>,
      "width": <meters, typically 0.9>,
      "height": <meters, typically 2.1>,
      "door_type": "standard",
      "opens_inward": true
    }}
  ],
  "windows": [
    {{
      "wall_id": "<room_id>_<n/s/e/w>_wall",
      "position_on_wall": <0.0-1.0>,
      "width": <meters, typically 1.0-1.5>,
      "height": <meters, typically 1.0-1.2>,
      "sill_height": <meters, typically 0.9>
    }}
  ]
}}

RULES:
- position_on_wall is fractional: 0.0 = start of wall, 0.5 = center, 1.0 = end
- Most rooms need at least one door
- Living spaces need windows for natural light
- Bathrooms/closets may skip windows
- Door width: 0.8-1.0m standard, 1.2-1.5m double
- Window sill height: 0.8-1.0m typical
- Doors/windows must fit within wall length
- On shared walls, doors must exist on BOTH sides

Return ONLY the JSON object."""


def operation_analysis_prompt(user_request: str, current_scene_info: str) -> str:
    """Classify a user operation as add/remove/replace and list affected objects."""
    return f"""You are analyzing a scene modification request.

CURRENT SCENE:
{current_scene_info}

USER REQUEST: {user_request}

Classify the operation and identify affected objects.

Return JSON:
{{
  "operation": "add|remove|replace|modify",
  "description": "<what the user wants to do>",
  "affected_objects": [
    {{
      "id": "<object_id or 'new'>",
      "action": "add|remove|replace|move",
      "details": "<specifics>"
    }}
  ]
}}

Return ONLY the JSON object."""
