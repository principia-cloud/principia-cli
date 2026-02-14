"""DFS-based object placement solver.

Ported from SAGE server/objects/object_placement_planner.py.
Implements grid-based DFS with constraint scoring for floor objects,
wall coordinate systems for wall objects, and surface scatter for on-object placement.
"""

import copy
import difflib
import logging
import math
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from rtree import index as rtree_index

from models import Object, Room, FloorPlan, Point3D, Euler, Dimensions, Door, Window

logger = logging.getLogger("scene-gen.placement_solver")

# ---------------------------------------------------------------------------
# Constraint type constants
# ---------------------------------------------------------------------------
EDGE = "edge"
MIDDLE = "middle"
CLOSE_TO = "close to"
NEAR = "near"
FAR = "far"
IN_FRONT_OF = "in front of"
BEHIND = "behind"
LEFT_OF = "left of"
RIGHT_OF = "right of"
SIDE_OF = "side of"
AROUND = "around"
CENTER_ALIGNED = "center aligned"
FACE_TO = "face to"
FACE_SAME_AS = "face same as"


# ===================================================================
# Constraint Parsing (SAGE lines 1465-1638)
# ===================================================================

def parse_constraints_from_json(
    constraints_list: List[Dict[str, Any]],
    new_object_ids: List[str],
    existing_object_ids: Optional[List[str]] = None,
    existing_id_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse constraints from JSON format into structured format for the solver.

    Accepts two input formats:
      1. List of dicts: [{"object_id": "bed_001", "constraints": ["edge", "close to, desk_001"]}]
      2. Dict mapping: {"bed_001": ["edge", "close to, desk_001"]}

    Returns:
        Dict mapping object_id to list of parsed constraint dicts.
    """
    if existing_object_ids is None:
        existing_object_ids = []
    if existing_id_mapping is None:
        existing_id_mapping = {}

    constraint_name2type = {
        "edge": "global",
        "middle": "global",
        "in front of": "relative",
        "behind": "relative",
        "left of": "relative",
        "right of": "relative",
        "side of": "relative",
        "around": "relative",
        "face to": "direction",
        "face same as": "direction",
        "aligned": "alignment",
        "center alignment": "alignment",
        "center aligned": "alignment",
        "aligned center": "alignment",
        "edge alignment": "alignment",
        "close to": "distance",
        "near": "distance",
        "far": "distance",
    }

    # Normalize input: if dict mapping, convert to list-of-dicts format
    if isinstance(constraints_list, dict):
        normalized = []
        for obj_id, constraint_strings in constraints_list.items():
            normalized.append({"object_id": obj_id, "constraints": constraint_strings})
        constraints_list = normalized

    object2constraints: Dict[str, List[Dict[str, Any]]] = {}

    for constraint_entry in constraints_list:
        object_id = constraint_entry.get("object_id", "").strip()
        constraint_strings = constraint_entry.get("constraints", [])

        if object_id not in new_object_ids:
            continue

        object2constraints[object_id] = []

        for constraint_str in constraint_strings:
            constraint_str = constraint_str.strip().lower()
            parts = [p.strip() for p in constraint_str.split(",")]
            constraint_name = parts[0]

            if constraint_name == "n/a":
                continue

            # Fuzzy match constraint name
            try:
                constraint_type = constraint_name2type[constraint_name]
            except KeyError:
                close_matches = difflib.get_close_matches(
                    constraint_name, constraint_name2type.keys(), n=1, cutoff=0.3
                )
                if close_matches:
                    constraint_name = close_matches[0]
                    constraint_type = constraint_name2type[constraint_name]
                else:
                    print(f"Constraint type not found for '{constraint_name}', skipping", file=sys.stderr)
                    continue

            if constraint_type == "global":
                object2constraints[object_id].append(
                    {"type": constraint_type, "constraint": constraint_name}
                )
            elif constraint_type in ("relative", "direction", "alignment", "distance"):
                if len(parts) < 2:
                    print(f"Constraint '{constraint_str}' missing target, skipping", file=sys.stderr)
                    continue

                target = parts[1].strip()
                target_is_valid = False
                actual_target = target

                if target in object2constraints:
                    target_is_valid = True
                elif target in existing_object_ids:
                    actual_target = existing_id_mapping.get(target, target)
                    target_is_valid = True
                else:
                    for eid in existing_object_ids:
                        if target in eid:
                            actual_target = existing_id_mapping.get(eid, eid)
                            target_is_valid = True
                            break
                    if not target_is_valid:
                        # Also check if target matches a new object id
                        for nid in new_object_ids:
                            if target in nid or nid in target:
                                actual_target = nid
                                target_is_valid = True
                                break

                if target_is_valid:
                    if constraint_name == "around":
                        object2constraints[object_id].append(
                            {"type": "distance", "constraint": "close to", "target": actual_target}
                        )
                        object2constraints[object_id].append(
                            {"type": "direction", "constraint": "face to", "target": actual_target}
                        )
                    elif constraint_name == "in front of":
                        object2constraints[object_id].append(
                            {"type": "relative", "constraint": "in front of", "target": actual_target}
                        )
                        object2constraints[object_id].append(
                            {"type": "alignment", "constraint": "center aligned", "target": actual_target}
                        )
                    else:
                        object2constraints[object_id].append(
                            {"type": constraint_type, "constraint": constraint_name, "target": actual_target}
                        )
                else:
                    print(f"Target object '{target}' not found, skipping constraint", file=sys.stderr)

    # Deduplicate
    cleaned: Dict[str, List[Dict[str, Any]]] = {}
    for object_id, constraints in object2constraints.items():
        seen = set()
        unique = []
        for c in constraints:
            key = (c["type"], c["constraint"], c.get("target"))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        cleaned[object_id] = unique

    return cleaned


# ===================================================================
# Door / Window obstacle generation (SAGE lines 1641-1843)
# ===================================================================

def get_door_window_placements(room: Room) -> Dict[str, Tuple]:
    """Compute door swing areas and window strips as obstacle polygons.

    Returns dict of ``{id: (center, rotation, vertices, weight)}``.
    """
    placements: Dict[str, Tuple] = {}
    i = 0
    wall_map = {wall.id: wall for wall in room.walls}

    for door in room.doors:
        wall = wall_map.get(door.wall_id)
        if wall is None:
            continue

        start_point = wall.start_point
        end_point = wall.end_point

        door_center_x = start_point.x + (end_point.x - start_point.x) * door.position_on_wall - room.position.x
        door_center_y = start_point.y + (end_point.y - start_point.y) * door.position_on_wall - room.position.y
        door_center_x_cm = door_center_x * 100
        door_center_y_cm = door_center_y * 100
        door_width_cm = door.width * 100

        room_center_x = room.position.x + room.dimensions.width / 2
        room_center_y = room.position.y + room.dimensions.length / 2
        mid_point_x = (start_point.x + end_point.x) / 2
        mid_point_y = (start_point.y + end_point.y) / 2
        wall_offset_x = room_center_x - mid_point_x
        wall_offset_y = room_center_y - mid_point_y

        if getattr(door, "opening", False):
            opening_space_cm = 50
            if abs(wall_offset_x) > abs(wall_offset_y):
                door_open_length_x_cm = opening_space_cm
                door_open_length_y_cm = door_width_cm
                sign = 1 if wall_offset_x > 0 else -1
                door_open_center_offset = (sign * opening_space_cm / 2, 0)
            else:
                door_open_length_x_cm = door_width_cm
                door_open_length_y_cm = opening_space_cm
                sign = 1 if wall_offset_y > 0 else -1
                door_open_center_offset = (0, sign * opening_space_cm / 2)

            ocx = door_center_x_cm + door_open_center_offset[0]
            ocy = door_center_y_cm + door_open_center_offset[1]
            vertices = [
                (ocx - door_open_length_x_cm / 2, ocy - door_open_length_y_cm / 2),
                (ocx + door_open_length_x_cm / 2, ocy - door_open_length_y_cm / 2),
                (ocx + door_open_length_x_cm / 2, ocy + door_open_length_y_cm / 2),
                (ocx - door_open_length_x_cm / 2, ocy + door_open_length_y_cm / 2),
            ]
        else:
            if abs(wall_offset_x) > abs(wall_offset_y):
                sign = 1 if wall_offset_x > 0 else -1
                door_open_center_offset = (sign * door_width_cm / 2, 0)
            else:
                sign = 1 if wall_offset_y > 0 else -1
                door_open_center_offset = (0, sign * door_width_cm / 2)

            ocx = door_center_x_cm + door_open_center_offset[0]
            ocy = door_center_y_cm + door_open_center_offset[1]
            vertices = [
                (ocx - door_width_cm / 2, ocy - door_width_cm / 2),
                (ocx + door_width_cm / 2, ocy - door_width_cm / 2),
                (ocx + door_width_cm / 2, ocy + door_width_cm / 2),
                (ocx - door_width_cm / 2, ocy + door_width_cm / 2),
            ]

        placements[f"door-{i}"] = ((ocx, ocy), 0, vertices, 1)
        i += 1

    for window in room.windows:
        wall = wall_map.get(window.wall_id)
        if wall is None:
            continue

        start_point = wall.start_point
        end_point = wall.end_point

        window_center_x = start_point.x + (end_point.x - start_point.x) * window.position_on_wall - room.position.x
        window_center_y = start_point.y + (end_point.y - start_point.y) * window.position_on_wall - room.position.y
        window_center_x_cm = window_center_x * 100
        window_center_y_cm = window_center_y * 100
        window_width_cm = window.width * 100

        room_center_x = room.position.x + room.dimensions.width / 2
        room_center_y = room.position.y + room.dimensions.length / 2
        mid_point_x = (start_point.x + end_point.x) / 2
        mid_point_y = (start_point.y + end_point.y) / 2
        wall_offset_x = room_center_x - mid_point_x
        wall_offset_y = room_center_y - mid_point_y

        wt = wall.thickness
        if abs(wall_offset_x) > abs(wall_offset_y):
            sign = 1 if wall_offset_x > 0 else -1
            window_open_center_offset = (sign * wt * 0.5 * 100 / 2, 0)
            window_length_x = wt * 0.45 * 100
            window_length_y = window_width_cm
        else:
            sign = 1 if wall_offset_y > 0 else -1
            window_open_center_offset = (0, sign * wt * 0.5 * 100 / 2)
            window_length_x = window_width_cm
            window_length_y = wt * 0.45 * 100

        wcx = window_center_x_cm + window_open_center_offset[0]
        wcy = window_center_y_cm + window_open_center_offset[1]
        vertices = [
            (wcx - window_length_x / 2, wcy - window_length_y / 2),
            (wcx + window_length_x / 2, wcy - window_length_y / 2),
            (wcx + window_length_x / 2, wcy + window_length_y / 2),
            (wcx - window_length_x / 2, wcy + window_length_y / 2),
        ]
        placements[f"window-{i}"] = ((wcx, wcy), 0, vertices, 0.3)
        i += 1

    return placements


# ===================================================================
# solution_to_objects (SAGE lines 1846-1893)
# ===================================================================

def solution_to_objects(
    solution: Dict[str, Any],
    solution_constraints: Dict[str, Any],
    floor_objects: List[Object],
    room: Room,
) -> List[Object]:
    """Convert DFS solver output back to Object instances."""
    placed = []
    for object_id, solution_data in solution.items():
        constraints = solution_constraints.get(object_id, [])
        if "door" in object_id or "window" in object_id or "open" in object_id:
            continue
        if object_id.startswith("existing-"):
            continue

        original_obj = next((o for o in floor_objects if o.id == object_id), None)
        if not original_obj:
            continue

        center_pos = solution_data[0]
        rotation_degrees = solution_data[1]

        x_meters = center_pos[0] / 100 + room.position.x
        y_meters = center_pos[1] / 100 + room.position.y
        z_meters = room.position.z

        placed_obj = Object(
            id=original_obj.id,
            room_id=room.id,
            type=original_obj.type,
            description=getattr(original_obj, "description", f"Placed {original_obj.type}"),
            position=Point3D(x=x_meters, y=y_meters, z=z_meters),
            rotation=Euler(x=0, y=0, z=rotation_degrees),
            dimensions=original_obj.dimensions,
            source=getattr(original_obj, "source", "placement"),
            source_id=getattr(original_obj, "source_id", original_obj.id),
            place_id=original_obj.place_id,
            placement_constraints=constraints,
            mass=getattr(original_obj, "mass", 1.0),
        )
        placed.append(placed_obj)

    return placed


# ===================================================================
# DFS_Solver_Floor (SAGE lines 3048-4021)
# ===================================================================

class DFS_Solver_Floor:
    """Grid-based DFS placement solver with constraint scoring."""

    def __init__(
        self,
        grid_size: int = 20,
        random_seed: int = 0,
        max_duration: float = 5,
        constraint_bouns: float = 0.2,
        room_id: Optional[str] = None,
    ):
        self.grid_size = grid_size
        self.random_seed = random_seed
        self.max_duration = max_duration
        self.constraint_bouns = constraint_bouns
        self.start_time: Optional[float] = None
        self.solutions: List[Dict] = []
        self.constraints_dict_list: List[Dict] = []

        self.func_dict = {
            "global": {"edge": self.place_edge},
            "relative": self.place_relative,
            "direction": self.place_face,
            "alignment": self.place_alignment_center,
            "distance": self.place_distance,
        }

        self.constraint_type2weight = {
            "global": 2.0,
            "relative": 1.0,
            "direction": 2.0,
            "alignment": 0.8,
            "distance": 1.0,
        }

        self.edge_bouns = 0.0
        self.room_id = room_id

    # ------------------------------------------------------------------
    # Top-level interface
    # ------------------------------------------------------------------

    def get_solution(self, bounds, objects_list, constraints, initial_state):
        self.start_time = time.time()
        self.solutions = []
        self.constraints_dict_list = []
        constraints_dict: Dict[str, Any] = {}

        result, result_constraints = self._attempt_solution(
            bounds, objects_list, constraints, initial_state, constraints_dict
        )
        if not result:
            result = {}
            result_constraints = {}
        return result, result_constraints

    def _attempt_solution(self, bounds, objects_list, constraints, initial_state, constraints_dict):
        grid_points = self.create_grids(bounds)
        grid_points = self.remove_points(grid_points, initial_state)
        branch_factor = 15 if len(objects_list) < 20 else 5
        try:
            self.dfs(bounds, objects_list, constraints, grid_points, initial_state, constraints_dict, branch_factor)
        except _TimeLimitReached:
            pass

        if self.solutions:
            return self._get_max_solution()
        return None, None

    def _get_max_solution(self):
        path_weights = [sum(obj[-1] for obj in sol.values()) for sol in self.solutions]
        max_idx = int(np.argmax(path_weights))
        return self.solutions[max_idx], self.constraints_dict_list[max_idx]

    # ------------------------------------------------------------------
    # DFS core
    # ------------------------------------------------------------------

    def dfs(self, room_poly, objects_list, constraints, grid_points, placed_objects, constraints_dict, branch_factor):
        if len(objects_list) == 0:
            self.solutions.append(placed_objects)
            self.constraints_dict_list.append(constraints_dict)
            return [placed_objects]

        if time.time() - self.start_time > self.max_duration:
            # Store partial solution before raising
            if placed_objects:
                self.solutions.append(placed_objects)
                self.constraints_dict_list.append(constraints_dict)
            raise _TimeLimitReached()

        object_id, object_dim = objects_list[0]
        print(f"dfs object_id: {object_id}, placed: {len(placed_objects)}, dim: {object_dim}", file=sys.stderr)

        object_constraints = constraints.get(object_id, [{"type": "global", "constraint": "edge"}])

        placements, placements_constraints = self.get_possible_placements(
            room_poly, object_dim, object_constraints, grid_points, placed_objects
        )

        if len(placements) == 0 and len(placed_objects) != 0:
            self.solutions.append(placed_objects)
            self.constraints_dict_list.append(constraints_dict)

        # Softmax sampling for branch selection
        if branch_factor > 1 and len(placements) > branch_factor:
            scores = np.array([p[-1] for p in placements])
            temperature = 0.4
            scores_scaled = scores / temperature
            exp_scores = np.exp(scores_scaled - np.max(scores_scaled))
            probabilities = exp_scores / np.sum(exp_scores)
            selected_indices = np.random.choice(
                len(placements), size=min(branch_factor, len(placements)), replace=False, p=probabilities
            )
            selected_placements = [placements[i] for i in selected_indices]
            selected_constraints = [placements_constraints[i] for i in selected_indices]
        else:
            selected_placements = placements[:branch_factor]
            selected_constraints = placements_constraints[:branch_factor]

        paths = []
        for placement, pc in zip(selected_placements, selected_constraints):
            placed_updated = copy.deepcopy(placed_objects)
            placed_updated[object_id] = placement
            grid_updated = self.remove_points(grid_points, placed_updated)
            constraints_dict_updated = copy.deepcopy(constraints_dict)
            constraints_dict_updated[object_id] = pc
            sub = self.dfs(room_poly, objects_list[1:], constraints, grid_updated, placed_updated, constraints_dict_updated, 1)
            paths.extend(sub)
        return paths

    # ------------------------------------------------------------------
    # Placement enumeration and scoring
    # ------------------------------------------------------------------

    def get_possible_placements(self, room_poly, object_dim, constraints, grid_points, placed_objects):
        solutions = self.filter_collision(
            placed_objects, self.get_all_solutions(room_poly, grid_points, object_dim)
        )
        solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
        edge_solutions = self.place_edge(room_poly, copy.deepcopy(solutions), object_dim, placed_objects=None)

        global_constraint = next(
            (c for c in constraints if c["type"] == "global"), None
        )
        if global_constraint is None:
            global_constraint = {"type": "global", "constraint": "edge"}

        if global_constraint["constraint"] == "edge":
            candidate_solutions = copy.deepcopy(edge_solutions)
        else:
            candidate_solutions = copy.deepcopy(solutions)

        candidate_solutions = self.filter_collision(placed_objects, candidate_solutions)

        if not candidate_solutions:
            return [], []

        random.shuffle(candidate_solutions)
        placement2score = {tuple(s[:3]): float(s[-1]) for s in candidate_solutions}
        placement2constraints = {tuple(s[:3]): [] for s in candidate_solutions}

        # Edge bonus
        for sol in candidate_solutions:
            if edge_solutions and sol in edge_solutions and len(constraints) >= 1:
                placement2score[tuple(sol[:3])] += self.edge_bouns
                placement2constraints[tuple(sol[:3])].append({"type": "global", "constraint": "edge"})

        # Score each constraint
        for constraint in constraints:
            if constraint["type"] == "global" and constraint["constraint"] == "middle":
                valid = self.place_middle(candidate_solutions, placed_objects, room_poly)
                for sol in valid:
                    placement2score[tuple(sol[:3])] += sol[-1] * 1.0
                    placement2constraints[tuple(sol[:3])].append(constraint)
                continue

            if "target" not in constraint:
                continue
            if constraint["target"] not in placed_objects:
                print(f"skipping constraint (target not placed): {constraint}", file=sys.stderr)
                continue

            func = self.func_dict.get(constraint["type"])
            if func is None:
                continue
            valid = func(constraint["constraint"], placed_objects[constraint["target"]], candidate_solutions)

            weight = self.constraint_type2weight[constraint["type"]]
            for sol in valid:
                bouns = sol[-1]
                if constraint["type"] == "direction" and constraint["constraint"] == "face to":
                    placement2score[tuple(sol[:3])] += bouns * self.constraint_bouns * weight
                elif constraint["type"] in ("distance", "relative"):
                    placement2score[tuple(sol[:3])] += bouns * weight
                else:
                    placement2score[tuple(sol[:3])] += self.constraint_bouns * weight
                placement2constraints[tuple(sol[:3])].append(constraint)

        # Sort by score, then distance to mean of top placements
        sorted_by_score = sorted(placement2score.keys(), key=lambda p: -placement2score[p])
        top_k = max(5, int(len(sorted_by_score) * 0.03))
        top_placements = sorted_by_score[:top_k]

        if top_placements:
            mean_x = sum(p[0][0] for p in top_placements) / len(top_placements)
            mean_y = sum(p[0][1] for p in top_placements) / len(top_placements)
        else:
            mean_x, mean_y = 0, 0

        sorted_placements = sorted(
            placement2score.keys(),
            key=lambda p: (
                -placement2score[p],
                ((p[0][0] - mean_x) ** 2 + (p[0][1] - mean_y) ** 2) ** 0.5,
            ),
        )

        sorted_solutions = [list(p) + [placement2score[p]] for p in sorted_placements]
        sorted_constraints = [list(p) + [placement2constraints[p]] for p in sorted_placements]

        if sorted_solutions[0][-1] < 0:
            return [], []

        return sorted_solutions, sorted_constraints

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def create_grids(self, room_poly):
        min_x, min_y, max_x, max_y = room_poly.bounds
        grid_points = []
        for x in range(int(min_x), int(max_x), self.grid_size):
            for y in range(int(min_y), int(max_y), self.grid_size):
                if room_poly.contains(Point(x, y)):
                    grid_points.append((x, y))
        return grid_points

    def remove_points(self, grid_points, objects_dict):
        idx = rtree_index.Index()
        polygons = []
        for i, val in enumerate(objects_dict.values()):
            _, _, obj_coords, _ = val
            poly = Polygon(obj_coords)
            idx.insert(i, poly.bounds)
            polygons.append(poly)

        valid = []
        for point in grid_points:
            p = Point(point)
            candidates = [polygons[i] for i in idx.intersection(p.bounds)]
            if not any(c.contains(p) for c in candidates):
                valid.append(point)
        return valid

    def get_all_solutions(self, room_poly, grid_points, object_dim):
        obj_length, obj_width = object_dim
        obj_half_length, obj_half_width = obj_length / 2, obj_width / 2

        rotation_adjustments = {
            0: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)),
            90: ((-obj_half_width, -obj_half_length), (obj_half_width, obj_half_length)),
            180: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)),
            270: ((-obj_half_width, -obj_half_length), (obj_half_width, obj_half_length)),
        }

        solutions = []
        for rotation in [0, 90, 180, 270]:
            for point in grid_points:
                cx, cy = point
                ll_adj, ur_adj = rotation_adjustments[rotation]
                ll = (cx + ll_adj[0], cy + ll_adj[1])
                ur = (cx + ur_adj[0], cy + ur_adj[1])
                obj_box = box(*ll, *ur)
                if room_poly.contains(obj_box):
                    solutions.append([point, rotation, tuple(obj_box.exterior.coords[:]), 1])
        return solutions

    def filter_collision(self, objects_dict, solutions):
        object_polygons = [Polygon(coords) for _, _, coords, _ in objects_dict.values()]
        valid = []
        for sol in solutions:
            sol_poly = Polygon(sol[2])
            if not any(sol_poly.intersects(p) for p in object_polygons):
                valid.append(sol)
        return valid

    def filter_facing_wall(self, room_poly, solutions, obj_dim):
        obj_half_width = obj_dim[1] / 2
        front_center_adjustments = {
            0: (0, obj_half_width),
            90: (-obj_half_width, 0),
            180: (0, -obj_half_width),
            270: (obj_half_width, 0),
        }
        valid = []
        for sol in solutions:
            cx, cy = sol[0]
            rotation = sol[1]
            adj = front_center_adjustments[rotation]
            fcx, fcy = cx + adj[0], cy + adj[1]
            if room_poly.boundary.distance(Point(fcx, fcy)) >= 10:
                valid.append(sol)
        return valid

    # ------------------------------------------------------------------
    # Constraint scoring methods
    # ------------------------------------------------------------------

    def place_edge(self, room_poly, solutions, obj_dim, placed_objects=None):
        """Snap back-center to wall with offset vector and lateral clearance bonus."""
        valid = []
        obj_half_width = obj_dim[1] / 2

        back_center_adjustments = {
            0: (0, -obj_half_width),
            90: (obj_half_width, 0),
            180: (0, obj_half_width),
            270: (-obj_half_width, 0),
        }
        two_side_vectors = {
            0: [(1, 0), (-1, 0)],
            90: [(0, 1), (0, -1)],
            180: [(1, 0), (-1, 0)],
            270: [(0, 1), (0, -1)],
        }

        for sol in solutions:
            cx, cy = sol[0]
            rotation = sol[1]
            adj = back_center_adjustments[rotation]
            bcx, bcy = cx + adj[0], cy + adj[1]

            back_dist = room_poly.boundary.distance(Point(bcx, bcy))
            center_dist = room_poly.boundary.distance(Point(cx, cy))

            if back_dist <= self.grid_size and back_dist < center_dist:
                sol[-1] += self.constraint_bouns

                # Snap to wall
                c2b = np.array([bcx - cx, bcy - cy])
                norm = np.linalg.norm(c2b)
                if norm > 0:
                    c2b /= norm
                    offset = c2b * back_dist
                    sol[0] = (cx + offset[0], cy + offset[1])
                    sol[2] = tuple(
                        (v[0] + offset[0], v[1] + offset[1]) for v in sol[2]
                    )

                # Lateral clearance bonus
                if placed_objects is not None:
                    sol_center_np = np.array(sol[0])
                    min_distances = []
                    for vec in two_side_vectors[rotation]:
                        far_point = sol_center_np + 1e6 * np.array(vec)
                        line = LineString([sol[0], tuple(far_point)])

                        # Room boundary
                        inter = line.intersection(room_poly.boundary)
                        for pt in _extract_points(inter):
                            min_distances.append(np.linalg.norm(sol_center_np - np.array(pt)))

                        # Placed objects
                        for obj_data in placed_objects.values():
                            obj_poly = Polygon(obj_data[2])
                            obj_inter = line.intersection(obj_poly)
                            for pt in _extract_points(obj_inter):
                                min_distances.append(np.linalg.norm(sol_center_np - np.array(pt)))

                    if min_distances:
                        sol[-1] += min(min(min_distances), 1000.0) / 1000.0 * 0.01

                valid.append(sol)
        return valid

    def place_relative(self, place_type, target_object, solutions):
        """3-tier relative position scoring (strict/loose/looser)."""
        valid = []
        _, target_rotation, target_coords, _ = target_object
        target_poly = Polygon(target_coords)
        min_x, min_y, max_x, max_y = target_poly.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2

        strict = _relative_comparisons(place_type, target_rotation, min_x, min_y, max_x, max_y, "strict")
        loose = _relative_comparisons(place_type, target_rotation, min_x, min_y, max_x, max_y, "loose")
        looser = _relative_comparisons(place_type, target_rotation, mean_x, mean_y, mean_x, mean_y, "looser")

        for sol in solutions:
            sc = sol[0]
            if strict(sc):
                sol[-1] = self.constraint_bouns
                valid.append(sol)
            elif loose(sc):
                sol[-1] = self.constraint_bouns * 0.2
                valid.append(sol)
            elif looser(sc):
                sol[-1] = self.constraint_bouns * 0.01
                valid.append(sol)

        return valid

    def place_distance(self, distance_type, target_object, solutions):
        """Interpolated distance scoring curves."""
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        distances = []
        valid = []

        for sol in solutions:
            sol_poly = Polygon(sol[2])
            d = target_poly.distance(sol_poly)
            distances.append(d)
            sol[-1] = d
            valid.append(sol)

        if not distances:
            return valid

        min_d, max_d = min(distances), max(distances)

        if distance_type == "close to":
            if min_d > 50:
                points = [(min_d, -1e6), (max_d, -1e6)]
            else:
                points = [(min_d, 1), (min(max_d, 50), 0), (max(max_d, 50), -1e6)]
        elif distance_type == "near":
            if min_d > 150:
                points = [(min_d, -1e6), (max_d, -1e6)]
            else:
                points = [(0, 0.2), (30, 1), (60, 1), (150, 0), (10000, 0)]
        elif distance_type == "far":
            points = [(min_d, 0), (max_d, 1)]
        else:
            return valid

        x_pts = [p[0] for p in points]
        y_pts = [p[1] for p in points]

        if len(x_pts) > 1:
            f = interp1d(x_pts, y_pts, kind="linear", fill_value="extrapolate")
            for sol in valid:
                sol[-1] = float(f(sol[-1]))
        else:
            for sol in valid:
                sol[-1] = y_pts[0] if y_pts else 0

        return valid

    def place_face(self, face_type, target_object, solutions):
        if face_type == "face to":
            return self.place_face_to(target_object, solutions)
        elif face_type == "face same as":
            return self.place_face_same(target_object, solutions)
        return []

    def place_face_to(self, target_object, solutions):
        """Ray casting face-to constraint with extended polygon fallback."""
        unit_vectors = {
            0: np.array([0.0, 1.0]),
            90: np.array([-1.0, 0.0]),
            180: np.array([0.0, -1.0]),
            270: np.array([1.0, 0.0]),
        }

        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        target_coords_np = np.array(target_coords).reshape(-1, 2)
        coords_x_max, coords_y_max = target_coords_np.max(axis=0)
        coords_x_min, coords_y_min = target_coords_np.min(axis=0)
        target_obj_x = (coords_x_max + coords_x_min) / 2
        target_obj_y = (coords_y_max + coords_y_min) / 2

        # Extended polygons for looser matching
        np_x_inf = target_coords_np.copy()
        np_x_inf[np_x_inf[:, 0] > target_obj_x, 0] = coords_x_max + 1e6
        np_x_inf[np_x_inf[:, 0] < target_obj_x, 0] = coords_x_min - 1e6
        np_y_inf = target_coords_np.copy()
        np_y_inf[np_y_inf[:, 1] > target_obj_y, 1] = coords_y_max + 1e6
        np_y_inf[np_y_inf[:, 1] < target_obj_y, 1] = coords_y_min - 1e6

        try:
            poly_x_inf = Polygon(np_x_inf.tolist())
            poly_y_inf = Polygon(np_y_inf.tolist())
        except Exception:
            poly_x_inf = target_poly
            poly_y_inf = target_poly

        valid = []
        for sol in solutions:
            sc = sol[0]
            rot = sol[1]
            far_point = np.array(sc) + 1e6 * unit_vectors[rot]
            half_line = LineString([sc, tuple(far_point)])
            sc_point = Point(sc[0], sc[1])

            if half_line.intersects(target_poly):
                sol[-1] = 1.0
                valid.append(sol)
            elif (not poly_x_inf.contains(sc_point)) and half_line.intersects(poly_x_inf):
                sol[-1] = 0.3
                valid.append(sol)
            elif (not poly_y_inf.contains(sc_point)) and half_line.intersects(poly_y_inf):
                sol[-1] = 0.3
                valid.append(sol)

        return valid

    def place_face_same(self, target_object, solutions):
        target_rotation = target_object[1]
        valid = []
        for sol in solutions:
            if abs(sol[1] - target_rotation) < 10:
                sol[-1] += self.constraint_bouns
                valid.append(sol)
        return valid

    def place_middle(self, candidate_solutions, placed_objects, room_poly):
        """Score by minimum distance from room edges and placed objects."""
        valid = []
        if not candidate_solutions:
            return valid

        middle_scores = []
        for sol in candidate_solutions:
            sol_poly = Polygon(sol[2])
            dist_room = room_poly.exterior.distance(sol_poly)
            min_obj = float("inf")
            for obj_data in placed_objects.values():
                d = sol_poly.distance(Polygon(obj_data[2]))
                min_obj = min(min_obj, d)
            if min_obj == float("inf"):
                middle_scores.append(dist_room)
            else:
                middle_scores.append(min(dist_room, min_obj))

        min_s, max_s = min(middle_scores), max(middle_scores)
        for i, sol in enumerate(candidate_solutions):
            if max_s > min_s:
                norm = (middle_scores[i] - min_s) / (max_s - min_s)
            else:
                norm = 0.5
            sol[-1] = norm * 0.02
            valid.append(sol)
        return valid

    def place_alignment_center(self, alignment_type, target_object, solutions):
        target_center = target_object[0]
        eps = self.grid_size / 2
        valid = []
        for sol in solutions:
            sc = sol[0]
            if abs(sc[0] - target_center[0]) < eps or abs(sc[1] - target_center[1]) < eps:
                sol[-1] += self.constraint_bouns
                valid.append(sol)
        return valid


class _TimeLimitReached(Exception):
    pass


# ===================================================================
# Floor placement orchestration
# ===================================================================

def place_floor_objects_dfs(
    objects_to_place: List[Object],
    room: Room,
    constraints: Dict[str, Any],
    max_duration: Optional[float] = None,
) -> List[Object]:
    """Place floor objects using the DFS grid solver.

    Args:
        objects_to_place: Objects that need floor positions.
        room: Target room.
        constraints: Either dict mapping id->constraint_list (string format)
                     or already-parsed structured constraints.
        max_duration: Max solver time in seconds (default: scales with object count).

    Returns:
        List of placed Object instances.
    """
    if not objects_to_place:
        return []

    # Separate existing (already have valid position) vs new objects
    existing_room_obj_ids = {o.id for o in room.objects}
    new_objects = [o for o in objects_to_place if o.id not in existing_room_obj_ids]
    existing_objects = [o for o in objects_to_place if o.id in existing_room_obj_ids]

    if not new_objects:
        return existing_objects

    # Parse constraints if in string format
    new_ids = [o.id for o in new_objects]
    existing_ids = list(existing_room_obj_ids)
    existing_id_mapping = {eid: f"existing-{eid}" for eid in existing_ids}

    if constraints and isinstance(constraints, dict):
        # Check if already structured (values are lists of dicts) or string format (values are lists of strings)
        sample_val = next(iter(constraints.values()), None)
        if sample_val and isinstance(sample_val, list) and sample_val and isinstance(sample_val[0], str):
            # String format — convert to structured
            structured = parse_constraints_from_json(constraints, new_ids, existing_ids, existing_id_mapping)
        elif sample_val and isinstance(sample_val, list) and sample_val and isinstance(sample_val[0], dict):
            structured = constraints
        else:
            structured = parse_constraints_from_json(constraints, new_ids, existing_ids, existing_id_mapping)
    elif isinstance(constraints, list):
        structured = parse_constraints_from_json(constraints, new_ids, existing_ids, existing_id_mapping)
    else:
        structured = {}

    # Compute max wall thickness and inner room polygon in cm
    max_wall_thickness = max((w.thickness for w in room.walls), default=0.1)
    padding_cm = max_wall_thickness * 100

    rw_cm = room.dimensions.width * 100 - 2 * padding_cm
    rl_cm = room.dimensions.length * 100 - 2 * padding_cm

    if rw_cm <= 0 or rl_cm <= 0:
        # Fallback: use full room dimensions
        rw_cm = room.dimensions.width * 100
        rl_cm = room.dimensions.length * 100
        padding_cm = 0

    room_poly = box(padding_cm, padding_cm, padding_cm + rw_cm, padding_cm + rl_cm)

    # Build objects list with dimensions in cm (+3.5cm padding)
    dim_padding = 3.5
    objects_list = []
    for obj in new_objects:
        w_cm = obj.dimensions.width * 100 + dim_padding
        l_cm = obj.dimensions.length * 100 + dim_padding
        objects_list.append((obj.id, (w_cm, l_cm)))

    # Build initial state from door/window placements + existing objects
    initial_state = get_door_window_placements(room)

    for obj in room.objects:
        obj_cx = (obj.position.x - room.position.x) * 100
        obj_cy = (obj.position.y - room.position.y) * 100
        rot = obj.rotation.z
        w_cm = obj.dimensions.width * 100
        l_cm = obj.dimensions.length * 100

        if rot in (0, 180):
            hw, hl = w_cm / 2, l_cm / 2
        else:
            hw, hl = l_cm / 2, w_cm / 2

        obj_box = box(obj_cx - hw, obj_cy - hl, obj_cx + hw, obj_cy + hl)
        key = f"existing-{obj.id}"
        initial_state[key] = (
            (obj_cx, obj_cy),
            rot,
            tuple(obj_box.exterior.coords[:]),
            1,
        )

    # Create solver and solve
    if max_duration is None:
        max_duration = max(300, len(objects_list) * 60)

    # Scale grid size with room area to keep search space manageable.
    # 20cm grid for a typical 4x5m room (~2000 grid points).
    # For a 36x44m warehouse the same grid would produce ~40k points,
    # causing combinatorial explosion in the DFS.  Scale up the grid
    # step so the total number of candidate points stays roughly constant.
    room_area_cm2 = room.dimensions.width * room.dimensions.length * 1e4  # m² → cm²
    reference_area_cm2 = 20 * 1e4  # 20 m² reference room
    grid_size = max(20, int(20 * (room_area_cm2 / reference_area_cm2) ** 0.5))

    solver = DFS_Solver_Floor(
        grid_size=grid_size,
        max_duration=max_duration,
        room_id=room.id,
    )

    solution, solution_constraints = solver.get_solution(
        room_poly, objects_list, structured, initial_state
    )

    if not solution:
        logger.warning("DFS solver found no solution for room %s", room.id)
        return existing_objects

    placed = solution_to_objects(solution, solution_constraints, new_objects, room)
    return existing_objects + placed


# ===================================================================
# Wall placement (SAGE lines 1988-2800)
# ===================================================================

def create_wall_coordinate_systems(room: Room) -> Dict[str, Dict]:
    """Create 2D coordinate systems for each wall."""
    wall_systems = {}
    for wall in room.walls:
        wall_vec = np.array([
            wall.end_point.x - wall.start_point.x,
            wall.end_point.y - wall.start_point.y,
            0,
        ])
        wall_length = np.linalg.norm(wall_vec[:2])
        wall_dir = wall_vec / max(np.linalg.norm(wall_vec), 1e-9)

        wall_normal_2d = np.array([-wall_dir[1], wall_dir[0]])
        wall_center = np.array([
            (wall.start_point.x + wall.end_point.x) / 2,
            (wall.start_point.y + wall.end_point.y) / 2,
        ])
        room_center = np.array([
            room.position.x + room.dimensions.width / 2,
            room.position.y + room.dimensions.length / 2,
        ])
        if np.dot(wall_normal_2d, room_center - wall_center) < 0:
            wall_normal_2d = -wall_normal_2d

        wall_normal = np.array([wall_normal_2d[0], wall_normal_2d[1], 0])
        rect_width = wall_length - wall.thickness
        rect_height = room.ceiling_height

        wall_systems[wall.id] = {
            "wall": wall,
            "wall_direction": wall_dir,
            "wall_normal": wall_normal,
            "wall_center_3d": np.array([wall_center[0], wall_center[1], room.position.z + rect_height / 2]),
            "rect_width": rect_width,
            "rect_height": rect_height,
            "wall_length": wall_length,
            "thickness": wall.thickness,
        }
    return wall_systems


def _wall_2d_to_3d(pos_2d, wall_info):
    x_2d, y_2d = pos_2d
    return (
        wall_info["wall_center_3d"]
        + x_2d * wall_info["wall_direction"]
        + y_2d * np.array([0, 0, 1])
        + (wall_info["thickness"] * 0.4) * wall_info["wall_normal"]
    )


def _world_3d_to_wall_2d(pos_3d, wall_info):
    rel = pos_3d - wall_info["wall_center_3d"]
    return np.array([np.dot(rel, wall_info["wall_direction"]), rel[2]])


def _get_object_3d_bbox(obj: Object):
    rot_z = obj.rotation.z
    if rot_z in (0, 180):
        wx, ly = obj.dimensions.width, obj.dimensions.length
    elif rot_z in (90, 270):
        wx, ly = obj.dimensions.length, obj.dimensions.width
    else:
        wx = ly = max(obj.dimensions.width, obj.dimensions.length)
    return {
        "center": np.array([obj.position.x, obj.position.y, obj.position.z + obj.dimensions.height / 2]),
        "half_extents": np.array([wx / 2, ly / 2, obj.dimensions.height / 2]),
        "min": np.array([obj.position.x - wx / 2, obj.position.y - ly / 2, obj.position.z]),
        "max": np.array([obj.position.x + wx / 2, obj.position.y + ly / 2, obj.position.z + obj.dimensions.height]),
    }


def _calc_distance_to_wall(obj_bbox, wall_info):
    wall = wall_info["wall"]
    wall_normal = wall_info["wall_normal"]
    wall_center_pt = np.array([
        (wall.start_point.x + wall.end_point.x) / 2,
        (wall.start_point.y + wall.end_point.y) / 2,
        0,
    ])
    wall_interior_pt = wall_center_pt + (wall_info["thickness"] / 2) * wall_normal
    center_to_wall = obj_bbox["center"] - wall_interior_pt
    dist = abs(np.dot(center_to_wall, wall_normal))
    extent = abs(np.dot(obj_bbox["half_extents"], np.abs(wall_normal)))
    return max(0, dist - extent)


def _rectangles_overlap(r1, r2):
    return not (r1["x_max"] <= r2["x_min"] or r2["x_max"] <= r1["x_min"]
                or r1["y_max"] <= r2["y_min"] or r2["y_max"] <= r1["y_min"])


def place_wall_objects(
    objects_to_place: List[Object],
    room: Room,
    existing_placed: List[Object],
) -> List[Object]:
    """Place wall-mounted objects using geometric scoring."""
    if not objects_to_place:
        return []

    placed_wall = []
    wall_systems = create_wall_coordinate_systems(room)

    # Create grid points on each wall
    wall_grids: Dict[str, List] = {}
    grid_density = 20
    for wall_id, info in wall_systems.items():
        rw, rh = info["rect_width"], info["rect_height"]
        margin = 0.1
        ew, eh = rw - 2 * margin, rh - 2 * margin
        if ew <= 0 or eh <= 0:
            continue
        x_step, y_step = ew / grid_density, eh / grid_density
        pts = []
        for i in range(grid_density):
            for j in range(grid_density):
                x2 = -ew / 2 + i * x_step + x_step / 2
                y2 = -eh / 2 + j * y_step + y_step / 2
                pts.append({"pos_2d": np.array([x2, y2]), "pos_3d": _wall_2d_to_3d(np.array([x2, y2]), info), "wall_id": wall_id})
        wall_grids[wall_id] = pts

    all_placed = list(existing_placed) + list(placed_wall)

    for wall_obj in objects_to_place:
        # Compute impossible regions (doors, windows, existing objects)
        impossible: Dict[str, List] = {wid: [] for wid in wall_systems}
        wall_obj_depth = wall_obj.dimensions.length + 0.20

        for wall_id, info in wall_systems.items():
            wall = info["wall"]
            for door in room.doors:
                if door.wall_id == wall.id:
                    dc = door.position_on_wall * info["wall_length"]
                    dx = dc - info["wall_length"] / 2
                    impossible[wall_id].append({
                        "type": "door",
                        "x_min": dx - door.width / 2, "x_max": dx + door.width / 2,
                        "y_min": -info["rect_height"] / 2, "y_max": -info["rect_height"] / 2 + door.height,
                    })
            for win in room.windows:
                if win.wall_id == wall.id:
                    wc = win.position_on_wall * info["wall_length"]
                    wx = wc - info["wall_length"] / 2
                    impossible[wall_id].append({
                        "type": "window",
                        "x_min": wx - win.width / 2, "x_max": wx + win.width / 2,
                        "y_min": -info["rect_height"] / 2 + win.sill_height,
                        "y_max": -info["rect_height"] / 2 + win.sill_height + win.height,
                    })

        for obj in all_placed:
            obj_bbox = _get_object_3d_bbox(obj)
            for wall_id, info in wall_systems.items():
                if _calc_distance_to_wall(obj_bbox, info) < wall_obj_depth:
                    corners = [
                        np.array([obj_bbox["min"][0], obj_bbox["min"][1], obj_bbox["min"][2]]),
                        np.array([obj_bbox["max"][0], obj_bbox["min"][1], obj_bbox["min"][2]]),
                        np.array([obj_bbox["min"][0], obj_bbox["max"][1], obj_bbox["min"][2]]),
                        np.array([obj_bbox["max"][0], obj_bbox["max"][1], obj_bbox["min"][2]]),
                        np.array([obj_bbox["min"][0], obj_bbox["min"][1], obj_bbox["max"][2]]),
                        np.array([obj_bbox["max"][0], obj_bbox["min"][1], obj_bbox["max"][2]]),
                        np.array([obj_bbox["min"][0], obj_bbox["max"][1], obj_bbox["max"][2]]),
                        np.array([obj_bbox["max"][0], obj_bbox["max"][1], obj_bbox["max"][2]]),
                    ]
                    proj = [_world_3d_to_wall_2d(c, info) for c in corners]
                    xs = [p[0] for p in proj]
                    ys = [p[1] for p in proj]
                    impossible[wall_id].append({
                        "type": "object",
                        "x_min": min(xs), "x_max": max(xs),
                        "y_min": min(ys), "y_max": max(ys),
                    })

        # Filter valid points
        ow2d = wall_obj.dimensions.width
        oh2d = wall_obj.dimensions.height
        min_gap = 0.20
        valid_points = []
        for wall_id, pts in wall_grids.items():
            for pt in pts:
                p2 = pt["pos_2d"]
                bbox = {
                    "x_min": p2[0] - ow2d / 2 - min_gap, "x_max": p2[0] + ow2d / 2 + min_gap,
                    "y_min": p2[1] - oh2d / 2 - min_gap, "y_max": p2[1] + oh2d / 2 + min_gap,
                }
                info = wall_systems[wall_id]
                if (bbox["x_min"] < -info["rect_width"] / 2 or bbox["x_max"] > info["rect_width"] / 2
                        or bbox["y_min"] < -info["rect_height"] / 2 or bbox["y_max"] > info["rect_height"] / 2):
                    continue
                if any(_rectangles_overlap(bbox, r) for r in impossible[wall_id]):
                    continue
                pt_copy = dict(pt)
                pt_copy["bbox_2d"] = bbox
                valid_points.append(pt_copy)

        if not valid_points:
            logger.warning("No valid wall placement for %s", wall_obj.id)
            continue

        # Score: prefer 55-65% wall height, horizontal alignment with floor objects
        scored = []
        for pt in valid_points:
            info = wall_systems[pt["wall_id"]]
            p2 = pt["pos_2d"]
            height_from_floor = p2[1] + info["rect_height"] / 2
            ideal_height = info["rect_height"] * 0.60
            height_score = 1.0 - abs(height_from_floor - ideal_height) / info["rect_height"]

            align_score = 0
            for obj in all_placed:
                if obj.place_id != "floor":
                    continue
                obj_bbox = _get_object_3d_bbox(obj)
                if _calc_distance_to_wall(obj_bbox, info) < 1.0:
                    obj_2d = _world_3d_to_wall_2d(
                        np.array([obj.position.x, obj.position.y, obj.position.z + obj.dimensions.height]),
                        info,
                    )
                    horiz_dist = abs(p2[0] - obj_2d[0])
                    if horiz_dist < 1.5:
                        align_score = max(align_score, 1.5 - horiz_dist)

            scored.append((height_score * 0.5 + align_score, pt))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_pt = scored[0][1]

        # Compute rotation from wall normal
        info = wall_systems[best_pt["wall_id"]]
        wn = info["wall_normal"][:2]
        angle = math.atan2(wn[1], wn[0]) - math.atan2(1, 0)
        rot_z = round(math.degrees(angle) / 90) * 90 % 360

        # Adjust position for object depth
        pos_3d = best_pt["pos_3d"]
        adjusted = pos_3d + info["wall_normal"] * (info["thickness"] * 0.4 + wall_obj.dimensions.length / 2)
        z_bottom = adjusted[2] - wall_obj.dimensions.height / 2

        placed_obj = Object(
            id=wall_obj.id,
            room_id=room.id,
            type=wall_obj.type,
            description=getattr(wall_obj, "description", f"Wall-mounted {wall_obj.type}"),
            position=Point3D(x=float(adjusted[0]), y=float(adjusted[1]), z=float(z_bottom)),
            rotation=Euler(x=0, y=0, z=rot_z),
            dimensions=wall_obj.dimensions,
            source=getattr(wall_obj, "source", "placement"),
            source_id=getattr(wall_obj, "source_id", wall_obj.id),
            place_id="wall",
            mass=getattr(wall_obj, "mass", 1.0),
        )
        placed_wall.append(placed_obj)
        all_placed.append(placed_obj)

    return placed_wall


# ===================================================================
# On-object placement with surface scatter
# ===================================================================

def place_on_object_objects(
    objects_to_place: List[Object],
    support_objects: List[Object],
    room: Room,
) -> List[Object]:
    """Place objects on top of supporting surfaces with collision avoidance.

    Uses 50 random position attempts on the support's top surface,
    rotated by support's rotation, with 2cm margin from edges.
    """
    placed = []
    supports_by_id = {o.id: o for o in support_objects}
    already_on_surface: Dict[str, List[Object]] = {}  # support_id -> placed objects

    for obj in objects_to_place:
        support = supports_by_id.get(obj.place_id)
        if support is None:
            logger.warning("Support %s not found for %s", obj.place_id, obj.id)
            continue

        margin = 0.02  # 2cm from edges
        sw, sl = support.dimensions.width, support.dimensions.length
        ow, ol = obj.dimensions.width, obj.dimensions.length

        # Determine Z based on place_location
        loc = getattr(obj, "place_location", "top")
        if loc == "inside":
            # Place inside the support volume (random height within)
            max_z = support.position.z + max(0, support.dimensions.height - obj.dimensions.height)
            surface_z = support.position.z + random.uniform(0, max(0, max_z - support.position.z))
        else:
            # "top" or "both" — place on top surface
            surface_z = support.position.z + support.dimensions.height

        # Available space on surface
        half_avail_w = max(0, (sw - ow) / 2 - margin)
        half_avail_l = max(0, (sl - ol) / 2 - margin)

        siblings = already_on_surface.get(obj.place_id, [])
        best_pos = None

        for _ in range(50):
            dx = random.uniform(-half_avail_w, half_avail_w) if half_avail_w > 0 else 0
            dy = random.uniform(-half_avail_l, half_avail_l) if half_avail_l > 0 else 0

            # Rotate offset by support's rotation
            rot_rad = math.radians(support.rotation.z)
            rx = dx * math.cos(rot_rad) - dy * math.sin(rot_rad)
            ry = dx * math.sin(rot_rad) + dy * math.cos(rot_rad)

            candidate_x = support.position.x + rx
            candidate_y = support.position.y + ry

            # Check collision with other objects on same surface
            collision = False
            for sib in siblings:
                dist = math.sqrt((candidate_x - sib.position.x) ** 2 + (candidate_y - sib.position.y) ** 2)
                min_dist = (max(ow, ol) + max(sib.dimensions.width, sib.dimensions.length)) / 2 + margin
                if dist < min_dist:
                    collision = True
                    break

            if not collision:
                best_pos = (candidate_x, candidate_y)
                break

        if best_pos is None:
            # Fallback: center of support
            best_pos = (support.position.x, support.position.y)

        obj.position = Point3D(x=best_pos[0], y=best_pos[1], z=surface_z)
        obj.rotation = Euler(x=0, y=0, z=support.rotation.z)
        placed.append(obj)

        if obj.place_id not in already_on_surface:
            already_on_surface[obj.place_id] = []
        already_on_surface[obj.place_id].append(obj)

    return placed


# ===================================================================
# Helpers
# ===================================================================

def _extract_points(geom) -> List[Tuple[float, float]]:
    """Extract point coordinates from a Shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    pts = []
    if isinstance(geom, Point):
        pts.append((geom.x, geom.y))
    elif isinstance(geom, MultiPoint):
        pts.extend([(p.x, p.y) for p in geom.geoms])
    elif isinstance(geom, LineString):
        pts.extend(list(geom.coords))
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            pts.extend(_extract_points(g))
    return pts


def _relative_comparisons(place_type, target_rotation, min_x, min_y, max_x, max_y, level):
    """Build a comparison function for relative placement constraints."""
    mean_x = (min_x + max_x) / 2
    mean_y = (min_y + max_y) / 2

    if level == "strict":
        defs = {
            "left of": {0: lambda s: s[0] < min_x and min_y <= s[1] <= max_y,
                        90: lambda s: s[1] < min_y and min_x <= s[0] <= max_x,
                        180: lambda s: s[0] > max_x and min_y <= s[1] <= max_y,
                        270: lambda s: s[1] > max_y and min_x <= s[0] <= max_x},
            "right of": {0: lambda s: s[0] > max_x and min_y <= s[1] <= max_y,
                         90: lambda s: s[1] > max_y and min_x <= s[0] <= max_x,
                         180: lambda s: s[0] < min_x and min_y <= s[1] <= max_y,
                         270: lambda s: s[1] < min_y and min_x <= s[0] <= max_x},
            "in front of": {0: lambda s: s[1] > max_y and min_x <= s[0] <= max_x,
                            90: lambda s: s[0] < min_x and min_y <= s[1] <= max_y,
                            180: lambda s: s[1] < min_y and min_x <= s[0] <= max_x,
                            270: lambda s: s[0] > max_x and min_y <= s[1] <= max_y},
            "behind": {0: lambda s: s[1] < min_y and min_x <= s[0] <= max_x,
                       90: lambda s: s[0] > max_x and min_y <= s[1] <= max_y,
                       180: lambda s: s[1] > max_y and min_x <= s[0] <= max_x,
                       270: lambda s: s[0] < min_x and min_y <= s[1] <= max_y},
            "side of": {0: lambda s: min_y <= s[1] <= max_y, 90: lambda s: min_x <= s[0] <= max_x,
                        180: lambda s: min_y <= s[1] <= max_y, 270: lambda s: min_x <= s[0] <= max_x},
        }
    elif level == "loose":
        defs = {
            "left of": {0: lambda s: s[0] < min_x, 90: lambda s: s[1] < min_y,
                        180: lambda s: s[0] > max_x, 270: lambda s: s[1] > max_y},
            "right of": {0: lambda s: s[0] > max_x, 90: lambda s: s[1] > max_y,
                         180: lambda s: s[0] < min_x, 270: lambda s: s[1] < min_y},
            "in front of": {0: lambda s: s[1] > max_y, 90: lambda s: s[0] < min_x,
                            180: lambda s: s[1] < min_y, 270: lambda s: s[0] > max_x},
            "behind": {0: lambda s: s[1] < min_y, 90: lambda s: s[0] > max_x,
                       180: lambda s: s[1] > max_y, 270: lambda s: s[0] < min_x},
            "side of": {0: lambda s: min_y <= s[1] <= max_y, 90: lambda s: min_x <= s[0] <= max_x,
                        180: lambda s: min_y <= s[1] <= max_y, 270: lambda s: min_x <= s[0] <= max_x},
        }
    else:  # looser
        defs = {
            "left of": {0: lambda s: s[0] < mean_x, 90: lambda s: s[1] < mean_y,
                        180: lambda s: s[0] > mean_x, 270: lambda s: s[1] > mean_y},
            "right of": {0: lambda s: s[0] > mean_x, 90: lambda s: s[1] > mean_y,
                         180: lambda s: s[0] < mean_x, 270: lambda s: s[1] < mean_y},
            "in front of": {0: lambda s: s[1] > mean_y, 90: lambda s: s[0] < mean_x,
                            180: lambda s: s[1] < mean_y, 270: lambda s: s[0] > mean_x},
            "behind": {0: lambda s: s[1] < mean_y, 90: lambda s: s[0] > mean_x,
                       180: lambda s: s[1] > mean_y, 270: lambda s: s[0] < mean_x},
            "side of": {0: lambda s: min_y <= s[1] <= max_y, 90: lambda s: min_x <= s[0] <= max_x,
                        180: lambda s: min_y <= s[1] <= max_y, 270: lambda s: min_x <= s[0] <= max_x},
        }

    return defs.get(place_type, {}).get(target_rotation, lambda s: False)
