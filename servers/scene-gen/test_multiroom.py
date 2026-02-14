"""Test multi-room support: shared walls, connecting doors, layout export."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    FloorPlan, Room, Wall, Door, Window, Object,
    Point3D, Euler, Dimensions,
    dict_to_room, floor_plan_to_dict,
)
from services.shared_walls import (
    rooms_to_dicts, find_shared_walls_between_rooms,
    find_all_shared_walls, calculate_room_wall_position_from_shared_wall,
    compute_connecting_doors, _wall_side_to_wall_id,
)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# Build a 2-room layout: living room (5x4.5) east-of bedroom (4x4.5)
# ---------------------------------------------------------------------------
print("\n=== Building 2-room layout ===")

def make_walls(rid, x, y, w, l, h=2.7):
    return [
        Wall(id=f"{rid}_s_wall", start_point=Point3D(x, y, 0), end_point=Point3D(x+w, y, 0), height=h),
        Wall(id=f"{rid}_e_wall", start_point=Point3D(x+w, y, 0), end_point=Point3D(x+w, y+l, 0), height=h),
        Wall(id=f"{rid}_n_wall", start_point=Point3D(x+w, y+l, 0), end_point=Point3D(x, y+l, 0), height=h),
        Wall(id=f"{rid}_w_wall", start_point=Point3D(x, y+l, 0), end_point=Point3D(x, y, 0), height=h),
    ]

room1 = Room(
    id="room_01", room_type="living_room",
    position=Point3D(0, 0, 0), dimensions=Dimensions(5.0, 4.5, 2.7),
    walls=make_walls("room_01", 0, 0, 5.0, 4.5),
    doors=[], objects=[], windows=[],
)
room2 = Room(
    id="room_02", room_type="bedroom",
    position=Point3D(5.0, 0, 0), dimensions=Dimensions(4.0, 4.5, 2.7),
    walls=make_walls("room_02", 5.0, 0, 4.0, 4.5),
    doors=[], objects=[], windows=[],
)
fp = FloorPlan(
    id="test_multi", rooms=[room1, room2],
    total_area=5.0*4.5 + 4.0*4.5,
    building_style="modern", description="Test multi-room",
    created_from_text="test",
)

# ---------------------------------------------------------------------------
# Test 1: rooms_to_dicts
# ---------------------------------------------------------------------------
print("\n--- Test 1: rooms_to_dicts ---")
rd = rooms_to_dicts(fp.rooms)
check("returns 2 rooms", len(rd) == 2)
check("room_01 id", rd[0]["id"] == "room_01")
check("room_02 position.x", rd[1]["position"]["x"] == 5.0)
check("room_01 has 4 walls", len(rd[0]["walls"]) == 4)

# ---------------------------------------------------------------------------
# Test 2: find_shared_walls_between_rooms
# ---------------------------------------------------------------------------
print("\n--- Test 2: find_shared_walls_between_rooms ---")
sw = find_shared_walls_between_rooms(rd[0], rd[1])
check("1 shared wall found", len(sw) == 1, f"got {len(sw)}")
if sw:
    check("room1 east wall", sw[0]["room1_wall"] == "east")
    check("room2 west wall", sw[0]["room2_wall"] == "west")
    check("overlap length ~4.5", abs(sw[0]["overlap_length"] - 4.5) < 0.01,
          f"got {sw[0]['overlap_length']}")

# ---------------------------------------------------------------------------
# Test 3: find_all_shared_walls
# ---------------------------------------------------------------------------
print("\n--- Test 3: find_all_shared_walls ---")
all_sw = find_all_shared_walls(rd)
check("1 room-room wall", all_sw["total_room_room_walls"] == 1)
# Each room has 4 walls, 1 is shared → 3 exterior each = 6 total
check("6 exterior walls", all_sw["total_room_exterior_walls"] == 6,
      f"got {all_sw['total_room_exterior_walls']}")

rr = all_sw["room_room_walls"][0]
check("room1 index 0", rr["room1"]["index"] == 0)
check("room2 index 1", rr["room2"]["index"] == 1)
check("direction is y", rr["direction"] == "y")

# Check exterior walls don't include the shared east/west walls
ext_sides = [(e["room"]["id"], e["wall_side"]) for e in all_sw["room_exterior_walls"]]
check("room_01 east not exterior", ("room_01", "east") not in ext_sides)
check("room_02 west not exterior", ("room_02", "west") not in ext_sides)
check("room_01 south is exterior", ("room_01", "south") in ext_sides)
check("room_02 north is exterior", ("room_02", "north") in ext_sides)

# ---------------------------------------------------------------------------
# Test 4: compute_connecting_doors (MST)
# ---------------------------------------------------------------------------
print("\n--- Test 4: compute_connecting_doors ---")
doors = compute_connecting_doors(rd, all_sw)
check("1 connecting door suggested", len(doors) == 1, f"got {len(doors)}")
if doors:
    d = doors[0]
    check("shared_wall_index 0", d["shared_wall_index"] == 0)
    check("position 0.5", d["center_position_on_shared_wall"] == 0.5)
    check("width <= overlap*0.8", d["width"] <= 4.5 * 0.8 + 0.01)
    check("door_type connecting", d["door_type"] == "connecting")

# ---------------------------------------------------------------------------
# Test 5: calculate_room_wall_position_from_shared_wall
# ---------------------------------------------------------------------------
print("\n--- Test 5: calculate_room_wall_position_from_shared_wall ---")
pos1 = calculate_room_wall_position_from_shared_wall(rr, 0.5, rd[0], True)
pos2 = calculate_room_wall_position_from_shared_wall(rr, 0.5, rd[1], False)
check("room1 wall position ~0.5", abs(pos1 - 0.5) < 0.01, f"got {pos1}")
check("room2 wall position ~0.5", abs(pos2 - 0.5) < 0.01, f"got {pos2}")

# Off-center position
pos1_off = calculate_room_wall_position_from_shared_wall(rr, 0.25, rd[0], True)
pos2_off = calculate_room_wall_position_from_shared_wall(rr, 0.25, rd[1], False)
check("room1 off-center ~0.25", abs(pos1_off - 0.25) < 0.01, f"got {pos1_off}")
check("room2 off-center ~0.25", abs(pos2_off - 0.25) < 0.01, f"got {pos2_off}")

# ---------------------------------------------------------------------------
# Test 6: _wall_side_to_wall_id
# ---------------------------------------------------------------------------
print("\n--- Test 6: _wall_side_to_wall_id ---")
check("north → n_wall", _wall_side_to_wall_id(room1, "north") == "room_01_n_wall")
check("south → s_wall", _wall_side_to_wall_id(room1, "south") == "room_01_s_wall")
check("east → e_wall", _wall_side_to_wall_id(room2, "east") == "room_02_e_wall")
check("west → w_wall", _wall_side_to_wall_id(room2, "west") == "room_02_w_wall")
check("invalid → None", _wall_side_to_wall_id(room1, "up") is None)

# ---------------------------------------------------------------------------
# Test 7: Add connecting doors to both rooms (simulating add_connecting_doors)
# ---------------------------------------------------------------------------
print("\n--- Test 7: Connecting door placement on both rooms ---")
sw_info = all_sw["room_room_walls"][0]
door_spec = doors[0]

final_w = min(door_spec["width"], sw_info["overlap_length"] * 0.8)
shared_pos = door_spec["center_position_on_shared_wall"]

p1 = calculate_room_wall_position_from_shared_wall(sw_info, shared_pos, rd[0], True)
p2 = calculate_room_wall_position_from_shared_wall(sw_info, shared_pos, rd[1], False)

wid1 = _wall_side_to_wall_id(room1, sw_info["room1_wall"])
wid2 = _wall_side_to_wall_id(room2, sw_info["room2_wall"])

door1 = Door(id="cdoor_test", wall_id=wid1, position_on_wall=p1,
             width=final_w, height=2.1, door_type="connecting")
door2 = Door(id="cdoor_test", wall_id=wid2, position_on_wall=p2,
             width=final_w, height=2.1, door_type="connecting")

room1.doors.append(door1)
room2.doors.append(door2)

check("room1 has 1 door", len(room1.doors) == 1)
check("room2 has 1 door", len(room2.doors) == 1)
check("same door id", room1.doors[0].id == room2.doors[0].id)
check("door on room1 east wall", room1.doors[0].wall_id == "room_01_e_wall")
check("door on room2 west wall", room2.doors[0].wall_id == "room_02_w_wall")

# ---------------------------------------------------------------------------
# Test 8: scene_export — create_room_meshes_with_openings with shared dedup
# ---------------------------------------------------------------------------
print("\n--- Test 8: create_room_meshes_with_openings with shared dedup ---")
from services.scene_export import create_room_meshes_with_openings, _get_door_unique_id

# Without shared sets (backward compat)
w1, d1, _, _, did1, _ = create_room_meshes_with_openings(room1)
check("room1: 4 walls", len(w1) == 4, f"got {len(w1)}")
check("room1: 1 door mesh", len(d1) == 1, f"got {len(d1)}")

# With shared sets (layout dedup)
shared_doors = set()
shared_windows = set()
w1b, d1b, _, _, did1b, _ = create_room_meshes_with_openings(room1, shared_doors, shared_windows)
w2b, d2b, _, _, did2b, _ = create_room_meshes_with_openings(room2, shared_doors, shared_windows)

check("shared dedup: room1 gets 1 door mesh", len(d1b) == 1, f"got {len(d1b)}")
check("shared dedup: room2 gets 0 door meshes (deduped)", len(d2b) == 0,
      f"got {len(d2b)}")
check("processed_doors has 1 entry", len(shared_doors) == 1, f"got {len(shared_doors)}")

# ---------------------------------------------------------------------------
# Test 9: build_layout_mesh_dict
# ---------------------------------------------------------------------------
print("\n--- Test 9: build_layout_mesh_dict ---")
from services.scene_export import build_layout_mesh_dict

with tempfile.TemporaryDirectory() as tmpdir:
    mesh_dict = build_layout_mesh_dict(fp, "test_multi", tmpdir)

    # Expected: 2 floors + 8 walls + 2 ceilings + 1 door = 13
    check("mesh_dict has 13 entries", len(mesh_dict) == 13,
          f"got {len(mesh_dict)}: {sorted(mesh_dict.keys())}")

    # Check key names
    check("floor_room_01 present", "floor_room_01" in mesh_dict)
    check("floor_room_02 present", "floor_room_02" in mesh_dict)
    check("ceiling_room_01 present", "ceiling_room_01" in mesh_dict)
    check("cdoor_test present (only once)", "cdoor_test" in mesh_dict)

    # Count door meshes — should be exactly 1
    door_keys = [k for k in mesh_dict if "door" in k.lower()]
    check("exactly 1 door in mesh_dict", len(door_keys) == 1,
          f"got {len(door_keys)}: {door_keys}")

# ---------------------------------------------------------------------------
# Test 10: Full layout export to USDZ
# ---------------------------------------------------------------------------
print("\n--- Test 10: export_layout_scene ---")
from services.scene_export import export_layout_scene

with tempfile.TemporaryDirectory() as tmpdir:
    results_dir = os.path.join(tmpdir, "results")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(results_dir, exist_ok=True)

    result = export_layout_scene(fp, "test_multi", results_dir, output_dir)
    check("result has usdz_path", "usdz_path" in result)
    check("result has room_count=2", result.get("room_count") == 2, f"got {result.get('room_count')}")
    check("usdz file exists", os.path.exists(result["usdz_path"]),
          f"path: {result.get('usdz_path')}")
    check("usd file exists", os.path.exists(result["usd_path"]))
    check("mesh_count == 13", result["mesh_count"] == 13,
          f"got {result['mesh_count']}")

    usdz_size = os.path.getsize(result["usdz_path"])
    check(f"usdz file not empty ({usdz_size} bytes)", usdz_size > 100)

# ---------------------------------------------------------------------------
# Test 11: 3-room L-shaped layout
# ---------------------------------------------------------------------------
print("\n--- Test 11: 3-room L-shaped layout ---")
#  room_01 (5x4) | room_02 (4x4)
#  room_03 (5x3) below room_01

room_a = Room(id="room_a", room_type="living_room",
    position=Point3D(0, 3, 0), dimensions=Dimensions(5, 4, 2.7),
    walls=make_walls("room_a", 0, 3, 5, 4), doors=[], objects=[], windows=[])
room_b = Room(id="room_b", room_type="bedroom",
    position=Point3D(5, 3, 0), dimensions=Dimensions(4, 4, 2.7),
    walls=make_walls("room_b", 5, 3, 4, 4), doors=[], objects=[], windows=[])
room_c = Room(id="room_c", room_type="kitchen",
    position=Point3D(0, 0, 0), dimensions=Dimensions(5, 3, 2.7),
    walls=make_walls("room_c", 0, 0, 5, 3), doors=[], objects=[], windows=[])

rd3 = rooms_to_dicts([room_a, room_b, room_c])
all3 = find_all_shared_walls(rd3)

check("2 room-room walls", all3["total_room_room_walls"] == 2,
      f"got {all3['total_room_room_walls']}")

# room_a↔room_b (east/west), room_a↔room_c (south/north)
rr_pairs = [(w["room1"]["id"], w["room2"]["id"]) for w in all3["room_room_walls"]]
check("room_a↔room_b shared", ("room_a", "room_b") in rr_pairs, f"pairs: {rr_pairs}")
check("room_a↔room_c shared", ("room_a", "room_c") in rr_pairs, f"pairs: {rr_pairs}")

doors3 = compute_connecting_doors(rd3, all3)
check("MST: 2 connecting doors for 3 rooms", len(doors3) == 2, f"got {len(doors3)}")

# ---------------------------------------------------------------------------
# Test 12: Single room backward compat (no shared walls)
# ---------------------------------------------------------------------------
print("\n--- Test 12: Single room backward compat ---")
rd_single = rooms_to_dicts([room1])
all_single = find_all_shared_walls(rd_single)
check("0 room-room walls", all_single["total_room_room_walls"] == 0)
check("4 exterior walls", all_single["total_room_exterior_walls"] == 4,
      f"got {all_single['total_room_exterior_walls']}")
doors_single = compute_connecting_doors(rd_single, all_single)
check("0 connecting doors", len(doors_single) == 0)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} checks")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
