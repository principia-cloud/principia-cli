"""2D floor plan visualization using matplotlib.

Generates annotated top-down PNG renderings of room layouts with:
- Color-coded bounding boxes per object
- Object ID + type labels with background
- Facing direction arrows (yellow)
- Door swing clearance zones
- Coordinate axes with meter scale
"""

import math
import os
import logging
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from models import FloorPlan, Room

logger = logging.getLogger("scene-gen.room_render")

# Color palette for objects (cycles)
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9",
]


def render_floor_plan(layout: FloorPlan, output_path: str, dpi: int = 150) -> str:
    """Render the full floor plan as a top-down 2D PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for room in layout.rooms:
        _draw_room(ax, room)

    ax.set_aspect("equal")
    ax.set_title(f"Floor Plan: {layout.description or layout.id}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.autoscale()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Floor plan rendered to %s", output_path)
    return output_path


def render_room(layout: FloorPlan, room_id: str, output_path: str, dpi: int = 150) -> str:
    """Render a single room with annotated objects as a top-down 2D PNG."""
    room = next((r for r in layout.rooms if r.id == room_id), None)
    if room is None:
        raise ValueError(f"Room {room_id} not found")

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    _draw_room(ax, room, draw_objects=True)

    # Add coordinate axes annotation
    x0, y0 = room.position.x, room.position.y
    w, l = room.dimensions.width, room.dimensions.length

    # Draw axis arrows
    arrow_len = min(w, l) * 0.12
    ax.annotate("", xy=(x0 + arrow_len, y0 - 0.15), xytext=(x0, y0 - 0.15),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(x0 + arrow_len + 0.05, y0 - 0.15, "X", color="red", fontsize=10, fontweight="bold", va="center")
    ax.annotate("", xy=(x0 - 0.15, y0 + arrow_len), xytext=(x0 - 0.15, y0),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(x0 - 0.15, y0 + arrow_len + 0.05, "Y", color="green", fontsize=10, fontweight="bold", ha="center")

    # Add meter scale ticks
    for m in range(int(w) + 1):
        ax.plot([x0 + m, x0 + m], [y0 - 0.05, y0 + 0.05], color="gray", linewidth=0.5)
        ax.text(x0 + m, y0 - 0.1, f"{m}m", ha="center", va="top", fontsize=6, color="gray")
    for m in range(int(l) + 1):
        ax.plot([x0 - 0.05, x0 + 0.05], [y0 + m, y0 + m], color="gray", linewidth=0.5)
        ax.text(x0 - 0.1, y0 + m, f"{m}m", ha="right", va="center", fontsize=6, color="gray")

    ax.set_aspect("equal")
    ax.set_title(f"{room.room_type} ({room.id}) — {w:.1f}m × {l:.1f}m — {len(room.objects)} objects")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.2)

    # Set axis limits with padding
    padding = max(0.3, min(w, l) * 0.05)
    ax.set_xlim(x0 - padding, x0 + w + padding)
    ax.set_ylim(y0 - padding, y0 + l + padding)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Room %s rendered to %s", room_id, output_path)
    return output_path


def _draw_room(ax, room: Room, draw_objects: bool = True):
    """Draw a room rectangle with doors, windows, and annotated objects."""
    x0, y0 = room.position.x, room.position.y
    w, l = room.dimensions.width, room.dimensions.length

    # Room outline
    rect = patches.Rectangle(
        (x0, y0), w, l, linewidth=2.5, edgecolor="black", facecolor="lightyellow", alpha=0.4
    )
    ax.add_patch(rect)
    ax.text(
        x0 + w / 2, y0 + l / 2, room.room_type,
        ha="center", va="center", fontsize=10, fontweight="bold", alpha=0.3,
    )

    # Doors with clearance zones
    for door in room.doors:
        _draw_door_on_wall(ax, room, door)

    # Windows
    for window in room.windows:
        _draw_window_on_wall(ax, room, window)

    # Objects
    if draw_objects:
        for idx, obj in enumerate(room.objects):
            color = _PALETTE[idx % len(_PALETTE)]
            _draw_object(ax, obj, room, color, idx)


def _draw_door_on_wall(ax, room: Room, door):
    """Draw a door indicator and swing clearance zone."""
    x0, y0 = room.position.x, room.position.y
    w, l = room.dimensions.width, room.dimensions.length

    wall_id = door.wall_id.lower() if door.wall_id else ""
    pos = door.position_on_wall
    dw = door.width

    # Draw door line
    if "south" in wall_id or "s_wall" in wall_id:
        dx = x0 + pos * w - dw / 2
        ax.plot([dx, dx + dw], [y0, y0], color="brown", linewidth=4, solid_capstyle="butt")
        # Clearance zone (semi-transparent)
        clearance = patches.Rectangle(
            (dx, y0), dw, dw, linewidth=0.5, edgecolor="brown",
            facecolor="brown", alpha=0.08, linestyle="--",
        )
        ax.add_patch(clearance)
    elif "north" in wall_id or "n_wall" in wall_id:
        dx = x0 + pos * w - dw / 2
        ax.plot([dx, dx + dw], [y0 + l, y0 + l], color="brown", linewidth=4, solid_capstyle="butt")
        clearance = patches.Rectangle(
            (dx, y0 + l - dw), dw, dw, linewidth=0.5, edgecolor="brown",
            facecolor="brown", alpha=0.08, linestyle="--",
        )
        ax.add_patch(clearance)
    elif "west" in wall_id or "w_wall" in wall_id:
        dy = y0 + pos * l - dw / 2
        ax.plot([x0, x0], [dy, dy + dw], color="brown", linewidth=4, solid_capstyle="butt")
        clearance = patches.Rectangle(
            (x0, dy), dw, dw, linewidth=0.5, edgecolor="brown",
            facecolor="brown", alpha=0.08, linestyle="--",
        )
        ax.add_patch(clearance)
    elif "east" in wall_id or "e_wall" in wall_id:
        dy = y0 + pos * l - dw / 2
        ax.plot([x0 + w, x0 + w], [dy, dy + dw], color="brown", linewidth=4, solid_capstyle="butt")
        clearance = patches.Rectangle(
            (x0 + w - dw, dy), dw, dw, linewidth=0.5, edgecolor="brown",
            facecolor="brown", alpha=0.08, linestyle="--",
        )
        ax.add_patch(clearance)


def _draw_window_on_wall(ax, room: Room, window):
    """Draw a window indicator on the appropriate wall."""
    x0, y0 = room.position.x, room.position.y
    w, l = room.dimensions.width, room.dimensions.length

    wall_id = window.wall_id.lower() if window.wall_id else ""
    pos = window.position_on_wall
    ww = window.width

    if "south" in wall_id or "s_wall" in wall_id:
        dx = x0 + pos * w - ww / 2
        ax.plot([dx, dx + ww], [y0, y0], color="deepskyblue", linewidth=3, solid_capstyle="butt")
    elif "north" in wall_id or "n_wall" in wall_id:
        dx = x0 + pos * w - ww / 2
        ax.plot([dx, dx + ww], [y0 + l, y0 + l], color="deepskyblue", linewidth=3, solid_capstyle="butt")
    elif "west" in wall_id or "w_wall" in wall_id:
        dy = y0 + pos * l - ww / 2
        ax.plot([x0, x0], [dy, dy + ww], color="deepskyblue", linewidth=3, solid_capstyle="butt")
    elif "east" in wall_id or "e_wall" in wall_id:
        dy = y0 + pos * l - ww / 2
        ax.plot([x0 + w, x0 + w], [dy, dy + ww], color="deepskyblue", linewidth=3, solid_capstyle="butt")


def _draw_object(ax, obj, room: Room, color: str, idx: int):
    """Draw an annotated object with bounding box, label, and facing arrow."""
    cx = obj.position.x
    cy = obj.position.y
    ow = obj.dimensions.width
    ol = obj.dimensions.length
    rot = obj.rotation.z

    # Compute rotated corners
    corners = [(-ow / 2, -ol / 2), (ow / 2, -ol / 2), (ow / 2, ol / 2), (-ow / 2, ol / 2)]
    rad = math.radians(rot)
    xs, ys = [], []
    for dx, dy in corners:
        rx = cx + dx * math.cos(rad) - dy * math.sin(rad)
        ry = cy + dx * math.sin(rad) + dy * math.cos(rad)
        xs.append(rx)
        ys.append(ry)
    xs.append(xs[0])
    ys.append(ys[0])

    # Filled bounding box
    ax.fill(xs, ys, alpha=0.25, color=color)
    ax.plot(xs, ys, color=color, linewidth=1.2)

    # Object ID + type label with white background
    label = f"{obj.id}\n{obj.type}"
    ax.text(
        cx, cy, label, ha="center", va="center", fontsize=5,
        fontweight="bold", color="black",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.85, linewidth=0.5),
    )

    # Facing direction arrow (yellow)
    # Convention: 0°=+Y, 90°=-X, 180°=-Y, 270°=+X
    arrow_len = min(ow, ol) * 0.4
    facing_vectors = {
        0: (0, 1), 90: (-1, 0), 180: (0, -1), 270: (1, 0),
    }
    # Get nearest standard rotation
    nearest_rot = round(rot / 90) * 90 % 360
    fv = facing_vectors.get(nearest_rot, (0, 1))

    ax.annotate(
        "",
        xy=(cx + fv[0] * arrow_len, cy + fv[1] * arrow_len),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color="#FFD700", lw=1.5, mutation_scale=8),
    )
