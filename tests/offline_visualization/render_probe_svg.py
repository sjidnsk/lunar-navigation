#!/usr/bin/env python3
"""Render an algorithm_probe result as a self-contained SVG map.

The input is intentionally limited to the canonical offline planner fixture and
the JSON produced by tests/parity/algorithm_probe.  It neither imports ROS nor
knows about ROS-side adapters such as trusted bridging.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


MARGIN = 48
CELL_PIXELS = 52


def point_to_svg(
    point: list[float], origin: list[float], height: int, resolution: float
) -> tuple[float, float]:
    x = MARGIN + (point[0] - origin[0]) / resolution * CELL_PIXELS
    y = MARGIN + (height - (point[1] - origin[1]) / resolution) * CELL_PIXELS
    return x, y


def render(fixture: dict[str, Any], probe: dict[str, Any]) -> str:
    local_map = fixture["map"]["local"]
    width = int(local_map["width"])
    height = int(local_map["height"])
    resolution = float(local_map["resolution_m"])
    origin = local_map["origin_m"]
    canvas_width = MARGIN * 2 + width * CELL_PIXELS
    canvas_height = MARGIN * 2 + height * CELL_PIXELS + 48
    forbidden = set(fixture["map"].get("forbidden_indices", []))

    cells: list[str] = []
    for row in range(height):
        for column in range(width):
            index = row * width + column
            x = MARGIN + column * CELL_PIXELS
            y = MARGIN + (height - 1 - row) * CELL_PIXELS
            fill = "#111827" if index in forbidden else "#f8fafc"
            cells.append(
                f'<rect x="{x}" y="{y}" width="{CELL_PIXELS}" '
                f'height="{CELL_PIXELS}" fill="{fill}" stroke="#cbd5e1" />'
            )

    start = fixture["start"]["position_m"]
    goal = fixture["goal"]["position_m"]
    start_x, start_y = point_to_svg(start, origin, height, resolution)
    goal_x, goal_y = point_to_svg(goal, origin, height, resolution)
    trajectory = probe.get("local_trajectory_points", [])
    route = ""
    message = ""
    if trajectory:
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in (point_to_svg(point, origin, height, resolution) for point in trajectory)
        )
        route = (
            f'<polyline points="{points}" fill="none" stroke="#1d4ed8" '
            'stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />'
        )
    else:
        message = (
            f'<text x="{MARGIN}" y="{canvas_height - 12}" fill="#b91c1c" '
            'font-family="sans-serif" font-size="16">No trajectory returned</text>'
        )

    reason = html.escape(str(probe.get("reason_code", "UNKNOWN")))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}" role="img" aria-label="Planner offline scenario">
  <title>Planner offline scenario</title>
  <rect width="100%" height="100%" fill="#ffffff" />
  {"".join(cells)}
  {route}
  <circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="10" fill="#16a34a" />
  <circle cx="{goal_x:.2f}" cy="{goal_y:.2f}" r="10" fill="#dc2626" />
  <text x="{MARGIN}" y="24" fill="#0f172a" font-family="sans-serif" font-size="18">Planner offline scenario — {reason}</text>
  <text x="{canvas_width - 210}" y="24" fill="#1d4ed8" font-family="sans-serif" font-size="14">blue: planned trajectory</text>
  {message}
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    fixture = json.loads(arguments.fixture.read_text(encoding="utf-8"))
    probe = json.loads(arguments.probe.read_text(encoding="utf-8"))
    arguments.output.write_text(render(fixture, probe), encoding="utf-8")


if __name__ == "__main__":
    main()
