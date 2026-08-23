#!/usr/bin/env python3
"""Contract tests for the standalone planner SVG renderer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


THIS_DIRECTORY = Path(__file__).resolve().parent
RENDERER = THIS_DIRECTORY / "render_probe_svg.py"


class RenderProbeSvgTest(unittest.TestCase):
    def test_renders_route_and_no_route_as_distinct_svg(self) -> None:
        fixture = {
            "map": {
                "local": {
                    "width": 6,
                    "height": 4,
                    "resolution_m": 1.0,
                    "origin_m": [0.0, 0.0, 0.0],
                },
                "forbidden_indices": [8, 9],
            },
            "start": {"position_m": [0.5, 0.5, 0.0]},
            "goal": {"position_m": [5.5, 3.5, 0.0]},
        }
        successful = {
            "outcome": 0,
            "reason_code": "WHEEL_PLAN_AVAILABLE",
            "local_trajectory_points": [[0.5, 0.5, 0.0], [2.5, 1.5, 0.0], [5.5, 3.5, 0.0]],
        }
        no_route = {
            "outcome": 3,
            "reason_code": "LOCAL_SEARCH_DOMAIN_EXHAUSTED",
            "local_trajectory_points": [],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            fixture_path = temporary / "fixture.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            for name, probe in (("route", successful), ("no-route", no_route)):
                probe_path = temporary / f"{name}.json"
                svg_path = temporary / f"{name}.svg"
                probe_path.write_text(json.dumps(probe), encoding="utf-8")
                subprocess.run(
                    [sys.executable, str(RENDERER), str(fixture_path), str(probe_path), str(svg_path)],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                svg = svg_path.read_text(encoding="utf-8")
                self.assertIn("Planner offline scenario", svg)
                self.assertIn("#1d4ed8", svg)
                self.assertIn("#dc2626", svg)
                self.assertIn("#16a34a", svg)
                self.assertIn(probe["reason_code"], svg)
                if name == "route":
                    self.assertIn("points=\"", svg)
                else:
                    self.assertIn("No trajectory returned", svg)


if __name__ == "__main__":
    unittest.main()
