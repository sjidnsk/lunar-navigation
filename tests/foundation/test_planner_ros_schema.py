from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    ROOT / "ros2_ws/src/lunar_planner_ros/config/planner_ros.schema.json"
)
RETIRED_PARAMETERS = {
    "global_search.maximum_expanded_states",
    "global_search.maximum_reopened_states",
    "global_search.maximum_generated_candidates",
    "global_search.maximum_open_states",
    "global_search.maximum_memory_bytes",
    "global_search.resources.maximum_expanded_states",
    "global_search.resources.maximum_reopened_states",
    "global_search.resources.maximum_generated_candidates",
    "global_search.resources.maximum_open_states",
    "global_search.resources.maximum_memory_bytes",
    "hopper.maximum_landing_regions",
    "hopper.maximum_graph_nodes",
    "hopper.maximum_graph_out_degree",
    "hopper.maximum_nominal_aim_points_per_region",
    "hopper.maximum_certification_attempts",
}


def test_schema_explicitly_retires_global_search_truncation_parameters() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert set(schema["x-retired-parameters"]) == RETIRED_PARAMETERS
    assert RETIRED_PARAMETERS.isdisjoint(schema["properties"])
