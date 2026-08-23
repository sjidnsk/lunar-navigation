"""Strict loader for canonical Task 15 inputs consumed by the C++ ROS runner.

This module validates fixtures only. Exploration selection, ranking, retry,
and completion behavior remain production C++ responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, get_type_hints

import pytest
import yaml


FIXTURE_DIRECTORY = Path(__file__).with_name("fixtures")
FIXTURE_NAMES = (
    "concave_region", "unreachable_frontier", "map_growth",
    "fully_known_completion",
)
AUTHORITY = "test-only/non-authoritative"
RESPONSE_KINDS = frozenset({"SUCCESS", "NO_PATH", "DELAYED"})
STATE_NAMES = frozenset({
    "IDLE", "WAITING_FOR_INPUT", "SELECTING_FRONTIER", "PLANNING",
    "EXECUTING", "REPLANNING", "PAUSED", "COMPLETED", "ERROR",
})


class FixtureNotFound(FileNotFoundError):
    """Raised when a committed Task 15 fixture is absent."""


class FixtureSchemaError(ValueError):
    """Raised when a fixture diverges from the canonical scenario contract."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise FixtureSchemaError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class ScenarioTrace:
    request_goal_cells: tuple[str, ...]
    request_ids: tuple[str, ...]
    status_sequence: tuple[str, ...]
    reference_plan_ids: tuple[str, ...]
    cancellation_plan_ids: tuple[str, ...]


class RosScenarioRunner(Protocol):
    def run(
        self, fixture: Mapping[str, Any], *, deadline_steady_s: float = 20.0
    ) -> ScenarioTrace: ...


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise FixtureSchemaError(f"{path} must be a string-keyed mapping")
    return value  # type: ignore[return-value]


def _exact(value: object, keys: set[str], path: str) -> Mapping[str, Any]:
    result = _mapping(value, path)
    if set(result) != keys:
        raise FixtureSchemaError(
            f"{path} keys must be {sorted(keys)}, got {sorted(result)}"
        )
    return result


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise FixtureSchemaError(f"{path} must be a non-empty string")
    return value


def _cell(value: object, path: str) -> str:
    result = _nonempty(value, path)
    try:
        x, y = result.split(",")
        int(x); int(y)
    except ValueError as error:
        raise FixtureSchemaError(f"{path} must be column,row") from error
    return result


def _candidate_key(value: object, path: str) -> str:
    result = _nonempty(value, path)
    try:
        x, y, yaw = result.split(",")
        int(x); int(y); int(yaw)
    except ValueError as error:
        raise FixtureSchemaError(
            f"{path} must be x_mm,y_mm,yaw_tenth_deg"
        ) from error
    return result


def _grid_data(grid: Mapping[str, Any]) -> list[int]:
    geometry = _mapping(grid["geometry"], "grid.geometry")
    width, height = geometry["width"], geometry["height"]
    if not isinstance(width, int) or not isinstance(height, int):
        raise FixtureSchemaError("grid dimensions must be integers")
    if "data" in grid:
        expanded = grid["data"]
        if not isinstance(expanded, list):
            raise FixtureSchemaError("grid.data must be a list")
    else:
        rows = grid["rows"]
        if not isinstance(rows, list) or len(rows) != height:
            raise FixtureSchemaError("grid.rows must contain exactly height rows")
        values = {".": 0, "#": 100, "?": -1}
        expanded = []
        for row in rows:
            if (
                not isinstance(row, str) or len(row) != width
                or set(row) - values.keys()
            ):
                raise FixtureSchemaError("grid row must use width cells from .#?")
            expanded.extend(values[item] for item in row)
    if (
        len(expanded) != width * height
        or any(item not in {-1, 0, 100} for item in expanded)
    ):
        raise FixtureSchemaError("grid cells must be native and match geometry")
    return list(expanded)


def _validate_grid(value: object) -> list[int]:
    grid = _mapping(value, "occupancy_grid")
    common = {"header", "geometry", "semantics", "content_id"}
    if set(grid) not in (common | {"data"}, common | {"rows"}):
        raise FixtureSchemaError("occupancy_grid must contain exactly data or rows")
    header = _exact(
        grid["header"], {"frame_id", "stamp_ns", "map_version"}, "grid.header"
    )
    if header["frame_id"] != "map" or not all(
        isinstance(header[key], int) for key in ("stamp_ns", "map_version")
    ):
        raise FixtureSchemaError("grid header metadata is invalid")
    geometry = _exact(
        grid["geometry"], {"resolution_m", "width", "height", "origin"},
        "grid.geometry",
    )
    if (
        not isinstance(geometry["resolution_m"], (int, float))
        or geometry["resolution_m"] <= 0
    ):
        raise FixtureSchemaError("grid resolution must be positive")
    _exact(geometry["origin"], {"x", "y"}, "grid.origin")
    if grid["semantics"] != {"unknown": -1, "free": 0, "occupied": 100}:
        raise FixtureSchemaError("grid semantics must be native -1/0/100")
    _nonempty(grid["content_id"], "grid.content_id")
    return _grid_data(grid)


def _validate_events(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FixtureSchemaError("events must be a non-empty list")
    if value[0] != {"kind": "publish_initial_inputs"}:
        raise FixtureSchemaError("events must begin with publish_initial_inputs")
    growth: list[Mapping[str, Any]] = []
    allowed = {
        "publish_initial_inputs": {"kind"},
        "publish_map_growth": {"kind", "occupancy_grid"},
        "publish_odometry": {"kind", "pose_xy_yaw"},
        "advance_steady_clock": {"kind", "seconds"},
        "poll_execution": {"kind"},
    }
    for index, raw in enumerate(value):
        event = _mapping(raw, f"events[{index}]")
        kind = _nonempty(event.get("kind"), f"events[{index}].kind")
        if kind not in allowed or set(event) != allowed[kind]:
            raise FixtureSchemaError(f"events[{index}] has an invalid shape")
        if kind == "publish_map_growth":
            _validate_grid(event["occupancy_grid"])
            growth.append(_mapping(event["occupancy_grid"], "growth grid"))
        if kind == "advance_steady_clock" and (
            not isinstance(event["seconds"], (int, float))
            or event["seconds"] < 0
        ):
            raise FixtureSchemaError("steady clock advance must be non-negative")
        if kind == "publish_odometry" and (
            not isinstance(event["pose_xy_yaw"], list)
            or len(event["pose_xy_yaw"]) != 3
        ):
            raise FixtureSchemaError("odometry event must contain x,y,yaw")
    return growth


def _validate_table(value: object) -> dict[str, tuple[str, str]]:
    table = _mapping(value, "candidate_goal_cell_response_table")
    by_request: dict[str, tuple[str, str]] = {}
    for goal_cell, raw_row in table.items():
        _cell(goal_cell, "response goal cell")
        row = _exact(raw_row, {"responses"}, f"response[{goal_cell}]")
        responses = row["responses"]
        if not isinstance(responses, list) or not responses:
            raise FixtureSchemaError("each response row must be non-empty")
        for raw_response in responses:
            response = _mapping(raw_response, f"response[{goal_cell}]")
            required = {"kind", "candidate_key", "request_id"}
            optional = {"executable_endpoint_xy", "path_segment"}
            if not required <= set(response) or set(response) - required - optional:
                raise FixtureSchemaError("PlanMotion response shape is invalid")
            kind = _nonempty(response["kind"], "response.kind")
            if kind not in RESPONSE_KINDS:
                raise FixtureSchemaError("PlanMotion response kind is invalid")
            key = _candidate_key(response["candidate_key"], "response.candidate_key")
            request_id = _nonempty(response["request_id"], "response.request_id")
            if request_id in by_request:
                raise FixtureSchemaError("response request ids must be unique")
            endpoint = response.get("executable_endpoint_xy")
            if endpoint is not None and (
                kind != "SUCCESS" or not isinstance(endpoint, list)
                or len(endpoint) != 2
            ):
                raise FixtureSchemaError("endpoint is only valid for SUCCESS")
            if "path_segment" in response and kind != "SUCCESS":
                raise FixtureSchemaError("path_segment is only valid for SUCCESS")
            by_request[request_id] = (goal_cell, key)
    return by_request


def _validate_expected(value: object, table: dict[str, tuple[str, str]]) -> None:
    expected = _exact(value, {
        "ordered_candidate_keys", "ordered_goal_cells", "ordered_requests",
        "status_sequence", "reference_plan_ids", "cancellation_plan_ids",
        "coverage_ratio", "completed", "requires_exhaustive_planner_evidence",
        "replan_count", "stuck_retries", "committed_goal_key",
        "semantic_mutations", "two_segment", "ordered_trace_id",
    }, "expected")
    requests = expected["ordered_requests"]
    if not isinstance(requests, list) or len(requests) != len(table):
        raise FixtureSchemaError("ordered requests must exactly cover response table")
    ids: list[str] = []
    keys: list[str] = []
    cells: list[str] = []
    reference_ids: list[str] = []
    for raw in requests:
        request = _exact(raw, {
            "request_id", "candidate_key", "goal_cell", "response_kind",
            "path_segments",
        }, "expected request")
        request_id = _nonempty(request["request_id"], "expected.request_id")
        key = _candidate_key(request["candidate_key"], "expected.candidate_key")
        cell = _cell(request["goal_cell"], "expected.goal_cell")
        if table.get(request_id) != (cell, key):
            raise FixtureSchemaError("ordered request must match response lookup")
        if request["response_kind"] not in {"SUCCESS", "NO_PATH"}:
            raise FixtureSchemaError("observed response kind must be terminal")
        if not isinstance(request["path_segments"], list):
            raise FixtureSchemaError("path_segments must be a list")
        for raw_segment in request["path_segments"]:
            segment = _exact(raw_segment, {
                "segment_index", "reference_plan_id", "endpoint_cell",
                "final_goal_cell",
            }, "path_segment")
            if not isinstance(segment["segment_index"], int):
                raise FixtureSchemaError("segment_index must be an integer")
            reference_ids.append(
                _nonempty(segment["reference_plan_id"], "reference_plan_id")
            )
            _cell(segment["endpoint_cell"], "endpoint_cell")
            _cell(segment["final_goal_cell"], "final_goal_cell")
        ids.append(request_id); keys.append(key); cells.append(cell)
    if (
        len(ids) != len(set(ids))
        or expected["ordered_candidate_keys"] != keys
        or expected["ordered_goal_cells"] != cells
    ):
        raise FixtureSchemaError("ordered candidate/cell/request trace is inconsistent")
    if expected["reference_plan_ids"] != reference_ids:
        raise FixtureSchemaError("reference plan ids must preserve segment order")
    statuses = expected["status_sequence"]
    if (
        not isinstance(statuses, list) or not statuses
        or any(state not in STATE_NAMES for state in statuses)
    ):
        raise FixtureSchemaError("status sequence contains an invalid state")
    if expected["completed"] != (statuses[-1] == "COMPLETED"):
        raise FixtureSchemaError("completion must agree with final state")
    coverage = expected["coverage_ratio"]
    if not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1:
        raise FixtureSchemaError("coverage ratio must be in [0,1]")
    trace_id = _nonempty(expected["ordered_trace_id"], "ordered_trace_id")
    if not trace_id.startswith("fnv1a64:"):
        raise FixtureSchemaError("ordered trace id must be a normalized FNV hash")
    committed = expected["committed_goal_key"]
    if committed != "none":
        _candidate_key(committed, "committed_goal_key")
    if not all(
        isinstance(expected[field], int)
        for field in ("replan_count", "stuck_retries")
    ):
        raise FixtureSchemaError("retry counters must be integers")
    mutations = _exact(
        expected["semantic_mutations"],
        {"timestamp_only", "map_version_only", "covariance_only"},
        "semantic_mutations",
    )
    if set(mutations.values()) != {trace_id}:
        raise FixtureSchemaError("metadata variants must retain exact trace hash")
    two_value = _mapping(expected["two_segment"], "two_segment")
    two_fields = {
        "enabled", "committed_candidate_key", "request_ids", "endpoint_cells",
        "final_goal_cell",
    }
    if two_value.get("enabled") is True:
        two_fields |= {
            "completed_goal_count", "active_goal_after_arrival",
            "failed_candidate_growth",
        }
    two = _exact(two_value, two_fields, "two_segment")
    if two["enabled"] and (
        two["committed_candidate_key"] not in keys
        or len(two["request_ids"]) != 2 or len(two["endpoint_cells"]) != 2
        or two["completed_goal_count"] != 1
        or two["active_goal_after_arrival"] is not False
        or two["failed_candidate_growth"] != 0
    ):
        raise FixtureSchemaError("two-segment commit must freeze exact requests")


def _validate_variants(value: object, expected: Mapping[str, Any]) -> None:
    for name, raw in _mapping(value, "variants").items():
        variant = _exact(
            raw, {"request_count", "success_request_index", "expected"},
            f"variant.{name}",
        )
        count, success = variant["request_count"], variant["success_request_index"]
        if (
            not isinstance(count, int) or not isinstance(success, int)
            or not 0 <= success < count <= len(expected["ordered_requests"])
        ):
            raise FixtureSchemaError("variant request bounds are invalid")
        variant_expected = _exact(variant["expected"], {
            "status_sequence", "reference_plan_ids", "cancellation_plan_ids",
            "coverage_ratio", "completed", "replan_count", "stuck_retries",
            "committed_goal_key", "ordered_trace_id",
        }, f"variant.{name}.expected")
        if not str(variant_expected["ordered_trace_id"]).startswith("fnv1a64:"):
            raise FixtureSchemaError("variant must freeze normalized trace hash")


def load_fixture(name: str) -> Mapping[str, Any]:
    path = FIXTURE_DIRECTORY / f"{name}.yaml"
    if not path.is_file():
        raise FixtureNotFound(f"Task 15 fixture not found: {path}")
    fixture = _mapping(
        yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader),
        "fixture",
    )
    required = {
        "schema_version", "fixture_id", "authority", "platform", "task",
        "initial_inputs", "events", "candidate_goal_cell_response_table",
        "expected",
    }
    if set(fixture) not in (required, required | {"variants"}):
        raise FixtureSchemaError("fixture root keys are invalid")
    if (
        fixture["schema_version"] != 1 or fixture["fixture_id"] != name
        or fixture["authority"] != AUTHORITY
    ):
        raise FixtureSchemaError("fixture identity is invalid")
    platform = _exact(
        fixture["platform"], {"platform_type", "geometry_reference"}, "platform"
    )
    if platform["platform_type"] != "wheel":
        raise FixtureSchemaError("Task 15 freezes wheel platform")
    task = _exact(fixture["task"], {"task_id", "boundary_xy"}, "task")
    if not isinstance(task["boundary_xy"], list) or len(task["boundary_xy"]) < 4:
        raise FixtureSchemaError("task boundary needs at least four points")
    inputs = _exact(
        fixture["initial_inputs"], {"occupancy_grid", "transforms", "odometry"},
        "inputs",
    )
    initial_data = _validate_grid(inputs["occupancy_grid"])
    transforms = _exact(inputs["transforms"], {"map_to_odom"}, "transforms")
    _exact(
        transforms["map_to_odom"], {"translation_xy", "yaw_rad"}, "map_to_odom"
    )
    odometry = _exact(inputs["odometry"], {
        "frame_id", "child_frame_id", "pose_xy_yaw", "covariance",
    }, "odometry")
    if (
        odometry["frame_id"] != "odom"
        or odometry["child_frame_id"] != "base_link"
        or not isinstance(odometry["covariance"], list)
        or len(odometry["covariance"]) != 36
    ):
        raise FixtureSchemaError("odometry contract is invalid")
    for grid in _validate_events(fixture["events"]):
        if (
            grid["content_id"] == inputs["occupancy_grid"]["content_id"]
            or _grid_data(grid) == initial_data
        ):
            raise FixtureSchemaError("map growth must change geometry/data content")
    table = _validate_table(fixture["candidate_goal_cell_response_table"])
    _validate_expected(fixture["expected"], table)
    if "variants" in fixture:
        _validate_variants(fixture["variants"], fixture["expected"])
    return fixture


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_fixture_loader_requires_each_committed_scenario(fixture_name: str) -> None:
    assert load_fixture(fixture_name)["fixture_id"] == fixture_name


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_fixture_load_is_deterministic_and_metadata_mutations_are_trace_invariant(
    fixture_name: str,
) -> None:
    first = load_fixture(fixture_name)
    assert first == load_fixture(fixture_name)
    assert set(first["expected"]["semantic_mutations"].values()) == {
        first["expected"]["ordered_trace_id"]
    }


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_fixture_preserves_native_occupancy_grid_semantics(fixture_name: str) -> None:
    grid = load_fixture(fixture_name)["initial_inputs"]["occupancy_grid"]
    assert grid["semantics"] == {"unknown": -1, "free": 0, "occupied": 100}
    assert len(_grid_data(grid)) == grid["geometry"]["width"] * grid["geometry"]["height"]


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_fixture_freezes_every_ordered_request_response_without_omission_or_repetition(
    fixture_name: str,
) -> None:
    requests = load_fixture(fixture_name)["expected"]["ordered_requests"]
    assert [row["request_id"] for row in requests] == list(
        dict.fromkeys(row["request_id"] for row in requests)
    )
    assert all(row["candidate_key"] and row["goal_cell"] for row in requests)


def test_concave_region_freezes_reference_without_coverage_completion_gate() -> None:
    expected = load_fixture("concave_region")["expected"]
    assert expected["status_sequence"][-1] == "EXECUTING"
    assert expected["completed"] is False
    assert expected["reference_plan_ids"] == [
        "scenario-plan:task15-concave-region/candidate/0"
    ]


def test_unreachable_frontier_freezes_narrow_continue_and_exhaustive_completion() -> None:
    fixture = load_fixture("unreachable_frontier")
    assert fixture["expected"]["completed"] is True
    assert fixture["expected"]["coverage_ratio"] < 1.0
    variant = fixture["variants"]["narrow_passage_continue"]
    assert variant["success_request_index"] == 1
    assert variant["expected"]["completed"] is False


def test_map_growth_freezes_reevaluation_and_two_segment_same_commit() -> None:
    fixture = load_fixture("map_growth")
    expected = fixture["expected"]
    growth = next(
        event for event in fixture["events"]
        if event["kind"] == "publish_map_growth"
    )["occupancy_grid"]
    assert _grid_data(growth) != _grid_data(
        fixture["initial_inputs"]["occupancy_grid"]
    )
    assert expected["two_segment"]["committed_candidate_key"] == "4081,2750,225"
    assert expected["two_segment"]["request_ids"] == [
        "task15-map-growth/candidate/15", "task15-map-growth/candidate/30"
    ]
    odometry_events = [
        event for event in fixture["events"]
        if event["kind"] == "publish_odometry"
    ]
    assert [event["pose_xy_yaw"] for event in odometry_events] == [
        [3.5, 2.75, 0.0], [4.081, 2.75, 0.39269908169872414]
    ]
    assert expected["two_segment"]["completed_goal_count"] == 1
    assert expected["two_segment"]["active_goal_after_arrival"] is False
    assert expected["two_segment"]["failed_candidate_growth"] == 0
    assert expected["replan_count"] == expected["stuck_retries"] == 0


def test_fully_known_completion_freezes_coverage_one_after_normal_exhaustion() -> None:
    fixture = load_fixture("fully_known_completion")
    assert -1 not in _grid_data(fixture["initial_inputs"]["occupancy_grid"])
    assert fixture["expected"]["coverage_ratio"] == 1.0
    assert fixture["expected"]["completed"] is True


def test_ros_runner_contract_has_a_fixed_steady_deadline() -> None:
    hints = get_type_hints(RosScenarioRunner.run)
    assert hints["deadline_steady_s"] is float
    assert hints["return"] is ScenarioTrace
