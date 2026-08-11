from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np
import pytest
from lunar_planner_training_bridge import PlannerBridge

from lunar_policy_training.environment.formal_builder import (
    FormalEnvironmentBuilder,
)
from lunar_policy_training.environment.macro_step import PolicyAction
from lunar_policy_training.environment.observation_boundary import (
    SensorBoundaryEvidence,
    SensorPathSample,
)
from lunar_policy_training.environment.observation_builder import Pose2
from lunar_policy_training.project_capability import (
    load_project_formal_capability,
)
from lunar_policy_training.polar_data.formal_cache import load_formal_cache


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
PINNED_HISTORICAL_ROOT = pathlib.Path(
    "/home/kai/CodexDownloads/lunar_navigation/platform_coverable_exploration/"
    "preflight-gate-a2ee1acb4676"
)
TRACE_NAME = "b0c2-legged-rejection-trace.jsonl"
TRACE_SHA256 = "a2f7b0ece14ae963bcbea77b14aa05b9e3d44850ca1693360eed4185e9975526"
STATE_NAME = "b0c2-legged-step193-state.json"
STATE_SHA256 = "d5ba1034f96b58f0e716b94207678ebf98894661eda49f6c6a58644e99c8c5a4"
HISTORICAL_COVERAGE = 0.039319027215242386
HISTORICAL_TARGET = np.asarray(
    (17286.0, 97490.0, 374.2354736328125), dtype=np.float64
)


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid_map_sha256(grid_map: object) -> str:
    digest = hashlib.sha256()
    origin = grid_map.origin_m
    header = {
        "frame_id": str(grid_map.frame_id),
        "width": int(grid_map.width),
        "height": int(grid_map.height),
        "resolution_m": float(grid_map.resolution_m),
        "origin_m": [float(origin.x), float(origin.y), float(origin.z)],
    }
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for name, layer in sorted(grid_map.layers.items()):
        values = np.ascontiguousarray(layer.values)
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _pose(raw: object) -> Pose2:
    assert isinstance(raw, dict)
    return Pose2(
        float(raw["x_m"]),
        float(raw["y_m"]),
        float(raw["yaw_rad"]),
        str(raw["frame_id"]),
        float(raw["elevation_m"]),
    )


def test_historical_boundary_failure_replays_against_v6_cache(
    record_property: pytest.FixtureRequest,
) -> None:
    historical_root_raw = os.environ.get("LUNAR_HISTORICAL_FAILURE_ROOT")
    cache_manifest_raw = os.environ.get("LUNAR_PHYSICAL_CACHE_MANIFEST")
    if not historical_root_raw or not cache_manifest_raw:
        pytest.skip(
            "exact historical replay requires LUNAR_HISTORICAL_FAILURE_ROOT "
            "and the Task 13 LUNAR_PHYSICAL_CACHE_MANIFEST"
        )

    historical_root = pathlib.Path(historical_root_raw).resolve(strict=True)
    assert historical_root == PINNED_HISTORICAL_ROOT
    trace_path = historical_root / TRACE_NAME
    state_path = historical_root / STATE_NAME
    assert _file_sha256(trace_path) == TRACE_SHA256
    assert _file_sha256(state_path) == STATE_SHA256

    trace = tuple(
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    historical = trace[-1]
    assert float(historical["coverage"]) == HISTORICAL_COVERAGE
    assert int(historical["oracle"]["opportunity_count"]) == 146
    assert historical["reason"] == "LOCAL_SEGMENT_INFEASIBLE"
    np.testing.assert_array_equal(
        np.asarray(historical["target"], dtype=np.float64),
        HISTORICAL_TARGET,
    )
    legacy_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert legacy_state["platform_type"] == "LEGGED"
    legacy_body_z_m = float(legacy_state["legged_body_z_m"])
    assert legacy_body_z_m == float(
        legacy_state["reveal_history"][-1]["legged_body_z_m"]
    )

    cache_manifest = pathlib.Path(cache_manifest_raw).resolve(strict=True)
    cache = load_formal_cache(cache_manifest, require_full=False)
    assert cache.manifest["schema"] == "lunar-formal-training-cache/v6"
    assert cache.manifest["materialization"] == "preflight"
    assert cache.manifest["formal_eligible"] is False
    capability_bundle = load_project_formal_capability(REPOSITORY_ROOT)
    assembly = FormalEnvironmentBuilder(
        cache_manifest_path=cache_manifest,
        capability_bundle=capability_bundle,
        split="train",
        allow_preflight=True,
    ).build()
    worker = assembly.factory(0, "LEGGED")
    episode = worker.episode
    assert episode.scene_id == legacy_state["scene_id"]
    assert list(episode.start_cell) == legacy_state["start_cell"]

    # The v4 worker payload is read-only evidence.  Rebuild a fresh v6 episode
    # exclusively through the public sensor-boundary API; never resume it.
    for reveal in legacy_state["reveal_history"]:
        pose = _pose(reveal["pose"])
        samples = tuple(
            SensorPathSample(_pose(sample["pose"]), float(sample["elapsed_s"]))
            for sample in reveal.get("path_samples", ())
        )
        evidence = SensorBoundaryEvidence(
            pose,
            float(reveal["elapsed_s"]),
            path_samples=samples,
        )
        episode.current_pose = pose
        boundary = episode.controller.after_execution(
            platform_type="LEGGED",
            execution_state=str(reveal["execution_state"]),
            evidence=evidence,
        )
        assert boundary.updated is True

    body_height = capability_bundle.for_platform(
        "LEGGED"
    ).typed_capability.body_height_m
    nominal_body_height_m = 0.5 * (
        float(body_height.lower) + float(body_height.upper)
    )
    episode._current_legged_body_z_m = (
        float(episode.current_pose.elevation_m) + nominal_body_height_m
    )
    rebuilt_body_height_m = (
        episode._current_legged_body_z_m
        - float(episode.current_pose.elevation_m)
    )
    assert float(body_height.lower) <= rebuilt_body_height_m
    assert rebuilt_body_height_m <= float(body_height.upper)
    assert rebuilt_body_height_m == pytest.approx(nominal_body_height_m)
    assert episode._current_legged_body_z_m != legacy_body_z_m

    snapshot = episode._snapshot
    assert snapshot is not None
    active = np.flatnonzero(snapshot.candidates.mask)
    matches = tuple(
        int(index)
        for index in active
        if np.array_equal(
            snapshot.candidates.target_positions_m[int(index)],
            HISTORICAL_TARGET,
        )
    )
    assert len(matches) == 1
    identity = episode.controller.current_observation.observation_identities[0]
    prepared = episode.begin_ground_option(
        PolicyAction(matches[0], 0.0), identity
    )
    request = prepared.request
    assert request.config.local_frontier.additional_corridor_margin_m == 2.0
    local_map_sha256 = _grid_map_sha256(request.world.local_map)

    output = PlannerBridge().plan(request)

    assert _grid_map_sha256(request.world.local_map) == local_map_sha256
    assert output.reason_code != "LOCAL_SEGMENT_INFEASIBLE"
    assert output.reason_code != "LOCAL_SEARCH_DOMAIN_EXHAUSTED"
    assert output.reference is not None
    assert output.diagnostics.expanded_states > 0
    metrics = output.diagnostics.hierarchical
    assert metrics is not None
    assert metrics.additional_corridor_margin_m == 2.0
    assert len(metrics.search_domain_sha256) == 64
    record_property("planner_outcome", str(output.outcome))
    record_property("planner_reason_code", output.reason_code)
    record_property("search_domain_sha256", metrics.search_domain_sha256)
    record_property("search_domain_cell_count", metrics.search_domain_cell_count)
    record_property("local_map_sha256", local_map_sha256)

    assert _file_sha256(trace_path) == TRACE_SHA256
    assert _file_sha256(state_path) == STATE_SHA256
