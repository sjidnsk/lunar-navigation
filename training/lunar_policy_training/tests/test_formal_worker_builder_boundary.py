from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lunar_policy_training.capability_freeze import ScenarioIdentity
from lunar_policy_training.config import TaskAreaConfig
from lunar_policy_training.environment import formal_builder as formal_builder_module


def test_worker_builder_defers_empty_initial_candidates_to_environment_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = formal_builder_module.FormalWorkerBuilder(
        cache_manifest_path="/unused/cache-manifest.json",
        split="train",
        allow_preflight=True,
        scenario_schedule_id="test-schedule/v1",
        task_area=TaskAreaConfig(
            minimum_size_m=100.0,
            maximum_size_m=500.0,
            sampling_algorithm="deterministic-uniform-square/v1",
        ),
    )
    capability = object()
    scenario = ScenarioIdentity(
        platform_type="WHEELED",
        scenario_schedule_id="test-schedule/v1",
        worker_index=2,
        episode_cursor=0,
        platform_worker_index=2,
        platform_worker_count=12,
        capability_version="test/v1",
        capability_sha256="f" * 64,
    )
    loaded = SimpleNamespace(
        task_geometry=SimpleNamespace(local_start_cell=(0, 0))
    )
    episode = SimpleNamespace(
        initial_observation=SimpleNamespace(
            candidate_mask=torch.zeros((1, 64), dtype=torch.bool)
        )
    )
    sentinel = object()
    monkeypatch.setattr(
        formal_builder_module.FormalWorkerBuilder,
        "_load_scheduled_scene",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        formal_builder_module,
        "FormalEpisode",
        lambda **_kwargs: episode,
    )
    monkeypatch.setattr(
        formal_builder_module.FormalWorkerBuilder,
        "_make_worker",
        staticmethod(lambda _episode: sentinel),
    )

    assert builder(2, "WHEELED", capability, scenario) is sentinel
