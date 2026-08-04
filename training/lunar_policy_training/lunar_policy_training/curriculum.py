"""Frozen 24-hour curriculum and deterministic proxy scenario schedule."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")
FORMAL_SEED = 4080
REWARD_CALIBRATION_SEEDS = (4081, 4082, 4083)
_COMPLETE_UPDATE_RESERVE_S = 600


@dataclass(frozen=True, slots=True)
class CurriculumScenario:
    platform_type: str
    capability_id: str
    terrain_id: str
    scenario_seed: int
    proxy: bool = True


@dataclass(frozen=True, slots=True)
class FrozenCurriculum:
    calibration_used_s: float
    warmup_used_s: Mapping[str, float]
    joint_budget_s: float
    total_gpu_limit_s: int
    formal_seed: int
    reward_calibration_seeds: tuple[int, int, int]
    scenario_schedule_id: str


@dataclass(frozen=True, slots=True)
class CurriculumSchedule:
    calibration_limit_s: int = 2 * 60 * 60
    warmup_limit_s: int = 6 * 60 * 60
    joint_minimum_s: int = 16 * 60 * 60
    platform_warmup_limit_s: int = 2 * 60 * 60
    evaluation_interval_s: int = 3 * 60 * 60

    def __post_init__(self) -> None:
        values = (
            self.calibration_limit_s,
            self.warmup_limit_s,
            self.joint_minimum_s,
            self.platform_warmup_limit_s,
            self.evaluation_interval_s,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("curriculum durations must be positive integer seconds")
        if self.warmup_limit_s != len(PLATFORMS) * self.platform_warmup_limit_s:
            raise ValueError("warmup budget must equal three platform warmups")
        if self.total_gpu_limit_s != 24 * 60 * 60:
            raise ValueError("curriculum total GPU budget must be 24 hours")

    @property
    def total_gpu_limit_s(self) -> int:
        return self.calibration_limit_s + self.warmup_limit_s + self.joint_minimum_s

    @property
    def platform_warmup_order(self) -> tuple[str, str, str]:
        return PLATFORMS

    @property
    def joint_worker_allocation(self) -> dict[str, int]:
        return {platform: 8 for platform in PLATFORMS}

    @property
    def scenario_schedule_id(self) -> str:
        payload = {
            "schema_version": "lunar-policy-proxy-scenario-schedule/v1",
            "proxy": True,
            "platforms": PLATFORMS,
            "terrain_cycle": ("flat_sparse", "rolling_sparse", "full_proxy"),
            "scenario_seeds": tuple(
                10_000 + platform_index * 1_000 + scenario_index
                for platform_index in range(len(PLATFORMS))
                for scenario_index in range(3)
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return f"proxy-scenario-schedule-v1:{digest}"

    def scenario_for(
        self, *, platform_type: str, scenario_index: int
    ) -> CurriculumScenario:
        if platform_type not in PLATFORMS:
            raise ValueError("unknown curriculum platform")
        if type(scenario_index) is not int or scenario_index < 0:
            raise ValueError("scenario index must be a non-negative integer")
        platform_index = PLATFORMS.index(platform_type)
        terrain_cycle = ("flat_sparse", "rolling_sparse", "full_proxy")
        return CurriculumScenario(
            platform_type=platform_type,
            capability_id=f"proxy-{platform_type.lower()}-v1",
            terrain_id=terrain_cycle[scenario_index % len(terrain_cycle)],
            scenario_seed=10_000 + platform_index * 1_000 + scenario_index,
        )

    def freeze(
        self,
        *,
        calibration_used_s: float,
        warmup_used_s: Mapping[str, float],
    ) -> FrozenCurriculum:
        calibration = _bounded_seconds(
            calibration_used_s,
            name="calibration",
            upper=float(self.calibration_limit_s),
        )
        if set(warmup_used_s) != set(PLATFORMS):
            raise ValueError("warmup usage must cover exactly three platforms")
        warmups = {
            platform: _bounded_seconds(
                warmup_used_s[platform],
                name=f"{platform} warmup",
                upper=float(self.platform_warmup_limit_s),
            )
            for platform in PLATFORMS
        }
        joint_budget = self.total_gpu_limit_s - calibration - sum(warmups.values())
        if joint_budget < self.joint_minimum_s:
            raise ValueError("early phases cannot reduce the joint minimum")
        return FrozenCurriculum(
            calibration_used_s=calibration,
            warmup_used_s=MappingProxyType(warmups),
            joint_budget_s=joint_budget,
            total_gpu_limit_s=self.total_gpu_limit_s,
            formal_seed=FORMAL_SEED,
            reward_calibration_seeds=REWARD_CALIBRATION_SEEDS,
            scenario_schedule_id=self.scenario_schedule_id,
        )

    def phase_for(
        self, *, consumed_gpu_s: float, calibration_end_gpu_s: float
    ) -> str:
        consumed = _bounded_absolute_seconds(consumed_gpu_s, "consumed")
        calibration_end = _bounded_absolute_seconds(
            calibration_end_gpu_s, "calibration end"
        )
        if consumed < calibration_end:
            raise ValueError("consumed GPU time precedes frozen calibration end")
        offset = consumed - calibration_end
        if offset < self.platform_warmup_limit_s - _COMPLETE_UPDATE_RESERVE_S:
            return "warmup_wheeled"
        if offset < 2 * self.platform_warmup_limit_s - _COMPLETE_UPDATE_RESERVE_S:
            return "warmup_legged"
        if offset < 3 * self.platform_warmup_limit_s - _COMPLETE_UPDATE_RESERVE_S:
            return "warmup_hopper"
        return "joint"

    def phase_end_gpu_s(
        self, phase: str, *, calibration_end_gpu_s: float
    ) -> float:
        calibration_end = _bounded_absolute_seconds(
            calibration_end_gpu_s, "calibration end"
        )
        warmup_number = {
            "warmup_wheeled": 1,
            "warmup_legged": 2,
            "warmup_hopper": 3,
        }.get(phase)
        if warmup_number is None:
            if phase != "joint":
                raise ValueError("unknown curriculum phase")
            return float(self.total_gpu_limit_s)
        return calibration_end + warmup_number * self.platform_warmup_limit_s

    def worker_allocation(
        self, phase: str, *, selected_workers: int
    ) -> dict[str, int]:
        if type(selected_workers) is not int or selected_workers not in (18, 24):
            raise ValueError("curriculum workers must be the frozen 18 or 24")
        warmup_platform = {
            "warmup_wheeled": "WHEELED",
            "warmup_legged": "LEGGED",
            "warmup_hopper": "HOPPER",
        }.get(phase)
        if warmup_platform is not None:
            return {warmup_platform: selected_workers}
        if phase != "joint":
            raise ValueError("unknown curriculum phase")
        per_platform = selected_workers // 3
        return {platform: per_platform for platform in PLATFORMS}


def _bounded_seconds(value: float, *, name: str, upper: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= upper
    ):
        raise ValueError(f"{name} GPU seconds are outside the frozen limit")
    return float(value)


def _bounded_absolute_seconds(value: float, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} GPU seconds must be finite and non-negative")
    return float(value)


__all__ = [
    "CurriculumScenario",
    "CurriculumSchedule",
    "FORMAL_SEED",
    "FrozenCurriculum",
    "PLATFORMS",
    "REWARD_CALIBRATION_SEEDS",
]
