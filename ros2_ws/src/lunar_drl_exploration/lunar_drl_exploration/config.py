"""Training-side view of the canonical native platform capability file."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from types import MappingProxyType
from typing import Mapping

from .contracts import Pose, SensorSpec


@dataclass(frozen=True)
class GraphConfig:
    """Graph geometry constants are independent of sensor range."""
    position_unit_m: float = 10.0
    frontier_unit_cells: float = 100.0
    history_tolerance_m: float = 0.1
    coverage_radius_m: float = 2.0
    connection_limit_m: float = 8.0
    stretch: float = 1.2
    platform: "PlatformConfig | None" = None

    def __post_init__(self):
        if not (0 < self.coverage_radius_m < self.connection_limit_m and self.stretch >= 1.0):
            raise ValueError("positive graph radius/connection length and stretch >= 1 required")
        if self.position_unit_m != 10.0 or self.frontier_unit_cells != 100.0:
            raise ValueError("task_graph_v3 fixes position and frontier normalization")
        if not np.isfinite(self.history_tolerance_m) or self.history_tolerance_m <= 0:
            raise ValueError("positive history tolerance required")


@dataclass(frozen=True)
class PlatformConfig:
    maximum_forward_speed_mps: float
    maximum_reverse_speed_mps: float
    maximum_spin_rate_radps: float
    footprint_radius_m: float
    capability: Mapping = field(default_factory=dict)
    capability_path: str = ""

    def actor_context(
        self,
        pose: Pose,
        *,
        linear_speed_mps: float,
        angular_speed_radps: float,
        sensor_range_m: float,
        sensor_fov_rad: float
    ) -> np.ndarray:
        return np.asarray(
            [
                np.cos(pose.yaw),
                np.sin(pose.yaw),
                linear_speed_mps / 0.2,
                angular_speed_radps / self.maximum_spin_rate_radps,
                sensor_range_m / 10.0,
                sensor_fov_rad / np.pi,
                self.footprint_radius_m,
                self.maximum_forward_speed_mps / 0.2,
            ],
            dtype=np.float32,
        )


def load_platform_config(path: Path | None = None) -> PlatformConfig:
    """Read wheel capability values from the native source of truth."""
    if path is None:
        try:
            from ament_index_python.packages import (
                get_package_share_directory,
                PackageNotFoundError,
            )
        except ImportError:
            path = Path(__file__).resolve().parents[4] / "config" / "wheel.yaml"
        else:
            try:
                path = (
                    Path(
                        get_package_share_directory("lunar_incremental_navigation_ros")
                    )
                    / "config"
                    / "wheel.yaml"
                )
            except PackageNotFoundError:
                path = Path(__file__).resolve().parents[4] / "config" / "wheel.yaml"
    with Path(path).open(encoding="utf-8") as stream:
        capability = yaml.safe_load(stream)["capability"]
    footprint = np.asarray(capability["footprint_xy_m"], dtype=np.float64)
    radius = float(np.max(np.linalg.norm(footprint, axis=1)))
    return PlatformConfig(
        maximum_forward_speed_mps=float(capability["maximum_forward_speed_mps"]),
        maximum_reverse_speed_mps=float(capability["maximum_reverse_speed_mps"]),
        maximum_spin_rate_radps=float(capability["maximum_spin_rate_radps"]),
        footprint_radius_m=radius,
        capability=MappingProxyType(capability),
        capability_path=str(Path(path).resolve()),
    )


@dataclass(frozen=True)
class ModelConfig:
    """Primitive network settings; importing observed workers never imports Torch."""
    width: int = 128
    heads: int = 8
    layers: int = 6
    actor_score_bound: float = 0.0

    def __post_init__(self):
        if self.width <= 0 or self.heads <= 0 or self.width % self.heads or self.layers <= 0:
            raise ValueError("positive width divisible by heads and positive layers required")
        if not np.isfinite(self.actor_score_bound) or self.actor_score_bound < 0:
            raise ValueError("actor_score_bound must be finite and nonnegative")


def canonical_model_config(record):
    """Only legacy v3's missing score parameterization defaults to unbounded."""
    result = dict(record)
    result.setdefault('actor_score_bound', 0.0)
    return result


@dataclass(frozen=True)
class LearningConfig:
    batch_size: int = 64
    microbatch_size: int = 16
    learning_rate: float = 1e-5
    gamma: float = 1.0
    polyak: float = 0.005
    initial_alpha: float = 5e-5
    maximum_alpha: float = 1e-4
    target_entropy_factor: float = 0.10

    def __post_init__(self):
        if self.batch_size != 64 or not 0 < self.microbatch_size <= self.batch_size:
            raise ValueError("effective batch is 64; microbatch must be in [1,64]")
        if not 0 < self.gamma <= 1:
            raise ValueError("task reward requires 0 < gamma <= 1")
        if not (0 < self.initial_alpha <= self.maximum_alpha and
                0 < self.polyak <= 1 and self.learning_rate > 0 and
                self.target_entropy_factor >= 0):
            raise ValueError("invalid SAC optimization settings")


@dataclass(frozen=True)
class TrainingConfig:
    """Operational defaults and explicit experience semantics; no Torch/native import."""
    model: ModelConfig = field(default_factory=ModelConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    completion_bonus: float = 5.0
    sensor: 'SensorSpec' = field(default_factory=lambda: SensorSpec())
    platform: PlatformConfig = field(default_factory=load_platform_config)
    seed: int = 20260915
    paired_scene_sequence: bool = False
    environments: int = 8
    target_rtf: float = 30.0
    integration_step_s: float = .05
    observation_hz: float = 2.0
    goal_position_tolerance_m: float = .05
    warmup: int = 1024
    update_ratio: float = .25
    max_update_credit: int = 32
    actor_publish_updates: int = 16
    worker_threads: int = 1
    collector_threads: int = 2
    learner_threads: int = 4
    dataloader_workers: int = 0
    curriculum_transition_boundaries: tuple = (20000, 60000)
    curriculum_mixtures: tuple = ((1., 0., 0.), (.25, .75, 0.), (.15, .25, .60))
    curriculum_budgets: tuple = (512, 2048, 8192)
    curriculum_extents_m: tuple = ((40, 80), (80, 150), (100, 300))
    replay_max_bytes: int = 8 * 1024**3
    snapshot_max_bytes: int = 2 * 1024**3
    total_output_max_bytes: int = 20 * 1024**3
    metrics_max_bytes: int = 8 * 1024**2
    save_interval_s: float = 1800.0
    output_dir: str = 'training-output/drl-metric-critic'
    system_reserve_bytes: int = 2 * 1024**3
    owned_pss_limit_bytes: int | None = None

    def __post_init__(self):
        boundaries = self.curriculum_transition_boundaries
        if not np.isfinite(self.completion_bonus) or self.completion_bonus < 0:
            raise ValueError('completion bonus must be finite and nonnegative')
        if (not isinstance(boundaries, (tuple, list)) or len(boundaries) != 2 or
                any(type(value) is not int or value < 0 for value in boundaries) or
                boundaries[0] >= boundaries[1]):
            raise ValueError('curriculum boundaries require two increasing nonnegative integers')
        mixtures = np.asarray(self.curriculum_mixtures, dtype=float)
        if (mixtures.shape != (3, 3) or not np.all(np.isfinite(mixtures)) or
                np.any(mixtures < 0) or not np.allclose(mixtures.sum(axis=1), 1., rtol=0., atol=1e-12)):
            raise ValueError('curriculum mixtures require three normalized nonnegative probability triples')
        for name in ('environments', 'max_update_credit', 'actor_publish_updates',
                     'worker_threads', 'collector_threads', 'learner_threads',
                     'replay_max_bytes', 'snapshot_max_bytes', 'total_output_max_bytes',
                     'metrics_max_bytes'):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'positive integer required: {name}')
        for name in ('target_rtf', 'integration_step_s', 'observation_hz',
                     'goal_position_tolerance_m', 'update_ratio', 'save_interval_s'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'positive finite value required: {name}')
        if self.warmup < 0 or self.dataloader_workers != 0 or self.system_reserve_bytes < 0:
            raise ValueError('invalid warmup, DataLoader or reserve settings')
        if self.owned_pss_limit_bytes is not None and self.owned_pss_limit_bytes <= 0:
            raise ValueError('positive owned PSS limit required')


def config_record(value):
    """Explicit canonical JSON/spawn record; asdict cannot deepcopy mappingproxy."""
    from dataclasses import fields, is_dataclass
    if is_dataclass(value):
        return {f.name: config_record(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): config_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [config_record(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f'nonprimitive config value: {type(value).__name__}')


def resume_semantics(config):
    """Full replay continuation only. Actor-only loads use actor schema/model shape."""
    record = config_record(config)
    learning = {key: value for key, value in record['learning'].items()
                if key not in ('microbatch_size','learning_rate','polyak','initial_alpha')}
    platform = {key: value for key, value in record['platform'].items()
                if key != 'capability_path'}
    result = dict(model=record['model'], learning=learning,
        observation_schema='task_graph_v3', action_schema='local_metric_pose_v3',
        privileged_schema='candidate_truth_v1',
        graph_geometry=dict(coverage_radius_m=2.,connection_limit_m=8.,stretch=1.2,
            base_order='clearance_desc_yx_v1'),
        observation_model='finite_center_tip_prefix_v1', generator_version=5,
        observation_origin='actual_pose_optical_offset',
        effective_measurement='classified_center_native_3x3_no_hidden_neighbors_v1',
        reward='delta_area/100-.02*distance/10-.005*absolute_turn/pi-.001+completion_bonus*terminated_v2',
        completion_bonus=config.completion_bonus,
        termination='current_reachable_task_opportunities_v1',
        graph_normalization=dict(position_m=10., frontier_cells=100.),
        sensor=record['sensor'], platform=platform,
        integration_step_s=config.integration_step_s, observation_hz=config.observation_hz,
        goal_position_tolerance_m=config.goal_position_tolerance_m,
        schedule=dict(warmup=config.warmup, ratio=config.update_ratio),
        curriculum_budgets=record['curriculum_budgets'],
        curriculum_extents_m=record['curriculum_extents_m'])
    if config.paired_scene_sequence:
        result['scene_sequence'] = dict(mode='paired_slot_episode_v1',
            seed=config.seed, environments=config.environments)
    return result


def training_config_from_record(record):
    """Reconstruct a JSON/spawn configuration without importing models or terrain."""
    values = dict(record)
    values['model'] = ModelConfig(**values['model'])
    values['learning'] = LearningConfig(**values['learning'])
    values['sensor'] = SensorSpec(**values['sensor'])
    platform = dict(values['platform'])
    platform['capability'] = MappingProxyType(dict(platform['capability']))
    values['platform'] = PlatformConfig(**platform)
    if 'curriculum_transition_boundaries' in values:
        values['curriculum_transition_boundaries'] = tuple(values['curriculum_transition_boundaries'])
    if 'curriculum_mixtures' in values:
        values['curriculum_mixtures'] = tuple(tuple(row) for row in values['curriculum_mixtures'])
    values['curriculum_budgets'] = tuple(values['curriculum_budgets'])
    values['curriculum_extents_m'] = tuple(tuple(pair) for pair in values['curriculum_extents_m'])
    return TrainingConfig(**values)
