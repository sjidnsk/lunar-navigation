from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .host import HostFacts


_INTERFACE_KEYS = (
    "map_global",
    "map_local",
    "odometry",
    "localization_status",
    "tf",
    "exploration_task",
    "motion_feedback",
    "plan_motion",
    "diagnostics",
    "certified_route_markers",
    "provisional_route_markers",
)
_TOP_LEVEL_KEYS = (
    "profile",
    "interfaces",
    "capabilities",
    "planner",
    "policy",
    "extensions",
    "input_adapters",
    "controller",
    "runtime",
)
_SNAPSHOT_KEYS = (
    "global_map_max_age",
    "local_map_max_age",
    "odometry_max_age",
    "localization_status_max_age",
    "tf_max_age",
    "max_pairwise_skew",
)
_FORBIDDEN_SAFETY_KEYS = {
    "disable_safety_projection",
    "disable_footprint_check",
    "disable_clearance_check",
    "disable_endpoint_check",
    "disable_path_certification",
    "disable_input_freshness_check",
    "disable_input_consistency_check",
}
_WHEELED_CONTROLLER_KEYS = (
    "enabled",
    "command_topic",
    "odometry_topic",
    "feedback_topic",
    "execution_goal_topic",
    "reference_topic",
    "control_rate_hz",
    "lookahead_m",
    "max_linear_mps",
    "max_angular_radps",
    "max_cross_track_error_m",
    "goal_position_tolerance_m",
    "goal_yaw_tolerance_rad",
    "reference_max_age_s",
    "odometry_max_age_s",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TargetProfile:
    id: str
    os_id: str
    os_version: str
    architecture: str
    ros_distro: str
    l4t_prefix: str | None
    jetpack_major: int | None
    policy_probe: str


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str
    interfaces: Mapping[str, str]
    capabilities: Mapping[str, str]
    planner: Mapping[str, Any]
    policy: Mapping[str, Any]
    extensions: Mapping[str, bool]
    input_adapters: Mapping[str, str | None]
    controller: Mapping[str, Mapping[str, str | float | bool]]
    runtime: Mapping[str, Any]

    def planner_ros_parameters(self) -> dict[str, Any]:
        return {
            **{f"interfaces.{key}": value for key, value in self.interfaces.items()},
            **dict(self.planner["snapshot_policy"]),
            "platform_capability_file": self.capabilities["platform_file"],
            "observation_capability_file": self.capabilities["observation_file"],
            "enable_nav2_adapter": self.planner["enable_nav2_adapter"],
            "log_level": self.runtime["log_level"],
        }

    def runtime_ros_parameters(self) -> dict[str, Any]:
        controller = self.controller["wheeled"]
        return {
            "/lunar_planner": {"ros__parameters": self.planner_ros_parameters()},
            "/luna_task_execution_coordinator": {"ros__parameters": {
                "exploration_task_topic": self.interfaces["exploration_task"],
                "execution_goal_topic": controller["execution_goal_topic"],
                "plan_motion_action": self.interfaces["plan_motion"],
                "reference_topic": controller["reference_topic"],
            }},
            "/luna_wheeled_controller": {"ros__parameters": dict(controller)},
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def load_profile(profile_id: str, repo_root: Path | None = None) -> TargetProfile:
    root = repo_root or _repo_root()
    path = root / "deployment" / "profiles" / f"{profile_id}.yaml"
    data = _load_yaml(path)
    expected = {"schema_version", "id", "host", "policy_probe"}
    if set(data) != expected:
        raise ConfigError(f"profile {profile_id} has unexpected keys")
    host = data["host"]
    if not isinstance(host, dict):
        raise ConfigError(f"profile {profile_id}.host must be a mapping")
    required = {"os_id", "os_version", "architecture", "ros_distro", "l4t_prefix", "jetpack_major"}
    if set(host) != required or data["id"] != profile_id:
        raise ConfigError(f"profile {profile_id} is incomplete")
    return TargetProfile(
        id=profile_id,
        os_id=str(host["os_id"]),
        os_version=str(host["os_version"]),
        architecture=str(host["architecture"]),
        ros_distro=str(host["ros_distro"]),
        l4t_prefix=None if host["l4t_prefix"] is None else str(host["l4t_prefix"]),
        jetpack_major=None if host["jetpack_major"] is None else int(host["jetpack_major"]),
        policy_probe=str(data["policy_probe"]),
    )


def validate_host(profile: TargetProfile, facts: HostFacts) -> tuple[str, ...]:
    reasons: list[str] = []
    if facts.os_id != profile.os_id:
        reasons.append("OS_ID_MISMATCH")
    if facts.os_version != profile.os_version:
        reasons.append("OS_VERSION_MISMATCH")
    if facts.architecture != profile.architecture:
        reasons.append("ARCHITECTURE_MISMATCH")
    if facts.ros_distro != profile.ros_distro:
        reasons.append("ROS_DISTRO_MISMATCH")
    if profile.l4t_prefix and not (facts.l4t or "").startswith(profile.l4t_prefix):
        reasons.append("L4T_MISMATCH")
    if profile.jetpack_major is not None and not (facts.jetpack or "").split(".", 1)[0] == str(profile.jetpack_major):
        reasons.append("JETPACK_MISMATCH")
    return tuple(reasons)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a mapping")
    return value


def load_runtime_config(path: Path) -> RuntimeConfig:
    data = _load_yaml(path)
    if set(data) != set(_TOP_LEVEL_KEYS):
        missing = sorted(set(_TOP_LEVEL_KEYS) - set(data))
        unknown = sorted(set(data) - set(_TOP_LEVEL_KEYS))
        raise ConfigError(f"runtime config top-level keys mismatch: missing={missing}, unknown={unknown}")

    interfaces = _require_mapping(data["interfaces"], "interfaces")
    for key in _INTERFACE_KEYS:
        if key not in interfaces:
            raise ConfigError(f"interfaces.{key} is required")
        value = interfaces[key]
        if not isinstance(value, str) or not value.startswith("/"):
            raise ConfigError(f"interfaces.{key} must be an absolute ROS name")
    if set(interfaces) - set(_INTERFACE_KEYS):
        raise ConfigError("interfaces contains an unknown endpoint")
    values = tuple(interfaces.values())
    if len(set(values)) != len(values):
        raise ConfigError("duplicate interface endpoint")

    capabilities = _require_mapping(data["capabilities"], "capabilities")
    if set(capabilities) != {"platform_file", "observation_file"}:
        raise ConfigError("capabilities must contain platform_file and observation_file")
    for key, value in capabilities.items():
        if not isinstance(value, str) or not value.startswith("/"):
            raise ConfigError(f"capabilities.{key} must be an absolute path")

    planner = _require_mapping(data["planner"], "planner")
    forbidden = _FORBIDDEN_SAFETY_KEYS & set(planner)
    if forbidden:
        raise ConfigError(f"safety switch is not configurable: {sorted(forbidden)[0]}")
    if set(planner) != {"enable_nav2_adapter", "snapshot_policy"}:
        raise ConfigError("planner has unsupported configuration")
    if not isinstance(planner["enable_nav2_adapter"], bool):
        raise ConfigError("planner.enable_nav2_adapter must be boolean")
    snapshot_policy = _require_mapping(planner["snapshot_policy"], "planner.snapshot_policy")
    if set(snapshot_policy) != set(_SNAPSHOT_KEYS):
        raise ConfigError("planner.snapshot_policy keys mismatch")
    for key, value in snapshot_policy.items():
        if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"planner.snapshot_policy.{key} must be positive")
    policy = _require_mapping(data["policy"], "policy")
    if set(policy) != {"mode", "model_id"} or policy["mode"] not in {"fallback", "onnx", "tensorrt"}:
        raise ConfigError("policy.mode must be fallback, onnx, or tensorrt")
    if policy["mode"] == "fallback" and policy["model_id"] is not None:
        raise ConfigError("fallback mode cannot select a model")
    if policy["mode"] != "fallback" and not isinstance(policy["model_id"], str):
        raise ConfigError("model mode requires policy.model_id")

    extensions = _require_mapping(data["extensions"], "extensions")
    if set(extensions) != {"map_pipeline", "path_tracking"} or not all(isinstance(v, bool) for v in extensions.values()):
        raise ConfigError("extensions must contain boolean map_pipeline and path_tracking")
    input_adapters = _require_mapping(data["input_adapters"], "input_adapters")
    if set(input_adapters) != {"mode", "task3_config_file"}:
        raise ConfigError("input_adapters must contain mode and task3_config_file")
    mode = input_adapters["mode"]
    config_file = input_adapters["task3_config_file"]
    if mode not in {"external_canonical", "task3_adapted"}:
        raise ConfigError("input_adapters.mode must be external_canonical or task3_adapted")
    if mode == "external_canonical" and config_file is not None:
        raise ConfigError("external_canonical input_adapters.task3_config_file must be null")
    if mode == "task3_adapted" and (
        not isinstance(config_file, str) or not config_file.startswith("/")
    ):
        raise ConfigError("input_adapters.task3_config_file must be an absolute path for task3_adapted")
    controller = _require_mapping(data["controller"], "controller")
    if set(controller) != {"wheeled"}:
        raise ConfigError("controller must contain wheeled")
    wheeled = _require_mapping(controller["wheeled"], "controller.wheeled")
    if set(wheeled) != set(_WHEELED_CONTROLLER_KEYS):
        raise ConfigError("controller.wheeled keys mismatch")
    if not isinstance(wheeled["enabled"], bool):
        raise ConfigError("controller.wheeled.enabled must be boolean")
    for key in ("command_topic", "odometry_topic", "feedback_topic", "execution_goal_topic", "reference_topic"):
        if not isinstance(wheeled[key], str) or not wheeled[key].startswith("/"):
            raise ConfigError(f"controller.wheeled.{key} must be an absolute ROS name")
    for key in _WHEELED_CONTROLLER_KEYS[6:]:
        value = wheeled[key]
        if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"controller.wheeled.{key} must be positive")
    runtime = _require_mapping(data["runtime"], "runtime")
    if set(runtime) != {"log_level"} or not isinstance(runtime["log_level"], str):
        raise ConfigError("runtime.log_level is required")

    return RuntimeConfig(
        profile=str(data["profile"]),
        interfaces={key: str(interfaces[key]) for key in _INTERFACE_KEYS},
        capabilities={key: str(capabilities[key]) for key in capabilities},
        planner={"enable_nav2_adapter": planner["enable_nav2_adapter"], "snapshot_policy": dict(snapshot_policy)},
        policy=dict(policy),
        extensions={key: bool(value) for key, value in extensions.items()},
        input_adapters={"mode": str(mode), "task3_config_file": config_file},
        controller={"wheeled": dict(wheeled)},
        runtime=dict(runtime),
    )


def atomic_write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def render_planner_params(config: RuntimeConfig, output_path: Path) -> None:
    atomic_write_yaml(output_path, config.runtime_ros_parameters())


def load_allowlist(path: Path) -> Mapping[str, Any]:
    data = _load_yaml(path)
    expected = {
        "schema_version",
        "runtime_files",
        "bundle_templates",
        "profile_directory",
        "runtime_roots",
        "optional_roots",
        "excluded_path_components",
    }
    if set(data) != expected:
        raise ConfigError("runtime source allowlist keys mismatch")
    return data
