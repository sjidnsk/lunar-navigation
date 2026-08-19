from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Protocol

import yaml

from .config import ConfigError, TargetProfile, _repo_root


class EnvironmentLockError(ValueError):
    pass


class RosdepResolver(Protocol):
    def resolve(self, roots: tuple[str, ...]) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class EnvironmentLock:
    sha256: str
    profile_id: str
    os_id: str
    os_version: str
    architecture: str
    ros_distro: str
    required_apt_packages: tuple[str, ...]
    optional_apt_groups: Mapping[str, tuple[str, ...]]
    rosdep_source_roots: tuple[str, ...]
    l4t_prefix: str | None
    jetpack_major: int | None
    fallback_required_executables: tuple[str, ...]
    model_required_executables: tuple[str, ...]


def load_environment_lock(profile_id: str, repo_root: Path | None = None) -> EnvironmentLock:
    root = repo_root or _repo_root()
    path = root / "deployment" / "profiles" / f"{profile_id}.environment.yaml"
    try:
        raw = path.read_bytes()
        data = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise EnvironmentLockError(f"cannot read environment lock {path}: {error}") from error
    if not isinstance(data, dict):
        raise EnvironmentLockError("environment lock must be a mapping")
    expected = {
        "schema_version", "profile_id", "host", "required_apt_packages", "optional_apt_groups",
        "rosdep_source_roots", "fallback_required_executables", "model_required_executables",
    }
    if set(data) != expected or data["profile_id"] != profile_id:
        raise EnvironmentLockError("environment lock schema mismatch")
    host = data["host"]
    if not isinstance(host, dict):
        raise EnvironmentLockError("environment lock host is invalid")
    packages = tuple(data["required_apt_packages"])
    if packages != tuple(sorted(packages)) or len(packages) != len(set(packages)):
        raise EnvironmentLockError("required apt packages must be unique and sorted")
    optional = {name: tuple(values) for name, values in data["optional_apt_groups"].items()}
    return EnvironmentLock(
        sha256=sha256(raw).hexdigest(),
        profile_id=profile_id,
        os_id=str(host["os_id"]),
        os_version=str(host["os_version"]),
        architecture=str(host["architecture"]),
        ros_distro=str(host["ros_distro"]),
        required_apt_packages=packages,
        optional_apt_groups=optional,
        rosdep_source_roots=tuple(data["rosdep_source_roots"]),
        l4t_prefix=None if host.get("l4t_prefix") is None else str(host["l4t_prefix"]),
        jetpack_major=None if host.get("jetpack_major") is None else int(host["jetpack_major"]),
        fallback_required_executables=tuple(data["fallback_required_executables"]),
        model_required_executables=tuple(data["model_required_executables"]),
    )


def verify_environment_lock(repo_root: Path, profile: TargetProfile, resolver: RosdepResolver) -> None:
    lock = load_environment_lock(profile.id, repo_root)
    if (lock.os_id, lock.os_version, lock.architecture, lock.ros_distro) != (
        profile.os_id, profile.os_version, profile.architecture, profile.ros_distro,
    ):
        raise EnvironmentLockError("ENVIRONMENT_LOCK_PROFILE_MISMATCH")
    if lock.l4t_prefix != profile.l4t_prefix or lock.jetpack_major != profile.jetpack_major:
        raise EnvironmentLockError("ENVIRONMENT_LOCK_PROFILE_MISMATCH")
    resolved = set(resolver.resolve(lock.rosdep_source_roots))
    missing = sorted(resolved - set(lock.required_apt_packages))
    if missing:
        raise EnvironmentLockError(f"DEPENDENCY_LOCK_DRIFT: missing {', '.join(missing)}")
