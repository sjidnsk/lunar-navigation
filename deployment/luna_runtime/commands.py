from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import yaml

from .config import RuntimeConfig, TargetProfile, load_profile, validate_host
from .environment import EnvironmentLock, load_environment_lock
from .host import HostFacts
from .state import RuntimePaths, atomic_write_json


class CommandRunner(Protocol):
    def is_installed(self, package: str) -> bool: ...

    def run(self, command: tuple[str, ...]) -> None: ...

    def package_versions(self, packages: tuple[str, ...]) -> dict[str, str]: ...


class RuntimeRefusal(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparePlan:
    profile_id: str
    environment_lock_sha256: str
    missing_apt_packages: tuple[str, ...]
    optional_groups: dict[str, tuple[str, ...]]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PrepareResult:
    plan: PreparePlan
    applied: bool


def make_prepare_plan(
    lock: EnvironmentLock,
    profile: TargetProfile,
    config: RuntimeConfig,
    facts: HostFacts,
    runner: CommandRunner,
) -> PreparePlan:
    reasons = list(validate_host(profile, facts))
    if reasons:
        return PreparePlan(
            profile_id=profile.id,
            environment_lock_sha256=lock.sha256,
            missing_apt_packages=(),
            optional_groups={},
            reasons=tuple(reasons),
        )
    missing = tuple(package for package in lock.required_apt_packages if not runner.is_installed(package))
    if missing:
        reasons.append("MISSING_APT_PACKAGES")
    optional = dict(lock.optional_apt_groups) if config.planner["enable_nav2_adapter"] else {}
    return PreparePlan(
        profile_id=profile.id,
        environment_lock_sha256=lock.sha256,
        missing_apt_packages=missing,
        optional_groups=optional,
        reasons=tuple(reasons),
    )


def write_environment_manifest(
    paths: RuntimePaths,
    lock: EnvironmentLock,
    facts: HostFacts,
    runner: CommandRunner,
) -> None:
    versions = runner.package_versions(lock.required_apt_packages)
    atomic_write_json(
        paths.data / "environment-manifest.json",
        {
            "schema_version": "luna-environment-manifest/v1",
            "source_commit": "development-tree",
            "profile_id": lock.profile_id,
            "environment_lock_sha256": lock.sha256,
            "host": asdict(facts),
            "ros_prefix": f"/opt/ros/{lock.ros_distro}",
            "apt_packages": [
                {"name": name, "version": version} for name, version in sorted(versions.items())
            ],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def prepare_environment(
    config: RuntimeConfig,
    paths: RuntimePaths,
    facts: HostFacts,
    *,
    apply: bool,
    confirmed: bool,
    runner: CommandRunner,
    repo_root: Path,
) -> PrepareResult:
    profile = load_profile(config.profile, repo_root)
    lock = load_environment_lock(config.profile, repo_root)
    plan = make_prepare_plan(lock, profile, config, facts, runner)
    host_reasons = validate_host(profile, facts)
    if not apply:
        return PrepareResult(plan=plan, applied=False)
    if host_reasons or not confirmed:
        raise RuntimeRefusal("PREPARE_REFUSED")
    if plan.missing_apt_packages:
        runner.run(("sudo", "apt-get", "update"))
        runner.run(("sudo", "apt-get", "install", "--no-install-recommends", *plan.missing_apt_packages))
    write_environment_manifest(paths, lock, facts, runner)
    return PrepareResult(plan=plan, applied=True)


def init_runtime(paths: RuntimePaths, profile_id: str, repo_root: Path) -> None:
    if paths.config.exists():
        raise RuntimeRefusal("CONFIG_EXISTS")
    template = yaml.safe_load(
        (repo_root / "deployment" / "config" / "runtime.default.yaml").read_text(encoding="utf-8")
    )
    template["profile"] = profile_id
    paths.config.parent.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    atomic_write_json(paths.data / "state.json", {"schema_version": "luna-runtime-state/v1", "profile_id": profile_id})


def doctor_runtime(config: RuntimeConfig, facts: HostFacts, runner: CommandRunner, repo_root: Path) -> PreparePlan:
    profile = load_profile(config.profile, repo_root)
    lock = load_environment_lock(config.profile, repo_root)
    return make_prepare_plan(lock, profile, config, facts, runner)
