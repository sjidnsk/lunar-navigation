from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .bundle import BundleRequest, build_bundle
from .commands import CommandRunner, RuntimeRefusal, doctor_runtime, init_runtime, prepare_environment
from .config import ConfigError, load_runtime_config, load_profile
from .host import HostFacts
from .state import resolve_runtime_paths


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    payload: dict[str, object]


class SystemRunner:
    def is_installed(self, package: str) -> bool:
        return subprocess.run(("dpkg-query", "-W", "-f=${db:Status-Abbrev}", package), text=True, capture_output=True).stdout.startswith("ii")

    def run(self, command: tuple[str, ...]) -> None:
        subprocess.run(command, check=True)

    def package_versions(self, packages: tuple[str, ...]) -> dict[str, str]:
        versions: dict[str, str] = {}
        for package in packages:
            result = subprocess.run(("dpkg-query", "-W", "-f=${Version}", package), text=True, capture_output=True)
            if result.returncode == 0:
                versions[package] = result.stdout.strip()
        return versions


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _host_facts() -> HostFacts:
    values: dict[str, str] = {}
    os_release = Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip('"')
    return HostFacts(
        os_id=values.get("ID", "unknown"),
        os_version=values.get("VERSION_ID", "unknown"),
        architecture=platform.machine(),
        ros_distro=os.environ.get("ROS_DISTRO", ""),
    )


def _payload(plan: object, *, applied: bool | None = None) -> dict[str, object]:
    data = json.loads(json.dumps(asdict(plan)))
    if applied is not None:
        data["applied"] = applied
    return data


def run_cli(
    argv: Sequence[str],
    *,
    facts: HostFacts | None = None,
    runner: CommandRunner | None = None,
    home: Path | None = None,
    repo_root: Path | None = None,
) -> CliResult:
    root = repo_root or _repo_root()
    host = facts or _host_facts()
    commands = runner or SystemRunner()
    parser = argparse.ArgumentParser(prog="luna", add_help=False)
    parser.add_argument("command", choices=("init", "prepare", "doctor", "config", "bundle"))
    parser.add_argument("--profile")
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--output")
    parser.add_argument("check", nargs="?")
    try:
        args = parser.parse_args(list(argv))
        if args.command == "bundle":
            if not args.target or not args.output:
                return CliResult(2, {"reason": "BUNDLE_TARGET_AND_OUTPUT_REQUIRED"})
            result = build_bundle(BundleRequest(root, "HEAD", args.target, Path(args.output)))
            return CliResult(0, {"archive": str(result.archive), "source_commit": result.source_commit})
        if args.command == "init":
            if not args.profile:
                return CliResult(2, {"reason": "PROFILE_REQUIRED"})
            profile = load_profile(args.profile, root)
            reasons = list(__import__("deployment.luna_runtime.config", fromlist=["validate_host"]).validate_host(profile, host))
            if reasons:
                return CliResult(3, {"reasons": reasons})
            paths = resolve_runtime_paths("dev", home=home)
            init_runtime(paths, args.profile, root)
            return CliResult(0, {"config": str(paths.config), "profile": args.profile})

        config_path = Path(args.config) if args.config else resolve_runtime_paths("dev", home=home).config
        config = load_runtime_config(config_path)
        if args.command == "config":
            if args.check != "check":
                return CliResult(2, {"reason": "CONFIG_SUBCOMMAND_REQUIRED"})
            return CliResult(0, {"config": str(config_path), "profile": config.profile})
        if args.command == "doctor":
            plan = doctor_runtime(config, host, commands, root)
            return CliResult(3 if plan.reasons else 0, _payload(plan))
        if args.command == "prepare":
            if args.dry_run == args.apply:
                return CliResult(2, {"reason": "SELECT_PREPARE_MODE"})
            result = prepare_environment(
                config,
                resolve_runtime_paths("dev", home=home),
                host,
                apply=args.apply,
                confirmed=args.yes,
                runner=commands,
                repo_root=root,
            )
            return CliResult(0 if result.applied or not result.plan.reasons else 3, _payload(result.plan, applied=result.applied))
    except (ConfigError, RuntimeRefusal) as error:
        return CliResult(2, {"reason": str(error)})
    return CliResult(2, {"reason": "COMMAND_NOT_IMPLEMENTED"})


def main(argv: Sequence[str] | None = None) -> int:
    result = run_cli(sys.argv[1:] if argv is None else argv)
    print(json.dumps(result.payload, ensure_ascii=False, sort_keys=True))
    return result.exit_code
