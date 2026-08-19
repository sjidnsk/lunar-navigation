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
from .build import build_runtime, make_build_plan
from .commands import (
    CommandRunner,
    RuntimeRefusal,
    doctor_runtime,
    extension_states,
    init_runtime,
    prepare_environment,
    set_extension_enabled,
)
from .config import ConfigError, load_runtime_config, load_profile
from .host import HostFacts
from .process import read_runtime_status, start_runtime, stop_runtime, tail_log
from .state import resolve_runtime_paths
from .model_store import ModelInstallError, ModelStore, model_store_root


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

    def popen(self, command: tuple[str, ...]):
        process = subprocess.Popen(command, start_new_session=True)
        fingerprint = Path(f"/proc/{process.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8").strip()
        return type("SystemChild", (), {
            "pid": process.pid,
            "fingerprint": fingerprint,
            "terminate": process.terminate,
        })()

    def lifecycle_state(self, node_name: str) -> str:
        result = subprocess.run(("ros2", "lifecycle", "get", node_name), text=True, capture_output=True, check=True)
        return result.stdout.strip().split()[-1]

    def process_matches(self, pid: int, fingerprint: str) -> bool:
        try:
            return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8").strip() == fingerprint
        except OSError:
            return False

    def terminate(self, pid: int, fingerprint: str) -> None:
        if self.process_matches(pid, fingerprint):
            os.kill(pid, 15)


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
    parser.add_argument("command", choices=("init", "prepare", "doctor", "config", "bundle", "build", "start", "stop", "status", "logs", "model", "extension"))
    parser.add_argument("--profile")
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--output")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("check", nargs="?")
    parser.add_argument("argument", nargs="?")
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
        paths = resolve_runtime_paths("dev", home=home)
        if args.command == "model":
            store = ModelStore(model_store_root(config_path, paths.data))
            if args.check == "install" and args.argument:
                installed = store.install(Path(args.argument))
                return CliResult(0, {"model_id": installed.model_id, "model_sha256": installed.model_sha256, "model_binding": "staged_not_connected"})
            if args.check == "activate" and args.argument:
                return CliResult(0, asdict(store.activate(args.argument)))
            if args.check == "rollback" and args.argument is None:
                return CliResult(0, asdict(store.rollback()))
            if args.check == "status" and args.argument is None:
                return CliResult(0, asdict(store.status()))
            return CliResult(2, {"reason": "MODEL_SUBCOMMAND_REQUIRED"})
        if args.command == "extension":
            if args.check in (None, "list") and args.argument is None:
                return CliResult(0, {key: asdict(value) for key, value in extension_states(config, commands).items()})
            if args.check in ("enable", "disable") and args.argument:
                states = set_extension_enabled(config_path, config, commands, args.argument, args.check == "enable")
                return CliResult(0, {key: asdict(value) for key, value in states.items()})
            return CliResult(2, {"reason": "EXTENSION_SUBCOMMAND_REQUIRED"})
        if args.command == "build":
            build_runtime(make_build_plan(config, paths, root), commands)
            return CliResult(0, {"build_base": str(paths.data / "build")})
        if args.command == "start":
            return CliResult(0, start_runtime(config, paths, commands))
        if args.command == "stop":
            return CliResult(0, stop_runtime(paths, commands))
        if args.command == "status":
            return CliResult(0, read_runtime_status(paths, commands))
        if args.command == "logs":
            return CliResult(0, {"log_directory": str(tail_log(paths))})
        if args.command == "config":
            if args.check != "check":
                return CliResult(2, {"reason": "CONFIG_SUBCOMMAND_REQUIRED"})
            return CliResult(0, {"config": str(config_path), "profile": config.profile})
        if args.command == "doctor":
            plan = doctor_runtime(config, host, commands, root)
            payload = _payload(plan)
            if args.live:
                payload["live_status"] = "WAITING_FOR_EXTERNAL_INPUT" if not plan.reasons else "HOST_OR_DEPENDENCY_UNREADY"
            return CliResult(3 if plan.reasons else 0, payload)
        if args.command == "prepare":
            if args.dry_run == args.apply:
                return CliResult(2, {"reason": "SELECT_PREPARE_MODE"})
            result = prepare_environment(
                config,
                paths,
                host,
                apply=args.apply,
                confirmed=args.yes,
                runner=commands,
                repo_root=root,
            )
            return CliResult(0 if result.applied or not result.plan.reasons else 3, _payload(result.plan, applied=result.applied))
    except (ConfigError, RuntimeRefusal, ModelInstallError) as error:
        return CliResult(2, {"reason": str(error)})
    return CliResult(2, {"reason": "COMMAND_NOT_IMPLEMENTED"})


def main(argv: Sequence[str] | None = None) -> int:
    result = run_cli(sys.argv[1:] if argv is None else argv)
    print(json.dumps(result.payload, ensure_ascii=False, sort_keys=True))
    return result.exit_code
