#!/usr/bin/env python3
"""Run and strictly accept the isolated 300 m ROS 2 Jazzy exploration demo."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO


RUN_ROOT = Path(
    "/home/kai/CodexDownloads/lunar_navigation/exploration_jazzy_runs"
)
REQUIRED_EXECUTABLES = {
    "lunar_pure_exploration_sim": {
        "simulation_node",
        "run_coordinator",
        "run_recorder",
        "simulation_hud_node",
    },
    "lunar_pure_planner_ros": {"lunar_pure_planner_node"},
    "lunar_pure_exploration_ros": {"pure_exploration_node"},
    "lunar_pure_wheeled_controller": {
        "lunar_pure_wheeled_controller_node.py"
    },
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_LAUNCH_RELATIVE = Path("launch/jazzy_300m_exploration_sim.launch.py")


class AcceptanceError(RuntimeError):
    """The run did not produce exact full-exploration acceptance evidence."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time_ticks: int
    pgid: int
    session_id: int


@dataclass
class ExactProcessGroup:
    pgid: int
    session_id: int
    leader: ProcessIdentity | None
    members: set[ProcessIdentity] = field(default_factory=set)


@dataclass(frozen=True)
class GroupTeardownResult:
    sigterm_sent: bool
    recorded_members: tuple[ProcessIdentity, ...]


@dataclass
class DomainReservation:
    domain_id: int
    environment: dict[str, str]
    lock: TextIO

    def close(self) -> None:
        self.lock.close()


def read_process_identity(pid: int) -> ProcessIdentity | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    command_end = stat.rfind(")")
    if command_end < 0:
        return None
    fields = stat[command_end + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return ProcessIdentity(
            pid=pid,
            start_time_ticks=int(fields[19]),
            pgid=int(fields[2]),
            session_id=int(fields[3]),
        )
    except ValueError:
        return None


def fallback_new_session_group(process: subprocess.Popen[bytes]) -> ExactProcessGroup:
    return ExactProcessGroup(process.pid, process.pid, None)


def observe_exact_group_members(group: ExactProcessGroup) -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        identity = read_process_identity(int(entry.name))
        if (
            identity is not None
            and identity.pgid == group.pgid
            and identity.session_id == group.session_id
        ):
            group.members.add(identity)


def capture_exact_process_group(
    process: subprocess.Popen[bytes],
) -> ExactProcessGroup:
    leader = read_process_identity(process.pid)
    if leader is None:
        raise RuntimeError("launch leader exited before process identity capture")
    pgid = os.getpgid(process.pid)
    if leader.pgid != pgid or leader.session_id != pgid:
        raise RuntimeError("launch did not create the expected new session/process group")
    group = ExactProcessGroup(pgid, leader.session_id, leader, {leader})
    observe_exact_group_members(group)
    return group


def identity_is_alive(identity: ProcessIdentity) -> bool:
    return read_process_identity(identity.pid) == identity


def remaining_exact_group_members(group: ExactProcessGroup) -> set[ProcessIdentity]:
    observe_exact_group_members(group)
    return {identity for identity in group.members if identity_is_alive(identity)}


def wait_for_exact_group_exit(
    process: subprocess.Popen[bytes], group: ExactProcessGroup, timeout: float
) -> set[ProcessIdentity]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        remaining = remaining_exact_group_members(group)
        if not remaining:
            return set()
        time.sleep(0.02)
    process.poll()
    return remaining_exact_group_members(group)


def terminate_exact_process_group(
    process: subprocess.Popen[bytes],
    group: ExactProcessGroup,
    *,
    interrupt_timeout: float = 10.0,
    terminate_timeout: float = 5.0,
) -> GroupTeardownResult:
    observe_exact_group_members(group)
    if remaining_exact_group_members(group):
        try:
            os.killpg(group.pgid, signal.SIGINT)
        except ProcessLookupError:
            pass
    remaining = wait_for_exact_group_exit(process, group, interrupt_timeout)
    sigterm_sent = bool(remaining)
    if remaining:
        try:
            os.killpg(group.pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        remaining = wait_for_exact_group_exit(process, group, terminate_timeout)
    if remaining:
        details = ", ".join(
            f"pid={item.pid}/start={item.start_time_ticks}"
            for item in sorted(remaining, key=lambda item: item.pid)
        )
        raise RuntimeError(
            f"exact launch process group survived bounded teardown: {details}"
        )
    return GroupTeardownResult(
        sigterm_sent,
        tuple(sorted(group.members, key=lambda item: item.pid)),
    )


def _run_cli(command: list[str], env: dict[str, str], timeout: float):
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def reserve_empty_domain() -> DomainReservation:
    first = 20 + ((os.getpid() * 37 + time.time_ns()) % 180)
    for offset in range(180):
        candidate = 20 + ((first - 20 + offset) % 180)
        lock = Path(f"/tmp/lunar_jazzy_operator_domain_{candidate}.lock").open(
            "a+", encoding="utf-8"
        )
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        env = os.environ.copy()
        env.update(
            {
                "ROS_DOMAIN_ID": str(candidate),
                "ROS_LOCALHOST_ONLY": "1",
                "ROS2CLI_NO_DAEMON": "1",
            }
        )
        try:
            probe = _run_cli(
                ["ros2", "node", "list", "--no-daemon"], env, timeout=3.0
            )
        except subprocess.TimeoutExpired:
            lock.close()
            continue
        if probe.returncode == 0 and not probe.stdout.strip():
            return DomainReservation(candidate, env, lock)
        lock.close()
    raise AcceptanceError("no empty candidate ROS_DOMAIN_ID passed preflight")


def wait_for_empty_graph(env: dict[str, str], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            probe = _run_cli(
                ["ros2", "node", "list", "--no-daemon"], env, timeout=2.0
            )
        except subprocess.TimeoutExpired:
            last = "node-list probe timed out"
            continue
        last = f"rc={probe.returncode}; stdout={probe.stdout!r}; stderr={probe.stderr!r}"
        if probe.returncode == 0 and not probe.stdout.strip():
            return
        time.sleep(0.1)
    raise AcceptanceError(f"selected ROS graph did not drain: {last}")


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise AcceptanceError(f"{label} must be positive")
    return number


def validate_results(output_dir: Path) -> dict[str, object]:
    paths = {
        name: output_dir / name
        for name in ("summary.json", "coverage.csv", "trajectory.csv")
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise AcceptanceError(f"missing or empty {name}")
    try:
        summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"invalid summary.json: {error}") from error
    if not isinstance(summary, dict):
        raise AcceptanceError("summary.json root must be an object")
    if summary.get("success") is not True:
        raise AcceptanceError("summary success is not true")
    if summary.get("terminal_state") != "COMPLETED":
        raise AcceptanceError("terminal state is not COMPLETED")
    if summary.get("reason_code") != "COMPLETED_NO_REACHABLE_FRONTIER":
        raise AcceptanceError("completion reason is not no reachable frontier")
    coverage = summary.get("coverage_ratio")
    if coverage != summary.get("status_coverage_ratio"):
        raise AcceptanceError("summary and status coverage are not exactly equal")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise AcceptanceError("coverage is not numeric")
    _positive_number(summary.get("distance_m"), "distance_m")
    _positive_number(summary.get("completed_goal_count"), "completed_goal_count")
    planner = summary.get("planner")
    if not isinstance(planner, dict):
        raise AcceptanceError("planner summary is missing")
    for stage in ("global", "local"):
        stage_summary = planner.get(stage)
        if not isinstance(stage_summary, dict):
            raise AcceptanceError(f"{stage} planner summary is missing")
        _positive_number(stage_summary.get("call_count"), f"{stage} planner call_count")
        _positive_number(
            stage_summary.get("total_elapsed_ms"),
            f"{stage} planner total_elapsed_ms",
        )
    try:
        with paths["coverage.csv"].open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as error:
        raise AcceptanceError(f"invalid coverage.csv: {error}") from error
    if not rows:
        raise AcceptanceError("coverage.csv has no data rows")
    try:
        csv_coverage = float(rows[-1]["coverage_ratio"])
    except (KeyError, TypeError, ValueError) as error:
        raise AcceptanceError("coverage.csv final coverage is invalid") from error
    if csv_coverage != coverage:
        raise AcceptanceError("CSV and summary coverage are not exactly equal")
    with paths["trajectory.csv"].open("r", encoding="utf-8") as stream:
        trajectory_rows = sum(1 for _ in stream)
    if trajectory_rows < 2:
        raise AcceptanceError("trajectory.csv has no data rows")
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _cmake_cache_value(cache_path: Path, key: str) -> str:
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise AcceptanceError(f"required CMakeCache is unavailable: {cache_path}") from error
    matches: list[str] = []
    for line in lines:
        name_and_type, separator, value = line.partition("=")
        if not separator:
            continue
        name, type_separator, _value_type = name_and_type.partition(":")
        if type_separator and name == key:
            matches.append(value)
    if len(matches) != 1:
        raise AcceptanceError(
            f"CMakeCache must contain exactly one {key}: {cache_path}"
        )
    return matches[0]


def verify_installation(
    start_rviz: bool,
    *,
    env: dict[str, str] | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    command_env = os.environ.copy() if env is None else env.copy()
    if command_env.get("ROS_DISTRO") != "jazzy":
        raise AcceptanceError("ROS_DISTRO must be jazzy after sourcing the overlay")
    overlay_value = command_env.get("LUNAR_JAZZY_OVERLAY")
    if not overlay_value:
        raise AcceptanceError("LUNAR_JAZZY_OVERLAY must identify the explicit overlay setup")
    try:
        overlay_setup = Path(overlay_value).expanduser().resolve(strict=True)
    except OSError as error:
        raise AcceptanceError(
            f"explicit overlay setup is unavailable: {overlay_value}"
        ) from error
    if not overlay_setup.is_file():
        raise AcceptanceError(f"explicit overlay setup is not a file: {overlay_setup}")
    overlay_install_root = overlay_setup.parent.resolve()
    build_root = overlay_install_root.parent / "build"

    resolved_prefixes: dict[str, str] = {}
    package_build_types: dict[str, str] = {}
    executable_evidence: dict[str, dict[str, dict[str, str]]] = {}
    simulation_prefix: Path | None = None
    for package, expected in REQUIRED_EXECUTABLES.items():
        prefix = _run_cli(["ros2", "pkg", "prefix", package], command_env, 5.0)
        if prefix.returncode != 0:
            raise AcceptanceError(f"required installed package is unavailable: {package}")
        prefix_text = prefix.stdout.strip()
        if not prefix_text or len(prefix_text.splitlines()) != 1:
            raise AcceptanceError(f"required package prefix is invalid: {package}")
        try:
            resolved_prefix = Path(prefix_text).expanduser().resolve(strict=True)
        except OSError as error:
            raise AcceptanceError(
                f"required package prefix is unavailable: {package}: {prefix_text}"
            ) from error
        if not resolved_prefix.is_dir() or not _is_within(
            resolved_prefix, overlay_install_root
        ):
            raise AcceptanceError(
                f"required package resolved outside explicit overlay: "
                f"{package}: {resolved_prefix}"
            )
        resolved_prefixes[package] = str(resolved_prefix)
        if package == "lunar_pure_exploration_sim":
            simulation_prefix = resolved_prefix

        cache_path = build_root / package / "CMakeCache.txt"
        build_type = _cmake_cache_value(cache_path, "CMAKE_BUILD_TYPE")
        cached_install_prefix = _cmake_cache_value(
            cache_path, "CMAKE_INSTALL_PREFIX"
        )
        try:
            cached_prefix = Path(cached_install_prefix).expanduser().resolve(strict=True)
        except OSError as error:
            raise AcceptanceError(
                f"CMakeCache install prefix is unavailable for {package}: "
                f"{cached_install_prefix}"
            ) from error
        if cached_prefix != resolved_prefix:
            raise AcceptanceError(
                f"CMakeCache install prefix mismatch for {package}: "
                f"cache={cached_prefix}; resolved={resolved_prefix}"
            )
        package_build_types[package] = build_type

        listed = _run_cli(
            ["ros2", "pkg", "executables", package], command_env, 5.0
        )
        available = {
            line.split(maxsplit=1)[1]
            for line in listed.stdout.splitlines()
            if len(line.split(maxsplit=1)) == 2
        }
        missing = expected - available
        if listed.returncode != 0 or missing:
            raise AcceptanceError(
                f"required executables are unavailable in {package}: {sorted(missing)}"
            )
        package_executables: dict[str, dict[str, str]] = {}
        for executable in sorted(expected):
            executable_path = (
                resolved_prefix / "lib" / package / executable
            )
            try:
                resolved_executable = executable_path.resolve(strict=True)
            except OSError as error:
                raise AcceptanceError(
                    f"required executable path is unavailable: {executable_path}"
                ) from error
            if not resolved_executable.is_file() or not os.access(
                resolved_executable, os.X_OK
            ):
                raise AcceptanceError(
                    f"required executable is not executable: {resolved_executable}"
                )
            if not _is_within(resolved_executable, resolved_prefix):
                raise AcceptanceError(
                    f"required executable resolved outside package prefix: "
                    f"{resolved_executable}"
                )
            package_executables[executable] = {
                "path": str(resolved_executable),
                "sha256": _sha256(resolved_executable),
            }
        executable_evidence[package] = package_executables

    assert simulation_prefix is not None
    source_launch = (repository_root / SIMULATION_LAUNCH_RELATIVE).resolve()
    installed_launch = (
        simulation_prefix
        / "share/lunar_pure_exploration_sim/launch"
        / SIMULATION_LAUNCH_RELATIVE.name
    ).resolve()
    if not source_launch.is_file():
        raise AcceptanceError(f"source launch file is unavailable: {source_launch}")
    if not installed_launch.is_file():
        raise AcceptanceError(f"installed launch file is unavailable: {installed_launch}")
    source_launch_hash = _sha256(source_launch)
    installed_launch_hash = _sha256(installed_launch)
    if source_launch_hash != installed_launch_hash:
        raise AcceptanceError(
            "source and installed launch file hash mismatch: "
            f"source={source_launch_hash}; installed={installed_launch_hash}"
        )

    git_result = _run_cli(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        command_env,
        5.0,
    )
    git_sha = git_result.stdout.strip()
    if git_result.returncode != 0 or len(git_sha) != 40:
        raise AcceptanceError("current worktree git SHA is unavailable")

    unique_build_types = set(package_build_types.values())
    if len(unique_build_types) != 1:
        raise AcceptanceError(
            f"required packages have inconsistent CMAKE_BUILD_TYPE values: "
            f"{package_build_types}"
        )
    if start_rviz:
        rviz = _run_cli(
            ["ros2", "pkg", "executables", "rviz2"], command_env, 5.0
        )
        if rviz.returncode != 0 or "rviz2 rviz2" not in rviz.stdout:
            raise AcceptanceError("rviz2 executable is unavailable")
    return {
        "overlay_setup": str(overlay_setup),
        "overlay_install_root": str(overlay_install_root),
        "resolved_package_prefixes": resolved_prefixes,
        "cmake_build_type": unique_build_types.pop(),
        "package_build_types": package_build_types,
        "worktree_git_sha": git_sha,
        "launch_files": {
            "source": {"path": str(source_launch), "sha256": source_launch_hash},
            "installed": {
                "path": str(installed_launch),
                "sha256": installed_launch_hash,
            },
        },
        "executables": executable_evidence,
    }


def make_run_directory(seed: int) -> tuple[Path, Path, Path]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUN_ROOT / f"run-{stamp}-seed-{seed}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir()
    output_dir = run_dir / "results"
    ros_log_dir = run_dir / "ros-logs"
    output_dir.mkdir()
    ros_log_dir.mkdir()
    return run_dir, output_dir, ros_log_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=21600.0)
    args = parser.parse_args(argv)
    if args.seed < 0 or not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error("seed must be nonnegative and max wall seconds must be positive")
    return args


def run(argv: list[str]) -> int:
    args = parse_args(argv)
    start_rviz = not args.no_rviz
    installation_identity = verify_installation(start_rviz)
    run_dir, output_dir, ros_log_dir = make_run_directory(args.seed)
    reservation = reserve_empty_domain()
    env = reservation.environment.copy()
    env["ROS_LOG_DIR"] = str(ros_log_dir)
    launch_log_path = run_dir / "launch.log"
    evidence_path = run_dir / "operator_evidence.json"
    interrupted = False

    def on_signal(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    previous_handlers = {
        number: signal.signal(number, on_signal)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    process: subprocess.Popen[bytes] | None = None
    group: ExactProcessGroup | None = None
    teardown: GroupTeardownResult | None = None
    summary: dict[str, object] | None = None
    outcome = "RUNNING"
    started = time.monotonic()
    rviz_argument = "start_rviz:=true" if start_rviz else "start_rviz:=false"
    try:
        with launch_log_path.open("wb") as launch_log:
            process = subprocess.Popen(
                [
                    "ros2",
                    "launch",
                    "lunar_pure_exploration_sim",
                    "jazzy_300m_exploration_sim.launch.py",
                    rviz_argument,
                    f"seed:={args.seed}",
                    "speed_multiplier:=20.0",
                    f"output_dir:={output_dir}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=launch_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            group = fallback_new_session_group(process)
            try:
                group = capture_exact_process_group(process)
                deadline = started + args.max_wall_seconds
                summary_path = output_dir / "summary.json"
                while True:
                    if interrupted:
                        outcome = "OPERATOR_INTERRUPT"
                        raise AcceptanceError("operator interrupt is not completion")
                    if summary_path.is_file():
                        summary = validate_results(output_dir)
                        outcome = "COMPLETED"
                        break
                    return_code = process.poll()
                    if return_code is not None:
                        outcome = "LAUNCH_EXIT"
                        raise AcceptanceError(
                            f"launch exited with rc={return_code} without a valid summary"
                        )
                    if time.monotonic() >= deadline:
                        outcome = "WALL_GUARD"
                        raise AcceptanceError("wall guard expired before completion")
                    time.sleep(0.2)
            finally:
                teardown = terminate_exact_process_group(process, group)
                wait_for_empty_graph(env)
    finally:
        elapsed = time.monotonic() - started
        evidence = {
            **installation_identity,
            "run_dir": str(run_dir),
            "domain_id": reservation.domain_id,
            "rviz_requested": start_rviz,
            "outcome": outcome,
            "operator_wall_elapsed_s": elapsed,
            "wall_guard_s": args.max_wall_seconds,
            "wall_guard_triggered": outcome == "WALL_GUARD",
            "summary_validated": summary is not None,
            "teardown": None
            if teardown is None
            else {
                "sigterm_sent": teardown.sigterm_sent,
                "recorded_members": [
                    {
                        "pid": item.pid,
                        "start_time_ticks": item.start_time_ticks,
                        "pgid": item.pgid,
                        "session_id": item.session_id,
                    }
                    for item in teardown.recorded_members
                ],
                "exact_members_gone": True,
                "ros_graph_empty": True,
            },
        }
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reservation.close()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    assert summary is not None
    print(f"RUN_DIR={run_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run(sys.argv[1:])
    except (AcceptanceError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"exploration acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
