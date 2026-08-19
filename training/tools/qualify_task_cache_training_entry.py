#!/usr/bin/env python3
"""Run the fixed task-cache suites and three task-level entry smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lunar_policy_training"
sys.path.insert(0, str(PACKAGE_ROOT))

from lunar_policy_training.polar_data.formal_cache import (  # noqa: E402
    FORMAL_CACHE_SCHEMA,
    GENERATOR_SHA256,
    load_formal_cache,
)
from lunar_policy_training.reward import reward_weights_sha256  # noqa: E402
from lunar_policy_training.training_semantics import (  # noqa: E402
    training_semantics_sha256,
)
import lunar_policy_training.cli as training_cli  # noqa: E402


SCHEMA_VERSION = "lunar-reward-v4-focused-qualification/v1"
TASK_RANGE = {"first": 1, "last": 11}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_RESULT_FIELDS = frozenset(
    (
        "name",
        "passed",
        "failed",
        "errors",
        "skipped",
        "elapsed_seconds",
        "command",
        "log_sha256",
    )
)
_FOCUSED_TARGETS = (
    "training/lunar_policy_training/tests/test_reward_v4_sentinel_boundary.py",
    "training/lunar_policy_training/tests/test_formal_ground_endpoint_authority.py",
    "training/lunar_policy_training/tests/"
    "test_reachable_constrained_ground_candidates.py",
    "training/lunar_policy_training/tests/test_formal_preflight.py",
)
_SHORT_SMOKES = (
    (
        "WHEELED",
        "training/lunar_policy_training/tests/"
        "test_reachable_constrained_ground_candidates.py::"
        "test_ground_frontier_qualifies_all_strip_witnesses_but_emits_three_actions",
    ),
    (
        "LEGGED",
        "training/lunar_policy_training/tests/test_formal_preflight.py::"
        "test_direct_checks_use_exact_common_training_world",
    ),
    (
        "HOPPER",
        "training/lunar_policy_training/tests/test_hopper_macro_step.py::"
        "test_hopper_resumes_policy_only_after_landed_hold",
    ),
)
_LIVE_REPAIR_FOCUSED_NODES = (
    (
        "reward-v4-curriculum",
        "training/lunar_policy_training/tests/"
        "test_reward_v4_curriculum.py::"
        "test_two_runtime_evaluations_advance_ground_stage_and_restore",
    ),
    (
        "pre-motion-suppression",
        "training/lunar_policy_training/tests/test_collector.py::"
        "test_pre_motion_suppression_reselects_without_journal_slot",
    ),
    (
        "evaluation-crash-candidate-first",
        "training/lunar_policy_training/tests/"
        "test_reward_v4_update_recovery.py::"
        "test_evaluation_fault_window_replays_one_atomic_decision"
        "[candidate_before_evaluation_report]",
    ),
    (
        "evaluation-crash-report-first",
        "training/lunar_policy_training/tests/"
        "test_reward_v4_update_recovery.py::"
        "test_evaluation_fault_window_replays_one_atomic_decision"
        "[evaluation_report_before_checkpoint]",
    ),
    (
        "runtime-diagnostics",
        "training/lunar_policy_training/tests/test_training_metrics.py::"
        "test_reward_v4_runtime_diagnostics_are_identity_bound_and_not_model_inputs",
    ),
    (
        "wheeled-request-latency",
        "training/lunar_policy_training/tests/test_formal_builder.py::"
        "test_frozen_wheeled_single_planner_request_under_one_second",
    ),
    (
        "legged-request-latency",
        "training/lunar_policy_training/tests/test_formal_builder.py::"
        "test_frozen_legged_single_planner_request_under_one_second",
    ),
)
_LIVE_REPAIR_SHORT_SMOKES = (
    (
        "WHEELED",
        "training/lunar_policy_training/tests/test_formal_builder.py::"
        "test_frozen_wheeled_coarse_fine_snapshot",
    ),
    (
        "LEGGED",
        "training/lunar_policy_training/tests/test_collector.py::"
        "test_frozen_legged_unsafe_start_snapshot",
    ),
    (
        "HOPPER",
        "training/lunar_policy_training/tests/test_hopper_macro_step.py::"
        "test_prepared_hopper_action_aggregates_until_landed_hold",
    ),
)


class TaskCacheQualificationError(ValueError):
    """Focused qualification cannot authorize training."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TaskCacheQualificationError(
            "qualification JSON is not canonical"
        ) from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _validate_result(value: object, *, smoke: bool) -> Mapping[str, object]:
    expected = _RESULT_FIELDS | ({"platform_type", "task_level"} if smoke else set())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TaskCacheQualificationError("qualification result fields are incomplete")
    if not isinstance(value["name"], str) or not value["name"]:
        raise TaskCacheQualificationError("qualification result name is invalid")
    for name in ("passed", "failed", "errors", "skipped"):
        if type(value[name]) is not int or value[name] < 0:
            raise TaskCacheQualificationError("qualification result counts are invalid")
    if (
        isinstance(value["elapsed_seconds"], bool)
        or not isinstance(value["elapsed_seconds"], (int, float))
        or not math.isfinite(float(value["elapsed_seconds"]))
        or float(value["elapsed_seconds"]) < 0.0
    ):
        raise TaskCacheQualificationError("qualification result timing is invalid")
    if (
        not isinstance(value["command"], (list, tuple))
        or not value["command"]
        or any(not isinstance(item, str) or not item for item in value["command"])
        or not _is_sha(value["log_sha256"])
    ):
        raise TaskCacheQualificationError("qualification result evidence is invalid")
    if smoke and (
        value["platform_type"] not in {"WHEELED", "LEGGED", "HOPPER"}
        or value["task_level"] is not True
    ):
        raise TaskCacheQualificationError("task-level smoke identity is invalid")
    return value


def build_qualification_report(
    *,
    source_commit: str,
    host_identity: Mapping[str, object],
    cache_manifest: str,
    cache_manifest_sha256: str,
    suite_results: Sequence[Mapping[str, object]],
    smoke_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not _is_sha(source_commit, 40):
        raise TaskCacheQualificationError("qualification source commit is invalid")
    required_host = {"hostname", "architecture", "logical_cpu_count"}
    if (
        not isinstance(host_identity, Mapping)
        or not required_host.issubset(host_identity)
        or not isinstance(host_identity["hostname"], str)
        or not host_identity["hostname"]
        or not isinstance(host_identity["architecture"], str)
        or not host_identity["architecture"]
        or type(host_identity["logical_cpu_count"]) is not int
        or host_identity["logical_cpu_count"] <= 0
    ):
        raise TaskCacheQualificationError("qualification host identity is incomplete")
    if not isinstance(cache_manifest, str) or not cache_manifest or not _is_sha(
        cache_manifest_sha256
    ):
        raise TaskCacheQualificationError("qualification cache identity is invalid")
    suites = tuple(_validate_result(value, smoke=False) for value in suite_results)
    smokes = tuple(_validate_result(value, smoke=True) for value in smoke_results)
    if not suites:
        raise TaskCacheQualificationError("focused qualification suites are missing")
    if tuple(value["platform_type"] for value in smokes) != (
        "WHEELED",
        "LEGGED",
        "HOPPER",
    ):
        raise TaskCacheQualificationError("three-platform short smokes are incomplete")
    all_results = suites + smokes
    summary = {
        "passed": sum(int(value["passed"]) for value in all_results),
        "failed": sum(int(value["failed"]) for value in all_results),
        "errors": sum(int(value["errors"]) for value in all_results),
        "unexpected_skips": sum(int(value["skipped"]) for value in all_results),
    }
    if (
        summary["passed"] <= 0
        or summary["failed"] != 0
        or summary["errors"] != 0
        or summary["unexpected_skips"] != 0
        or any(int(value["passed"]) <= 0 for value in smokes)
    ):
        raise TaskCacheQualificationError(
            "focused suite or task-level smoke did not pass"
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "host_identity": dict(host_identity),
        "cache_manifest": cache_manifest,
        "cache_manifest_sha256": cache_manifest_sha256,
        "task_range": dict(TASK_RANGE),
        "focused_suites": [dict(value) for value in suites],
        "short_smokes": [dict(value) for value in smokes],
        "test_summary": summary,
        "passed": True,
    }
    return {**body, "report_sha256": _digest(body)}


def validate_qualification_report(
    report: object,
    *,
    expected_source_commit: str,
) -> str:
    if not isinstance(report, Mapping):
        raise TaskCacheQualificationError("qualification report is not an object")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    digest = _digest(body)
    if report.get("report_sha256") != digest:
        raise TaskCacheQualificationError("qualification report hash differs")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("source_commit") != expected_source_commit
        or report.get("task_range") != TASK_RANGE
        or report.get("passed") is not True
    ):
        raise TaskCacheQualificationError("qualification report identity is stale")
    rebuilt = build_qualification_report(
        source_commit=str(report["source_commit"]),
        host_identity=report["host_identity"],
        cache_manifest=str(report["cache_manifest"]),
        cache_manifest_sha256=str(report["cache_manifest_sha256"]),
        suite_results=report["focused_suites"],
        smoke_results=report["short_smokes"],
    )
    if rebuilt != dict(report):
        raise TaskCacheQualificationError("qualification report body differs")
    return digest


def write_qualification_report_atomic(
    path: Path,
    report: Mapping[str, object],
) -> str:
    target = Path(path)
    digest = validate_qualification_report(
        report,
        expected_source_commit=str(report.get("source_commit")),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json_bytes(dict(report)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        if temporary.exists():
            temporary.unlink()
        raise TaskCacheQualificationError("qualification report write failed") from error
    return digest


def require_stable_source_commit(before: str, after: str) -> str:
    if not _is_sha(before, 40) or not _is_sha(after, 40) or before != after:
        raise TaskCacheQualificationError("repository HEAD changed during qualification")
    return before


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TaskCacheQualificationError("qualification source is unavailable") from error
    value = completed.stdout.strip()
    if not _is_sha(value, 40):
        raise TaskCacheQualificationError("qualification source commit is invalid")
    return value


def _require_clean_source(repository_root: Path) -> str:
    commit = _source_commit(repository_root)
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TaskCacheQualificationError("qualification cleanliness is unavailable") from error
    if completed.stdout:
        raise TaskCacheQualificationError("qualification requires a clean current source")
    return commit


def _host_identity() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpu_count": int(os.cpu_count() or 0),
    }


def _external_path(
    path: Path,
    repository_root: Path,
    *,
    require_file: bool,
) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise TaskCacheQualificationError("qualification path must be absolute")
    target = path.resolve(strict=require_file)
    repository = repository_root.resolve(strict=True)
    if target == repository or repository in target.parents:
        raise TaskCacheQualificationError(
            "qualification artifacts must remain outside the repository"
        )
    if require_file and not target.is_file():
        raise TaskCacheQualificationError("qualification input file is missing")
    return target


def _external_root(path: Path, repository_root: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise TaskCacheQualificationError("qualification artifact root is invalid")
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            raise TaskCacheQualificationError("qualification artifact parent is missing")
        probe = probe.parent
    if probe.is_symlink() or not probe.is_dir():
        raise TaskCacheQualificationError("qualification artifact parent is unsafe")
    repository = repository_root.resolve(strict=True)
    target = path.resolve(strict=False)
    if target == repository or repository in target.parents:
        raise TaskCacheQualificationError(
            "qualification artifacts must remain outside the repository"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve(strict=True)


def _parse_junit(path: Path) -> tuple[int, int, int, int]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise TaskCacheQualificationError("qualification JUnit is invalid") from error
    suites = (root,) if root.tag == "testsuite" else tuple(root.findall("testsuite"))
    if not suites:
        raise TaskCacheQualificationError("qualification JUnit has no test suites")
    try:
        tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
        failed = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    except ValueError as error:
        raise TaskCacheQualificationError("qualification JUnit counts are invalid") from error
    passed = tests - failed - errors - skipped
    if min(tests, passed, failed, errors, skipped) < 0:
        raise TaskCacheQualificationError("qualification JUnit counts differ")
    return passed, failed, errors, skipped


def _write_log(path: Path, content: str) -> str:
    raw = content.encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _run_pytest(
    *,
    name: str,
    targets: Sequence[str],
    artifact_root: Path,
    repository_root: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    safe_name = name.replace("/", "-")
    junit = artifact_root / f"{safe_name}.junit.xml"
    log = artifact_root / f"{safe_name}.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *targets,
        f"--junitxml={junit}",
    ]
    environment = dict(os.environ)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TaskCacheQualificationError(f"qualification command {name} failed") from error
    elapsed = time.perf_counter() - started
    log_sha256 = _write_log(
        log,
        completed.stdout
        + ("\n" if completed.stdout and completed.stderr else "")
        + completed.stderr,
    )
    passed, failed, errors, skipped = _parse_junit(junit)
    if completed.returncode != 0 and failed == 0 and errors == 0:
        errors = 1
    return {
        "name": name,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "command": command,
        "log_sha256": log_sha256,
    }


def _require_passing_node(result: Mapping[str, object]) -> None:
    if (
        int(result["passed"]) <= 0
        or int(result["failed"]) != 0
        or int(result["errors"]) != 0
        or int(result["skipped"]) != 0
    ):
        raise TaskCacheQualificationError(
            f"qualification node {result['name']} did not pass"
        )


def _run_qualification_profile(
    *,
    profile: str,
    artifact_root: Path,
    repository_root: Path,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if profile == "task-cache-v1":
        suites = (
            _run_pytest(
                name="tasks-1-11-focused",
                targets=_FOCUSED_TARGETS,
                artifact_root=artifact_root,
                repository_root=repository_root,
                timeout_seconds=1200.0,
            ),
        )
        smokes = tuple(
            {
                **_run_pytest(
                    name=f"{platform_type.lower()}-task-smoke",
                    targets=(target,),
                    artifact_root=artifact_root,
                    repository_root=repository_root,
                    timeout_seconds=300.0,
                ),
                "platform_type": platform_type,
                "task_level": True,
            }
            for platform_type, target in _SHORT_SMOKES
        )
        return suites, smokes
    if profile != "live-repair-v1":
        raise TaskCacheQualificationError("qualification profile is invalid")

    suite_results: list[dict[str, object]] = []
    for name, target in _LIVE_REPAIR_FOCUSED_NODES:
        result = _run_pytest(
            name=name,
            targets=(target,),
            artifact_root=artifact_root,
            repository_root=repository_root,
            timeout_seconds=120.0,
        )
        _require_passing_node(result)
        suite_results.append(result)
    smoke_results: list[dict[str, object]] = []
    for platform_type, target in _LIVE_REPAIR_SHORT_SMOKES:
        result = _run_pytest(
            name=f"{platform_type.lower()}-task-smoke",
            targets=(target,),
            artifact_root=artifact_root,
            repository_root=repository_root,
            timeout_seconds=120.0,
        )
        _require_passing_node(result)
        smoke_results.append(
            {**result, "platform_type": platform_type, "task_level": True}
        )
    return tuple(suite_results), tuple(smoke_results)


def _validate_scene_index(
    cache_manifest: Path,
    repository_root: Path,
) -> None:
    cache = load_formal_cache(cache_manifest, require_full=True)
    capability = training_cli._formal_capability_preflight(repository_root)
    v3_commit, v3_sha256 = training_cli.current_v3_identity(repository_root)
    expected = {
        "generator_sha256": GENERATOR_SHA256,
        "capability_sha256": capability.bundle_sha256,
        "reward_sha256": reward_weights_sha256(),
        "training_semantics_sha256": training_semantics_sha256(),
        "v3_source_commit": v3_commit,
        "v3_sha256": v3_sha256,
    }
    if cache.manifest.get("schema") != FORMAL_CACHE_SCHEMA or any(
        getattr(cache.identity, name) != value for name, value in expected.items()
    ):
        raise TaskCacheQualificationError("qualification scene index is stale")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("task-cache-v1", "live-repair-v1"),
        default="task-cache-v1",
    )
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        cache_manifest = _external_path(
            arguments.cache_manifest,
            repository_root,
            require_file=True,
        )
        artifact_root = _external_root(arguments.artifact_root, repository_root)
        output = _external_path(
            arguments.output,
            repository_root,
            require_file=False,
        )
        if output.suffix != ".json" or (
            output != artifact_root and artifact_root not in output.parents
        ):
            raise TaskCacheQualificationError(
                "qualification output must be JSON under the artifact root"
            )
        source_before = _require_clean_source(repository_root)
        _validate_scene_index(cache_manifest, repository_root)
        suite_results, smoke_results = _run_qualification_profile(
            profile=arguments.profile,
            artifact_root=artifact_root,
            repository_root=repository_root,
        )
        source_commit = require_stable_source_commit(
            source_before,
            _require_clean_source(repository_root),
        )
        report = build_qualification_report(
            source_commit=source_commit,
            host_identity=_host_identity(),
            cache_manifest=str(cache_manifest),
            cache_manifest_sha256=_file_sha256(cache_manifest),
            suite_results=suite_results,
            smoke_results=smoke_results,
        )
        digest = write_qualification_report_atomic(output, report)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "report_sha256": digest,
                    "test_summary": report["test_summary"],
                    "passed": True,
                },
                sort_keys=True,
            )
        )
    except (TaskCacheQualificationError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "TaskCacheQualificationError",
    "build_parser",
    "build_qualification_report",
    "main",
    "require_stable_source_commit",
    "validate_qualification_report",
    "write_qualification_report_atomic",
]
