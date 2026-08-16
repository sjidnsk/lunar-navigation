#!/usr/bin/env python3
"""Measure task-cache cold builds, warm loads, and safe builder concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lunar_policy_training"
sys.path.insert(0, str(PACKAGE_ROOT))

from lunar_policy_training.config import TaskAreaConfig, load_training_config  # noqa: E402
from lunar_policy_training.environment.task_area import (  # noqa: E402
    derive_task_evidence_halo,
    freeze_formal_task_geometry,
    sample_task_area_span_cells,
)
from lunar_policy_training.polar_data.formal_cache import (  # noqa: E402
    GENERATOR_SHA256,
    load_formal_cache,
)
from lunar_policy_training.polar_data.task_cache import (  # noqa: E402
    PlatformTaskKey,
    TaskCacheStore,
    TaskCommonKey,
)
from lunar_policy_training.polar_data.task_cache_scheduler import (  # noqa: E402
    TaskBuildPriority,
    TaskBuildProduct,
    TaskCacheCoordinator,
)
from lunar_policy_training.polar_data.task_coverability import (  # noqa: E402
    build_ground_task_coverability,
    build_task_common,
)
from lunar_policy_training.reward import reward_weights_sha256  # noqa: E402
from lunar_policy_training.reward_contract import TaskScaleBucket  # noqa: E402
from lunar_policy_training.training_semantics import (  # noqa: E402
    training_semantics_sha256,
)
import lunar_policy_training.cli as training_cli  # noqa: E402


SCHEMA_VERSION = "lunar-task-cache-benchmark/v1"
BUILDER_CANDIDATES = (1, 2, 4, 6)
TASK_SIZES_M = (100, 248, 348, 448, 500)
_SHA256 = frozenset("0123456789abcdef")
_REQUIRED_RUN_METRICS = frozenset(
    (
        "builder_workers",
        "elapsed_seconds",
        "cpu_seconds",
        "throughput_tasks_per_second",
        "p95_task_wall_seconds",
        "peak_total_rss_bytes",
        "swap_growth_bytes",
        "max_in_flight_bytes",
        "max_native_calls",
        "coordinator_metrics",
        "task_rows",
    )
)
_REQUIRED_TASK_FIELDS = frozenset(
    (
        "phase",
        "platform_type",
        "task_size_m",
        "scale_bucket",
        "task_common_key_sha256",
        "platform_task_key_sha256",
        "task_common_artifact_sha256",
        "platform_task_artifact_sha256",
        "task_common_blob_sha256",
        "platform_blob_sha256",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "native_call_count",
        "heavy_callback_count",
        "artifact_bytes",
    )
)


class TaskCacheBenchmarkError(ValueError):
    """Benchmark evidence is incomplete, stale, or unsafe."""


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
        raise TaskCacheBenchmarkError("benchmark JSON is not canonical") from error


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in _SHA256 for character in value)
    )


def _positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _blob_hashes(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(name, str) and name and _is_sha(digest)
            for name, digest in value.items()
        )
    )


def _validate_task_row(row: object) -> Mapping[str, object]:
    if not isinstance(row, Mapping) or set(row) != _REQUIRED_TASK_FIELDS:
        raise TaskCacheBenchmarkError("task benchmark row metrics are incomplete")
    if row["phase"] not in {"cold", "warm"}:
        raise TaskCacheBenchmarkError("task benchmark phase is invalid")
    if row["platform_type"] not in {"WHEELED", "LEGGED"}:
        raise TaskCacheBenchmarkError("task benchmark platform is invalid")
    if type(row["task_size_m"]) is not int or row["task_size_m"] not in TASK_SIZES_M:
        raise TaskCacheBenchmarkError("task benchmark size is invalid")
    expected_bucket = _bucket_for_size(int(row["task_size_m"])).value
    if row["scale_bucket"] != expected_bucket:
        raise TaskCacheBenchmarkError("task benchmark scale bucket differs")
    for name in (
        "task_common_key_sha256",
        "platform_task_key_sha256",
        "task_common_artifact_sha256",
        "platform_task_artifact_sha256",
    ):
        if not _is_sha(row[name]):
            raise TaskCacheBenchmarkError("task benchmark artifact identity is invalid")
    if not _blob_hashes(row["task_common_blob_sha256"]) or not _blob_hashes(
        row["platform_blob_sha256"]
    ):
        raise TaskCacheBenchmarkError("task benchmark blob identities are invalid")
    for name in ("wall_seconds", "cpu_seconds"):
        if not _nonnegative_number(row[name]):
            raise TaskCacheBenchmarkError("task benchmark timing metrics are invalid")
    for name in (
        "peak_rss_bytes",
        "native_call_count",
        "heavy_callback_count",
        "artifact_bytes",
    ):
        if type(row[name]) is not int or int(row[name]) < 0:
            raise TaskCacheBenchmarkError("task benchmark count metrics are invalid")
    if row["artifact_bytes"] <= 0:
        raise TaskCacheBenchmarkError("task benchmark artifact bytes are missing")
    if row["phase"] == "warm" and (
        row["native_call_count"] != 0 or row["heavy_callback_count"] != 0
    ):
        raise TaskCacheBenchmarkError("warm task cache invoked a heavy callback")
    return row


def validate_builder_run(
    run: object,
    *,
    total_ram_bytes: int,
) -> Mapping[str, object]:
    if not isinstance(run, Mapping) or set(run) != _REQUIRED_RUN_METRICS:
        raise TaskCacheBenchmarkError("builder benchmark metrics are incomplete")
    workers = run["builder_workers"]
    if type(workers) is not int or workers not in BUILDER_CANDIDATES:
        raise TaskCacheBenchmarkError("builder benchmark candidate is invalid")
    for name in (
        "elapsed_seconds",
        "cpu_seconds",
        "throughput_tasks_per_second",
        "p95_task_wall_seconds",
    ):
        if not _positive_number(run[name]):
            raise TaskCacheBenchmarkError("builder benchmark timing metrics are invalid")
    for name in (
        "peak_total_rss_bytes",
        "swap_growth_bytes",
        "max_in_flight_bytes",
        "max_native_calls",
    ):
        if type(run[name]) is not int or int(run[name]) < 0:
            raise TaskCacheBenchmarkError("builder benchmark resource metrics are invalid")
    if type(total_ram_bytes) is not int or total_ram_bytes <= 0:
        raise TaskCacheBenchmarkError("host total RAM metric is invalid")
    metrics = run["coordinator_metrics"]
    required_metrics = {
        "peak_builder_count",
        "peak_in_flight_bytes",
        "peak_in_flight_native_calls",
        "actual_native_call_count",
        "artifact_bytes",
        "build_count",
        "cache_hit_count",
    }
    if (
        not isinstance(metrics, Mapping)
        or not required_metrics.issubset(metrics)
        or any(type(metrics[name]) is not int or metrics[name] < 0 for name in required_metrics)
    ):
        raise TaskCacheBenchmarkError("coordinator metrics are incomplete")
    if (
        metrics["peak_builder_count"] > workers
        or metrics["peak_in_flight_bytes"] > run["max_in_flight_bytes"]
        or metrics["peak_in_flight_native_calls"] > run["max_native_calls"]
    ):
        raise TaskCacheBenchmarkError("coordinator exceeded a frozen resource limit")
    rows = run["task_rows"]
    if not isinstance(rows, (list, tuple)) or not rows:
        raise TaskCacheBenchmarkError("task benchmark rows are missing")
    validated = tuple(_validate_task_row(row) for row in rows)
    if {int(row["task_size_m"]) for row in validated} != set(TASK_SIZES_M):
        raise TaskCacheBenchmarkError("task benchmark scale coverage is incomplete")
    pairs: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = {}
    for row in validated:
        identity = (
            row["platform_type"],
            row["task_size_m"],
            row["task_common_key_sha256"],
            row["platform_task_key_sha256"],
        )
        phases = pairs.setdefault(identity, {})
        if row["phase"] in phases:
            raise TaskCacheBenchmarkError("task benchmark phase is duplicated")
        phases[str(row["phase"])] = row
    for phases in pairs.values():
        if set(phases) != {"cold", "warm"}:
            raise TaskCacheBenchmarkError("task benchmark cold/warm pair is incomplete")
        cold, warm = phases["cold"], phases["warm"]
        for name in (
            "task_common_artifact_sha256",
            "platform_task_artifact_sha256",
            "task_common_blob_sha256",
            "platform_blob_sha256",
            "artifact_bytes",
        ):
            if cold[name] != warm[name]:
                raise TaskCacheBenchmarkError("warm artifact identity differs from cold")
    return run


def select_builder_workers(
    builder_runs: Sequence[Mapping[str, object]],
    *,
    total_ram_bytes: int,
) -> int:
    runs = tuple(
        validate_builder_run(run, total_ram_bytes=total_ram_bytes)
        for run in builder_runs
    )
    if tuple(int(run["builder_workers"]) for run in runs) != BUILDER_CANDIDATES:
        raise TaskCacheBenchmarkError("builder candidates must be exactly 1 2 4 6")
    best_p95 = min(float(run["p95_task_wall_seconds"]) for run in runs)
    eligible = tuple(
        run
        for run in runs
        if run["swap_growth_bytes"] == 0
        and int(run["peak_total_rss_bytes"]) * 4 < total_ram_bytes * 3
        and float(run["p95_task_wall_seconds"]) <= best_p95 * 1.15
    )
    if not eligible:
        raise TaskCacheBenchmarkError("no builder candidate is resource eligible")
    selected = max(
        eligible,
        key=lambda run: (
            float(run["throughput_tasks_per_second"]),
            -int(run["builder_workers"]),
        ),
    )
    return int(selected["builder_workers"])


def build_benchmark_report(
    *,
    source_commit: str,
    host_identity: Mapping[str, object],
    cache_manifest: str,
    cache_manifest_sha256: str,
    builder_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not _is_sha(source_commit, 40):
        raise TaskCacheBenchmarkError("benchmark source commit is invalid")
    required_host = {
        "hostname",
        "architecture",
        "logical_cpu_count",
        "total_ram_bytes",
    }
    if (
        not isinstance(host_identity, Mapping)
        or not required_host.issubset(host_identity)
        or not isinstance(host_identity["hostname"], str)
        or not host_identity["hostname"]
        or not isinstance(host_identity["architecture"], str)
        or not host_identity["architecture"]
        or type(host_identity["logical_cpu_count"]) is not int
        or host_identity["logical_cpu_count"] <= 0
        or type(host_identity["total_ram_bytes"]) is not int
        or host_identity["total_ram_bytes"] <= 0
    ):
        raise TaskCacheBenchmarkError("benchmark host identity is incomplete")
    if not isinstance(cache_manifest, str) or not cache_manifest:
        raise TaskCacheBenchmarkError("benchmark cache manifest path is invalid")
    if not _is_sha(cache_manifest_sha256):
        raise TaskCacheBenchmarkError("benchmark cache manifest identity is invalid")
    runs = json.loads(_canonical_json_bytes(list(builder_runs)).decode("utf-8"))
    selected = select_builder_workers(
        runs,
        total_ram_bytes=int(host_identity["total_ram_bytes"]),
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "host_identity": dict(host_identity),
        "cache_manifest": cache_manifest,
        "cache_manifest_sha256": cache_manifest_sha256,
        "benchmark_scope": "GROUND_R1_TASK_CACHE",
        "task_sizes_m": list(TASK_SIZES_M),
        "builder_candidates": list(BUILDER_CANDIDATES),
        "builder_runs": runs,
        "selected_builder_workers": selected,
        "passed": True,
    }
    return {**body, "report_sha256": _canonical_digest(body)}


def validate_benchmark_report(
    report: object,
    *,
    expected_source_commit: str,
) -> str:
    if not isinstance(report, Mapping):
        raise TaskCacheBenchmarkError("benchmark report is not an object")
    claimed = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    digest = _canonical_digest(body)
    if claimed != digest:
        raise TaskCacheBenchmarkError("benchmark report hash differs")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("source_commit") != expected_source_commit
        or report.get("builder_candidates") != list(BUILDER_CANDIDATES)
        or report.get("task_sizes_m") != list(TASK_SIZES_M)
        or report.get("passed") is not True
    ):
        raise TaskCacheBenchmarkError("benchmark report identity is stale")
    rebuilt = build_benchmark_report(
        source_commit=str(report["source_commit"]),
        host_identity=report["host_identity"],
        cache_manifest=str(report["cache_manifest"]),
        cache_manifest_sha256=str(report["cache_manifest_sha256"]),
        builder_runs=report["builder_runs"],
    )
    if rebuilt != dict(report):
        raise TaskCacheBenchmarkError("benchmark report selection differs")
    return digest


def write_benchmark_report_atomic(path: Path, report: Mapping[str, object]) -> str:
    target = Path(path)
    digest = validate_benchmark_report(
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
        raise TaskCacheBenchmarkError("benchmark report write failed") from error
    return digest


def _bucket_for_size(size_m: int) -> TaskScaleBucket:
    if size_m < 200:
        return TaskScaleBucket.M100_200
    if size_m < 300:
        return TaskScaleBucket.M200_300
    if size_m < 400:
        return TaskScaleBucket.M300_400
    return TaskScaleBucket.M400_500


def _seed_for_exact_span(
    *,
    scene_id: str,
    platform_type: str,
    size_m: int,
    config: TaskAreaConfig,
) -> str:
    """Find a deterministic episode seed whose formal sampler hits one span."""
    if not _is_sha(scene_id) or platform_type not in {"WHEELED", "LEGGED"}:
        raise TaskCacheBenchmarkError("benchmark seed authority is invalid")
    if size_m not in TASK_SIZES_M:
        raise TaskCacheBenchmarkError("benchmark seed size is invalid")
    bucket = _bucket_for_size(size_m)
    target_span = size_m // 4
    for attempt in range(4096):
        seed = hashlib.sha256(
            (
                f"task-cache-benchmark/v1\0{scene_id}\0{platform_type}\0"
                f"{size_m}\0{attempt}"
            ).encode("utf-8")
        ).hexdigest()
        if (
            sample_task_area_span_cells(
                config,
                seed,
                scale_bucket=bucket,
            )
            == target_span
        ):
            return seed
    raise TaskCacheBenchmarkError("benchmark could not freeze the exact task span")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_source_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TaskCacheBenchmarkError("benchmark source commit is unavailable") from error
    value = completed.stdout.strip()
    if not _is_sha(value, 40):
        raise TaskCacheBenchmarkError("benchmark source commit is invalid")
    return value


def _require_clean_source(repository_root: Path) -> str:
    commit = _current_source_commit(repository_root)
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TaskCacheBenchmarkError("benchmark source cleanliness is unavailable") from error
    if completed.stdout:
        raise TaskCacheBenchmarkError("benchmark requires a clean current source")
    return commit


def _memory_info() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            pieces = raw.strip().split()
            if pieces:
                values[name] = int(pieces[0]) * 1024
    except (OSError, UnicodeError, ValueError) as error:
        raise TaskCacheBenchmarkError("host memory metrics are unavailable") from error
    if values.get("MemTotal", 0) <= 0:
        raise TaskCacheBenchmarkError("host total RAM metric is unavailable")
    return values


def current_host_identity() -> dict[str, object]:
    memory = _memory_info()
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpu_count": int(os.cpu_count() or 0),
        "total_ram_bytes": memory["MemTotal"],
    }


def _current_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, UnicodeError, ValueError):
        pass
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum) * 1024


class _RssMonitor:
    def __init__(self) -> None:
        self.peak_bytes = _current_rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())


def _external_file(path: Path, repository_root: Path, *, must_exist: bool) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise TaskCacheBenchmarkError("benchmark path must be absolute and external")
    target = path.resolve(strict=must_exist)
    repository = repository_root.resolve(strict=True)
    if target == repository or repository in target.parents:
        raise TaskCacheBenchmarkError("benchmark path must remain outside the repository")
    if must_exist and not target.is_file():
        raise TaskCacheBenchmarkError("benchmark input file is missing")
    return target


def _external_directory(path: Path, repository_root: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise TaskCacheBenchmarkError("benchmark artifact root must be absolute")
    repository = repository_root.resolve(strict=True)
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            raise TaskCacheBenchmarkError("benchmark artifact parent is unavailable")
        probe = probe.parent
    if probe.is_symlink() or not probe.is_dir():
        raise TaskCacheBenchmarkError("benchmark artifact parent is unsafe")
    resolved = path.resolve(strict=False)
    if resolved == repository or repository in resolved.parents:
        raise TaskCacheBenchmarkError("benchmark artifacts must remain outside the repository")
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise TaskCacheBenchmarkError("benchmark artifact root must be empty")
    return path.resolve(strict=True)


def _artifact_blob_sha256(artifact: object) -> dict[str, str]:
    arrays = getattr(artifact, "manifest", {}).get("arrays")
    if not isinstance(arrays, Mapping) or not arrays:
        raise TaskCacheBenchmarkError("task artifact blob manifest is missing")
    output: dict[str, str] = {}
    for name, value in arrays.items():
        if not isinstance(value, Mapping) or not _is_sha(value.get("sha256")):
            raise TaskCacheBenchmarkError("task artifact blob hash is invalid")
        output[str(name)] = str(value["sha256"])
    return output


def _artifact_bytes(*artifacts: object) -> int:
    roots = {Path(getattr(artifact, "root")) for artifact in artifacts}
    return sum(
        path.stat().st_size
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _benchmark_specifications(
    *,
    config_path: Path,
    cache_manifest_path: Path,
    repository_root: Path,
    source_commit: str,
):
    config = load_training_config(config_path)
    if config.run_kind != "formal":
        raise TaskCacheBenchmarkError("task cache benchmark requires formal config")
    cache = load_formal_cache(cache_manifest_path, require_full=True)
    capability_bundle = training_cli._formal_capability_preflight(repository_root)
    current_v3_commit, current_v3_sha256 = training_cli.current_v3_identity(
        repository_root
    )
    expected = {
        "generator_sha256": GENERATOR_SHA256,
        "capability_sha256": capability_bundle.bundle_sha256,
        "reward_sha256": reward_weights_sha256(),
        "training_semantics_sha256": training_semantics_sha256(),
        "v3_source_commit": current_v3_commit,
        "v3_sha256": current_v3_sha256,
    }
    if any(getattr(cache.identity, name) != value for name, value in expected.items()):
        raise TaskCacheBenchmarkError("scene index differs from current source")
    halo = derive_task_evidence_halo(capability_bundle)
    halo_sha256 = training_cli._task_halo_contract_sha256(halo)
    eligible = tuple(
        entry
        for entry in cache.manifest["scenes"]
        if entry["split"] == "train"
        and all(
            entry["platform_starts"][name]["qualified"] is True
            for name in ("WHEELED", "LEGGED")
        )
    )
    if not eligible:
        raise TaskCacheBenchmarkError("benchmark has no jointly eligible ground scene")
    scene = min(eligible, key=lambda entry: str(entry["scene_id"]))
    specifications = []
    for size_m in TASK_SIZES_M:
        for platform_type in ("WHEELED", "LEGGED"):
            platform_capability = capability_bundle.for_platform(platform_type)
            qualified_start = scene["platform_starts"][platform_type]
            start = tuple(int(value) for value in qualified_start["qualified_start_cell"])
            seed = _seed_for_exact_span(
                scene_id=str(scene["scene_id"]),
                platform_type=platform_type,
                size_m=size_m,
                config=config.task_area,
            )
            geometry = freeze_formal_task_geometry(
                start_cell=start,
                config=config.task_area,
                episode_seed=seed,
                scale_bucket=_bucket_for_size(size_m),
                halo=halo,
            )
            common_key = TaskCommonKey(
                scene_id=str(scene["scene_id"]),
                scenario_identity_sha256=seed,
                source_identity_sha256=cache.identity.source_lock_file_sha256,
                coarse_bounds_half_open=geometry.coarse_bounds_half_open,
                detail_bounds_half_open=geometry.detail_bounds_half_open,
                task_span_cells=geometry.span_cells,
                scale_bucket=geometry.scale_bucket.value,
                geometry_sha256=geometry.geometry_sha256,
                halo_contract_sha256=halo_sha256,
                capability_bundle_sha256=capability_bundle.bundle_sha256,
                generator_sha256=GENERATOR_SHA256,
                source_commit=source_commit,
            )
            specifications.append(
                training_cli._ScheduledTaskBuild(
                    worker_index=len(specifications),
                    episode_offset=0,
                    scene_id=str(scene["scene_id"]),
                    platform_type=platform_type,
                    geometry=geometry,
                    common_key=common_key,
                    platform_key=training_cli._task_platform_key(
                        common_key=common_key,
                        platform=platform_capability,
                        qualified_start=qualified_start,
                    ),
                    platform=platform_capability,
                    qualified_start=qualified_start,
                    priority=TaskBuildPriority.CURRENT_GROUND,
                )
            )
    return cache, capability_bundle, tuple(specifications)


def _task_row(
    *,
    specification: object,
    phase: str,
    wall_seconds: float,
    cpu_seconds: float,
    peak_rss_bytes: int,
    native_call_count: int,
    heavy_callback_count: int,
    store: TaskCacheStore,
) -> dict[str, object]:
    common = store.load_common(specification.common_key)
    platform_artifact = store.load_platform(specification.platform_key)
    if common is None or platform_artifact is None:
        raise TaskCacheBenchmarkError("task benchmark artifact is missing")
    if platform_artifact.common_artifact_sha256 != common.artifact_sha256:
        raise TaskCacheBenchmarkError("task benchmark common artifact differs")
    return {
        "phase": phase,
        "platform_type": specification.platform_type,
        "task_size_m": specification.geometry.span_cells * 4,
        "scale_bucket": specification.geometry.scale_bucket.value,
        "task_common_key_sha256": specification.common_key.sha256(),
        "platform_task_key_sha256": specification.platform_key.sha256(),
        "task_common_artifact_sha256": common.artifact_sha256,
        "platform_task_artifact_sha256": platform_artifact.artifact_sha256,
        "task_common_blob_sha256": _artifact_blob_sha256(common),
        "platform_blob_sha256": _artifact_blob_sha256(platform_artifact),
        "wall_seconds": max(0.0, float(wall_seconds)),
        "cpu_seconds": max(0.0, float(cpu_seconds)),
        "peak_rss_bytes": int(peak_rss_bytes),
        "native_call_count": int(native_call_count),
        "heavy_callback_count": int(heavy_callback_count),
        "artifact_bytes": _artifact_bytes(common, platform_artifact),
    }


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise TaskCacheBenchmarkError("task wall measurements are missing")
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _run_builder_candidate(
    *,
    workers: int,
    root: Path,
    cache: object,
    capability_bundle: object,
    specifications: Sequence[object],
) -> dict[str, object]:
    store = TaskCacheStore(root)
    by_key = {item.platform_key.sha256(): item for item in specifications}
    estimates = {
        identity: training_cli._task_build_estimate(specification)
        for identity, specification in by_key.items()
    }
    ordered = sorted(
        estimates.values(),
        key=lambda value: (value.in_flight_bytes, value.native_calls),
        reverse=True,
    )
    capacity = min(workers, len(ordered))
    max_in_flight_bytes = sum(value.in_flight_bytes for value in ordered[:capacity])
    max_native_calls = sum(value.native_calls for value in ordered[:capacity])
    bridge_state = threading.local()
    native_calls: dict[str, int] = {}
    native_lock = threading.Lock()
    coordinator = None

    def build(key: PlatformTaskKey) -> TaskBuildProduct:
        assert coordinator is not None
        specification = by_key[key.sha256()]
        common = coordinator.request_common(
            specification.common_key,
            lambda: build_task_common(
                scene_index=cache,
                scene_id=specification.scene_id,
                geometry=specification.geometry,
                capability_bundle=capability_bundle,
            ),
        )
        bridge = getattr(bridge_state, "bridge", None)
        if bridge is None:
            import lunar_planner_training_bridge as bridge_api

            bridge = bridge_api.PlannerBridge()
            bridge_state.bridge = bridge
        payload = build_ground_task_coverability(
            common=common,
            platform=specification.platform,
            qualified_start=specification.qualified_start,
            bridge=bridge,
        )
        training_cli._validate_task_payload_identity(key, payload)
        calls = training_cli._task_native_call_count(
            specification.platform_type,
            payload,
        )
        with native_lock:
            native_calls[key.sha256()] = calls
        return TaskBuildProduct(payload=payload, native_calls=calls)

    memory_before = _memory_info()
    cold_rows: list[dict[str, object]] = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    with _RssMonitor() as monitor:
        coordinator = TaskCacheCoordinator(
            store=store,
            builder=build,
            estimator=lambda key: estimates[key.sha256()],
            max_builder_workers=workers,
            max_in_flight_bytes=max_in_flight_bytes,
            max_native_calls=max_native_calls,
        )
        try:
            def cold_one(specification):
                row_wall = time.perf_counter()
                row_cpu = time.process_time()
                reference = coordinator.request(
                    specification.platform_key,
                    specification.priority,
                )
                reference.load()
                return specification, time.perf_counter() - row_wall, time.process_time() - row_cpu

            with ThreadPoolExecutor(max_workers=len(specifications)) as executor:
                measured = tuple(executor.map(cold_one, specifications))
            cold_metrics = dict(coordinator.metrics())
        finally:
            coordinator.close()
    cold_elapsed = time.perf_counter() - wall_started
    cold_cpu = time.process_time() - cpu_started
    for specification, wall_seconds, cpu_seconds in measured:
        cold_rows.append(
            _task_row(
                specification=specification,
                phase="cold",
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                peak_rss_bytes=monitor.peak_bytes,
                native_call_count=native_calls[specification.platform_key.sha256()],
                heavy_callback_count=1,
                store=store,
            )
        )

    def forbidden(_key):
        raise AssertionError("warm task cache invoked heavy builder")

    warm_rows: list[dict[str, object]] = []
    warm_coordinator = TaskCacheCoordinator(
        store=store,
        builder=forbidden,
        estimator=lambda key: estimates[key.sha256()],
        max_builder_workers=workers,
        max_in_flight_bytes=max_in_flight_bytes,
        max_native_calls=max_native_calls,
    )
    try:
        for specification in specifications:
            row_wall = time.perf_counter()
            row_cpu = time.process_time()
            reference = warm_coordinator.request(
                specification.platform_key,
                specification.priority,
            )
            reference.load()
            if store.load_common(specification.common_key) is None:
                raise TaskCacheBenchmarkError("warm common artifact is missing")
            warm_rows.append(
                _task_row(
                    specification=specification,
                    phase="warm",
                    wall_seconds=time.perf_counter() - row_wall,
                    cpu_seconds=time.process_time() - row_cpu,
                    peak_rss_bytes=_current_rss_bytes(),
                    native_call_count=0,
                    heavy_callback_count=0,
                    store=store,
                )
            )
        warm_metrics = dict(warm_coordinator.metrics())
    finally:
        warm_coordinator.close()
    if warm_metrics["build_count"] != 0 or warm_metrics["actual_native_call_count"] != 0:
        raise TaskCacheBenchmarkError("warm task cache repeated builder work")
    memory_after = _memory_info()
    coordinator_metrics = {
        **cold_metrics,
        "cache_hit_count": int(warm_metrics["cache_hit_count"]),
    }
    swap_growth = max(
        0,
        int(memory_before.get("SwapFree", 0)) - int(memory_after.get("SwapFree", 0)),
    )
    run = {
        "builder_workers": workers,
        "elapsed_seconds": cold_elapsed,
        "cpu_seconds": cold_cpu,
        "throughput_tasks_per_second": len(specifications) / cold_elapsed,
        "p95_task_wall_seconds": _p95([float(row["wall_seconds"]) for row in cold_rows]),
        "peak_total_rss_bytes": monitor.peak_bytes,
        "swap_growth_bytes": swap_growth,
        "max_in_flight_bytes": max_in_flight_bytes,
        "max_native_calls": max_native_calls,
        "coordinator_metrics": coordinator_metrics,
        "task_rows": cold_rows + warm_rows,
    }
    validate_builder_run(run, total_ram_bytes=memory_before["MemTotal"])
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--builder-candidates", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        if tuple(arguments.builder_candidates) != BUILDER_CANDIDATES:
            raise TaskCacheBenchmarkError("builder candidates must be exactly 1 2 4 6")
        config = arguments.config.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
        if not config.is_file() or not config.is_relative_to(repository):
            raise TaskCacheBenchmarkError("benchmark config must be a repository file")
        cache_manifest = _external_file(
            arguments.cache_manifest,
            repository_root,
            must_exist=True,
        )
        output = _external_file(
            arguments.output,
            repository_root,
            must_exist=False,
        )
        if output.suffix != ".json":
            raise TaskCacheBenchmarkError("benchmark output must be JSON")
        artifact_root = _external_directory(arguments.artifact_root, repository_root)
        source_commit = _require_clean_source(repository_root)
        host = current_host_identity()
        cache, capability_bundle, specifications = _benchmark_specifications(
            config_path=config,
            cache_manifest_path=cache_manifest,
            repository_root=repository_root,
            source_commit=source_commit,
        )
        runs = []
        for workers in BUILDER_CANDIDATES:
            candidate_root = artifact_root / f"builder-{workers}"
            candidate_root.mkdir()
            runs.append(
                _run_builder_candidate(
                    workers=workers,
                    root=candidate_root,
                    cache=cache,
                    capability_bundle=capability_bundle,
                    specifications=specifications,
                )
            )
        if _current_source_commit(repository_root) != source_commit:
            raise TaskCacheBenchmarkError("benchmark repository HEAD changed")
        report = build_benchmark_report(
            source_commit=source_commit,
            host_identity=host,
            cache_manifest=str(cache_manifest),
            cache_manifest_sha256=_file_sha256(cache_manifest),
            builder_runs=runs,
        )
        digest = write_benchmark_report_atomic(output, report)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "report_sha256": digest,
                    "selected_builder_workers": report["selected_builder_workers"],
                    "passed": True,
                },
                sort_keys=True,
            )
        )
    except (TaskCacheBenchmarkError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUILDER_CANDIDATES",
    "SCHEMA_VERSION",
    "TASK_SIZES_M",
    "TaskCacheBenchmarkError",
    "build_benchmark_report",
    "build_parser",
    "current_host_identity",
    "main",
    "select_builder_workers",
    "validate_benchmark_report",
    "validate_builder_run",
    "write_benchmark_report_atomic",
]
