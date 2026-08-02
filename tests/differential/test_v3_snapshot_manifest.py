from __future__ import annotations

import json
import hashlib
import math
import subprocess
import sys
from pathlib import Path, PurePosixPath

import yaml
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FILE_MAP_PATH = REPOSITORY_ROOT / "migration" / "v3_file_map.yaml"
SOURCE_INVENTORY_PATH = REPOSITORY_ROOT / "migration" / "source_inventory.yaml"
IMPORT_SCRIPT_PATH = REPOSITORY_ROOT / "tools" / "import_v3_snapshot.py"
CORE_PACKAGE_ROOT = REPOSITORY_ROOT / "ros2_ws" / "src" / "lunar_planner_core"
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "differential" / "fixtures"
LEGACY_EXPECTED_PATH = REPOSITORY_ROOT / "tests" / "differential" / "legacy_expected.json"
IMPORT_RESULT_PATH = REPOSITORY_ROOT / "migration" / "v3_import_result.json"
EXPECTED_GROUPS = {
    "shared",
    "wheel",
    "legged",
    "hopper",
    "temporary_contract_adapter",
    "tests",
}
LEGACY_TARGET_ROOT = PurePosixPath(
    "ros2_ws/src/lunar_planner_core/src/migration/legacy_v3/upstream"
)


def _source_inventory() -> dict[str, object]:
    return json.loads(SOURCE_INVENTORY_PATH.read_text(encoding="utf-8"))


def _file_map() -> dict[str, object]:
    return yaml.safe_load(FILE_MAP_PATH.read_text(encoding="utf-8"))


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_source_repository(
    tmp_path: Path,
    payload: bytes,
    *,
    source_path: str = "cpp/src/example.cpp",
    symlink_target: str | None = None,
) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q", "-b", "mainline")
    _git(source, "config", "user.name", "Snapshot Test")
    _git(source, "config", "user.email", "snapshot@example.invalid")
    _git(source, "remote", "add", "origin", "git@example.invalid:path-planner.git")
    source_file = source / source_path
    source_file.parent.mkdir(parents=True)
    if symlink_target is None:
        source_file.write_bytes(payload)
    else:
        source_file.symlink_to(symlink_target)
    _git(source, "add", source_path)
    _git(source, "commit", "-q", "-m", "fixture")
    return source, _git(source, "rev-parse", "HEAD")


def _write_import_contracts(
    tmp_path: Path,
    *,
    commit: str,
    payload: bytes,
    source_path: str = "cpp/src/example.cpp",
) -> tuple[Path, Path, str]:
    target_path = (
        LEGACY_TARGET_ROOT / PurePosixPath(*PurePosixPath(source_path).parts[1:])
    ).as_posix()
    inventory = {
        "schema_version": "lunar-migration-source-inventory/v1",
        "repositories": {
            "path_planner": {
                "commit": commit,
                "origin": "git@example.invalid:path-planner.git",
            }
        },
        "files": [
            {
                "repository": "path_planner",
                "path": source_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "migration_role": "migration_source",
            }
        ],
    }
    groups = {group: [] for group in EXPECTED_GROUPS}
    groups["shared"] = [{"source": source_path, "target": target_path}]
    file_map = {
        "schema_version": "lunar-v3-file-map/v1",
        "source": {"repository": "path_planner", "commit": commit},
        "target_root": LEGACY_TARGET_ROOT.as_posix(),
        "exclude_roots": ["schemas", "benchmarks", "python_bindings"],
        "groups": groups,
    }
    inventory_path = tmp_path / "source_inventory.json"
    file_map_path = tmp_path / "v3_file_map.yaml"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    file_map_path.write_text(
        yaml.safe_dump(file_map, sort_keys=False), encoding="utf-8"
    )
    return inventory_path, file_map_path, target_path


def _run_importer(
    *,
    source: Path,
    repository_root: Path,
    inventory_path: Path,
    file_map_path: Path,
    result_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(IMPORT_SCRIPT_PATH),
            "--source-git",
            str(source),
            "--repository-root",
            str(repository_root),
            "--source-inventory",
            str(inventory_path),
            "--file-map",
            str(file_map_path),
            "--result",
            str(result_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )


def test_v3_file_map_selects_only_frozen_inventory_and_excludes_governance() -> None:
    """Catch an omitted inventory blob or an import outside the private snapshot."""

    inventory = _source_inventory()
    manifest = _file_map()

    source = manifest["source"]
    assert source == {
        "repository": "path_planner",
        "commit": inventory["repositories"]["path_planner"]["commit"],
    }
    assert set(manifest["exclude_roots"]) >= {
        "schemas",
        "benchmarks",
        "python_bindings",
    }
    assert set(manifest["groups"]) == EXPECTED_GROUPS

    entries = [
        entry
        for group in EXPECTED_GROUPS
        for entry in manifest["groups"][group]
    ]
    selected_sources = {
        entry["path"]
        for entry in inventory["files"]
        if entry["repository"] == "path_planner"
    }
    mapped_sources = {entry["source"] for entry in entries}
    mapped_targets = {entry["target"] for entry in entries}

    assert mapped_sources == selected_sources
    assert len(entries) == len(mapped_sources) == len(mapped_targets)
    for entry in entries:
        source_path = PurePosixPath(entry["source"])
        target_path = PurePosixPath(entry["target"])
        assert not source_path.is_absolute()
        assert not target_path.is_absolute()
        assert ".." not in source_path.parts
        assert ".." not in target_path.parts
        assert target_path.is_relative_to(LEGACY_TARGET_ROOT)
        assert not any(
            excluded in source_path.parts
            for excluded in manifest["exclude_roots"]
        )


def test_import_result_matches_every_private_snapshot_blob() -> None:
    """Catch any post-import edit, omission, or unrecorded file in the snapshot."""

    result = json.loads(IMPORT_RESULT_PATH.read_text(encoding="utf-8"))
    manifest = _file_map()
    assert result["schema_version"] == "lunar-v3-import-result/v1"
    assert result["source_repository"] == manifest["source"]["repository"]
    assert result["source_commit"] == manifest["source"]["commit"]

    recorded_targets = {PurePosixPath(entry["target"]) for entry in result["files"]}
    mapped_targets = {
        PurePosixPath(entry["target"])
        for entries in manifest["groups"].values()
        for entry in entries
    }
    target_root = REPOSITORY_ROOT / manifest["target_root"]
    actual_targets = {
        PurePosixPath(path.relative_to(REPOSITORY_ROOT).as_posix())
        for path in target_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert recorded_targets == mapped_targets == actual_targets
    for entry in result["files"]:
        payload = (REPOSITORY_ROOT / entry["target"]).read_bytes()
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_importer_copies_manifest_blob_and_records_verified_hash(tmp_path: Path) -> None:
    """Catch copying checkout bytes or emitting an unaudited import result."""

    payload = b"int imported_snapshot() { return 3; }\n"
    source, commit = _make_source_repository(tmp_path, payload)
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path, commit=commit, payload=payload
    )
    repository_root = tmp_path / "destination"
    repository_root.mkdir()
    result_path = repository_root / "migration" / "v3_import_result.json"

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=result_path,
    )

    assert completed.returncode == 0, completed.stderr
    imported = repository_root / target_path
    assert imported.read_bytes() == payload
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result == {
        "schema_version": "lunar-v3-import-result/v1",
        "source_repository": "path_planner",
        "source_commit": commit,
        "files": [
            {
                "group": "shared",
                "source": "cpp/src/example.cpp",
                "target": target_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
    }


def test_importer_rejects_unlisted_file_in_snapshot_target(tmp_path: Path) -> None:
    """Catch an import that silently preserves a file outside the manifest."""

    payload = b"int selected_snapshot() { return 3; }\n"
    source, commit = _make_source_repository(tmp_path, payload)
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path, commit=commit, payload=payload
    )
    repository_root = tmp_path / "destination"
    extra = repository_root / LEGACY_TARGET_ROOT / "src" / "extra.cpp"
    extra.parent.mkdir(parents=True)
    extra.write_text("unlisted\n", encoding="utf-8")
    result_path = repository_root / "migration" / "v3_import_result.json"

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=result_path,
    )

    assert completed.returncode == 1
    assert "unlisted target file" in completed.stderr
    assert extra.read_text(encoding="utf-8") == "unlisted\n"
    assert not (repository_root / target_path).exists()
    assert not result_path.exists()


@pytest.mark.parametrize(
    ("source_path", "expected_error"),
    [
        ("cpp/src/example.o", "forbidden source suffix"),
        ("cpp/benchmarks/example.cpp", "excluded source root"),
    ],
)
def test_importer_rejects_object_and_excluded_source_roots(
    tmp_path: Path, source_path: str, expected_error: str
) -> None:
    """Catch a manifest edit that imports build objects or governance surfaces."""

    payload = b"not an authorized source file\n"
    source, commit = _make_source_repository(
        tmp_path, payload, source_path=source_path
    )
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path,
        commit=commit,
        payload=payload,
        source_path=source_path,
    )
    repository_root = tmp_path / "destination"
    repository_root.mkdir()

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=repository_root / "migration" / "v3_import_result.json",
    )

    assert completed.returncode == 1
    assert expected_error in completed.stderr
    assert not (repository_root / target_path).exists()


def test_importer_never_overwrites_modified_mapped_target(tmp_path: Path) -> None:
    """Catch an import that destroys a local edit at an otherwise mapped path."""

    payload = b"int frozen_snapshot() { return 3; }\n"
    source, commit = _make_source_repository(tmp_path, payload)
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path, commit=commit, payload=payload
    )
    repository_root = tmp_path / "destination"
    existing = repository_root / target_path
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"local edit\n")
    result_path = repository_root / "migration" / "v3_import_result.json"

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=result_path,
    )

    assert completed.returncode == 1
    assert "existing target differs from frozen blob" in completed.stderr
    assert existing.read_bytes() == b"local edit\n"
    assert not result_path.exists()


def test_importer_rejects_duplicate_source_or_target_mapping(tmp_path: Path) -> None:
    """Catch ambiguous group overlap before it reaches the audit result."""

    payload = b"int frozen_snapshot() { return 3; }\n"
    source, commit = _make_source_repository(tmp_path, payload)
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path, commit=commit, payload=payload
    )
    file_map = yaml.safe_load(file_map_path.read_text(encoding="utf-8"))
    file_map["groups"]["tests"].append(
        {"source": "cpp/src/example.cpp", "target": target_path}
    )
    file_map_path.write_text(
        yaml.safe_dump(file_map, sort_keys=False), encoding="utf-8"
    )
    repository_root = tmp_path / "destination"
    repository_root.mkdir()

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=repository_root / "migration" / "v3_import_result.json",
    )

    assert completed.returncode == 1
    assert "duplicate source or target mapping" in completed.stderr
    assert not (repository_root / target_path).exists()


def test_importer_validates_result_path_before_copying_any_file(tmp_path: Path) -> None:
    """Catch a failed invocation that leaves a partially imported snapshot."""

    payload = b"int frozen_snapshot() { return 3; }\n"
    source, commit = _make_source_repository(tmp_path, payload)
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path, commit=commit, payload=payload
    )
    repository_root = tmp_path / "destination"
    repository_root.mkdir()

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=tmp_path / "outside-result.json",
    )

    assert completed.returncode == 1
    assert "result path must be inside repository root" in completed.stderr
    assert not (repository_root / target_path).exists()


def test_importer_rejects_source_checkout_at_another_commit(tmp_path: Path) -> None:
    """Catch accidentally importing a later checkout with an unchanged manifest."""

    payload = b"int frozen_snapshot() { return 3; }\n"
    source, frozen_commit = _make_source_repository(tmp_path, payload)
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path, commit=frozen_commit, payload=payload
    )
    later_file = source / "cpp" / "src" / "later.cpp"
    later_file.write_text("int later() { return 4; }\n", encoding="utf-8")
    _git(source, "add", "cpp/src/later.cpp")
    _git(source, "commit", "-q", "-m", "later")
    repository_root = tmp_path / "destination"
    repository_root.mkdir()

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=repository_root / "migration" / "v3_import_result.json",
    )

    assert completed.returncode == 1
    assert "source HEAD mismatch" in completed.stderr
    assert not (repository_root / target_path).exists()


def test_importer_rejects_symbolic_link_source(tmp_path: Path) -> None:
    """Catch a manifest that would import a Git link instead of a regular blob."""

    payload = b"real.cpp"
    source, commit = _make_source_repository(
        tmp_path, payload, symlink_target=payload.decode("ascii")
    )
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path, commit=commit, payload=payload
    )
    repository_root = tmp_path / "destination"
    repository_root.mkdir()

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=repository_root / "migration" / "v3_import_result.json",
    )

    assert completed.returncode == 1
    assert "source mode is not a regular file" in completed.stderr
    assert not (repository_root / target_path).exists()


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (b"not utf-8: \xff\n", "source is not UTF-8 text"),
        (b"windows line ending\r\n", "source is not canonical LF text"),
        (b"embedded\x00nul\n", "source is not canonical LF text"),
    ],
)
def test_importer_rejects_noncanonical_text(
    tmp_path: Path, payload: bytes, expected_error: str
) -> None:
    """Catch binary, non-UTF-8, or CRLF payloads before any target write."""

    source, commit = _make_source_repository(tmp_path, payload)
    inventory_path, file_map_path, target_path = _write_import_contracts(
        tmp_path, commit=commit, payload=payload
    )
    repository_root = tmp_path / "destination"
    repository_root.mkdir()

    completed = _run_importer(
        source=source,
        repository_root=repository_root,
        inventory_path=inventory_path,
        file_map_path=file_map_path,
        result_path=repository_root / "migration" / "v3_import_result.json",
    )

    assert completed.returncode == 1
    assert expected_error in completed.stderr
    assert not (repository_root / target_path).exists()


def test_legacy_snapshot_has_an_opt_in_private_ament_build_boundary() -> None:
    """Catch exposing migration-only headers or building the snapshot by default."""

    package_xml = (CORE_PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")
    cmake = (CORE_PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "<name>lunar_planner_core</name>" in package_xml
    assert "<build_type>ament_cmake</build_type>" in package_xml
    assert 'option(LUNAR_BUILD_LEGACY_V3 "' in cmake
    assert "add_library(lunar_planner_legacy_v3 STATIC" in cmake
    assert "target_include_directories(lunar_planner_legacy_v3 PRIVATE" in cmake
    assert "install(TARGETS lunar_planner_legacy_v3" not in cmake
    assert "legacy_v3/upstream/include/" not in cmake.replace(
        "${LEGACY_V3_ROOT}/include", ""
    )


def test_legacy_differential_summary_has_only_stable_cross_version_semantics() -> None:
    """Catch missing platform cases or coupling the baseline to legacy handles."""

    case_documents = [
        json.loads((FIXTURE_ROOT / filename).read_text(encoding="utf-8"))
        for filename in ("wheel_cases.json", "legged_cases.json", "hopper_cases.json")
    ]
    expected = json.loads(LEGACY_EXPECTED_PATH.read_text(encoding="utf-8"))
    required_setups = {
        "open_known",
        "no_safe_route",
        "goal_infeasible",
        "numerical_failure",
        "deterministic_repeat",
    }
    all_case_ids: set[str] = set()
    for document, platform in zip(
        case_documents, ("wheel", "legged", "hopper"), strict=True
    ):
        assert document["schema_version"] == "lunar-v3-differential-cases/v1"
        assert document["platform"] == platform
        assert {case["setup"] for case in document["cases"]} == required_setups
        assert len(document["cases"]) == len(required_setups)
        for case in document["cases"]:
            assert case["case_id"].startswith(f"{platform}-")
            assert case["case_id"] not in all_case_ids
            assert case["repetitions"] == (
                3 if case["setup"] == "deterministic_repeat" else 1
            )
            all_case_ids.add(case["case_id"])

    assert expected["schema_version"] == "lunar-v3-legacy-summary/v1"
    summaries = expected["summaries"]
    assert {summary["case_id"] for summary in summaries} == all_case_ids
    allowed_fields = {
        "case_id",
        "reachable",
        "planning_outcome",
        "collision_free",
        "goal_reached",
        "platform_constraints_satisfied",
        "cost",
    }
    by_id = {}
    for summary in summaries:
        assert set(summary) == allowed_fields
        assert isinstance(summary["reachable"], bool)
        assert isinstance(summary["collision_free"], bool)
        assert isinstance(summary["goal_reached"], bool)
        assert isinstance(summary["platform_constraints_satisfied"], bool)
        assert summary["cost"] is None or (
            isinstance(summary["cost"], (int, float))
            and math.isfinite(summary["cost"])
            and summary["cost"] >= 0.0
        )
        by_id[summary["case_id"]] = summary

    for platform in ("wheel", "legged", "hopper"):
        safe = by_id[f"{platform}-safe-corridor"]
        repeated = by_id[f"{platform}-deterministic-repeat"]
        assert {key: value for key, value in repeated.items() if key != "case_id"} == {
            key: value for key, value in safe.items() if key != "case_id"
        }
        assert safe["planning_outcome"] == "NEW_REFERENCE_AVAILABLE"
        assert by_id[f"{platform}-no-safe-route"]["planning_outcome"] == (
            "NO_KNOWN_SAFE_ROUTE"
        )
        assert by_id[f"{platform}-goal-infeasible"]["planning_outcome"] == (
            "GOAL_INFEASIBLE"
        )
        assert by_id[f"{platform}-numerical-failure"]["planning_outcome"] == (
            "NUMERICAL_FAILURE"
        )

    serialized = LEGACY_EXPECTED_PATH.read_text(encoding="utf-8")
    for forbidden in ("ContentRef", "content_hash", "registry", "bundle_hash"):
        assert forbidden not in serialized
