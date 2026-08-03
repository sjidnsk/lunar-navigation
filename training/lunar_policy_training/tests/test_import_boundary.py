from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest
import yaml


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from import_ppo_core import ImportError, import_snapshot  # noqa: E402


def _git(repository: pathlib.Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True)


def _frozen_source(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Task 1 test")
    _git(source, "remote", "add", "origin", "git@example.invalid:legacy.git")
    payloads = {
        "src/lunar_exploration_ppo/policy/observation.py": b"VALUE = 3\n",
        "src/lunar_exploration_ppo/policy/cross_attention.py": (
            b"from lunar_exploration_ppo.policy.observation import VALUE\nRESULT = VALUE\n"
        ),
    }
    for relative, payload in payloads.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    _git(source, "add", ".")
    _git(source, "commit", "-m", "frozen source")
    return source, _git(source, "rev-parse", "HEAD").strip(), "git@example.invalid:legacy.git"


def _inventory(commit: str, origin: str, *, corrupt_hash: bool = False) -> dict[str, object]:
    paths = (
        "src/lunar_exploration_ppo/policy/observation.py",
        "src/lunar_exploration_ppo/policy/cross_attention.py",
    )
    files = []
    for path in paths:
        payload = b"VALUE = 3\n" if path.endswith("observation.py") else b"from lunar_exploration_ppo.policy.observation import VALUE\nRESULT = VALUE\n"
        files.append(
            {
                "repository": "legacy_root",
                "path": path,
                "sha256": "0" * 64 if corrupt_hash else hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "migration_role": "migration_source",
            }
        )
    return {"repositories": {"legacy_root": {"commit": commit, "origin": origin}}, "files": files}


def _map(commit: str, *, dependency: str | None = None, target: str = "policy/observation.py") -> dict[str, object]:
    return {
        "schema_version": "lunar-ppo-file-map/v1",
        "source": {"repository": "legacy_root", "commit": commit},
        "target_root": "training/lunar_policy_training/lunar_policy_training",
        "allow_dependencies": ["torch", "numpy"],
        "groups": {
            "policy": [
                {
                    "source": "src/lunar_exploration_ppo/policy/observation.py",
                    "target": target,
                },
                {
                    "source": "src/lunar_exploration_ppo/policy/cross_attention.py",
                    "target": "policy/cross_attention.py",
                },
            ]
        },
    }


def _write_contracts(
    tmp_path: pathlib.Path, inventory: dict[str, object], file_map: dict[str, object]
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    inventory_path = tmp_path / "inventory.yaml"
    map_path = tmp_path / "map.yaml"
    result_path = tmp_path / "result.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    map_path.write_text(yaml.safe_dump(file_map), encoding="utf-8")
    return inventory_path, map_path, result_path


def test_imports_only_verified_allowlisted_files_and_rewrites_internal_imports(tmp_path: pathlib.Path) -> None:
    """Would fail if an unverified file or absolute legacy import crossed the migration boundary."""
    source, commit, origin = _frozen_source(tmp_path)
    inventory_path, map_path, result_path = _write_contracts(tmp_path, _inventory(commit, origin), _map(commit))

    result = import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    imported = tmp_path / "training/lunar_policy_training/lunar_policy_training/policy/cross_attention.py"
    assert result["source_commit"] == commit
    assert imported.read_text(encoding="utf-8") == "from .observation import VALUE\nRESULT = VALUE\n"
    assert json.loads(result_path.read_text(encoding="utf-8"))["files"][0]["source"].endswith("observation.py")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("origin", "source origin mismatch"),
        ("commit", "source HEAD mismatch"),
        ("hash", "source SHA-256 mismatch"),
        ("unlisted", "source is not selected by inventory"),
        ("escape", "unsafe target"),
    ),
)
def test_import_rejects_identity_hash_allowlist_and_path_boundary_violations(
    tmp_path: pathlib.Path, mutation: str, message: str
) -> None:
    """Would fail if a tampered source could be imported outside the frozen boundary."""
    source, commit, origin = _frozen_source(tmp_path)
    inventory = _inventory(commit, origin, corrupt_hash=mutation == "hash")
    file_map = _map(commit)
    if mutation == "origin":
        _git(source, "remote", "set-url", "origin", "git@example.invalid:wrong.git")
    elif mutation == "commit":
        file_map["source"]["commit"] = "0" * 40  # type: ignore[index]
        inventory["repositories"]["legacy_root"]["commit"] = "0" * 40  # type: ignore[index]
    elif mutation == "unlisted":
        inventory["files"] = inventory["files"][:-1]  # type: ignore[index]
    elif mutation == "escape":
        file_map["groups"]["policy"][0]["target"] = "../escape.py"  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(tmp_path, inventory, file_map)

    with pytest.raises(ImportError, match=message):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_import_rejects_forbidden_legacy_dependency_with_its_source_path(tmp_path: pathlib.Path) -> None:
    """Would fail if workflows or authority code entered the PPO mathematical core."""
    source, commit, origin = _frozen_source(tmp_path)
    blocked = source / "src/lunar_exploration_ppo/policy/observation.py"
    blocked.write_text("from lunar_exploration_ppo.workflows.runner import Runner\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "forbidden dependency")
    blocked_commit = _git(source, "rev-parse", "HEAD").strip()
    payload = blocked.read_bytes()
    inventory = _inventory(blocked_commit, origin)
    inventory["files"][0]["sha256"] = hashlib.sha256(payload).hexdigest()  # type: ignore[index]
    inventory["files"][0]["size_bytes"] = len(payload)  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(tmp_path, inventory, _map(blocked_commit))

    with pytest.raises(ImportError, match=r"forbidden dependency.*policy/observation.py"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )
