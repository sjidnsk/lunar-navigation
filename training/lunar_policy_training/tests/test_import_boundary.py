from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
import yaml


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
REAL_FROZEN_SOURCE_ENV = "LUNAR_PPO_FROZEN_SOURCE_GIT"
REAL_FROZEN_COMMIT = "7309e93fdb85c60ff3736efe1a7f3c7eb640ee78"
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from import_ppo_core import ImportError, import_snapshot  # noqa: E402


def _git(repository: pathlib.Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True)


def _frozen_source(
    tmp_path: pathlib.Path,
    *,
    observation_payload: bytes = b"VALUE = 3\n",
    cross_attention_payload: bytes = (
        b"from lunar_exploration_ppo.policy.observation import VALUE\nRESULT = VALUE\n"
    ),
) -> tuple[pathlib.Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Task 1 test")
    _git(source, "remote", "add", "origin", "git@example.invalid:legacy.git")
    payloads = {
        "src/lunar_exploration_ppo/policy/observation.py": observation_payload,
        "src/lunar_exploration_ppo/policy/cross_attention.py": cross_attention_payload,
    }
    for relative, payload in payloads.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    _git(source, "add", ".")
    _git(source, "commit", "-m", "frozen source")
    return source, _git(source, "rev-parse", "HEAD").strip(), "git@example.invalid:legacy.git"


def _inventory_for_source(
    source: pathlib.Path, commit: str, origin: str
) -> dict[str, object]:
    files = []
    for path in (
        "src/lunar_exploration_ppo/policy/observation.py",
        "src/lunar_exploration_ppo/policy/cross_attention.py",
    ):
        payload = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{commit}:{path}"]
        )
        files.append(
            {
                "repository": "legacy_root",
                "path": path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "migration_role": "migration_source",
            }
        )
    return {"repositories": {"legacy_root": {"commit": commit, "origin": origin}}, "files": files}


def _inventory(
    commit: str,
    origin: str,
    *,
    corrupt_hash: bool = False,
    corrupt_size: bool = False,
) -> dict[str, object]:
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
                "size_bytes": len(payload) + (1 if corrupt_size else 0),
                "migration_role": "migration_source",
            }
        )
    return {"repositories": {"legacy_root": {"commit": commit, "origin": origin}}, "files": files}


def _frozen_payload(
    source: pathlib.Path, commit: str, relative: str
) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(source), "show", f"{commit}:{relative}"]
    )


def _map(
    source: pathlib.Path,
    commit: str,
    *,
    target: str = "policy/observation.py",
) -> dict[str, object]:
    observation_source = "src/lunar_exploration_ppo/policy/observation.py"
    cross_attention_source = "src/lunar_exploration_ppo/policy/cross_attention.py"
    return {
        "schema_version": "lunar-ppo-file-map/v1",
        "source": {"repository": "legacy_root", "commit": commit},
        "target_root": "training/lunar_policy_training/lunar_policy_training",
        "allow_dependencies": ["torch", "numpy"],
        "groups": {
            "policy": [
                {
                    "source": observation_source,
                    "sha256": hashlib.sha256(
                        _frozen_payload(source, commit, observation_source)
                    ).hexdigest(),
                    "target": target,
                    "allow_symbols": ["VALUE"],
                    "forbid_imports": ["lunar_exploration_ppo.workflows"],
                    "adaptation": {"kind": "symbol_core"},
                },
                {
                    "source": cross_attention_source,
                    "sha256": hashlib.sha256(
                        _frozen_payload(source, commit, cross_attention_source)
                    ).hexdigest(),
                    "target": "policy/cross_attention.py",
                    "allow_symbols": ["RESULT"],
                    "forbid_imports": ["lunar_exploration_ppo.workflows"],
                    "adaptation": {"kind": "symbol_core"},
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
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory(commit, origin), _map(source, commit)
    )

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
        ("map_hash", "file-map SHA-256 mismatch"),
        ("size", "source size mismatch"),
        ("unlisted", "source is not selected by inventory"),
        ("escape", "unsafe target"),
    ),
)
def test_import_rejects_identity_hash_allowlist_and_path_boundary_violations(
    tmp_path: pathlib.Path, mutation: str, message: str
) -> None:
    """Would fail if a tampered source could be imported outside the frozen boundary."""
    source, commit, origin = _frozen_source(tmp_path)
    inventory = _inventory(
        commit,
        origin,
        corrupt_hash=mutation == "hash",
        corrupt_size=mutation == "size",
    )
    file_map = _map(source, commit)
    if mutation == "origin":
        _git(source, "remote", "set-url", "origin", "git@example.invalid:wrong.git")
    elif mutation == "commit":
        file_map["source"]["commit"] = "0" * 40  # type: ignore[index]
        inventory["repositories"]["legacy_root"]["commit"] = "0" * 40  # type: ignore[index]
    elif mutation == "unlisted":
        inventory["files"] = inventory["files"][:-1]  # type: ignore[index]
    elif mutation == "map_hash":
        file_map["groups"]["policy"][0]["sha256"] = "0" * 64  # type: ignore[index]
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
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, inventory, _map(source, blocked_commit)
    )

    with pytest.raises(ImportError, match=r"forbidden dependency.*policy/observation.py"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_import_rejects_absolute_legacy_import_statement(tmp_path: pathlib.Path) -> None:
    """Would fail if ``import lunar_exploration_ppo...`` survived the package boundary."""
    source, commit, origin = _frozen_source(
        tmp_path,
        cross_attention_payload=(
            b"import lunar_exploration_ppo.policy.observation as observation\n"
            b"RESULT = observation.VALUE\n"
        ),
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), _map(source, commit)
    )

    with pytest.raises(ImportError, match=r"absolute legacy import.*cross_attention.py"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_import_rewrites_multiline_from_legacy_import(tmp_path: pathlib.Path) -> None:
    """Would fail if a parenthesized legacy import bypassed relative rewriting."""
    source, commit, origin = _frozen_source(
        tmp_path,
        cross_attention_payload=(
            b"from lunar_exploration_ppo.policy.observation import (\n"
            b"    VALUE,\n"
            b")\nRESULT = VALUE\n"
        ),
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), _map(source, commit)
    )

    import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    imported = tmp_path / "training/lunar_policy_training/lunar_policy_training/policy/cross_attention.py"
    assert imported.read_text(encoding="utf-8") == "from .observation import VALUE\nRESULT = VALUE\n"


@pytest.mark.parametrize(
    ("statement", "message"),
    (
        (
            "import importlib\nimportlib.import_module('lunar_exploration_ppo.workflows.runner')\n",
            "forbidden dependency",
        ),
        (
            "__import__('lunar_exploration_ppo.authority.guard')\n",
            "forbidden dependency",
        ),
        ("import pandas\n", "unlisted dependency"),
        ("import importlib\nimportlib.import_module('pandas')\n", "unlisted dependency"),
    ),
)
def test_import_rejects_forbidden_and_unlisted_static_or_dynamic_dependencies(
    tmp_path: pathlib.Path, statement: str, message: str
) -> None:
    """Would fail if a dynamic import bypassed the same dependency boundary."""
    source, commit, origin = _frozen_source(
        tmp_path, observation_payload=statement.encode("utf-8")
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), _map(source, commit)
    )

    with pytest.raises(ImportError, match=message):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_import_rejects_dynamic_import_even_for_allowlisted_dependency(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if an allowlisted module made dynamic execution acceptable."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"import importlib\nNUMPY = importlib.import_module('numpy')\n"
        ),
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), _map(source, commit)
    )

    with pytest.raises(ImportError, match="dynamic import"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_import_rejects_type_checking_any_dependency_substitution(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if TYPE_CHECKING/Any could conceal an unmapped runtime dependency."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"from __future__ import annotations\n"
            b"from lunar_exploration_ppo.env.frontier import FrontierActionSet\n"
            b"def name(value: FrontierActionSet) -> str:\n"
            b"    return value.__class__.__name__\n"
        ),
    )
    file_map = _map(source, commit)
    file_map["type_checking_dependencies"] = [
        "lunar_exploration_ppo.env.frontier"
    ]
    file_map["groups"]["policy"][0]["allow_symbols"] = ["name"]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=r"type_checking_dependencies|unmapped dependency"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize(
    "statement",
    (
        "import builtins\nbuiltins.__import__('lunar_exploration_ppo.workflows.runner')\n",
        "import builtins\nload = builtins.__import__\nload('lunar_exploration_ppo.workflows.runner')\n",
        "import builtins\nload = getattr(builtins, '__import__')\nload('lunar_exploration_ppo.workflows.runner')\n",
    ),
)
def test_import_rejects_builtins_dynamic_import_bypasses(
    tmp_path: pathlib.Path, statement: str
) -> None:
    """Would fail if builtins attribute or aliases bypassed forbidden dependency checks."""
    source, commit, origin = _frozen_source(
        tmp_path, observation_payload=statement.encode("utf-8")
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), _map(source, commit)
    )

    with pytest.raises(ImportError, match="forbidden dependency|dynamic import"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize(
    "payload",
    (
        (
            "def probe():\n"
            "    loader = __import__\n"
            "    return loader('torch')\n"
            "VALUE = 3\n"
        ),
        (
            "import importlib\n"
            "def probe():\n"
            "    loader = importlib.import_module\n"
            "    return loader('torch')\n"
            "VALUE = 3\n"
        ),
        (
            "import builtins\n"
            "def probe():\n"
            "    loader = builtins.__import__\n"
            "    return loader('torch')\n"
            "VALUE = 3\n"
        ),
        (
            "import builtins\n"
            "def probe():\n"
            "    loader = getattr(builtins, '__import__')\n"
            "    def nested():\n"
            "        return loader('torch')\n"
            "    return nested()\n"
            "VALUE = 3\n"
        ),
    ),
)
def test_import_rejects_dynamic_import_aliases_in_nested_scopes(
    tmp_path: pathlib.Path, payload: str
) -> None:
    """Would fail if a local callable alias escaped recursive dynamic-import checks."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=payload.encode("utf-8"),
        cross_attention_payload=b"RESULT = 3\n",
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), _map(source, commit)
    )

    with pytest.raises(ImportError, match="dynamic import"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_import_keeps_ordinary_local_assignment_legal(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if fail-closed alias tracking rejected unrelated local data flow."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"def probe():\n"
            b"    local_value = 3\n"
            b"    return local_value\n"
            b"VALUE = 3\n"
        ),
        cross_attention_payload=b"RESULT = 3\n",
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), _map(source, commit)
    )

    result = import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    assert result["files"][0]["resolved_symbols"] == ["VALUE"]


def test_symbol_extractor_emits_minimal_dependency_closure_and_provenance(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if extraction copied unused symbols/imports or lost source provenance."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b'"""Selected observation math."""\n'
            b"import math\n"
            b"import numpy as np\n"
            b"UNUSED = (1, 2)\n"
            b"VALUE = 3\n"
            b"def helper(value):\n"
            b"    local_only = math.floor(value)\n"
            b"    return local_only + VALUE\n"
            b"def public(value):\n"
            b"    return helper(value)\n"
        ),
        cross_attention_payload=(
            b"from lunar_exploration_ppo.policy.observation import public\n"
            b"def run(value):\n"
            b"    return public(value)\n"
        ),
    )
    file_map = _map(source, commit)
    policy_entries = file_map["groups"]["policy"]  # type: ignore[index]
    policy_entries[0]["allow_symbols"] = ["VALUE", "helper", "public"]
    policy_entries[1]["allow_symbols"] = ["run"]
    policy_entries[1]["adaptation"] = {"kind": "renamed_backbone"}
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    result = import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    observation = (
        tmp_path
        / "training/lunar_policy_training/lunar_policy_training/policy/observation.py"
    )
    cross_attention = observation.with_name("cross_attention.py")
    observation_text = observation.read_text(encoding="utf-8")
    cross_attention_text = cross_attention.read_text(encoding="utf-8")
    compile(observation_text, str(observation), "exec")
    compile(cross_attention_text, str(cross_attention), "exec")
    assert ast_names(observation_text) >= {"VALUE", "helper", "math"}
    assert "UNUSED" not in observation_text
    assert "numpy" not in observation_text
    assert "from .observation import public" in cross_attention_text
    first = result["files"][0]
    assert first["requested_symbols"] == ["VALUE", "helper", "public"]
    assert first["resolved_symbols"] == ["VALUE", "helper", "public"]
    assert first["source_sha256"] == file_map["groups"]["policy"][0]["sha256"]  # type: ignore[index]
    assert first["source_blob_oid"] == _git(
        source,
        "rev-parse",
        f"{commit}:src/lunar_exploration_ppo/policy/observation.py",
    ).strip()
    assert first["target"].endswith("policy/observation.py")
    assert first["adaptation"] == {"kind": "symbol_core"}
    assert first["provenance"] == {
        "commit": commit,
        "origin": origin,
        "repository": "legacy_root",
    }


def test_symbol_extractor_omits_unused_future_annotations_import(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a semantically unused future import survived minimal extraction."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"from __future__ import annotations\n"
            b"VALUE = 3\n"
        ),
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path,
        _inventory_for_source(source, commit, origin),
        _map(source, commit),
    )

    import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    imported = (
        tmp_path
        / "training/lunar_policy_training/lunar_policy_training/policy/observation.py"
    ).read_text(encoding="utf-8")
    assert imported == "VALUE = 3\n"


def test_symbol_extractor_allows_only_known_pure_constant_constructors(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if frozen numeric constants required unsafe arbitrary top-level calls."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"import math\n"
            b"import numpy as np\n"
            b"VALUE = np.float32(math.log(math.expm1(0.1)))\n"
        ),
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path,
        _inventory_for_source(source, commit, origin),
        _map(source, commit),
    )

    import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    imported = (
        tmp_path
        / "training/lunar_policy_training/lunar_policy_training/policy/observation.py"
    ).read_text(encoding="utf-8")
    assert "import math" in imported
    assert "import numpy as np" in imported
    assert "np.float32(math.log(math.expm1(0.1)))" in imported


def ast_names(text: str) -> set[str]:
    return {
        node.id
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Name)
    }


def test_symbol_extractor_does_not_treat_function_locals_as_global_dependencies(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a parameter or assignment shadowing a module name expanded the closure."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"VALUE = 99\n"
            b"def public(VALUE):\n"
            b"    local_only = VALUE + 1\n"
            b"    return local_only\n"
        ),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    file_map["groups"]["policy"][0]["allow_symbols"] = ["public"]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    result = import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    imported = (
        tmp_path
        / "training/lunar_policy_training/lunar_policy_training/policy/observation.py"
    ).read_text(encoding="utf-8")
    assert result["files"][0]["resolved_symbols"] == ["public"]
    assert "VALUE = 99" not in imported


def test_symbol_extractor_rejects_unallowlisted_helper_in_dependency_closure(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a helper could enter output without explicit symbol approval."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"def helper(value):\n    return value + 1\n"
            b"def public(value):\n    return helper(value)\n"
        ),
    )
    file_map = _map(source, commit)
    file_map["groups"]["policy"][0]["allow_symbols"] = ["public"]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=r"observation\.py:public.*helper"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_symbol_extractor_rejects_unresolved_global_name(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if an unresolved runtime name survived extraction or compile checks."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=b"def public():\n    return MISSING_NAME\n",
    )
    file_map = _map(source, commit)
    file_map["groups"]["policy"][0]["allow_symbols"] = ["public"]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=r"observation\.py:public.*MISSING_NAME"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize(
    ("payload", "dependency"),
    (
        (
            b"def decorate(value):\n    return value\n"
            b"@decorate\nclass Public:\n    pass\n",
            "decorate",
        ),
        (b"class Base:\n    pass\nclass Public(Base):\n    pass\n", "Base"),
        (
            b"DEFAULT = 3\nclass Public:\n"
            b"    def method(self, value=DEFAULT):\n        return value\n",
            "DEFAULT",
        ),
        (
            b"class Annotation:\n    pass\n"
            b"class Public:\n    value: Annotation\n",
            "Annotation",
        ),
        (b"BODY_VALUE = 3\nclass Public:\n    value = BODY_VALUE\n", "BODY_VALUE"),
    ),
)
def test_symbol_extractor_counts_every_class_definition_dependency(
    tmp_path: pathlib.Path, payload: bytes, dependency: str
) -> None:
    """Would fail if class decorators, bases, defaults, annotations, or body escaped closure."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=payload,
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    file_map["groups"]["policy"][0]["allow_symbols"] = ["Public"]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(
        ImportError, match=rf"observation\.py:Public.*{dependency}"
    ):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize(
    "payload",
    (
        b"VALUE = 3\nprint(VALUE)\n",
        b"VALUE = side_effect()\n",
    ),
)
def test_symbol_extractor_rejects_top_level_execution(
    tmp_path: pathlib.Path, payload: bytes
) -> None:
    """Would fail if importing generated code could execute source-side behavior."""
    source, commit, origin = _frozen_source(tmp_path, observation_payload=payload)
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path,
        _inventory_for_source(source, commit, origin),
        _map(source, commit),
    )

    with pytest.raises(ImportError, match=r"top-level.*observation\.py"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize("call", ("eval('1 + 1')", "exec('VALUE = 4')"))
def test_symbol_extractor_rejects_eval_and_exec(
    tmp_path: pathlib.Path, call: str
) -> None:
    """Would fail if selected symbols retained dynamic code evaluation."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=f"def public():\n    return {call}\n".encode("utf-8"),
    )
    file_map = _map(source, commit)
    file_map["groups"]["policy"][0]["allow_symbols"] = ["public"]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=r"observation\.py:public.*eval|exec"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize(
    ("payload", "symbol", "message"),
    (
        (
            b"def public(value):\n    artifact = value\n    return artifact\n",
            "public",
            "artifact",
        ),
        (
            b"def public():\n    return 'Stage4 ContentRef durable repair authority'\n",
            "public",
            "Stage4",
        ),
        (
            b"class CrossAttentionFrontierPolicy:\n    pass\n",
            "CrossAttentionFrontierPolicy",
            "CrossAttentionFrontierPolicy",
        ),
    ),
)
def test_symbol_extractor_rejects_forbidden_symbols_and_content(
    tmp_path: pathlib.Path, payload: bytes, symbol: str, message: str
) -> None:
    """Would fail if governance, durable-state, or old public names entered output."""
    source, commit, origin = _frozen_source(tmp_path, observation_payload=payload)
    file_map = _map(source, commit)
    file_map["groups"]["policy"][0]["allow_symbols"] = [symbol]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(
        ImportError, match=rf"observation\.py:{symbol}.*{message}"
    ):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_symbol_extractor_applies_explicit_rename_to_definition_and_references(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a renamed old class leaked through its definition or Name references."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"class LegacyPolicy:\n    pass\n"
            b"def build():\n    return LegacyPolicy()\n"
        ),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    observation_entry = file_map["groups"]["policy"][0]  # type: ignore[index]
    observation_entry["allow_symbols"] = ["LegacyPolicy", "build"]
    observation_entry["rename"] = {"LegacyPolicy": "StablePolicy"}
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    result = import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    imported = (
        tmp_path
        / "training/lunar_policy_training/lunar_policy_training/policy/observation.py"
    ).read_text(encoding="utf-8")
    assert "LegacyPolicy" not in imported
    assert "class StablePolicy" in imported
    assert "return StablePolicy()" in imported
    assert result["files"][0]["rename"] == {"LegacyPolicy": "StablePolicy"}


@pytest.mark.parametrize(
    "old_symbol",
    ("PolicyBatch", "PolicyForwardOutput", "CrossAttentionFrontierPolicy"),
)
def test_symbol_extractor_rejects_old_policy_types_even_when_renamed(
    tmp_path: pathlib.Path, old_symbol: str
) -> None:
    """Would fail if rename could disguise a frozen six-input policy type as safe core."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=f"class {old_symbol}:\n    pass\n".encode("utf-8"),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    observation_entry = file_map["groups"]["policy"][0]  # type: ignore[index]
    observation_entry["allow_symbols"] = [old_symbol]
    observation_entry["rename"] = {old_symbol: "RenamedCore"}
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=rf"observation\.py:{old_symbol}"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_symbol_extractor_rename_does_not_capture_shadowing_local_parameter(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if top-level rename changed a shadowing local into a global reference."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"class LegacyMath:\n    pass\n"
            b"def echo(LegacyMath):\n    return LegacyMath\n"
        ),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    observation_entry = file_map["groups"]["policy"][0]  # type: ignore[index]
    observation_entry["allow_symbols"] = ["LegacyMath", "echo"]
    observation_entry["rename"] = {"LegacyMath": "StableMath"}
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    imported = (
        tmp_path
        / "training/lunar_policy_training/lunar_policy_training/policy/observation.py"
    ).read_text(encoding="utf-8")
    assert "class StableMath" in imported
    assert "def echo(LegacyMath):" in imported
    assert "return LegacyMath" in imported


def test_symbol_extractor_is_byte_deterministic_for_output_and_result(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if traversal order, filesystem paths, or timestamps changed generated bytes."""
    source, commit, origin = _frozen_source(tmp_path)
    inventory_path, map_path, _result_path = _write_contracts(
        tmp_path,
        _inventory_for_source(source, commit, origin),
        _map(source, commit),
    )
    output_bytes: list[bytes] = []
    result_bytes: list[bytes] = []
    for name in ("run-a", "run-b"):
        repository_root = tmp_path / name
        repository_root.mkdir()
        result_path = repository_root / "migration/result.json"
        import_snapshot(
            source_git=source,
            repository_root=repository_root,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )
        output_bytes.append(
            (
                repository_root
                / "training/lunar_policy_training/lunar_policy_training/policy/observation.py"
            ).read_bytes()
        )
        result_bytes.append(result_path.read_bytes())

    assert output_bytes[0] == output_bytes[1]
    assert result_bytes[0] == result_bytes[1]


def test_symbol_extractor_applies_only_explicit_unused_import_and_string_adaptations(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if an exact frozen-source adaptation were ignored or broadened."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b'"""Stage 9 policy math."""\n'
            b"from lunar_exploration_ppo.utils.geometry import PoseXYTheta\n"
            b"VALUE = 3\n"
        ),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    entry = file_map["groups"]["policy"][0]  # type: ignore[index]
    entry["omit_imports"] = ["lunar_exploration_ppo.utils.geometry"]
    entry["string_replacements"] = [
        {
            "old": "Stage 9 policy math.",
            "new": "Frozen policy mathematical core.",
            "count": 1,
        }
    ]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    result = import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    generated = (
        tmp_path
        / "training/lunar_policy_training/lunar_policy_training/policy/observation.py"
    ).read_text(encoding="utf-8")
    assert ast.get_docstring(ast.parse(generated), clean=False) == (
        "Frozen policy mathematical core."
    )
    assert "lunar_exploration_ppo" not in generated
    assert result["files"][0]["omit_imports"] == [
        "lunar_exploration_ppo.utils.geometry"
    ]
    assert result["files"][0]["string_replacements"] == entry[
        "string_replacements"
    ]


def test_symbol_extractor_rejects_required_omitted_import(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if omit_imports could replace a selected runtime dependency with a stub."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(
            b"from lunar_exploration_ppo.utils.geometry import PoseXYTheta\n"
            b"def public():\n    return PoseXYTheta(1.0, 2.0, 3.0)\n"
        ),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    entry = file_map["groups"]["policy"][0]  # type: ignore[index]
    entry["allow_symbols"] = ["public"]
    entry["omit_imports"] = ["lunar_exploration_ppo.utils.geometry"]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(
        ImportError,
        match=r"public.*omitted import is required.*PoseXYTheta",
    ):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_symbol_extractor_rejects_unmatched_omit_import_declaration(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if stale adaptation metadata survived a frozen-source import drift."""
    source, commit, origin = _frozen_source(tmp_path)
    file_map = _map(source, commit)
    file_map["groups"]["policy"][0]["omit_imports"] = [  # type: ignore[index]
        "lunar_exploration_ppo.utils.geometry"
    ]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=r"omit_imports not found.*utils\.geometry"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize("omit_import", (False, True), ids=("direct", "exact-omit"))
@pytest.mark.parametrize(
    ("statement", "module", "message"),
    (
        (
            "import lunar_exploration_ppo.content_ref\nVALUE = 3\n",
            "lunar_exploration_ppo.content_ref",
            r"forbidden dependency.*content_ref",
        ),
        (
            "from lunar_exploration_ppo.content_ref import X as SafeX\nVALUE = 3\n",
            "lunar_exploration_ppo.content_ref",
            r"forbidden dependency.*content_ref",
        ),
        (
            "from benign.module import ContentRef as ContentReference\nVALUE = 3\n",
            "benign.module",
            r"forbidden imported symbol.*ContentRef",
        ),
    ),
)
def test_import_rejects_content_ref_modules_and_symbols_before_exact_omit(
    tmp_path: pathlib.Path,
    statement: str,
    module: str,
    message: str,
    omit_import: bool,
) -> None:
    """Would fail if exact omit metadata or an asname hid a forbidden ContentRef."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=statement.encode("utf-8"),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    file_map["allow_dependencies"].append("benign")  # type: ignore[union-attr]
    if omit_import:
        file_map["groups"]["policy"][0]["omit_imports"] = [module]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=message):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize("omit_import", (False, True), ids=("direct", "exact-omit"))
@pytest.mark.parametrize(
    ("statement", "module"),
    (
        (
            "import benign.content_reference\nVALUE = 3\n",
            "benign.content_reference",
        ),
        (
            "from benign.module import ContentReference as SafeReference\nVALUE = 3\n",
            "benign.module",
        ),
    ),
)
def test_import_allows_nearby_content_reference_names(
    tmp_path: pathlib.Path,
    statement: str,
    module: str,
    omit_import: bool,
) -> None:
    """Would fail if exact forbidden-name normalization became a substring gate."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=statement.encode("utf-8"),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    file_map["allow_dependencies"].append("benign")  # type: ignore[union-attr]
    if omit_import:
        file_map["groups"]["policy"][0]["omit_imports"] = [module]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    result = import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    assert result["files"][0]["resolved_symbols"] == ["VALUE"]


@pytest.mark.parametrize(
    ("module", "forbid_imports"),
    (
        (
            "lunar_exploration_ppo.workflows.runner",
            ["lunar_exploration_ppo.workflows"],
        ),
        (
            "lunar_exploration_ppo.env.frontier",
            ["lunar_exploration_ppo.env"],
        ),
        ("lunar_exploration_ppo.authority.guard", []),
        ("lunar_exploration_ppo.utils.artifact_registry", []),
    ),
)
def test_omit_imports_cannot_exempt_explicit_or_global_forbidden_dependency(
    tmp_path: pathlib.Path,
    module: str,
    forbid_imports: list[str],
) -> None:
    """Would fail if omit metadata were consumed before the forbidden gate."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=(f"from {module} import Ignored\nVALUE = 3\n").encode(
            "utf-8"
        ),
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    entry = file_map["groups"]["policy"][0]  # type: ignore[index]
    entry["forbid_imports"] = forbid_imports
    entry["omit_imports"] = [module]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match="forbid"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_symbol_extractor_rejects_string_replacement_count_drift(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if text adaptation silently matched more or fewer frozen strings."""
    source, commit, origin = _frozen_source(
        tmp_path,
        observation_payload=b'"""Stage 9 policy math."""\nVALUE = 3\n',
        cross_attention_payload=b"RESULT = 3\n",
    )
    file_map = _map(source, commit)
    file_map["groups"]["policy"][0]["string_replacements"] = [  # type: ignore[index]
        {
            "old": "Stage 9 policy math.",
            "new": "Frozen policy mathematical core.",
            "count": 2,
        }
    ]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=r"string replacement count.*expected 2, got 1"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def _real_frozen_source() -> pathlib.Path:
    configured = os.environ.get(REAL_FROZEN_SOURCE_ENV)
    if not configured:
        pytest.skip(f"set {REAL_FROZEN_SOURCE_ENV} to run frozen-source integration")
    source = pathlib.Path(configured)
    if not source.is_absolute():
        pytest.fail(f"{REAL_FROZEN_SOURCE_ENV} must be an absolute path")
    if not source.is_dir():
        pytest.fail(f"{REAL_FROZEN_SOURCE_ENV} does not name a directory")
    return source.resolve()


@pytest.mark.integration
def test_real_frozen_commit_extracts_six_clean_importable_cores_with_exact_provenance(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if real frozen behavior, exact provenance, or a clean core import were lost."""
    real_frozen_source = _real_frozen_source()
    expected_symbols = {
        "src/lunar_exploration_ppo/policy/observation.py": [
            "OBSERVATION_SCHEMA_VERSION",
            "GLOBAL_PRIOR_CHANNELS",
            "COVERAGE_SUMMARY_CHANNELS",
            "LOCAL_CROP_CHANNELS",
            "FRONTIER_FEATURE_FIELDS",
            "POSE_FEATURE_FIELDS",
        ],
        "src/lunar_exploration_ppo/policy/cross_attention.py": [
            "TOKEN_DIM",
            "MAP_POOL_SHAPE",
            "ATTENTION_HEADS",
            "CROSS_ATTENTION_LAYERS",
            "FFN_HIDDEN_DIM",
            "ACTION_HIDDEN_DIM",
            "DROPOUT",
            "INVALID_LOGIT_VALUE",
            "INITIAL_KAPPA_RAW",
            "PolicyActionError",
            "MapEncoder",
            "CrossAttentionBlock",
            "_normalize_angle",
            "normalize_theta",
            "_validated_masked_logits",
            "_gather_candidate",
            "_safe_theta_mu",
            "_masked_mean_max",
            "_map_position_encoding",
        ],
        "src/lunar_exploration_ppo/ppo/rollout.py": [
            "GAMMA",
            "GAE_LAMBDA",
            "RolloutContractError",
            "GAEResult",
            "compute_gae",
        ],
        "src/lunar_exploration_ppo/ppo/trainer.py": [
            "PPOTrainingError",
            "PPOLossTerms",
            "compute_ppo_loss_terms",
            "policy_state_sha256",
            "physical_microbatch_slices",
            "_gradient_norm",
            "_parameter_change_l2",
        ],
        "src/lunar_exploration_ppo/eval/baselines.py": [
            "BASELINE_METHODS",
            "ALL_METHODS",
            "_DISTANCE_FEATURE",
            "_POTENTIAL_GAIN_FEATURE",
            "_RECOMMENDED_THETA_SIN_FEATURE",
            "_RECOMMENDED_THETA_COS_FEATURE",
            "_REACHABLE_COST_FEATURE",
            "_MIN_DIRECTION_NORM",
            "BaselineSelectionError",
            "NoCandidateAction",
            "reconstruct_recommended_theta",
            "validate_selected_index",
        ],
        "src/lunar_exploration_ppo/eval/metrics.py": [
            "EPISODE_FIELDS",
            "BOOTSTRAP_METRICS",
            "ZERO_DISTANCE_POLICY",
            "_TERMINATION_REASONS",
            "MetricError",
            "EpisodeResult",
            "build_episode_result",
            "validate_episode_result",
            "episode_record",
            "bootstrap_episode_indices",
            "bootstrap_indices_sha256",
            "summarize_episodes",
            "_episode_order_key",
            "_summary_values",
            "_finite_mean",
        ],
    }
    expected_targets = {
        "policy/observation_core.py": "lunar_policy_training.policy.observation_core",
        "policy/backbone_core.py": "lunar_policy_training.policy.backbone_core",
        "ppo/rollout_core.py": "lunar_policy_training.ppo.rollout_core",
        "ppo/trainer_core.py": "lunar_policy_training.ppo.trainer_core",
        "eval/baseline_core.py": "lunar_policy_training.eval.baseline_core",
        "eval/metrics_core.py": "lunar_policy_training.eval.metrics_core",
    }
    manual_targets = {
        "src/lunar_exploration_ppo/ppo/collector.py": "ppo/collector.py",
        "src/lunar_exploration_ppo/ppo/checkpoint.py": "ppo/checkpoint.py",
    }
    package_root = tmp_path / "training/lunar_policy_training/lunar_policy_training"
    repository_package_root = (
        REPOSITORY_ROOT
        / "training/lunar_policy_training/lunar_policy_training"
    )
    for target in manual_targets.values():
        destination = package_root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repository_package_root / target).read_bytes())
    result_path = tmp_path / "migration/ppo_import_result.json"

    result = import_snapshot(
        source_git=real_frozen_source,
        repository_root=tmp_path,
        source_inventory_path=REPOSITORY_ROOT / "migration/source_inventory.yaml",
        file_map_path=REPOSITORY_ROOT / "migration/ppo_file_map.yaml",
        result_path=result_path,
    )

    assert result["source_repository"] == "legacy_root"
    assert result["source_origin"] == "git@github.com:sjidnsk/lunar-path-planning.git"
    assert result["source_commit"] == REAL_FROZEN_COMMIT
    assert result == json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["files"]) == 8
    result_by_source = {entry["source"]: entry for entry in result["files"]}
    for source, requested in expected_symbols.items():
        entry = result_by_source[source]
        assert entry["requested_symbols"] == requested
        assert entry["resolved_symbols"] == requested
        assert entry["provenance"] == {
            "commit": REAL_FROZEN_COMMIT,
            "origin": "git@github.com:sjidnsk/lunar-path-planning.git",
            "repository": "legacy_root",
        }
        assert entry["source_blob_oid"] == _git(
            real_frozen_source, "rev-parse", f"{REAL_FROZEN_COMMIT}:{source}"
        ).strip()
    for source, target in manual_targets.items():
        entry = result_by_source[source]
        target_payload = (package_root / target).read_bytes()
        assert entry["mode"] == "manual_thin_adapter"
        assert entry["requested_symbols"] == []
        assert entry["resolved_symbols"] == []
        assert entry["target_sha256"] == hashlib.sha256(target_payload).hexdigest()
        assert entry["target_size_bytes"] == len(target_payload)
        assert entry["target_validation"] == {
            "ast": "parsed-and-compiled",
            "imports": "static-allowlist-and-no-dynamic-imports",
            "kind": "manual-python-adapter",
        }

    forbidden = re.compile(
        r"\b(?:stage\d*|contentref|authority|repair|artifact|durable|"
        r"stub|policybatch|policyforwardoutput|"
        r"crossattentionfrontierpolicy)\b",
        re.IGNORECASE,
    )
    clean_env = os.environ.copy()
    clean_env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(package_root.parent),
        }
    )
    for target, module in expected_targets.items():
        generated = package_root / target
        text = generated.read_text(encoding="utf-8")
        compile(text, generated.as_posix(), "exec")
        assert forbidden.search(text) is None
        names = {
            node.id for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Name)
        }
        assert names.isdisjoint({"Any", "TYPE_CHECKING"})
        subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=tmp_path,
            check=True,
            env=clean_env,
        )


def _write_manual_target(
    repository_root: pathlib.Path,
    payload: str = (
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class CollectorConfig:\n"
        "    horizon: int\n"
    ),
) -> pathlib.Path:
    target = (
        repository_root
        / "training/lunar_policy_training/lunar_policy_training/ppo/collector.py"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return target


def _manual_collector_map(source: pathlib.Path, commit: str) -> dict[str, object]:
    file_map = _map(source, commit)
    observation_entry = file_map["groups"]["policy"][0]  # type: ignore[index]
    observation_entry["target"] = "ppo/collector.py"
    observation_entry["allow_symbols"] = []
    observation_entry["mode"] = "manual_thin_adapter"
    observation_entry["adaptation"] = {
        "kind": "in_memory_vector_env",
        "require_target_in_2b": True,
    }
    return file_map


def test_manual_thin_adapter_validates_target_and_records_both_provenances(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a legal adapter was copied from legacy or its target proof was lost."""
    source, commit, origin = _frozen_source(
        tmp_path, cross_attention_payload=b"RESULT = 3\n"
    )
    file_map = _manual_collector_map(source, commit)
    observation_entry = file_map["groups"]["policy"][0]  # type: ignore[index]
    target = _write_manual_target(tmp_path)
    target_before = target.read_bytes()
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    result = import_snapshot(
        source_git=source,
        repository_root=tmp_path,
        source_inventory_path=inventory_path,
        file_map_path=map_path,
        result_path=result_path,
    )

    manual = result["files"][0]
    assert manual["mode"] == "manual_thin_adapter"
    assert manual["requested_symbols"] == []
    assert manual["resolved_symbols"] == []
    assert manual["source_sha256"] == observation_entry["sha256"]
    assert manual["target_sha256"] == hashlib.sha256(target_before).hexdigest()
    assert manual["target_size_bytes"] == len(target_before)
    assert manual["target_validation"] == {
        "ast": "parsed-and-compiled",
        "imports": "static-allowlist-and-no-dynamic-imports",
        "kind": "manual-python-adapter",
    }
    assert target.read_bytes() == target_before


def test_manual_thin_adapter_rejects_missing_target(tmp_path: pathlib.Path) -> None:
    """Would fail if provenance could claim validation for an absent public adapter."""
    source, commit, origin = _frozen_source(
        tmp_path, cross_attention_payload=b"RESULT = 3\n"
    )
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path,
        _inventory_for_source(source, commit, origin),
        _manual_collector_map(source, commit),
    )

    with pytest.raises(ImportError, match=r"manual target is missing.*ppo/collector\.py"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            "from lunar_exploration_ppo.workflows.runner import Runner\n",
            "forbidden dependency",
        ),
        ("import pandas\n", "unlisted dependency"),
        ("VALUE = __import__('torch')\n", "dynamic import"),
        ("from typing import Any\nVALUE: Any = 1\n", "Any"),
        ("class PolicyBatch:\n    pass\n", "PolicyBatch"),
        ("def run():\n    return 1\nrun()\n", "top-level execution"),
        (
            "def run(value):\n    return value\n@run(1)\nclass Adapter:\n    value = 1\n",
            "definition-time execution",
        ),
        (
            "def run():\n    return 1\ndef collect(value=run()):\n    return value\n",
            "definition-time execution",
        ),
    ),
)
def test_manual_thin_adapter_rejects_unsafe_target_behavior(
    tmp_path: pathlib.Path, payload: str, message: str
) -> None:
    """Would fail if an adapter target bypassed the extracted-core safety boundary."""
    source, commit, origin = _frozen_source(
        tmp_path, cross_attention_payload=b"RESULT = 3\n"
    )
    file_map = _manual_collector_map(source, commit)
    _write_manual_target(tmp_path, payload)
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=message):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_manual_thin_adapter_result_is_byte_deterministic(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if target verification introduced paths, timestamps, or unstable order."""
    source, commit, origin = _frozen_source(
        tmp_path, cross_attention_payload=b"RESULT = 3\n"
    )
    inventory_path, map_path, _ = _write_contracts(
        tmp_path,
        _inventory_for_source(source, commit, origin),
        _manual_collector_map(source, commit),
    )
    results: list[bytes] = []
    for name in ("manual-a", "manual-b"):
        repository_root = tmp_path / name
        repository_root.mkdir()
        _write_manual_target(repository_root)
        result_path = repository_root / "migration/result.json"
        import_snapshot(
            source_git=source,
            repository_root=repository_root,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )
        results.append(result_path.read_bytes())

    assert results[0] == results[1]


@pytest.mark.parametrize("field", ("sha256", "target", "allow_symbols"))
def test_file_map_requires_explicit_symbol_extraction_fields_per_source(
    tmp_path: pathlib.Path, field: str
) -> None:
    """Would fail if a mapping silently fell back to whole-file copy defaults."""
    source, commit, origin = _frozen_source(tmp_path)
    file_map = _map(source, commit)
    del file_map["groups"]["policy"][0][field]  # type: ignore[index]
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path, _inventory_for_source(source, commit, origin), file_map
    )

    with pytest.raises(ImportError, match=field):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_import_rejects_non_regular_git_blob(tmp_path: pathlib.Path) -> None:
    """Would fail if a symlink could be imported as though it were a frozen regular blob."""
    source, _commit, origin = _frozen_source(tmp_path)
    observation = source / "src/lunar_exploration_ppo/policy/observation.py"
    observation.unlink()
    observation.symlink_to("cross_attention.py")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "symlink source")
    commit = _git(source, "rev-parse", "HEAD").strip()
    inventory_path, map_path, result_path = _write_contracts(
        tmp_path,
        _inventory_for_source(source, commit, origin),
        _map(source, commit),
    )

    with pytest.raises(ImportError, match=r"not a regular blob.*observation\.py"):
        import_snapshot(
            source_git=source,
            repository_root=tmp_path,
            source_inventory_path=inventory_path,
            file_map_path=map_path,
            result_path=result_path,
        )


def test_repository_file_map_declares_core_targets_and_manual_adapter_boundaries() -> None:
    """Would fail if 2B targets regressed to frozen copies or overwrote public adapters."""
    document = yaml.safe_load(
        (REPOSITORY_ROOT / "migration/ppo_file_map.yaml").read_text(encoding="utf-8")
    )
    entries = {
        entry["source"]: entry
        for group in document["groups"].values()
        for entry in group
    }
    inventory = json.loads(
        (REPOSITORY_ROOT / "migration/source_inventory.yaml").read_text(
            encoding="utf-8"
        )
    )
    inventory_hashes = {
        entry["path"]: entry["sha256"]
        for entry in inventory["files"]
        if entry["repository"] == "legacy_root"
    }
    expected_targets = {
        "src/lunar_exploration_ppo/policy/observation.py": "policy/observation_core.py",
        "src/lunar_exploration_ppo/policy/cross_attention.py": "policy/backbone_core.py",
        "src/lunar_exploration_ppo/ppo/rollout.py": "ppo/rollout_core.py",
        "src/lunar_exploration_ppo/ppo/trainer.py": "ppo/trainer_core.py",
        "src/lunar_exploration_ppo/eval/baselines.py": "eval/baseline_core.py",
        "src/lunar_exploration_ppo/eval/metrics.py": "eval/metrics_core.py",
        "src/lunar_exploration_ppo/ppo/collector.py": "ppo/collector.py",
        "src/lunar_exploration_ppo/ppo/checkpoint.py": "ppo/checkpoint.py",
    }
    assert {source: entry["target"] for source, entry in entries.items()} == expected_targets
    for source, entry in entries.items():
        assert len(entry["sha256"]) == 64
        assert entry["sha256"] == inventory_hashes[source]
        assert isinstance(entry["allow_symbols"], list)
        assert isinstance(entry["forbid_imports"], list)
        old_policy_symbols = {
            "PolicyBatch",
            "PolicyForwardOutput",
            "CrossAttentionFrontierPolicy",
        }
        assert old_policy_symbols.isdisjoint(entry["allow_symbols"])
        assert old_policy_symbols.isdisjoint(entry.get("rename", {}))
        assert all(
            word not in entry["target"]
            for word in ("frozen", "legacy", "reference")
        )
        expected_mode = (
            "manual_thin_adapter"
            if source.endswith(("collector.py", "checkpoint.py"))
            else "extract"
        )
        assert entry.get("mode", "extract") == expected_mode
        if expected_mode == "manual_thin_adapter":
            assert entry["allow_symbols"] == []
            assert entry["adaptation"]["require_target_in_2b"] is True
    assert "type_checking_dependencies" not in document
