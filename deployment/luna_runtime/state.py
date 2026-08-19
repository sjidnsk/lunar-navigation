from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimePaths:
    config: Path
    data: Path


def resolve_runtime_paths(
    release_id: str,
    *,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimePaths:
    environment = os.environ if environ is None else environ
    override = environment.get("LUNA_HOME")
    if override:
        root = Path(override).expanduser().resolve()
        return RuntimePaths(config=root / "config" / "runtime.yaml", data=root / "state" / release_id)
    user_home = home or Path.home()
    return RuntimePaths(
        config=user_home / ".config" / "luna" / "runtime.yaml",
        data=user_home / ".local" / "share" / "luna" / release_id,
    )


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
