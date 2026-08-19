from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import TargetProfile, load_allowlist, load_profile
from .environment import load_environment_lock


@dataclass(frozen=True)
class BundleFile:
    source_path: str
    archive_path: str
    content: bytes
    mode: int = 0o644


@dataclass(frozen=True)
class BundleRequest:
    repo_root: Path
    revision: str
    target: str
    output_dir: Path


@dataclass(frozen=True)
class BundleResult:
    archive: Path
    source_commit: str
    source_list_sha256: str


class BundleError(ValueError):
    pass


def _git_lines(repo_root: Path, args: tuple[str, ...]) -> tuple[str, ...]:
    result = subprocess.run(("git", *args), cwd=repo_root, text=True, capture_output=True)
    if result.returncode:
        raise BundleError(result.stderr.strip() or "git command failed")
    return tuple(line for line in result.stdout.splitlines() if line)


def _git_bytes(repo_root: Path, revision: str, source_path: str) -> bytes:
    result = subprocess.run(
        ("git", "show", f"{revision}:{source_path}"), cwd=repo_root, capture_output=True
    )
    if result.returncode:
        raise BundleError(f"cannot read {source_path} from {revision}")
    return result.stdout


def _component_excluded(path: str, excluded: tuple[str, ...]) -> bool:
    return bool(set(PurePosixPath(path).parts) & set(excluded))


def _in_root(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _archive_path(source_path: str, profile: TargetProfile) -> tuple[str, int] | None:
    if source_path == "scripts/luna":
        return "luna", 0o755
    if _in_root(source_path, "deployment/luna_runtime"):
        return source_path.removeprefix("deployment/"), 0o644
    if _in_root(source_path, "deployment/config"):
        return source_path.removeprefix("deployment/"), 0o644
    if source_path == "deployment/runtime_source_allowlist.yaml":
        return "runtime_source_allowlist.yaml", 0o644
    if source_path == f"deployment/profiles/{profile.id}.yaml" or source_path == f"deployment/profiles/{profile.id}.environment.yaml":
        return source_path.removeprefix("deployment/"), 0o644
    if source_path == "deployment/docs/README.runtime.md":
        return "README.md", 0o644
    if source_path == "deployment/docs/COMMANDS.runtime.md":
        return "COMMANDS.md", 0o644
    return source_path, 0o644


def _is_allowed(path: str, profile: TargetProfile, allowlist: dict[str, object]) -> bool:
    if path == "scripts/luna":
        return True
    fixed = tuple(allowlist["runtime_files"]) + tuple(allowlist["bundle_templates"])
    if any(_in_root(path, root) for root in fixed):
        return True
    if path in {
        f"deployment/profiles/{profile.id}.yaml",
        f"deployment/profiles/{profile.id}.environment.yaml",
    }:
        return True
    roots = tuple(allowlist["runtime_roots"])
    if any(_in_root(path, root) for root in roots):
        return True
    optional = allowlist["optional_roots"]
    return profile.id == "ubuntu22-humble-amd64" and _in_root(path, optional["nav2_adapter"])


def collect_bundle_files(repo_root: Path, revision: str, target: TargetProfile) -> tuple[BundleFile, ...]:
    allowlist = dict(load_allowlist(repo_root / "deployment" / "runtime_source_allowlist.yaml"))
    tracked = _git_lines(repo_root, ("ls-tree", "-r", "--name-only", revision))
    excluded = tuple(allowlist["excluded_path_components"])
    templates = set(allowlist["bundle_templates"])
    selected = [
        path
        for path in tracked
        if (path in templates or not _component_excluded(path, excluded))
        and _is_allowed(path, target, allowlist)
    ]
    required_roots = tuple(allowlist["runtime_roots"])
    for root in required_roots:
        if not any(_in_root(path, root) for path in selected):
            raise BundleError(f"required runtime root missing from {revision}: {root}")
    dirty = set(_git_lines(repo_root, ("diff", "--name-only")))
    if dirty & set(selected):
        raise BundleError("ALLOWLISTED_PATH_DIRTY")
    files: list[BundleFile] = []
    for path in sorted(selected):
        archive_path, mode = _archive_path(path, target)
        files.append(BundleFile(path, archive_path, _git_bytes(repo_root, revision, path), mode))
    return tuple(files)


def _source_list_hash(files: tuple[BundleFile, ...]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.archive_path):
        digest.update(item.archive_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.content).digest())
    return digest.hexdigest()


def build_bundle(request: BundleRequest) -> BundleResult:
    profile = load_profile(request.target, request.repo_root)
    files = collect_bundle_files(request.repo_root, request.revision, profile)
    source_commit = _git_lines(request.repo_root, ("rev-parse", request.revision))[0]
    source_hash = _source_list_hash(files)
    lock = load_environment_lock(request.target, request.repo_root)
    manifest = {
        "schema_version": "luna-release-manifest/v1",
        "source_commit": source_commit,
        "source_list_sha256": source_hash,
        "target_profile": profile.id,
        "environment_lock_sha256": lock.sha256,
        "interface_contract": "lunar-external-interfaces/v5",
        "observation_contract": "lunar-observation-contract/v4",
        "action_contract": "lunar-action-contract/v2",
        "optional_extensions": ["map_pipeline", "path_tracking", "nav2_adapter"],
        "policy_mode": "fallback",
    }
    files = (*files, BundleFile("generated", "release-manifest.json", (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")))
    prefix = f"lunar-runtime-{profile.id}-src"
    request.output_dir.mkdir(parents=True, exist_ok=True)
    archive = request.output_dir / f"{prefix}.tar.gz"
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for item in sorted(files, key=lambda value: value.archive_path):
                    info = tarfile.TarInfo(f"{prefix}/{item.archive_path}")
                    info.size = len(item.content)
                    info.mode = item.mode
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(item.content))
    return BundleResult(archive=archive, source_commit=source_commit, source_list_sha256=source_hash)
