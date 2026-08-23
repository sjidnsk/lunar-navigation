#!/usr/bin/env python3
"""Validate a repository-external pure-exploration deployment environment file."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
from typing import Any


class ProfileValidationError(ValueError):
    """Raised when deployment evidence cannot safely admit a launch."""


RESOURCE_LIMITS = (
    "maximum_position_probes",
    "maximum_candidate_views",
    "maximum_collision_work_units",
    "maximum_visibility_work_units",
    "maximum_path_preview_poses",
    "maximum_executable_path_points",
    "maximum_failure_entries",
    "maximum_failure_patch_cells_per_entry",
    "maximum_failure_total_patch_cells",
)
FAILURE_SLOTS = {
    "entries": ("entry", "maximum_entries", "maximum_failure_entries"),
    "patch_per_entry": (
        "per_entry",
        "maximum_patch_cells_per_entry",
        "maximum_failure_patch_cells_per_entry",
    ),
    "total_patch": (
        "total",
        "maximum_total_patch_cells",
        "maximum_failure_total_patch_cells",
    ),
}
SCHEMA = "lunar-pure-exploration/failure-saturation-evidence/v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
POSITIVE_INTEGER_RE = re.compile(r"[1-9][0-9]*\Z")
PROFILE_LINE_RE = re.compile(r"(?:export[ \t]+)?([A-Z0-9_]+)=(.*)\Z")


def _resource_env_name(name: str) -> str:
    return f"PURE_EXPLORATION_{name.upper()}"


def _slot_prefix(slot: str) -> str:
    return f"PURE_EXPLORATION_FAILURE_{slot.upper()}"


REQUIRED_PROFILE_KEYS = frozenset(
    {
        "PURE_EXPLORATION_ORIN_VALIDATED",
        *(_resource_env_name(name) for name in RESOURCE_LIMITS),
        "PURE_EXPLORATION_VALIDATED_VISIBILITY_PRODUCT",
        "PURE_EXPLORATION_VALIDATED_WHOLE_CYCLE_ELAPSED_MS",
        "PURE_EXPLORATION_VALIDATED_WHOLE_CYCLE_PEAK_RSS_MIB",
        "PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_ENVIRONMENT",
        "PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_IMAGE",
        *(
            key
            for slot in FAILURE_SLOTS
            for key in (
                f"{_slot_prefix(slot)}_EVIDENCE_PATH",
                f"{_slot_prefix(slot)}_EVIDENCE_SHA256",
                f"{_slot_prefix(slot)}_VALIDATED_LIMIT",
                f"{_slot_prefix(slot)}_SATURATION_ELAPSED_MS",
                f"{_slot_prefix(slot)}_PEAK_RETAINED_RSS_MIB",
                f"{_slot_prefix(slot)}_OVER_LIMIT_REJECTED",
            )
        ),
    }
)


def _fail(message: str) -> None:
    raise ProfileValidationError(message)


def _read_profile(path: Path) -> dict[str, str]:
    """Read a deliberately small, non-executable external environment profile."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        _fail(f"cannot read profile: {error}")
    profile: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROFILE_LINE_RE.fullmatch(line)
        if match is None:
            _fail(f"profile line {line_number} must be KEY=value or export KEY=value")
        key, value = match.groups()
        if key in profile:
            _fail(f"duplicate profile key: {key}")
        if not value:
            _fail(f"profile value is empty: {key}")
        profile[key] = value
    missing = REQUIRED_PROFILE_KEYS - profile.keys()
    unknown = profile.keys() - REQUIRED_PROFILE_KEYS
    if missing:
        _fail(f"profile missing required keys: {', '.join(sorted(missing))}")
    if unknown:
        _fail(f"profile has unknown keys: {', '.join(sorted(unknown))}")
    return profile


def _validate_external_profile_path(profile_path: Path, repository_root: Path) -> Path:
    """Resolve and reject a profile in this or any other Git worktree."""
    if not profile_path.is_absolute():
        _fail("profile path must be absolute")
    resolved_profile = profile_path.resolve()
    if resolved_profile.is_relative_to(repository_root.resolve()):
        _fail("profile path must be outside the repository root")
    try:
        git_root = subprocess.run(
            ["git", "-C", str(resolved_profile.parent), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        _fail(f"cannot establish Git repository/worktree boundary: {error}")
    if git_root.returncode == 0:
        root_text = git_root.stdout.strip()
        if not root_text:
            _fail("cannot establish Git repository/worktree boundary")
        if resolved_profile.is_relative_to(Path(root_text).resolve()):
            _fail("profile path must be outside every Git repository/worktree")
    return resolved_profile


def _positive_integer(value: str, name: str) -> int:
    if POSITIVE_INTEGER_RE.fullmatch(value) is None:
        _fail(f"{name} must be a positive integer")
    return int(value)


def _finite_positive(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError:
        _fail(f"{name} must be a finite positive number")
    if not math.isfinite(result) or result <= 0.0:
        _fail(f"{name} must be a finite positive number")
    return result


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_exact_keys(record: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(record)
    if actual != expected:
        missing = expected - actual
        unknown = actual - expected
        _fail(f"{label} keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _json_positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _fail(f"{label} must be a finite positive number")
    return result


def _json_positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{label} must be a positive integer")
    return value


def _load_evidence(path: Path, digest: str, repository_root: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute():
        _fail(f"{label} evidence path must be absolute")
    resolved_path = path.resolve()
    resolved_repo = repository_root.resolve()
    if resolved_path.is_relative_to(resolved_repo):
        _fail(f"{label} evidence path must be outside the repository")
    if SHA256_RE.fullmatch(digest) is None:
        _fail(f"{label} evidence sha256 must be lowercase hexadecimal")
    try:
        evidence_bytes = resolved_path.read_bytes()
    except OSError as error:
        _fail(f"cannot read {label} evidence: {error}")
    actual_digest = hashlib.sha256(evidence_bytes).hexdigest()
    if actual_digest != digest:
        _fail(f"{label} evidence sha256 does not match bytes")
    try:
        decoded = json.loads(evidence_bytes.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} evidence is not strict UTF-8 JSON: {error}")
    if not isinstance(decoded, dict):
        _fail(f"{label} evidence must be an object")
    return decoded


def _validate_evidence(
    evidence: dict[str, Any],
    *,
    slot: str,
    profile: dict[str, str],
    failure_limits: dict[str, int],
) -> dict[str, Any]:
    kind, target, resource_name = FAILURE_SLOTS[slot]
    prefix = _slot_prefix(slot)
    _require_exact_keys(
        evidence,
        {
            "schema", "fixture_kind", "configured_limits", "target_limit_name",
            "reached_value", "over_one_attempted_value", "over_one_rejected",
            "elapsed_ms", "peak_retained_rss_mib", "authority", "generated_at",
        },
        f"{slot} evidence",
    )
    if evidence["schema"] != SCHEMA:
        _fail(f"{slot} evidence schema mismatch")
    if evidence["fixture_kind"] != kind or evidence["target_limit_name"] != target:
        _fail(f"{slot} evidence slot/kind/target mismatch")
    configured_limits = evidence["configured_limits"]
    if not isinstance(configured_limits, dict):
        _fail(f"{slot} configured_limits must be an object")
    _require_exact_keys(configured_limits, set(failure_limits), f"{slot} configured_limits")
    parsed_limits = {
        name: _json_positive_integer(value, f"{slot} configured_limits.{name}")
        for name, value in configured_limits.items()
    }
    if parsed_limits != failure_limits:
        _fail(f"{slot} configured_limits do not match profile")
    reached = _json_positive_integer(evidence["reached_value"], f"{slot} reached_value")
    if reached != failure_limits[target]:
        _fail(f"{slot} reached_value does not equal target limit")
    if _json_positive_integer(evidence["over_one_attempted_value"], f"{slot} over_one_attempted_value") != reached + 1:
        _fail(f"{slot} over_one_attempted_value must be reached_value + 1")
    if evidence["over_one_rejected"] is not True:
        _fail(f"{slot} over_one_rejected must be true")
    elapsed = _json_positive_number(evidence["elapsed_ms"], f"{slot} elapsed_ms")
    rss = _json_positive_number(evidence["peak_retained_rss_mib"], f"{slot} peak_retained_rss_mib")
    if elapsed != _finite_positive(profile[f"{prefix}_SATURATION_ELAPSED_MS"], f"{slot} saturation elapsed"):
        _fail(f"{slot} saturation elapsed does not match evidence")
    if rss != _finite_positive(profile[f"{prefix}_PEAK_RETAINED_RSS_MIB"], f"{slot} peak retained RSS"):
        _fail(f"{slot} peak retained RSS does not match evidence")
    if _positive_integer(profile[f"{prefix}_VALIDATED_LIMIT"], f"{slot} validated limit") != reached:
        _fail(f"{slot} validated limit does not match evidence")
    if profile[f"{prefix}_OVER_LIMIT_REJECTED"] != "1":
        _fail(f"{slot} over-limit rejected must be 1")
    authority = evidence["authority"]
    if not isinstance(authority, dict):
        _fail(f"{slot} authority must be an object")
    _require_exact_keys(authority, {"environment", "image"}, f"{slot} authority")
    expected_authority = {
        "environment": profile["PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_ENVIRONMENT"],
        "image": profile["PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_IMAGE"],
    }
    if not all(isinstance(value, str) and value for value in authority.values()):
        _fail(f"{slot} authority fields must be nonempty strings")
    if authority != expected_authority:
        _fail(f"{slot} authority does not match profile")
    if not isinstance(evidence["generated_at"], str) or not evidence["generated_at"]:
        _fail(f"{slot} generated_at must be a nonempty record string")
    return evidence


def validate_profile(profile_path: Path, *, repository_root: Path) -> dict[str, Any]:
    """Validate external evidence without importing deployment runtime admission code."""
    profile = _read_profile(
        _validate_external_profile_path(profile_path, repository_root)
    )
    if profile["PURE_EXPLORATION_ORIN_VALIDATED"] != "1":
        _fail("PURE_EXPLORATION_ORIN_VALIDATED must be 1")
    resources = {
        name: _positive_integer(profile[_resource_env_name(name)], name)
        for name in RESOURCE_LIMITS
    }
    expected_product = resources["maximum_candidate_views"] * resources["maximum_visibility_work_units"]
    if _positive_integer(profile["PURE_EXPLORATION_VALIDATED_VISIBILITY_PRODUCT"], "validated visibility product") != expected_product:
        _fail("validated visibility product does not match candidate-view and visibility limits")
    whole_cycle_elapsed = _finite_positive(profile["PURE_EXPLORATION_VALIDATED_WHOLE_CYCLE_ELAPSED_MS"], "whole-cycle elapsed")
    whole_cycle_rss = _finite_positive(profile["PURE_EXPLORATION_VALIDATED_WHOLE_CYCLE_PEAK_RSS_MIB"], "whole-cycle peak RSS")
    if not profile["PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_ENVIRONMENT"] or not profile["PURE_EXPLORATION_FAILURE_EVIDENCE_AUTHORITY_IMAGE"]:
        _fail("failure evidence authority environment and image must be nonempty")
    failure_limits = {
        "maximum_entries": resources["maximum_failure_entries"],
        "maximum_patch_cells_per_entry": resources["maximum_failure_patch_cells_per_entry"],
        "maximum_total_patch_cells": resources["maximum_failure_total_patch_cells"],
    }
    paths: set[Path] = set()
    digests: set[str] = set()
    evidence: dict[str, dict[str, Any]] = {}
    for slot in FAILURE_SLOTS:
        prefix = _slot_prefix(slot)
        path = Path(profile[f"{prefix}_EVIDENCE_PATH"])
        digest = profile[f"{prefix}_EVIDENCE_SHA256"]
        if path.resolve() in paths:
            _fail(f"duplicate {slot} evidence path")
        if digest in digests:
            _fail(f"duplicate {slot} evidence sha256")
        paths.add(path.resolve())
        digests.add(digest)
        evidence[slot] = _validate_evidence(
            _load_evidence(path, digest, repository_root, slot),
            slot=slot,
            profile=profile,
            failure_limits=failure_limits,
        )
    return {
        "visibility_product": expected_product,
        "whole_cycle_elapsed_ms": whole_cycle_elapsed,
        "whole_cycle_peak_rss_mib": whole_cycle_rss,
        "failure_evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path, help="external env profile")
    parser.add_argument("--repository-root", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        validate_profile(arguments.profile, repository_root=arguments.repository_root)
    except ProfileValidationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
