"""Behavioral checks for the approved three-platform capability freeze."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = (
    ROOT
    / "ros2_ws/src/lunar_navigation_config/config/platform_capability_schema_v2.yaml"
)
FREEZE = (
    ROOT
    / "ros2_ws/src/lunar_navigation_config/config/three_platform_capability_freeze_v1.yaml"
)
CHECKER = ROOT / "tools/check_platform_capability_freeze.py"


def run_checker(schema: Path, freeze: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(CHECKER),
            "--schema",
            str(schema),
            "--freeze",
            str(freeze),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def write_mutated_freeze(tmp_path: Path, mutate) -> Path:
    document = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    mutate(document)
    output = tmp_path / "freeze.yaml"
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return output


def test_approved_freeze_is_complete_and_digest_verified():
    """Missing a platform, provenance, or matching digest must block the freeze."""
    result = run_checker(SCHEMA, FREEZE)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("capability freeze: OK sha256=")


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda document: document["platforms"].pop("HOPPER"),
            "platform set must be exactly LEGGED, HOPPER, WHEELED",
        ),
        (
            lambda document: document["platforms"]["LEGGED"]["terrain"].update(
                maximum_step_height_m=0.16
            ),
            "LEGGED.terrain.maximum_step_height_m must be 0.5",
        ),
        (
            lambda document: document["platforms"]["WHEELED"]["unknown_fields"].remove(
                "bare_mass_kg"
            ),
            "WHEELED unknown_fields",
        ),
        (
            lambda document: document["platforms"]["WHEELED"].update(
                platform_mass_kg=50.0
            ),
            "WHEELED has unexpected keys",
        ),
        (
            lambda document: document["platforms"]["HOPPER"].update(
                total_mass_kg=20.0
            ),
            "HOPPER has unexpected keys",
        ),
        (
            lambda document: document["platforms"]["HOPPER"]["sources"].update(
                specific_impulse_s="manufacturer_verified"
            ),
            "unsupported source type",
        ),
    ],
)
def test_checker_rejects_unapproved_capability_mutations(
    tmp_path: Path, mutate, reason: str
):
    """Proxy values, retired fields, and unsupported provenance must not pass."""
    result = run_checker(SCHEMA, write_mutated_freeze(tmp_path, mutate))

    assert result.returncode == 1
    assert reason in result.stdout


def test_checker_rejects_payload_drift_without_a_new_digest(tmp_path: Path):
    """Changing an approved value without regenerating the freeze must be detected."""
    mutated = write_mutated_freeze(
        tmp_path,
        lambda document: document["platforms"]["WHEELED"]["kinematics"].update(
            maximum_forward_speed_mps=1.4
        ),
    )

    result = run_checker(SCHEMA, mutated)

    assert result.returncode == 1
    assert "WHEELED.kinematics.maximum_forward_speed_mps must be 1.5" in result.stdout
    assert "freeze_digest_sha256 does not match platform payload" in result.stdout


def test_checker_rejects_missing_field_provenance(tmp_path: Path):
    """A runtime capability without its evidence class is not auditable."""
    mutated = write_mutated_freeze(
        tmp_path,
        lambda document: document["platforms"]["WHEELED"]["sources"].pop(
            "wheel_diameter_m"
        ),
    )

    result = run_checker(SCHEMA, mutated)

    assert result.returncode == 1
    assert "WHEELED sources missing: wheel_diameter_m" in result.stdout


def test_schema_and_freeze_are_utf8_yaml_documents():
    """A non-UTF-8 or non-mapping contract cannot be consumed deterministically."""
    for path in (SCHEMA, FREEZE):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(document, dict)
