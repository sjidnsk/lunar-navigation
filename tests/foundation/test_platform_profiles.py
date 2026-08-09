"""三个可直接替换平台 profile 的行为合同。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "ros2_ws/src/lunar_navigation_config/config"
SCHEMA = CONFIG_ROOT / "platform_profile.schema.json"
FREEZE = CONFIG_ROOT / "three_platform_capability_freeze_v1.yaml"
PROFILE_ROOT = CONFIG_ROOT / "platform_profiles"
CHECKER = ROOT / "tools/check_platform_profiles.py"
PROFILE_FILES = {
    "WHEELED": PROFILE_ROOT / "wheeled.yaml",
    "LEGGED": PROFILE_ROOT / "legged.yaml",
    "HOPPER": PROFILE_ROOT / "hopper.yaml",
}


def run_checker(*profiles: Path) -> subprocess.CompletedProcess[str]:
    command = [
        "python3",
        str(CHECKER),
        "--schema",
        str(SCHEMA),
        "--freeze",
        str(FREEZE),
    ]
    for profile in profiles:
        command.extend(("--profile", str(profile)))
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def load_profile(platform_type: str) -> dict[str, object]:
    path = PROFILE_FILES[platform_type]
    assert path.is_file(), f"missing formal profile: {path}"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def write_mutated_profile(
    tmp_path: Path,
    platform_type: str,
    mutate,
) -> Path:
    document = load_profile(platform_type)
    mutate(document)
    output = tmp_path / PROFILE_FILES[platform_type].name
    output.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def test_approved_profiles_are_complete_and_equivalent_to_training_freeze() -> None:
    """遗漏平台或改变正式值时，独立 profile 不能继续通过。"""
    result = run_checker(*PROFILE_FILES.values())

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("platform profiles: OK ")
    assert "WHEELED=" in result.stdout
    assert "LEGGED=" in result.stdout
    assert "HOPPER=" in result.stdout


@pytest.mark.parametrize("platform_type", tuple(PROFILE_FILES))
def test_each_profile_is_a_self_contained_runtime_file(
    tmp_path: Path,
    platform_type: str,
) -> None:
    """设备只复制一个 YAML 时仍应拥有平台和观测的全部运行字段。"""
    copied = tmp_path / "platform_profile.yaml"
    source = PROFILE_FILES[platform_type]
    assert source.is_file(), f"missing formal profile: {source}"
    shutil.copyfile(source, copied)

    result = run_checker(copied)
    document = yaml.safe_load(copied.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert set(document) == {
        "schema_version",
        "ownership",
        "platform",
        "observation",
        "assets",
        "sources",
        platform_type.lower(),
    }
    assert document["observation"] == {
        "sensor_range_m": 30.0,
        "sensor_fov_deg": 360.0,
    }
    assert document["assets"] == {"urdf": None, "meshes": []}


def test_checker_rejects_profile_value_drift(tmp_path: Path) -> None:
    """运行 profile 不能悄悄改变训练使用的轮式速度能力。"""
    mutated = write_mutated_profile(
        tmp_path,
        "WHEELED",
        lambda document: document["wheeled"].update(
            maximum_forward_speed_mps=9.0
        ),
    )

    result = run_checker(mutated)

    assert result.returncode == 1
    assert "maximum_forward_speed_mps must equal approved freeze" in result.stdout


def test_checker_rejects_missing_observation_field(tmp_path: Path) -> None:
    """缺少 FOV 时不能用隐式默认值启动。"""
    mutated = write_mutated_profile(
        tmp_path,
        "LEGGED",
        lambda document: document["observation"].pop("sensor_fov_deg"),
    )

    result = run_checker(mutated)

    assert result.returncode == 1
    assert "observation missing required key: sensor_fov_deg" in result.stdout


def test_hopper_reference_propellant_is_not_runtime_inventory() -> None:
    """飞跃 profile 只允许非递减的单跳参考推进剂语义。"""
    document = load_profile("HOPPER")

    assert document["hopper"]["reference_propellant_mass_kg"] == 0.2
    assert "remaining_usable_fuel_mass_kg" not in document["hopper"]
    assert "runtime_fuel" not in document["hopper"]


def test_profile_contract_files_are_utf8_documents() -> None:
    """固定配置必须能在 AGX 上按 UTF-8 直接读取。"""
    assert SCHEMA.is_file(), f"missing profile schema: {SCHEMA}"
    for path in (SCHEMA, *PROFILE_FILES.values()):
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text
