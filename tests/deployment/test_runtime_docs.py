from __future__ import annotations

import json
from pathlib import Path


def test_runtime_document_templates_describe_fallback_and_safe_prepare() -> None:
    root = Path(__file__).resolve().parents[2] / "deployment" / "docs"
    readme = (root / "README.runtime.md").read_text(encoding="utf-8")
    commands = (root / "COMMANDS.runtime.md").read_text(encoding="utf-8")
    assert "fallback" in readme
    assert "luna prepare --dry-run" in commands
    assert "POLICY_RUNTIME_UNBOUND" in commands


def test_rendered_commands_document_every_public_subcommand_and_safe_policy_boundary() -> None:
    commands = (Path(__file__).resolve().parents[2] / "deployment" / "docs" / "COMMANDS.runtime.md").read_text(encoding="utf-8")
    for token in (
        "init", "prepare --dry-run", "prepare --apply", "doctor", "doctor --live",
        "config check", "build", "start", "stop", "status", "logs", "model install",
        "model activate", "model rollback", "extension", "bundle",
    ):
        assert f"luna {token}" in commands
    assert "ROSDEP_NOT_INITIALIZED" in commands
    assert "WAITING_FOR_EXTERNAL_INPUT" in commands
    assert "formal preflight" not in commands.lower()


def test_task3_documentation_declares_direct_inputs_and_plan_feedback_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    readme = (root / "deployment" / "docs" / "README.runtime.md").read_text(encoding="utf-8")
    commands = (root / "deployment" / "docs" / "COMMANDS.runtime.md").read_text(encoding="utf-8")
    smoke_client = root / "ros2_ws" / "src" / "luna_t3_map_adapter" / "scripts" / "luna_plan_smoke_client.py"

    for token in (
        "/Car/T3/mapping/grid_map",
        "/Car/T3/semantic/current_pose",
        "map -> odom -> base_link",
        "MotionReference",
        "MotionExecutionFeedback",
        "no generic controller command topic",
    ):
        assert token in readme
    for token in (
        "task3-adapted.runtime.yaml",
        "luna_plan_smoke_client.py",
        "--mission-id",
        "--mission-revision",
        "--goal-id",
        "--x",
        "--y",
        "--tolerance",
    ):
        assert token in commands
    assert smoke_client.is_file()


def test_runtime_schema_and_operator_docs_cover_optional_wheeled_execution() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "deployment" / "config" / "runtime.schema.json").read_text(encoding="utf-8"))
    readme = (root / "deployment" / "docs" / "README.runtime.md").read_text(encoding="utf-8")
    commands = (root / "deployment" / "docs" / "COMMANDS.runtime.md").read_text(encoding="utf-8")

    assert "controller" in schema["required"]
    assert schema["properties"]["controller"]["required"] == ["wheeled"]
    for token in (
        "controller.wheeled.enabled",
        "/Car/T5/Car_Cmd_Vel",
        "/mission/execution_goal",
        "/execution/wheeled_reference",
        "keyboard_teleop.py",
        "STALE_INPUT",
    ):
        assert token in readme or token in commands
