from __future__ import annotations

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
