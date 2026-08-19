from __future__ import annotations

from pathlib import Path


def test_runtime_document_templates_describe_fallback_and_safe_prepare() -> None:
    root = Path(__file__).resolve().parents[2] / "deployment" / "docs"
    readme = (root / "README.runtime.md").read_text(encoding="utf-8")
    commands = (root / "COMMANDS.runtime.md").read_text(encoding="utf-8")
    assert "fallback" in readme
    assert "luna prepare --dry-run" in commands
    assert "POLICY_RUNTIME_UNBOUND" in commands
