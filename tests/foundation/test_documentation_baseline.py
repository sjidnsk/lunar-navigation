from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_GPU_DOCS = (
    "AGENTS.md",
    "README.md",
    "docs/architecture/system-ownership.md",
    "docs/superpowers/specs/2026-08-02-lunar-navigation-greenfield-ros2-jetson-design.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-greenfield-roadmap.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-1-foundation.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-3-policy-pipeline.md",
    "docs/superpowers/plans/2026-08-02-lunar-navigation-volume-4-integration-cutover.md",
)


def test_active_documents_use_rtx4080_super_names():
    stale = []
    for relative in ACTIVE_GPU_DOCS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if re.search(r"RTX 4080(?! SUPER)", text) or re.search(r"rtx4080(?!_super)", text):
            stale.append(relative)
    assert stale == []


def test_external_baseline_declares_provisional_schema_authority():
    text = (ROOT / "docs/interfaces/external-input-baseline.md").read_text(encoding="utf-8")
    assert "暂定消息字段与语义的唯一权威基线" in text
    assert "Topic 数据生产者仍由外部项目拥有" in text
    assert "不得据此复制" not in text


def test_agent_rules_record_the_approved_provisional_exception():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "暂定提供同名 schema" in text
    assert "不得与上游同名包共存" in text
