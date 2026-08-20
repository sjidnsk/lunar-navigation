from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree


def test_smoke_client_declares_plan_motion_message_runtime_dependency() -> None:
    package_xml = Path(__file__).resolve().parents[1] / "package.xml"
    root = ElementTree.parse(package_xml).getroot()

    assert "lunar_planning_msgs" in {
        dependency.text for dependency in root.findall("exec_depend")
    }
