from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros2_ws/src/lunar_navigation_msgs"
EXPECTED = {
    "LocalizationStatus.msg": """uint8 UNKNOWN=0
uint8 VALID=1
uint8 DEGRADED=2
uint8 INVALID=3
uint8 RELOCALIZING=4
std_msgs/Header header
uint8 status
""",
    "ScienceTargetRegion.msg": """string region_id
string objective_id
geometry_msgs/Polygon boundary
float64 priority
""",
    "ExplorationTask.msg": """uint8 ACTIVE=1
uint8 PAUSED=2
uint8 CANCELED=3
std_msgs/Header header
string mission_id
uint64 revision
uint8 desired_state
float64 roi_min_x_m
float64 roi_min_y_m
float64 roi_max_x_m
float64 roi_max_y_m
lunar_navigation_msgs/ScienceTargetRegion[<=64] science_regions
""",
}


def test_provisional_message_sources_are_exact():
    actual = {
        path.name: path.read_text(encoding="utf-8")
        for path in (PACKAGE / "msg").glob("*.msg")
    }
    assert actual == EXPECTED


def test_package_declares_only_required_rosidl_dependencies():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    names = [element.text for element in root]
    assert root.findtext("name") == "lunar_navigation_msgs"
    assert root.findtext("version") == "0.1.0"
    assert "rosidl_default_generators" in names
    assert "rosidl_default_runtime" in names
    assert "geometry_msgs" in names
    assert "std_msgs" in names
    assert root.findtext("member_of_group") == "rosidl_interface_packages"


def test_package_declares_ament_cmake_build_type():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    assert root.findtext("export/build_type") == "ament_cmake"


def test_message_sources_match_the_authoritative_baseline():
    baseline = (ROOT / "docs/interfaces/external-input-baseline.md").read_text(encoding="utf-8")
    for source in EXPECTED.values():
        assert f"```text\n{source}```" in baseline


def test_no_same_name_upstream_is_pinned():
    assert (ROOT / "dependencies.repos").read_text(encoding="utf-8") == "repositories: {}\n"
