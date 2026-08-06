from pathlib import Path
import re
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
    "MotionExecutionFeedback.msg": """uint8 WHEELED=1
uint8 LEGGED=2
uint8 HOPPER=3
uint8 IDLE=0
uint8 ACCEPTED=1
uint8 EXECUTING=2
uint8 SEGMENT_COMPLETE=3
uint8 LANDED_HOLD=4
uint8 FAILED=5
uint8 CANCELED=6
std_msgs/Header header
uint64 sequence
uint8 platform_type
string plan_id
string segment_id
uint8 state
string reason_code
""",
    "HopperPropellantState.msg": """std_msgs/Header header
string platform_id
string capability_version
float64 total_mass_kg
float64 remaining_usable_fuel_mass_kg
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
    assert root.findtext("name") == "lunar_navigation_msgs"
    assert root.findtext("version") == "0.1.0"
    assert [node.text for node in root.findall("buildtool_depend")] == [
        "ament_cmake",
        "rosidl_default_generators",
    ]
    assert {node.text for node in root.findall("depend")} == {
        "geometry_msgs",
        "std_msgs",
    }
    assert [node.text for node in root.findall("exec_depend")] == [
        "rosidl_default_runtime"
    ]
    dependency_tags = {
        node.tag
        for node in root
        if node.tag.endswith("depend") or node.tag == "buildtool_depend"
    }
    assert dependency_tags == {"buildtool_depend", "depend", "exec_depend"}
    assert root.findtext("member_of_group") == "rosidl_interface_packages"

    cmake = (PACKAGE / "CMakeLists.txt").read_text(encoding="utf-8")
    found_packages = set(re.findall(r"find_package\((\w+) REQUIRED\)", cmake))
    assert found_packages == {
        "ament_cmake",
        "geometry_msgs",
        "rosidl_default_generators",
        "std_msgs",
    }
    dependencies = re.search(
        r"rosidl_generate_interfaces\(\$\{PROJECT_NAME\}.*?DEPENDENCIES\s+([^\n]+)",
        cmake,
        re.DOTALL,
    )
    assert dependencies is not None
    assert set(dependencies.group(1).split()) == {"geometry_msgs", "std_msgs"}


def test_package_declares_ament_cmake_build_type():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    assert root.findtext("export/build_type") == "ament_cmake"


def test_message_sources_match_the_authoritative_baseline():
    baseline = (ROOT / "docs/interfaces/external-input-baseline.md").read_text(encoding="utf-8")
    for source in EXPECTED.values():
        assert f"```text\n{source}```" in baseline


def test_no_same_name_upstream_is_pinned():
    assert (ROOT / "dependencies.repos").read_text(encoding="utf-8") == "repositories: {}\n"
