from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_ROOT = REPOSITORY_ROOT / "ros2_ws/src/lunar_planning_msgs"


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_legacy_plan_motion_action_keeps_its_abi_and_is_not_extended():
    lines = _lines(INTERFACE_ROOT / "action/PlanMotion.action")

    assert "uint8 environment_mode" in lines
    assert "lunar_planning_msgs/MotionReference reference" in lines
    assert "uint8 planning_outcome" in lines
    assert "float64 target_x_m" not in lines


def test_navigate_to_pose_action_matches_the_independent_session_contract():
    assert _lines(INTERFACE_ROOT / "action/NavigateToPose.action") == [
        "float64 target_x_m",
        "float64 target_y_m",
        "bool has_target_yaw",
        "float64 target_yaw_rad",
        "---",
        "uint8 GOAL_REACHED=0",
        "uint8 NO_PATH=1",
        "uint8 INVALID_GOAL=2",
        "uint8 MAP_UNAVAILABLE=3",
        "uint8 TIMEOUT=4",
        "uint8 CANCELED=5",
        "uint8 INTERNAL_ERROR=6",
        "uint8 outcome",
        "string reason_code",
        "uint64 last_segment_revision",
        "---",
        "uint8 PLANNING=0",
        "uint8 EXECUTING=1",
        "uint8 REPLANNING=2",
        "uint8 session_state",
        "uint64 planning_cycle",
        "uint64 active_segment_revision",
        "string reason_code",
    ]


def test_path_reference_matches_the_session_segment_contract():
    assert _lines(INTERFACE_ROOT / "msg/PathReference.msg") == [
        "uint8 ACTIVE=0",
        "uint8 INVALIDATED=1",
        "unique_identifier_msgs/UUID session_id",
        "uint64 segment_revision",
        "uint64 traversability_revision",
        "uint8 state",
        "bool reaches_final_goal",
        "nav_msgs/Path path",
    ]


def test_cmake_and_package_register_new_interfaces_and_uuid_dependency():
    cmake = (INTERFACE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package = (INTERFACE_ROOT / "package.xml").read_text(encoding="utf-8")

    assert '"action/PlanMotion.action"' in cmake
    assert '"action/NavigateToPose.action"' in cmake
    assert '"msg/PathReference.msg"' in cmake
    assert "find_package(unique_identifier_msgs REQUIRED)" in cmake
    assert "unique_identifier_msgs" in cmake.split("rosidl_generate_interfaces", 1)[1]
    assert "<depend>unique_identifier_msgs</depend>" in package
