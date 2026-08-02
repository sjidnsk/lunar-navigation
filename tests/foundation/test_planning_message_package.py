from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros2_ws/src/lunar_planning_msgs"
EXPECTED_MESSAGES = {
    "GoalRegion.msg": """uint8 POINT=1
uint8 PLANAR_REGION=2
std_msgs/Header header
string goal_id
uint8 goal_type
geometry_msgs/Point point
geometry_msgs/Polygon planar_region
float64 position_tolerance_m
bool has_yaw_constraint
float64 yaw_rad
float64 yaw_tolerance_rad
""",
    "HopSegment.msg": """std_msgs/Header header
string segment_id
geometry_msgs/Pose launch_pose
geometry_msgs/Polygon landing_region
builtin_interfaces/Duration flight_time
geometry_msgs/Vector3 launch_velocity
float64 flight_tube_radius_m
""",
    "MotionReference.msg": """uint8 WHEELED=1
uint8 LEGGED=2
uint8 HOPPER=3
std_msgs/Header header
string plan_id
uint8 platform_type
builtin_interfaces/Time input_time
nav_msgs/Path path_preview
trajectory_msgs/MultiDOFJointTrajectory trajectory
lunar_planning_msgs/HopSegment[] hops
""",
    "PlannerDiagnostics.msg": """string planner_name
float64 elapsed_s
uint64 expanded_states
bool has_best_cost
float64 best_cost
string[] warning_codes
""",
}
EXPECTED_ACTION = """string request_id
string mission_id
uint64 mission_revision
lunar_planning_msgs/GoalRegion goal
bool replace_active_request
---
uint8 NEW_REFERENCE_AVAILABLE=0
uint8 SAFE_FRONTIER_REFERENCE_AVAILABLE=1
uint8 NO_KNOWN_SAFE_ROUTE=2
uint8 GOAL_INFEASIBLE=3
uint8 INVALID_REQUEST=4
uint8 STALE_INPUT=5
uint8 NUMERICAL_FAILURE=6
uint8 RESOURCE_EXHAUSTED=7
uint8 ACTIVE_REFERENCE_INVALIDATED=8
uint8 CANCELED=9
uint8 ACTIVATE_NEW_REFERENCE=0
uint8 CONTINUE_ACTIVE_REFERENCE=1
uint8 HOLD_POSITION=2
uint8 CONTINUE_COMMITTED_HOP=3
uint8 NO_SAFE_REFERENCE=4
uint8 planning_outcome
uint8 execution_directive
string reason_code
builtin_interfaces/Time global_map_stamp
builtin_interfaces/Time local_map_stamp
builtin_interfaces/Time state_stamp
uint64 mission_revision
bool has_reference
lunar_planning_msgs/MotionReference reference
lunar_planning_msgs/PlannerDiagnostics diagnostics
---
uint8 VALIDATING_INPUT=0
uint8 BUILDING_SNAPSHOT=1
uint8 SEARCHING=2
uint8 OPTIMIZING=3
uint8 CERTIFYING=4
uint8 phase
float64 elapsed_s
uint64 expanded_states
bool has_best_cost
float64 best_cost
"""


def test_internal_planning_message_sources_are_exact():
    actual = {
        path.name: path.read_text(encoding="utf-8")
        for path in (PACKAGE / "msg").glob("*.msg")
    }
    assert actual == EXPECTED_MESSAGES
    assert (PACKAGE / "action/PlanMotion.action").read_text(encoding="utf-8") == EXPECTED_ACTION


def test_package_declares_ament_cmake_build_type():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    assert root.findtext("export/build_type") == "ament_cmake"
