"""Static contracts for the pure-frontier exploration ROS interfaces."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MESSAGE_DIRECTORY = (
    REPOSITORY_ROOT
    / "ros2_ws/src/lunar_pure_exploration_msgs/msg"
)

TASK_FIELDS = (
    "uint8 START=1",
    "uint8 PAUSE=2",
    "uint8 RESUME=3",
    "uint8 CANCEL=4",
    "std_msgs/Header header",
    "string task_id",
    "uint8 command",
    "geometry_msgs/Polygon boundary",
)

STATUS_FIELDS = (
    "uint8 IDLE=0",
    "uint8 WAITING_FOR_INPUT=1",
    "uint8 SELECTING_FRONTIER=2",
    "uint8 PLANNING=3",
    "uint8 EXECUTING=4",
    "uint8 REPLANNING=5",
    "uint8 PAUSED=6",
    "uint8 COMPLETED=7",
    "uint8 ERROR=8",
    "std_msgs/Header header",
    "string task_id",
    "uint8 state",
    "string reason_code",
    "float64 polygon_area_m2",
    "float64 task_raster_area_m2",
    "float64 known_free_area_m2",
    "float64 known_occupied_area_m2",
    "float64 unknown_area_m2",
    "float64 outside_map_area_m2",
    "float64 coverage_ratio",
    "uint32 frontier_cluster_count",
    "uint32 candidate_count",
    "uint32 reachable_candidate_count",
    "uint32 failed_candidate_count",
    "uint32 completed_goal_count",
    "uint32 replan_count",
    "string current_plan_id",
    "geometry_msgs/Pose current_goal",
    "float64 active_elapsed_s",
)


def message_lines(message_name: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in (MESSAGE_DIRECTORY / message_name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def assert_exact_message_contract(
    message_name: str, expected: tuple[str, ...]
) -> None:
    assert message_lines(message_name) == expected


def test_pure_exploration_task_has_exact_command_and_payload_contract() -> None:
    assert_exact_message_contract("PureExplorationTask.msg", TASK_FIELDS)


def test_pure_exploration_status_has_exact_state_and_metrics_contract() -> None:
    assert_exact_message_contract("PureExplorationStatus.msg", STATUS_FIELDS)


def test_message_contract_rejects_reordered_or_duplicate_expected_fields() -> None:
    actual = message_lines("PureExplorationTask.msg")
    assert actual == TASK_FIELDS
    assert actual != TASK_FIELDS[1:] + TASK_FIELDS[:1]
    assert actual != TASK_FIELDS + (TASK_FIELDS[-1],)
