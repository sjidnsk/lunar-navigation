from lunar_planning_msgs.action import PlanMotion
from lunar_planning_msgs.msg import GoalRegion, MotionReference


def test_plan_motion_goal_surface_and_result_constants():
    assert PlanMotion.Goal.get_fields_and_field_types() == {
        "request_id": "string",
        "mission_id": "string",
        "mission_revision": "uint64",
        "goal": "lunar_planning_msgs/GoalRegion",
        "replace_active_request": "boolean",
    }
    assert PlanMotion.Result.STALE_INPUT == 5
    assert PlanMotion.Result.CONTINUE_COMMITTED_HOP == 3


def test_goal_and_motion_reference_constants():
    assert (GoalRegion.POINT, GoalRegion.PLANAR_REGION) == (1, 2)
    assert (MotionReference.WHEELED, MotionReference.LEGGED, MotionReference.HOPPER) == (1, 2, 3)
