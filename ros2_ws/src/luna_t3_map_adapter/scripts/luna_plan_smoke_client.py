#!/usr/bin/env python3
"""Send one explicit point request to the Luna PlanMotion action.

This client does not command a vehicle. An external controller owns execution
and must use the returned MotionReference identity in its feedback.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

import rclpy
from rclpy.action import ActionClient
from lunar_planning_msgs.action import PlanMotion


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-id", required=True)
    parser.add_argument("--mission-revision", required=True, type=int)
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--x", required=True, type=float)
    parser.add_argument("--y", required=True, type=float)
    parser.add_argument("--tolerance", required=True, type=float)
    parser.add_argument("--action-name", default="/plan_motion")
    parser.add_argument("--wait-server-s", default=5.0, type=float)
    parser.add_argument("--wait-result-s", default=30.0, type=float)
    parsed = parser.parse_args(arguments)
    if parsed.mission_revision < 0 or parsed.tolerance < 0.0:
        parser.error("mission revision and tolerance must be non-negative")
    return parsed


def _result_summary(result: PlanMotion.Result) -> dict[str, object]:
    reference = result.reference
    return {
        "planning_outcome": int(result.planning_outcome),
        "execution_directive": int(result.execution_directive),
        "reason_code": result.reason_code,
        "has_reference": bool(result.has_reference),
        "plan_id": reference.plan_id if result.has_reference else None,
        "platform_type": int(reference.platform_type) if result.has_reference else None,
    }


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    rclpy.init(args=None)
    node = rclpy.create_node("luna_plan_smoke_client")
    client = ActionClient(node, PlanMotion, args.action_name)
    try:
        if not client.wait_for_server(timeout_sec=args.wait_server_s):
            print(json.dumps({"reason": "PLAN_MOTION_SERVER_UNAVAILABLE"}))
            return 2
        goal = PlanMotion.Goal()
        goal.request_id = str(uuid.uuid4())
        goal.mission_id = args.mission_id
        goal.mission_revision = args.mission_revision
        goal.goal.goal_id = args.goal_id
        goal.goal.goal_type = goal.goal.POINT
        goal.goal.point.x = args.x
        goal.goal.point.y = args.y
        goal.goal.position_tolerance_m = args.tolerance
        goal.goal.has_yaw_constraint = False
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send, timeout_sec=args.wait_result_s)
        if not send.done() or send.result() is None or not send.result().accepted:
            print(json.dumps({"reason": "PLAN_MOTION_GOAL_REJECTED"}))
            return 3
        result = client.get_result_async(send.result())
        rclpy.spin_until_future_complete(node, result, timeout_sec=args.wait_result_s)
        if not result.done() or result.result() is None:
            print(json.dumps({"reason": "PLAN_MOTION_RESULT_TIMEOUT"}))
            return 4
        print(json.dumps(_result_summary(result.result().result), sort_keys=True))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
