"""Action-driven evidence recorder; no command or tracking publisher."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from grid_map_msgs.msg import GridMap
from lunar_planning_msgs.action import NavigateToPose
from lunar_planning_msgs.msg import PathReference, TrackingStatus
from diagnostic_msgs.msg import DiagnosticArray
from .plant import Scene
from .evidence import plan_found_with_active_reference, matching_stopped_completion
from .vehicle_sim import PREFIX, LATCHED


class Recorder(Node):
    def __init__(self):
        super().__init__('incremental_controller_case_recorder')
        self.client = ActionClient(self, NavigateToPose, PREFIX + '/navigate_to_pose')
        self.plant = {}
        self.map_seen = False
        self.references = {}
        self.tracking = []
        self.feedback = []
        self.diagnostics = []
        self.create_subscription(String, PREFIX + '/plant_state', self.on_plant, 10)
        self.create_subscription(GridMap, PREFIX + '/grid_map', self.on_map, LATCHED)
        self.create_subscription(PathReference, PREFIX + '/path_reference', self.on_reference, LATCHED)
        self.create_subscription(TrackingStatus, PREFIX + '/tracking_status', self.on_tracking, 10)
        self.create_subscription(DiagnosticArray, PREFIX + '/diagnostics', self.on_diagnostics, 10)

    def on_plant(self, msg):
        self.plant = json.loads(msg.data)

    def on_map(self, _):
        self.map_seen = True

    def on_reference(self, msg):
        if msg.state == PathReference.ACTIVE and msg.path.poses:
            key = (bytes(msg.session_id.uuid).hex(), msg.segment_revision)
            self.references[key] = dict(session_id=key[0], segment_revision=key[1],
                                        reaches_final_goal=msg.reaches_final_goal,
                                        pose_count=len(msg.path.poses))

    def on_tracking(self, msg):
        item = dict(session_id=bytes(msg.session_id.uuid).hex(), segment_revision=msg.segment_revision,
                    state=msg.state, direction=msg.direction, reason=msg.reason,
                    linear_speed_mps=msg.linear_speed_mps, angular_speed_radps=msg.angular_speed_radps)
        if not self.tracking or any(item[k] != self.tracking[-1][k]
                                   for k in ('session_id', 'segment_revision', 'state', 'direction', 'reason')):
            self.tracking.append(item)

    def on_feedback(self, msg):
        f = msg.feedback
        item = dict(reason_code=f.reason_code, planning_cycle=f.planning_cycle,
                    active_segment_revision=f.active_segment_revision, session_state=f.session_state)
        if not self.feedback or item != self.feedback[-1]:
            self.feedback.append(item)

    def on_diagnostics(self, msg):
        for status in msg.status:
            data = {v.key: v.value for v in status.values}
            if data and (not self.diagnostics or data != self.diagnostics[-1]):
                self.diagnostics.append(data)


def spin_until(node, predicate, deadline):
    while rclpy.ok() and time.monotonic() < deadline:
        if predicate():
            return True
        rclpy.spin_once(node, timeout_sec=.1)
    return predicate()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=Scene.GOALS, default='forward')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=240.)
    opts, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = Recorder()
    start = time.monotonic()
    result = None
    failure = None
    goal_handle = None
    try:
        if not spin_until(node, lambda: node.map_seen and bool(node.plant) and node.client.server_is_ready(), start+30):
            failure = 'READINESS_TIMEOUT'
        elif node.plant.get('case') != opts.case:
            failure = 'SCENE_MISMATCH'
        else:
            # Allow the first elevation snapshot to reach the formal planner.
            spin_until(node, lambda: False, time.monotonic()+2)
            x, y, yaw = Scene(opts.case).goal
            goal = NavigateToPose.Goal(target_x_m=x, target_y_m=y,
                                       has_target_yaw=yaw is not None, target_yaw_rad=yaw or 0.)
            future = node.client.send_goal_async(goal, feedback_callback=node.on_feedback)
            deadline = time.monotonic()+opts.timeout
            if not spin_until(node, future.done, deadline):
                failure = 'GOAL_RESPONSE_TIMEOUT'
            else:
                goal_handle = future.result()
                if not goal_handle.accepted:
                    failure = 'GOAL_REJECTED'
                else:
                    future = goal_handle.get_result_async()
                    if not spin_until(node, future.done, deadline):
                        failure = 'RUN_TIMEOUT'
                        cancel = goal_handle.cancel_goal_async()
                        spin_until(node, cancel.done, time.monotonic()+5)
                    else:
                        response = future.result()
                        result = dict(action_status=response.status, outcome=response.result.outcome,
                                      reason_code=response.result.reason_code,
                                      last_segment_revision=response.result.last_segment_revision)
                    spin_until(node, lambda: False, time.monotonic()+1)
    finally:
        expected_session = bytes(goal_handle.goal_id.uuid).hex() if goal_handle else None
        refs = [r for r in node.references.values() if r['session_id'] == expected_session]
        stopped = bool(node.plant) and abs(node.plant['v']) <= .01 and abs(node.plant['w']) <= .02
        completion = [t for t in node.tracking if result is not None and
                      matching_stopped_completion(t, expected_session,
                          result['last_segment_revision'], node.references, TrackingStatus.COMPLETED)]
        planned = plan_found_with_active_reference(expected_session, node.references,
                                                   node.feedback, node.diagnostics)
        safe = bool(node.plant) and all(node.plant.get(k, 1) == 0 for k in
                    ('raw_command_limit_violations', 'raw_invalid_commands', 'collisions'))
        reverse = opts.case != 'reverse' or node.plant.get('max_actual_reverse', 0) > .05
        rolling = opts.case != 'multiple_exits' or any(not r['reaches_final_goal'] for r in refs)
        x, y, yaw = Scene(opts.case).goal
        distance = math.hypot(node.plant.get('x', math.inf)-x, node.plant.get('y', math.inf)-y)
        yaw_error = abs(math.remainder(node.plant.get('yaw', 0)-(yaw or 0), 2*math.pi)) if yaw is not None else None
        geometry = distance <= .35 and (yaw_error is None or yaw_error <= .2)
        passed = (failure is None and result is not None and result['action_status'] == 4 and result['outcome'] == 0 and
                  result['reason_code'] == 'GOAL_REACHED' and planned and bool(completion) and
                  stopped and safe and reverse and rolling and geometry)
        report = dict(schema='lunar-controller-demo-evidence/v1', case=opts.case,
                      requested_goal=dict(x_m=x, y_m=y, yaw_rad=yaw),
                      runtime=f"{os.environ.get('ROS_DISTRO', 'unknown')} command-driven simulation", passed=passed, failure=failure,
                      elapsed_wall_s=time.monotonic()-start, expected_session_id=expected_session, action_result=result,
                      gates=dict(plan_found_with_active_reference=planned, matching_final_completion=bool(completion),
                                 actual_stopped=stopped, no_speed_violation_or_collision=safe,
                                 reverse_motion_observed=reverse, nonfinal_reference_observed=rolling,
                                 goal_geometry=geometry), goal_distance_m=distance, goal_yaw_error_rad=yaw_error,
                      plant=node.plant, references=refs, tracking_transitions=node.tracking,
                      navigation_feedback=node.feedback, diagnostics=node.diagnostics,
                      boundaries=dict(native_Orin='NOT_RUN', vehicle='NOT_RUN'))
        output = Path(opts.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(dict(output=str(output), passed=passed, failure=failure, gates=report['gates'])))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(0 if passed else 1)
