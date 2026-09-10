"""Evidence must join real ROS messages without counting scans or unsafe motion.

The recorder and task starter are the production nodes. A small ROS Action
server supplies controlled navigation results; it does not simulate planning.
"""
import json
import math
from pathlib import Path
import sys
import time

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

pytest.importorskip('rclpy')
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionServer, GoalResponse
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from unique_identifier_msgs.msg import UUID

from lunar_planning_msgs.action import NavigateToPose
from lunar_planning_msgs.msg import PathReference, TrackingStatus
from lunar_pure_exploration_msgs.msg import PureExplorationStatus, PureExplorationTask
from lunar_integrated_exploration_demo.record import Recorder
from lunar_integrated_exploration_demo.task import PREFIX, TaskStarter

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
FIRST = '00112233445566778899aabbccddeeff'
SECOND = '102132435465768798a9bacbdcedfe0f'
SCAN = 'ffeeddccbbaa99887766554433221100'


class RosRuntime:
    """Keep test-owned ROS resources and bounded spinning outside the nodes."""

    def __init__(self):
        self.executor = SingleThreadedExecutor()
        self.nodes = []
        self.receipt_publisher = None

    def add(self, node):
        self.nodes.append(node)
        self.executor.add_node(node)
        return node

    def until(self, predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)
        assert predicate(), 'expected ROS observation was not received before the deadline'

    def spin_for(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=max(0.0, min(0.02, deadline - time.monotonic())))

    def receipt(self, node, sessions):
        if self.receipt_publisher is None:
            writer = self.add(Node('bootstrap_receipt_fixture'))
            self.receipt_publisher = writer.create_publisher(
                String, PREFIX + '/bootstrap_sessions', LATCHED)
        self.receipt_publisher.publish(String(data=json.dumps(sessions)))
        self.until(lambda: node.bootstrap_ids == set(sessions))

    def close(self):
        for node in reversed(self.nodes):
            self.executor.remove_node(node)
            node.destroy_node()
        self.executor.shutdown()


@pytest.fixture
def runtime(request):
    # This domain is distinct from both the interactive demos and vehicle tests.
    rclpy.init(args=getattr(request, 'param', []), domain_id=188)
    result = RosRuntime()
    try:
        yield result
    finally:
        result.close()
        rclpy.try_shutdown()


def uuid(session):
    return UUID(uuid=list(bytes.fromhex(session)))


def reference(session, segment=1, fine=17, final=True):
    message = PathReference(session_id=uuid(session), segment_revision=segment,
                            traversability_revision=fine, state=PathReference.ACTIVE,
                            reaches_final_goal=final)
    message.path.header.frame_id = 'map'
    message.path.poses = [PoseStamped()]
    return message


def diagnostic(session, segment=1, fine=17, result='PLAN_FOUND'):
    values = {'session_id': session, 'segment_revision': str(segment),
              'fine_traversability_revision': str(fine),
              'cycle_result': result, 'path_state': 'ACTIVE'}
    return DiagnosticArray(status=[DiagnosticStatus(
        name='lunar_incremental_navigation/planning_cycle',
        values=[KeyValue(key=key, value=value) for key, value in values.items()])])


def feedback(session, segment=1, linear=0.0, angular=0.0):
    return TrackingStatus(session_id=uuid(session), segment_revision=segment,
                          state=TrackingStatus.COMPLETED,
                          linear_speed_mps=linear, angular_speed_radps=angular)


def completed_reference(node, session, segment=1, fine=17, final=True):
    node.on_reference(reference(session, segment, fine, final))
    node.on_diag(diagnostic(session, segment, fine))
    node.on_tracking(feedback(session, segment))


def prepare_measurements(node):
    node.plant = {'raw_limit_violations': 0, 'raw_invalid_commands': 0,
                  'raw_angular_limit_violations': 0, 'collisions': 0,
                  'max_actual_forward': 0.2, 'max_actual_reverse': 0.2}
    node.coarse_res, node.fine_res = 1.0, 0.20000000298023224
    for revision in range(1, 6):
        node.on_diag(DiagnosticArray(status=[DiagnosticStatus(values=[
            KeyValue(key='fine_traversability_revision', value=str(revision))])]))
    node.on_status(PureExplorationStatus(
        task_id='test-exploration', state=PureExplorationStatus.EXECUTING,
        completed_goal_count=2, reason_code='NAVIGATION_EXECUTING'))


@pytest.fixture
def recorder(runtime):
    node = runtime.add(Recorder())
    prepare_measurements(node)
    runtime.receipt(node, [SCAN])
    return node


def test_recorder_waits_for_explicit_bootstrap_receipt_even_when_list_is_empty(runtime):
    node = runtime.add(Recorder())
    prepare_measurements(node)
    completed_reference(node, FIRST)
    completed_reference(node, SECOND)
    assert not all(node.gates(2, False).values()), (
        'independent publishers may deliver completion evidence before the scan receipt')

    runtime.receipt(node, [])
    runtime.until(lambda: all(node.gates(2, False).values()))


def test_matching_final_sessions_pass_but_completion_requires_real_terminal_state(recorder):
    completed_reference(recorder, FIRST)
    completed_reference(recorder, SECOND)
    assert all(recorder.gates(2, False).values())
    assert not recorder.gates(2, True)['requested_terminal_condition']

    recorder.on_status(PureExplorationStatus(
        task_id='test-exploration', state=PureExplorationStatus.COMPLETED,
        completed_goal_count=2, reason_code='COVERAGE_TARGET_REACHED'))
    assert all(recorder.gates(2, True).values())


@pytest.mark.parametrize('missing', ['reference', 'diagnostic', 'feedback'])
def test_each_formal_evidence_source_is_required(recorder, missing):
    completed_reference(recorder, FIRST)
    if missing != 'reference':
        recorder.on_reference(reference(SECOND))
    if missing != 'diagnostic':
        recorder.on_diag(diagnostic(SECOND))
    if missing != 'feedback':
        recorder.on_tracking(feedback(SECOND))
    assert not recorder.gates(2, False)['formal_plan_and_stopped_feedback']


@pytest.mark.parametrize('invalid', [
    'nonfinal', 'invalidated', 'empty_path', 'fine_mismatch', 'not_plan_found',
    'different_session', 'different_segment', 'moving', 'rotating',
    'nonfinite_linear', 'nonfinite_angular', 'not_completed',
])
def test_local_or_mismatched_or_unstopped_evidence_cannot_complete_a_goal(recorder, invalid):
    completed_reference(recorder, FIRST)
    path, plan, stopped = reference(SECOND), diagnostic(SECOND), feedback(SECOND)
    if invalid == 'nonfinal':
        path.reaches_final_goal = False
    elif invalid == 'invalidated':
        path.state = PathReference.INVALIDATED
    elif invalid == 'empty_path':
        path.path.poses = []
    elif invalid == 'fine_mismatch':
        plan = diagnostic(SECOND, fine=18)
    elif invalid == 'not_plan_found':
        plan = diagnostic(SECOND, result='LOCAL_NO_PROGRESS')
    elif invalid == 'different_session':
        stopped.session_id = uuid(FIRST)
    elif invalid == 'different_segment':
        stopped.segment_revision = 2
    elif invalid == 'moving':
        stopped.linear_speed_mps = 0.02
    elif invalid == 'rotating':
        stopped.angular_speed_radps = 0.04
    elif invalid == 'nonfinite_linear':
        stopped.linear_speed_mps = math.nan
    elif invalid == 'nonfinite_angular':
        stopped.angular_speed_radps = math.inf
    elif invalid == 'not_completed':
        stopped.state = TrackingStatus.FINAL_ALIGN
    recorder.on_reference(path)
    recorder.on_diag(plan)
    recorder.on_tracking(stopped)
    assert not recorder.gates(2, False)['formal_plan_and_stopped_feedback']


def test_multiple_completed_revisions_of_one_session_count_as_one_goal(recorder):
    completed_reference(recorder, FIRST, segment=1)
    completed_reference(recorder, FIRST, segment=2, fine=18)
    assert not recorder.gates(2, False)['formal_plan_and_stopped_feedback']
    completed_reference(recorder, SECOND)
    assert recorder.gates(2, False)['formal_plan_and_stopped_feedback']


def test_bootstrap_session_never_contributes_to_exploration_goal_count(recorder):
    completed_reference(recorder, SCAN)
    completed_reference(recorder, FIRST)
    assert not recorder.gates(2, False)['formal_plan_and_stopped_feedback']
    completed_reference(recorder, SECOND)
    assert recorder.gates(2, False)['formal_plan_and_stopped_feedback']


@pytest.mark.parametrize('field,value', [
    ('raw_limit_violations', 1), ('raw_invalid_commands', 1),
    ('raw_angular_limit_violations', 1), ('collisions', 1),
    ('max_actual_forward', 0.201), ('max_actual_reverse', 0.201),
    ('max_actual_forward', math.nan), ('max_actual_reverse', math.inf),
])
def test_bad_command_or_actual_motion_evidence_fails_the_safety_gate(recorder, field, value):
    completed_reference(recorder, FIRST)
    completed_reference(recorder, SECOND)
    assert all(recorder.gates(2, False).values())
    recorder.plant[field] = value
    assert not recorder.gates(2, False)['no_raw_speed_violation_or_collision']


def start_bootstrap(runtime, *, outcome=NavigateToPose.Result.GOAL_REACHED,
                    reason='GOAL_REACHED', reject=False):
    server_node = runtime.add(Node('controlled_navigation_server'))
    requests, sessions, tasks = [], [], []

    def execute(handle):
        requests.append(handle.request)
        sessions.append(bytes(handle.goal_id.uuid).hex())
        # Deliberately succeed at the transport level even for a business error.
        # TaskStarter must inspect the navigation result, not action status alone.
        handle.succeed()
        return NavigateToPose.Result(outcome=outcome, reason_code=reason,
                                     last_segment_revision=1)

    server = ActionServer(server_node, NavigateToPose, PREFIX + '/navigate_to_pose',
                          execute_callback=execute,
                          goal_callback=lambda _: GoalResponse.REJECT if reject else GoalResponse.ACCEPT)
    server_node.create_subscription(PureExplorationTask, PREFIX + '/exploration/task',
                                     tasks.append, 10)
    starter = runtime.add(TaskStarter())
    starter.on_map(OccupancyGrid(data=[-1, 0, 100]))
    pose = Odometry()
    pose.pose.pose.position.x, pose.pose.pose.position.y = 1.25, -0.5
    starter.on_pose(pose)
    return starter, server, requests, sessions, tasks


def test_completed_scan_sessions_are_latched_and_excluded_by_late_recorder(runtime):
    starter, server, requests, sessions, tasks = start_bootstrap(runtime)
    try:
        runtime.until(lambda: bool(tasks))
        assert starter.phase == 'EXPLORATION_TASK_STARTED'
        assert len(requests) == len(sessions) == 4
        assert [request.target_yaw_rad for request in requests] == pytest.approx(
            [math.pi / 2, math.pi, -math.pi / 2, 0.0])
        assert all(request.has_target_yaw and request.target_x_m == 1.25 and
                   request.target_y_m == -0.5 for request in requests)
        assert len(tasks) == 1
        assert tasks[0].command == PureExplorationTask.START
        assert tasks[0].header.frame_id == 'map'
        assert [(p.x, p.y) for p in tasks[0].boundary.points] == [
            (-150.0, -150.0), (150.0, -150.0), (150.0, 150.0), (-150.0, 150.0)]

        late = runtime.add(Recorder())
        runtime.until(lambda: late.bootstrap_ids == set(sessions))
        prepare_measurements(late)
        for session in sessions:
            completed_reference(late, session)
        assert not late.gates(2, False)['formal_plan_and_stopped_feedback']
        completed_reference(late, FIRST)
        completed_reference(late, SECOND)
        assert all(late.gates(2, False).values())
    finally:
        server.destroy()


def test_pending_scan_does_not_publish_an_empty_success_receipt(runtime):
    starter = runtime.add(TaskStarter())
    observer = runtime.add(Recorder())
    runtime.until(lambda: starter.sessions_pub.get_subscription_count() == 1)
    runtime.spin_for(0.1)
    assert starter.phase == 'WAITING_FOR_MAP_AND_STATE'
    assert not observer.bootstrap_receipt_seen


@pytest.mark.parametrize('runtime', [
    ['--ros-args', '-p', 'integrated_task_starter:initial_scan:=false'],
], indirect=True)
def test_disabled_scan_publishes_explicit_empty_receipt_when_starting_task(runtime):
    starter, server, requests, sessions, tasks = start_bootstrap(runtime)
    try:
        runtime.until(lambda: bool(tasks))
        assert starter.phase == 'EXPLORATION_TASK_STARTED'
        assert not requests and not sessions
        late = runtime.add(Recorder())
        runtime.until(lambda: late.bootstrap_receipt_seen)
        assert late.bootstrap_ids == set()
        prepare_measurements(late)
        completed_reference(late, FIRST)
        completed_reference(late, SECOND)
        assert all(late.gates(2, False).values())
    finally:
        server.destroy()


@pytest.mark.parametrize('outcome,reason', [
    (NavigateToPose.Result.NO_PATH, 'NO_PATH'),
    (NavigateToPose.Result.GOAL_REACHED, 'PLAN_FOUND'),
    (NavigateToPose.Result.INTERNAL_ERROR, 'INTERNAL_ERROR'),
])
def test_transport_success_with_failed_scan_never_starts_exploration(runtime, outcome, reason):
    starter, server, requests, sessions, tasks = start_bootstrap(
        runtime, outcome=outcome, reason=reason)
    try:
        runtime.until(lambda: starter.phase == 'BOOTSTRAP_FAILED')
        for _ in range(3):
            starter.tick()
        assert not tasks
        assert len(requests) == 1
        late = runtime.add(Recorder())
        runtime.until(lambda: starter.sessions_pub.get_subscription_count() == 1)
        runtime.spin_for(0.1)
        assert len(sessions) == 1
        assert not late.bootstrap_receipt_seen
    finally:
        server.destroy()


def test_rejected_scan_never_publishes_exploration_task(runtime):
    starter, server, requests, sessions, tasks = start_bootstrap(runtime, reject=True)
    try:
        runtime.until(lambda: starter.phase == 'BOOTSTRAP_FAILED')
        for _ in range(3):
            starter.tick()
        assert not tasks and not requests and not sessions
        assert not starter.bootstrap_ids
    finally:
        server.destroy()
