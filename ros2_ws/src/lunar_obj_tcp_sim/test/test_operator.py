import signal
from types import SimpleNamespace
import time

from lunar_obj_tcp_sim.operator import (
    _cancel_response_has_no_active_goals,
    _receipt_is_fresh,
    _twist_is_stopped,
    finish_process_group,
)


class FakeProcess:
    pid = 4242

    def __init__(self, returncode=None, events=None):
        self.returncode = returncode
        self.events = events if events is not None else []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def send_signal(self, sig):
        self.events.append(("process_signal", self.pid, sig))


class FakeStopClient:
    def __init__(self, events, confirmed=True):
        self.events = events
        self.confirmed = confirmed

    def request_and_confirm_stop(self, timeout_s):
        self.events.append(("cancel_and_confirm", timeout_s))
        return self.confirmed


def test_ordered_shutdown_confirms_ros_stop_before_interrupting_launch_group():
    events = []
    process = FakeProcess(events=events)

    confirmed = finish_process_group(
        process,
        FakeStopClient(events),
        timeout_s=5.0,
        signal_group=lambda pid, sig: events.append(("signal", pid, sig)),
    )

    assert confirmed is True
    assert events == [
        ("cancel_and_confirm", 4.0),
        ("process_signal", 4242, signal.SIGINT),
    ]


def test_unconfirmed_stop_is_reported_but_launch_group_is_still_cleaned_up():
    events = []
    process = FakeProcess(returncode=7, events=events)

    confirmed = finish_process_group(
        process,
        FakeStopClient(events, confirmed=False),
        timeout_s=2.0,
        signal_group=lambda pid, sig: events.append(("signal", pid, sig)),
    )

    assert confirmed is False
    assert events == [
        ("cancel_and_confirm", 1.0),
        ("process_signal", 4242, signal.SIGINT),
    ]


def test_stopped_twist_checks_lateral_and_all_angular_components():
    zero = SimpleNamespace(
        linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    lateral = SimpleNamespace(
        linear=SimpleNamespace(x=0.0, y=0.02, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    roll = SimpleNamespace(
        linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        angular=SimpleNamespace(x=0.01, y=0.0, z=0.0),
    )
    assert _twist_is_stopped(zero)
    assert not _twist_is_stopped(lateral)
    assert not _twist_is_stopped(roll)


def test_empty_cancel_list_is_terminal_only_when_request_was_accepted():
    accepted = SimpleNamespace(return_code=0, goals_canceling=[])
    rejected = SimpleNamespace(return_code=1, goals_canceling=[])
    assert _cancel_response_has_no_active_goals(accepted, error_none=0)
    assert not _cancel_response_has_no_active_goals(rejected, error_none=0)


def test_stop_receipt_must_still_be_fresh_when_confirmation_is_checked():
    assert _receipt_is_fresh(received_at=10.2, started_at=10.0, now=11.1)
    assert not _receipt_is_fresh(received_at=10.2, started_at=10.0, now=11.3)
    assert not _receipt_is_fresh(received_at=9.9, started_at=10.0, now=10.1)


def test_cancel_waits_for_discovery_and_retries_before_confirmation():
    from lunar_obj_tcp_sim.operator import RosStopClient

    events = []
    client = RosStopClient.__new__(RosStopClient)
    client.PureExplorationTask = type(
        "Task", (), {"CANCEL": 4, "__init__": lambda self: setattr(
            self, "header", SimpleNamespace(stamp=None)
        )}
    )
    client.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: 1))
    )
    client.task_publisher = SimpleNamespace(
        get_subscription_count=lambda: int("discovered" in events),
        publish=lambda _task: events.append("publish"),
    )
    client.cancel_client = SimpleNamespace(service_is_ready=lambda: False)
    client.exploration_terminal = False
    client.goals_terminal = False
    client.odom_stopped = client.command_stopped = False
    spins = 0

    def spin_once(_node, timeout_sec):
        nonlocal spins
        time.sleep(timeout_sec)
        spins += 1
        if spins == 1:
            events.append("discovered")
        if spins >= 8:
            now = time.monotonic()
            client.exploration_terminal = client.goals_terminal = True
            client.odom_stopped = client.command_stopped = True
            client.odom_received_at = client.command_received_at = now

    client.executor = SimpleNamespace(spin_once=lambda timeout_sec: spin_once(None, timeout_sec))

    assert client.request_and_confirm_stop(0.8)
    assert events[0] == "discovered"
    assert events.count("publish") >= 2


def test_odometry_stop_requires_a_fresh_header_and_new_receipt():
    from nav_msgs.msg import Odometry
    from rclpy.clock import Clock
    from lunar_obj_tcp_sim.operator import RosStopClient

    client = RosStopClient.__new__(RosStopClient)
    clock = Clock()
    client.node = SimpleNamespace(get_clock=lambda: clock)
    client.started_at = time.monotonic()
    stale = Odometry()
    client._on_odometry(stale)
    assert not client.odom_stopped

    fresh = Odometry()
    fresh.header.stamp = clock.now().to_msg()
    client._on_odometry(fresh)
    assert client.odom_stopped
    assert client.odom_received_at >= client.started_at


def test_transient_stop_client_uses_its_custom_context_executor():
    from lunar_obj_tcp_sim.operator import RosStopClient

    client = RosStopClient("/operator_loopback", "nav")
    try:
        client.executor.spin_once(timeout_sec=0.01)
    finally:
        client.close()


def test_real_idle_status_confirms_only_an_observed_cancel_transition():
    from lunar_pure_exploration_msgs.msg import PureExplorationStatus
    from lunar_obj_tcp_sim.operator import RosStopClient

    client = RosStopClient.__new__(RosStopClient)
    client.exploration_terminal = False
    initial = PureExplorationStatus()
    initial.state = initial.IDLE
    initial.reason_code = "IDLE"
    client._on_exploration(initial)
    assert not client.exploration_terminal

    canceled = PureExplorationStatus()
    canceled.state = canceled.IDLE
    canceled.reason_code = "CANCELED"
    client._on_exploration(canceled)
    assert client.exploration_terminal
