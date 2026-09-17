"""Regression coverage for the recovered relay dependency (no ROS or vehicle)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "p4_target_command_gate", Path(__file__).resolve().parents[1] / "joint/command_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_localization_wait_retains_execution_and_recovers():
    policy = gate.StopPolicy()
    assert policy.evaluate(0., True, False, "P3_ODOMETRY_STALE")[0] == "WAITING"
    assert policy.evaluate(3600., True, False, "CONTROLLER_NOT_READY")[0] == "WAITING"
    assert policy.evaluate(3601., True, True, "READY")[0] == "FORWARDING"
    assert policy.latched is None


def test_manual_stop_requires_explicit_authorization():
    policy = gate.StopPolicy()
    policy.external_stop()
    assert policy.evaluate(100., True, True, "READY")[0] == "STOPPED"
    policy.authorize()
    assert policy.evaluate(101., True, True, "READY")[0] == "FORWARDING"


def test_unrelated_persistent_failure_and_idle_are_distinct():
    policy = gate.StopPolicy()
    assert policy.evaluate(0., False, False, "NO_PATH")[0] == "IDLE"
    assert policy.evaluate(1., True, False, "INVALID_COMMAND")[0] == "WAITING"
    assert policy.evaluate(12., True, False, "INVALID_COMMAND") == (
        "STOPPED", "EXECUTION_UNAVAILABLE:INVALID_COMMAND")
