"""Plant acceptance: command-only motion, limits, braking, obstacle contact."""
import sys
from pathlib import Path
import math
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lunar_incremental_controller_demo.plant import Plant, Scene


def test_no_command_never_moves():
    p = Plant()
    for _ in range(100):
        p.step(0.02, 0.0, 0.0)
    assert (p.x, p.y, p.yaw, p.v, p.w) == (0.0,) * 5


def test_limits_and_raw_command_violation_are_both_visible():
    p = Plant()
    for _ in range(100):
        p.step(0.02, 0.7, 0.0)
    assert p.v == pytest.approx(0.2)
    assert p.max_raw_forward == 0.7
    assert p.speed_limit_violations == 100
    assert 0 < p.x < 0.4


def test_reverse_limit_and_acceleration():
    p = Plant()
    p.step(0.1, -0.2, 0.0)
    assert p.v == pytest.approx(-0.03)
    for _ in range(100):
        p.step(0.02, -0.2, 0.0)
    assert p.v == pytest.approx(-0.2)
    assert p.x < 0


def test_braking_has_real_stopping_distance():
    p = Plant()
    for _ in range(100):
        p.step(0.02, 0.2, 0)
    x = p.x
    p.step(0.1, 0, 0)
    assert p.v == pytest.approx(0.15)
    assert p.x > x
    for _ in range(10):
        p.step(0.1, 0, 0)
    assert p.v == 0


def test_turning_integrates_actual_twist():
    p = Plant()
    for _ in range(100):
        p.step(0.02, 0.1, 0.3)
    assert p.y > 0.02
    assert 0.5 < p.yaw < 0.6


def test_footprint_contact_stops_pose_and_records_failure():
    scene = Scene('detour')
    p = Plant(x=1.35)
    for _ in range(200):
        p.step(0.02, 0.2, 0.0, scene.collides)
    assert p.collisions > 0
    assert p.x < 1.5
    assert p.v == 0


def test_nonfinite_command_stops_and_is_evidence():
    p = Plant()
    p.step(0.02, math.nan, 0)
    assert p.invalid_commands == 1
    assert math.isfinite(p.x)
