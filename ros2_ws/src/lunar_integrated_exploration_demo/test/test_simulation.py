"""Clock and command-only motion contracts, without a ROS executor."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE.parent / 'lunar_incremental_controller_demo'))


def clock_type():
    from lunar_integrated_exploration_demo.clock import SimulationClock
    return SimulationClock


def simulation_type():
    from lunar_integrated_exploration_demo.physics import VehicleSimulation
    return VehicleSimulation


@pytest.mark.parametrize('factor', [1.0, 7.5, 30.0, 60.0])
def test_clock_matches_requested_factor_when_wall_callbacks_keep_up(factor):
    clock = clock_type()(time_scale=factor, start_wall_s=0.0)
    for index in range(1, 501):
        clock.advance(index * 0.01 / factor)
    assert clock.elapsed_sim_s == pytest.approx(5.0, abs=1e-6)
    assert clock.snapshot()['actual_time_scale'] == pytest.approx(factor, abs=1e-5)
    assert clock.snapshot()['discarded_sim_lag_s'] == pytest.approx(0.0)


def test_clock_never_jumps_over_controller_opportunities_after_wall_stall():
    clock = clock_type()(time_scale=30.0, start_wall_s=0.0)
    assert clock.advance(1.0) == pytest.approx(0.02)
    assert clock.elapsed_sim_s == pytest.approx(0.02)
    assert clock.snapshot()['actual_time_scale'] == pytest.approx(0.02)
    assert clock.snapshot()['discarded_sim_lag_s'] == pytest.approx(29.98)
    assert clock.advance(1.0) == 0.0
    assert clock.advance(1.0 + 0.01 / 30.0) == pytest.approx(0.01, abs=1e-8)
    assert clock.elapsed_sim_s == pytest.approx(0.03, abs=1e-8)


def test_clock_scale_change_preserves_time_and_does_not_catch_up_old_debt():
    clock = clock_type()(time_scale=30.0, start_wall_s=0.0)
    clock.advance(0.01 / 30.0)
    before = clock.sim_time_ns
    clock.set_time_scale(1.0)
    assert clock.sim_time_ns == before
    assert clock.advance(0.01 / 30.0 + 0.01) == pytest.approx(0.01)
    assert clock.snapshot()['requested_time_scale'] == 1.0
    assert clock.wall_period_s == pytest.approx(0.02)


@pytest.mark.parametrize('value', [0.0, 0.99, 60.01, math.inf, -math.inf, math.nan, True, '30'])
def test_clock_rejects_invalid_scale_without_changing_active_scale(value):
    clock = clock_type()(time_scale=30.0, start_wall_s=0.0)
    with pytest.raises(ValueError, match='time_scale'):
        clock.set_time_scale(value)
    assert clock.time_scale == 30.0


def test_clock_rejects_backwards_or_nonfinite_wall_time_atomically():
    clock = clock_type()(time_scale=1.0, start_wall_s=1.0)
    clock.advance(1.01)
    before = clock.sim_time_ns
    for value in [1.0, math.nan, math.inf]:
        with pytest.raises(ValueError, match='wall'):
            clock.advance(value)
        assert clock.sim_time_ns == before


def test_sim_time_stamp_starts_nonzero_and_normalizes_nanoseconds():
    clock = clock_type()(time_scale=1.0, start_wall_s=0.0)
    assert clock.stamp_parts() == (1, 0)
    for index in range(1, 102):
        clock.advance(index * 0.01)
    assert clock.stamp_parts() == (2, 10_000_000)


@pytest.mark.parametrize('factor', [1.0, 30.0, 60.0])
def test_accelerated_simulation_keeps_same_physical_velocity_and_distance(factor):
    sim = simulation_type()(time_scale=factor, start_wall_s=0.0)
    for index in range(1, 201):
        sim.receive_command(0.2, 0.0)
        sim.advance(index * 0.01 / factor)
    assert sim.plant.v == pytest.approx(0.2)
    # 0.3 m/s² acceleration to 0.2 m/s takes 2/3 s; then cruise.
    assert sim.plant.x == pytest.approx(1.0 / 3.0, abs=2e-5)
    assert sim.plant.y == 0.0
    assert sim.plant.max_actual_forward <= 0.2
    assert sim.state()['distance_m'] == pytest.approx(sim.plant.x)


def test_command_timeout_uses_simulation_seconds_and_real_braking():
    sim = simulation_type()(time_scale=30.0, start_wall_s=0.0)
    for index in range(1, 101):
        sim.receive_command(0.2, 0.0)
        sim.advance(index * 0.01 / 30.0)
    before = sim.plant.x
    for index in range(101, 201):
        sim.advance(index * 0.01 / 30.0)
    assert sim.plant.v == 0.0
    assert before < sim.plant.x < before + 0.11
    assert sim.state()['command_timed_out'] is True


def test_invalid_command_is_counted_once_and_telemetry_contains_no_nan_or_infinity():
    sim = simulation_type()(time_scale=1.0, start_wall_s=0.0)
    for v, w in [(math.nan, 0.0), (0.0, math.inf), (0.0, -math.inf)]:
        sim.receive_command(v, w)
        sim.advance(sim.clock.last_wall_s + 0.01)
    state = sim.state()
    assert state['raw_invalid_commands'] == 3
    assert state['command_messages'] == 3
    assert state['x'] == state['y'] == state['v'] == state['w'] == 0.0
    json.dumps(state, allow_nan=False)


def test_raw_limit_counts_and_actual_forward_reverse_limits_remain_distinct():
    sim = simulation_type()(time_scale=30.0, start_wall_s=0.0)
    for index in range(1, 501):
        sim.receive_command(0.7 if index <= 250 else -0.8, 1.2)
        sim.advance(index * 0.01 / 30.0)
    state = sim.state()
    assert state['raw_limit_violations'] == 500
    assert state['raw_command_forward_max'] == 0.7
    assert state['raw_command_reverse_max'] == 0.8
    assert state['max_actual_forward'] <= 0.2
    assert state['max_actual_reverse'] <= 0.2
    assert state['w'] <= 0.6
    assert state['max_actual_angular_acceleration'] <= 0.5 + 1e-9
    assert state['max_actual_linear_acceleration'] <= 0.5 + 1e-9


def test_collision_from_accelerated_motion_stops_at_footprint_boundary():
    sim = simulation_type()(time_scale=30.0, start_wall_s=0.0,
                            collision=lambda x, _y, _yaw: x + 0.591 >= 1.0)
    for index in range(1, 501):
        sim.receive_command(0.2, 0.0)
        sim.advance(index * 0.01 / 30.0)
    assert sim.plant.collisions > 0
    assert sim.plant.x < 0.409
    assert sim.plant.v == 0.0


def test_map_publication_gate_limits_wall_rate_without_throttling_simulation_sampling():
    clock_type()
    from lunar_integrated_exploration_demo.clock import PublicationSchedule
    gate = PublicationSchedule(wall_cap_hz=5.0)
    assert gate.take(1.0, 0.0)
    assert not gate.take(2.0, 0.01)
    assert gate.take(7.0, 0.2)
    assert not gate.take(7.0, 0.2)
    assert gate.take(7.1, 0.4)
    state = gate.snapshot()
    assert state['publication_count'] == 3
    assert state['map_actual_sim_rate_hz'] == pytest.approx(2.0 / 6.1)
    assert state['map_actual_wall_rate_hz'] == pytest.approx(5.0)


def observation_batch():
    from lunar_integrated_exploration_demo.physics import ObservationBatch
    return ObservationBatch


def observed_value(observation, x, y):
    ix = math.floor((x - observation['origin_x']) / observation['resolution'])
    iy = math.floor((y - observation['origin_y']) / observation['resolution'])
    return observation['values'][iy, ix]


def test_opposite_heading_samples_preserve_both_sides_without_filling_unseen_sector():
    from lunar_integrated_exploration_demo.terrain import Terrain
    batch = observation_batch()()
    terrain = Terrain()
    ahead = terrain.observe(0.0, 0.0, 0.0)
    behind = terrain.observe(0.0, 0.0, math.pi)
    batch.add(ahead, sample_sim_s=1.0)
    batch.add(behind, sample_sim_s=2.0)
    merged = batch.snapshot()
    assert np.isfinite(observed_value(merged, 6.1, 0.1))
    assert np.isfinite(observed_value(merged, -6.1, 0.1))
    assert np.isnan(observed_value(merged, 0.1, 8.1))
    assert np.isnan(observed_value(merged, 0.1, -8.1))
    assert batch.sample_count == 2
    assert merged['first_sample_sim_s'] == 1.0
    assert merged['last_sample_sim_s'] == 2.0
    np.testing.assert_array_equal(np.isfinite(merged['values']),
                                  np.isfinite(ahead['values']) | np.isfinite(behind['values']))


def test_repeated_occluded_samples_remain_unknown_in_a_batch():
    from lunar_integrated_exploration_demo.terrain import Terrain
    batch = observation_batch()()
    terrain = Terrain()
    batch.add(terrain.observe(4.0, 5.0, 0.0), sample_sim_s=1.0)
    batch.add(terrain.observe(4.0, 5.0, 0.02), sample_sim_s=2.0)
    merged = batch.snapshot()
    assert np.isfinite(observed_value(merged, 7.9, 5.1))
    assert np.isnan(observed_value(merged, 12.1, 5.1))
    assert np.isfinite(observed_value(merged, 12.1, 9.1))


def test_moving_sample_window_keeps_old_world_evidence_outside_current_window():
    from lunar_integrated_exploration_demo.terrain import Terrain
    batch = observation_batch()()
    terrain = Terrain()
    first = terrain.observe(0.0, 0.0, math.pi)
    second = terrain.observe(20.0, 0.0, 0.0)
    batch.add(first, sample_sim_s=1.0)
    batch.add(second, sample_sim_s=2.0)
    merged = batch.snapshot()
    assert second['origin_x'] > -6.1
    assert observed_value(merged, -6.1, 0.1) == observed_value(first, -6.1, 0.1)
    assert observed_value(merged, 26.1, 0.1) == observed_value(second, 26.1, 0.1)
    assert np.isnan(observed_value(merged, 14.1, 9.1))
    assert merged['origin_x'] == -14.0
    assert merged['values'].shape == (140, 240)


def test_batch_clear_releases_only_published_evidence_and_next_batch_starts_empty():
    batch = observation_batch()()
    one = {'origin_x': -0.4, 'origin_y': 0.6, 'resolution': 0.2,
           'values': np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)}
    batch.add(one, sample_sim_s=1.0)
    published = batch.snapshot()
    batch.clear()
    assert batch.sample_count == batch.cell_count == 0
    assert batch.snapshot() is None
    batch.add(dict(one, values=np.full((2, 2), np.nan)), sample_sim_s=2.0)
    assert np.isnan(batch.snapshot()['values']).all()
    assert observed_value(published, -0.3, 0.7) == 1.0


def test_batch_bound_failure_preserves_pending_evidence_instead_of_dropping_it():
    batch = observation_batch()(max_cells=4)
    one = {'origin_x': 0.0, 'origin_y': 0.0, 'resolution': 0.2,
           'values': np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)}
    batch.add(one, sample_sim_s=1.0)
    with pytest.raises(ValueError, match='bound'):
        batch.add(dict(one, origin_x=100.0), sample_sim_s=2.0)
    assert batch.sample_count == 1
    assert batch.snapshot()['origin_x'] == 0.0
    assert observed_value(batch.snapshot(), 0.1, 0.1) == 1.0


def test_30x_turn_samples_each_simulated_second_while_wall_publication_batches_them():
    from lunar_integrated_exploration_demo.clock import PublicationSchedule
    from lunar_integrated_exploration_demo.terrain import Terrain
    terrain = Terrain()
    observed_yaws = []

    def sensor(x, y, yaw):
        observed_yaws.append(yaw)
        return terrain.observe(x, y, yaw)

    sim = simulation_type()(time_scale=30.0, start_wall_s=0.0, sensor=sensor)
    publication = PublicationSchedule(wall_cap_hz=5.0)
    sample_times = [sim.state()['last_sample_sim_s']]
    prior_samples = sim.state()['sample_count']
    batch_sizes = []
    for index in range(1, 601):
        sim.receive_command(0.0, 0.5)
        sim.advance(index * 0.02 / 30.0)
        state = sim.state()
        if state['sample_count'] > prior_samples:
            sample_times.append(state['last_sample_sim_s'])
            prior_samples = state['sample_count']
        if sim.pending_observation.sample_count and publication.take(
                sim.clock.sim_time_s, sim.clock.last_wall_s):
            batch_sizes.append(sim.pending_observation.sample_count)
            sim.pending_observation.clear()
    assert len(sample_times) == 13
    assert np.diff(sample_times) == pytest.approx(np.ones(12), abs=0.020000001)
    assert max(batch_sizes) >= 5
    assert sim.state()['sample_sim_rate_hz'] == pytest.approx(1.0)
    assert sim.state()['missed_sample_deadlines'] == 0
    assert abs(math.remainder(observed_yaws[-1] - observed_yaws[0], 2 * math.pi)) > 0.1
    assert sim.plant.x == sim.plant.y == 0.0


def test_dynamic_scale_changes_do_not_change_sensor_simulation_cadence():
    minimal = {'origin_x': 0.0, 'origin_y': 0.0, 'resolution': 0.2,
               'values': np.array([[0.0]], dtype=np.float32)}
    sim = simulation_type()(time_scale=30.0, start_wall_s=0.0,
                            sensor=lambda _x, _y, _yaw: minimal)
    for scale in (30.0, 1.0, 60.0):
        sim.clock.set_time_scale(scale)
        for _ in range(100):
            sim.advance(sim.clock.last_wall_s + 0.02 / scale)
    state = sim.state()
    assert state['sample_count'] == 7
    assert state['sample_sim_rate_hz'] == pytest.approx(1.0)
    assert state['max_sample_interval_sim_s'] <= 1.02
    assert state['missed_sample_deadlines'] == 0
