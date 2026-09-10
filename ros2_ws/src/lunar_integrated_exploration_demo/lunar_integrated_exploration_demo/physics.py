"""Command-driven integration of the unchanged simple-demo vehicle Plant."""
from dataclasses import asdict
import math
import time

import numpy as np

from lunar_incremental_controller_demo.plant import Plant

from .clock import ObservationSchedule, SimulationClock, finite_number


class ObservationBatch:
    """Pending measured cells, anchored by integer global grid coordinates.

    The matrix expands to the union of pending windows. Only finite incoming
    samples overwrite cells, so a later NaN cannot erase an earlier observation.
    A successful publisher clears the batch; this is not a persistent truth map.
    """

    def __init__(self, resolution=0.2, max_cells=1_000_000):
        self.resolution = finite_number(resolution, 'batch resolution', 0.2, 0.2)
        if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells <= 0:
            raise ValueError('batch bound must be a positive integer cell count')
        self.max_cells = max_cells
        self.clear()

    @property
    def cell_count(self):
        return 0 if self._values is None else self._values.size

    def add(self, observation, sample_sim_s):
        if observation['resolution'] != self.resolution:
            raise ValueError('observation resolution differs from global batch grid')
        sample_sim_s = finite_number(sample_sim_s, 'sample simulation time', 0.0, 1e12)
        if self.last_sample_sim_s is not None and sample_sim_s < self.last_sample_sim_s:
            raise ValueError('sample simulation time must not run backwards')
        origins = [float(observation[name]) / self.resolution
                   for name in ('origin_x', 'origin_y')]
        if not all(math.isfinite(value) and math.isclose(value, round(value), abs_tol=1e-7)
                   for value in origins):
            raise ValueError('observation origin must be snapped to the global grid')
        x0, y0 = (round(value) for value in origins)
        values = np.asarray(observation['values'], dtype=np.float32)
        if values.ndim != 2 or not values.size or np.isinf(values).any():
            raise ValueError('observation must contain a nonempty finite-or-NaN matrix')
        height, width = values.shape
        x1, y1 = x0 + width, y0 + height
        if self._values is not None:
            old_x, old_y = self._origin_cells
            old_h, old_w = self._values.shape
            new_x, new_y = min(x0, old_x), min(y0, old_y)
            new_w, new_h = max(x1, old_x + old_w) - new_x, max(y1, old_y + old_h) - new_y
        else:
            new_x, new_y, new_w, new_h = x0, y0, width, height
        if new_w * new_h > self.max_cells:
            # Fail visibly with the existing pending batch intact; never evict
            # measured cells merely to fit the next publication window.
            raise ValueError('pending observation batch exceeds its bounded cell capacity')
        if (self._values is None or self._origin_cells != (new_x, new_y)
                or self._values.shape != (new_h, new_w)):
            expanded = np.full((new_h, new_w), np.nan, dtype=np.float32)
            if self._values is not None:
                expanded[old_y - new_y:old_y - new_y + old_h,
                         old_x - new_x:old_x - new_x + old_w] = self._values
            self._values, self._origin_cells = expanded, (new_x, new_y)
        target = self._values[y0 - new_y:y0 - new_y + height, x0 - new_x:x0 - new_x + width]
        np.copyto(target, values, where=np.isfinite(values))
        if self.first_sample_sim_s is None:
            self.first_sample_sim_s = sample_sim_s
        self.last_sample_sim_s = sample_sim_s
        self.sample_count += 1

    def snapshot(self):
        if self._values is None:
            return None
        return {
            'origin_x': self._origin_cells[0] * self.resolution,
            'origin_y': self._origin_cells[1] * self.resolution,
            'resolution': self.resolution,
            'values': self._values.copy(),
            'sample_count': self.sample_count,
            'first_sample_sim_s': self.first_sample_sim_s,
            'last_sample_sim_s': self.last_sample_sim_s,
        }

    def clear(self):
        self._values = self._origin_cells = None
        self.sample_count = 0
        self.first_sample_sim_s = self.last_sample_sim_s = None


class VehicleSimulation:
    """The only motion input is a velocity command; there are no path inputs."""

    def __init__(self, time_scale=30.0, start_wall_s=0.0, collision=None,
                 command_timeout_s=0.3, max_substep_s=0.01, sensor=None,
                 sample_sim_rate_hz=1.0):
        self.clock = SimulationClock(time_scale=time_scale, start_wall_s=start_wall_s)
        self.plant = Plant()
        self.collision = collision
        self.command_timeout_s = finite_number(command_timeout_s, 'command_timeout_s', 0.05, 2.0)
        self.max_substep_s = finite_number(max_substep_s, 'max_substep_s', 0.001, 0.02)
        self.command = (0.0, 0.0)
        self.command_time_s = None
        self.command_messages = 0
        self.raw_invalid_commands = 0
        self.raw_limit_violations = 0
        self.raw_angular_limit_violations = 0
        self.raw_command_forward_max = 0.0
        self.raw_command_reverse_max = 0.0
        self.raw_command_angular_max = 0.0
        self.distance_m = 0.0
        self.max_actual_linear_acceleration = 0.0
        self.max_actual_angular_acceleration = 0.0
        self.integration_steps = 0
        self.sensor = sensor
        self.sample_schedule = ObservationSchedule(sample_sim_rate_hz)
        self.pending_observation = ObservationBatch()
        self.last_observed_cells = 0
        self.last_observation_wall_ms = self.max_observation_wall_ms = 0.0
        self.max_pending_cells = 0
        self.sample_observation()

    def receive_command(self, linear, angular, other_components_finite=True):
        self.command_messages += 1
        if not (math.isfinite(linear) and math.isfinite(angular) and other_components_finite):
            self.raw_invalid_commands += 1
            linear = angular = 0.0
        else:
            self.raw_command_forward_max = max(self.raw_command_forward_max, linear)
            self.raw_command_reverse_max = max(self.raw_command_reverse_max, -linear)
            self.raw_command_angular_max = max(self.raw_command_angular_max, abs(angular))
            self.raw_limit_violations += int(abs(linear) > 0.2 + 1e-9)
            self.raw_angular_limit_violations += int(abs(angular) > 0.6 + 1e-9)
        self.command = (linear, angular)
        self.command_time_s = self.clock.sim_time_s

    def advance(self, wall_s):
        cursor_s = self.clock.sim_time_s
        remaining_s = self.clock.advance(wall_s)
        advanced_s = remaining_s
        while remaining_s > 1e-10:
            dt = min(remaining_s, self.max_substep_s)
            valid = (self.command_time_s is not None
                     and cursor_s < self.command_time_s + self.command_timeout_s - 1e-10)
            if valid:
                # Do not hold a stale command through a substep crossing its expiry.
                dt = min(dt, self.command_time_s + self.command_timeout_s - cursor_s)
            command = self.command if valid else (0.0, 0.0)
            p = self.plant
            old_x, old_y, old_v, old_w, contacts = p.x, p.y, p.v, p.w, p.collisions
            # A stationary footprint remains at its already checked initial pose.
            collision = self.collision if any((p.v, p.w, *command)) else None
            p.step(dt, *command, collision=collision)
            self.distance_m += math.hypot(p.x - old_x, p.y - old_y)
            if p.collisions == contacts:
                self.max_actual_linear_acceleration = max(
                    self.max_actual_linear_acceleration, abs(p.v - old_v) / dt)
                self.max_actual_angular_acceleration = max(
                    self.max_actual_angular_acceleration, abs(p.w - old_w) / dt)
            self.integration_steps += 1
            cursor_s += dt
            remaining_s -= dt
        self.sample_observation()
        return advanced_s

    def sample_observation(self):
        if self.sensor is None or not self.sample_schedule.take(self.clock.sim_time_s):
            return
        start = time.perf_counter()
        observation = self.sensor(self.plant.x, self.plant.y, self.plant.yaw)
        self.pending_observation.add(observation, sample_sim_s=self.clock.sim_time_s)
        self.last_observed_cells = int(np.isfinite(observation['values']).sum())
        self.max_pending_cells = max(self.max_pending_cells, self.pending_observation.cell_count)
        self.last_observation_wall_ms = (time.perf_counter() - start) * 1000
        self.max_observation_wall_ms = max(self.max_observation_wall_ms, self.last_observation_wall_ms)

    def state(self):
        state = asdict(self.plant)
        age = (None if self.command_time_s is None
               else self.clock.sim_time_s - self.command_time_s)
        state.update(self.clock.snapshot())
        state.update(self.sample_schedule.snapshot())
        state.update({
            'command_messages': self.command_messages,
            'command_v': self.command[0],
            'command_w': self.command[1],
            'command_age_sim_s': age,
            'command_timed_out': age is None or age >= self.command_timeout_s,
            'command_timeout_sim_s': self.command_timeout_s,
            'raw_invalid_commands': self.raw_invalid_commands,
            'raw_limit_violations': self.raw_limit_violations,
            'raw_angular_limit_violations': self.raw_angular_limit_violations,
            'raw_command_forward_max': self.raw_command_forward_max,
            'raw_command_reverse_max': self.raw_command_reverse_max,
            'raw_command_angular_max': self.raw_command_angular_max,
            'distance_m': self.distance_m,
            'max_actual_linear_acceleration': self.max_actual_linear_acceleration,
            'max_actual_angular_acceleration': self.max_actual_angular_acceleration,
            'integration_steps': self.integration_steps,
            'speed_limit_mps': 0.2,
            'angular_speed_limit_radps': 0.6,
            'observation_count': self.sample_schedule.sample_count,
            'pending_sample_count': self.pending_observation.sample_count,
            'pending_cell_count': self.pending_observation.cell_count,
            'max_pending_cell_count': self.max_pending_cells,
            'observed_cells': self.last_observed_cells,
            'observation_wall_ms': self.last_observation_wall_ms,
            'max_observation_wall_ms': self.max_observation_wall_ms,
        })
        return state
