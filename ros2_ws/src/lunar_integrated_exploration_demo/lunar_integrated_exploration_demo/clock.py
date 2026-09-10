"""Bounded simulation time, driven only by monotonically increasing wall time."""
from collections import deque
import math


def finite_number(value, name, minimum, maximum):
    """Validate parameters without accepting booleans or numeric strings."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f'{name} must be finite and in [{minimum}, {maximum}]')
    return float(value)


class SimulationClock:
    """Never accumulate debt which would skip future controller opportunities.

    If a wall callback runs late, advance at most ``max_step_s`` and report the
    discarded requested time. Physical speed always remains per simulated second.
    The wall timer is paced at max_step_s / requested scale; its producer must use
    a steady clock, even when the node has use_sim_time enabled.
    """

    def __init__(self, time_scale=30.0, start_wall_s=0.0, max_step_s=0.02):
        self.time_scale = finite_number(time_scale, 'time_scale', 1.0, 60.0)
        self.max_step_s = finite_number(max_step_s, 'max_step_s', 1e-4, 0.02)
        self.start_wall_s = finite_number(start_wall_s, 'start wall time', 0.0, 1e15)
        self.last_wall_s = self.start_wall_s
        # A nonzero initial stamp lets ROS distinguish valid time from unstarted /clock.
        self.sim_time_ns = 1_000_000_000
        self._initial_sim_ns = self.sim_time_ns
        self.discarded_sim_lag_s = 0.0
        self.clock_steps = 0
        self._rate_samples = deque([(self.start_wall_s, 0.0)])

    @property
    def wall_period_s(self):
        return self.max_step_s / self.time_scale

    @property
    def sim_time_s(self):
        return self.sim_time_ns / 1e9

    @property
    def elapsed_sim_s(self):
        return (self.sim_time_ns - self._initial_sim_ns) / 1e9

    def set_time_scale(self, value):
        self.time_scale = finite_number(value, 'time_scale', 1.0, 60.0)

    def advance(self, wall_s):
        wall_s = finite_number(wall_s, 'wall time', self.last_wall_s, 1e15)
        requested_s = (wall_s - self.last_wall_s) * self.time_scale
        advance_ns = round(min(requested_s, self.max_step_s) * 1e9)
        self.discarded_sim_lag_s += max(0.0, requested_s - self.max_step_s)
        self.last_wall_s = wall_s
        self.sim_time_ns += advance_ns
        self.clock_steps += int(advance_ns > 0)
        if wall_s - self._rate_samples[-1][0] >= 0.05:
            self._rate_samples.append((wall_s, self.elapsed_sim_s))
        while len(self._rate_samples) > 1 and self._rate_samples[1][0] < wall_s - 2.0:
            self._rate_samples.popleft()
        return advance_ns / 1e9

    def stamp_parts(self):
        return divmod(self.sim_time_ns, 1_000_000_000)

    def snapshot(self):
        elapsed_wall = self.last_wall_s - self.start_wall_s
        sample_wall, sample_sim = self._rate_samples[0]
        recent_wall = self.last_wall_s - sample_wall
        recent_factor = ((self.elapsed_sim_s - sample_sim) / recent_wall
                         if recent_wall > 0.0 else 0.0)
        return {
            'sim_time_s': self.sim_time_s,
            'elapsed_sim_s': self.elapsed_sim_s,
            'wall_time_s': elapsed_wall,
            'elapsed_wall_s': elapsed_wall,
            'requested_time_scale': self.time_scale,
            'actual_time_scale': recent_factor,
            'average_time_scale': self.elapsed_sim_s / elapsed_wall if elapsed_wall > 0 else 0.0,
            'discarded_sim_lag_s': self.discarded_sim_lag_s,
            'max_clock_step_sim_s': self.max_step_s,
            'clock_steps': self.clock_steps,
        }


class ObservationSchedule:
    """Sample the actual pose at simulation deadlines, independently of publishing.

    A late caller samples only its current pose. It never manufactures samples
    for past poses; missed deadlines are evidence. Vehicle clock steps <= .02 s
    ensure that the configured sensor period cannot be skipped in normal use.
    """

    def __init__(self, sim_rate_hz=1.0):
        self.sim_rate_hz = finite_number(sim_rate_hz, 'sample_sim_rate_hz', 0.05, 20.0)
        self.period_ns = round(1e9 / self.sim_rate_hz)
        self.sample_count = 0
        self.missed_deadlines = 0
        self.max_interval_ns = 0
        self._next_ns = self._first_ns = self._last_ns = None

    def take(self, sim_s):
        now_ns = round(finite_number(sim_s, 'sample simulation time', 0.0, 1e12) * 1e9)
        if self._next_ns is None:
            self._next_ns = now_ns
        if now_ns < self._next_ns:
            return False
        missed = (now_ns - self._next_ns) // self.period_ns
        self.missed_deadlines += missed
        self._next_ns += (missed + 1) * self.period_ns
        if self._last_ns is not None:
            self.max_interval_ns = max(self.max_interval_ns, now_ns - self._last_ns)
        else:
            self._first_ns = now_ns
        self._last_ns = now_ns
        self.sample_count += 1
        return True

    def snapshot(self):
        actual_rate = 0.0
        if self.sample_count > 1:
            actual_rate = (self.sample_count - 1) * 1e9 / (self._last_ns - self._first_ns)
        return {
            'sample_count': self.sample_count,
            'sample_requested_sim_rate_hz': self.sim_rate_hz,
            'sample_sim_rate_hz': actual_rate,
            'last_sample_sim_s': None if self._last_ns is None else self._last_ns / 1e9,
            'max_sample_interval_sim_s': self.max_interval_ns / 1e9,
            'missed_sample_deadlines': self.missed_deadlines,
        }


class PublicationSchedule:
    """Limit transport wall rate only; sampling belongs to ObservationSchedule."""

    def __init__(self, wall_cap_hz=5.0):
        self.wall_cap_hz = finite_number(wall_cap_hz, 'map_wall_cap_hz', 0.5, 30.0)
        self.publications = 0
        self._first = None
        self._last = None

    def take(self, sim_s, wall_s):
        if not self.ready(wall_s):
            return False
        if self._last is None:
            self._first = (sim_s, wall_s)
        self._last = (sim_s, wall_s)
        self.publications += 1
        return True

    def ready(self, wall_s):
        return self._last is None or wall_s - self._last[1] + 1e-9 >= 1.0 / self.wall_cap_hz

    def snapshot(self):
        sim_rate = wall_rate = 0.0
        if self.publications > 1:
            sim_rate = (self.publications - 1) / (self._last[0] - self._first[0])
            wall_rate = (self.publications - 1) / (self._last[1] - self._first[1])
        return {
            'publication_count': self.publications,
            'map_publications': self.publications,
            'map_wall_cap_hz': self.wall_cap_hz,
            'map_actual_sim_rate_hz': sim_rate,
            'map_actual_wall_rate_hz': wall_rate,
        }
