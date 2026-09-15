"""Light SE(2) actuator; native footprint-inflated M owns terrain safety.

Distances are actual planar arc length, not a three-dimensional surface odometer.
There is no controller here: only the public controller's v/omega commands enter.
"""
import math
from .contracts import Pose
from .geometry import world_to_cell


class GeometryFailure(RuntimeError):
    pass


class KinematicPlant:
    def __init__(self, terrain, pose, platform=None):
        self.terrain = terrain
        if platform is None:
            from .config import load_platform_config
            platform = load_platform_config()
        self.platform = platform
        self.pose = pose
        self.linear_mps = self.angular_radps = 0.0
        self.distance_m = self.turn_rad = self.simulation_s = 0.0
        self.collision = False
        if not self._free(*self._cell(pose.x, pose.y)):
            raise GeometryFailure('initial pose is not native M FREE')

    def _cell(self, x, y):
        return world_to_cell(x, y, self.terrain.origin, self.terrain.resolution_m)

    def _free(self, x, y):
        m = self.terrain.navigation
        return 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] == 1

    def _swept(self, a, b):
        # Each integration interval is <=.05 s, hence <=.01 m. Check every
        # crossed lattice boundary; exact corners require both cardinal sides,
        # matching native grid-step connectivity (no corner cutting).
        x0, y0 = self._cell(a.x, a.y)
        x1, y1 = self._cell(b.x, b.y)
        if not self._free(x1, y1): return False
        if x0 != x1 and y0 != y1:
            return self._free(x0, y1) and self._free(x1, y0)
        return True

    def advance(self, command_v, command_w, dt):
        if not all(math.isfinite(v) for v in (command_v, command_w, dt)) or not 0 < dt <= .05:
            raise ValueError('finite commands and 0 < dt <= .05 required')
        if self.collision: raise GeometryFailure('COLLISION')
        c = self.platform.capability
        target_v = max(-self.platform.maximum_reverse_speed_mps,
                       min(self.platform.maximum_forward_speed_mps, command_v))
        target_w = max(-self.platform.maximum_spin_rate_radps,
                       min(self.platform.maximum_spin_rate_radps, command_w))
        def slew(old, target, amount): return old + max(-amount, min(amount, target-old))
        accel = c['maximum_acceleration_mps2'] if abs(target_v) > abs(self.linear_mps) else c['maximum_braking_deceleration_mps2']
        v = slew(self.linear_mps, target_v, accel * dt)
        w = slew(self.angular_radps, target_w, c['maximum_yaw_acceleration_radps2'] * dt)
        theta = w * dt
        yaw = self.pose.yaw
        if abs(w) < 1e-12:
            dx, dy = v * dt * math.cos(yaw), v * dt * math.sin(yaw)
        else:
            dx = v / w * (math.sin(yaw + theta) - math.sin(yaw))
            dy = v / w * (math.cos(yaw) - math.cos(yaw + theta))
        candidate = Pose(self.pose.x+dx, self.pose.y+dy, math.atan2(math.sin(yaw+theta), math.cos(yaw+theta)))
        if not self._swept(self.pose, candidate):
            self.collision = True
            self.linear_mps = self.angular_radps = 0.0
            raise GeometryFailure('COLLISION: native M swept connection rejected')
        self.pose = candidate
        self.linear_mps, self.angular_radps = v, w
        self.distance_m += abs(v) * dt
        self.turn_rad += abs(theta)
        self.simulation_s += dt
        return self.pose
