"""ROS-free SE(2) plant. No path or goal is an input to the vehicle model."""
from dataclasses import dataclass
import math


class Scene:
    GOALS = {'forward': (3.0, 0.0, None), 'reverse': (-3.0, 0.0, 0.0),
             'final_yaw': (3.0, 0.0, math.pi / 2), 'detour': (6.0, 0.0, None),
             'multiple_exits': (12.0, 0.0, None), 'manual': (3.0, 0.0, None)}

    def __init__(self, name='forward'):
        if name not in self.GOALS:
            raise ValueError(f'Unknown scene {name}; choose {list(self.GOALS)}')
        self.name = name
        self.goal = self.GOALS[name]
        self.obstacles = []
        if name == 'detour':
            self.obstacles = [(2.0, 3.0, -1.8, 1.8)]
        elif name == 'multiple_exits':
            self.obstacles = [(2.0, 3.0, -3.0, -0.9),
                              (2.0, 3.0, 0.9, 3.0), (5.0, 6.0, -0.7, 0.7)]

    def elevation(self, x, y):
        return 0.9 if any(a <= x <= b and c <= y <= d
                          for a, b, c, d in self.obstacles) else 0.0

    def collides(self, x, y, yaw):
        """Separating-axis test for 1.182 x .818 m wheel footprint and rocks."""
        cs, sn = math.cos(yaw), math.sin(yaw)
        axes = [(1., 0.), (0., 1.), (cs, sn), (-sn, cs)]
        vehicle = [(x + cs * u - sn * v, y + sn * u + cs * v)
                   for u in (-0.591, 0.591) for v in (-0.409, 0.409)]
        for a, b, c, d in self.obstacles:
            rock = [(u, v) for u in (a, b) for v in (c, d)]
            for ax, ay in axes:
                vp = [ax * u + ay * v for u, v in vehicle]
                rp = [ax * u + ay * v for u, v in rock]
                if max(vp) < min(rp) or max(rp) < min(vp):
                    break
            else:
                return True
        return False


@dataclass
class Plant:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    w: float = 0.0
    acceleration: float = 0.3
    braking: float = 0.5
    angular_acceleration: float = 0.5
    max_raw_forward: float = 0.0
    max_raw_reverse: float = 0.0
    max_actual_forward: float = 0.0
    max_actual_reverse: float = 0.0
    speed_limit_violations: int = 0
    invalid_commands: int = 0
    collisions: int = 0

    def step(self, dt, command_v, command_w, collision=None):
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError('integration dt must be finite and in (0, .1]')
        if not math.isfinite(command_v) or not math.isfinite(command_w):
            self.invalid_commands += 1
            command_v = command_w = 0.0
        self.max_raw_forward = max(self.max_raw_forward, command_v)
        self.max_raw_reverse = max(self.max_raw_reverse, -command_v)
        if abs(command_v) > 0.2 + 1e-9:
            self.speed_limit_violations += 1
        target_v = max(-0.2, min(0.2, command_v))
        target_w = max(-0.6, min(0.6, command_w))
        rate = self.braking if (self.v * target_v < 0 or abs(target_v) < abs(self.v)) else self.acceleration
        next_v = self.v + max(-rate * dt, min(rate * dt, target_v - self.v))
        next_w = self.w + max(-self.angular_acceleration * dt,
                             min(self.angular_acceleration * dt, target_w - self.w))
        v, w = (self.v + next_v) / 2, (self.w + next_w) / 2
        angle = self.yaw + 0.5 * w * dt
        x, y = self.x + v * math.cos(angle) * dt, self.y + v * math.sin(angle) * dt
        yaw = math.remainder(self.yaw + w * dt, 2 * math.pi)
        if collision and collision(x, y, yaw):
            self.collisions += 1
            self.v = self.w = 0.0
        else:
            self.x, self.y, self.yaw = x, y, yaw
            self.v, self.w = next_v, next_w
        self.max_actual_forward = max(self.max_actual_forward, self.v)
        self.max_actual_reverse = max(self.max_actual_reverse, -self.v)
