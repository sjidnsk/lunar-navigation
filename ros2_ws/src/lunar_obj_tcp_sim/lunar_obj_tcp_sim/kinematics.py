"""Four independently driven/steered wheels, ROS body X forward/Y left.

Canonical order: front-left, front-right, rear-left, rear-right. Wire order
lists give the canonical wheel index for each corresponding UE array slot.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import yaml


@dataclass(frozen=True)
class WheelCommand:
    wheel_speeds: tuple
    turn_angles: tuple
    scale: float

    @property
    def saturated(self):
        return self.scale < 1.


def validate_steering_options(wheel_order, steering_order, wheel_signs, steering_signs,
                              steering_zero_rad, max_steering_angle_rad, steering_tolerance_rad):
    for name,order in [('wheel_order',wheel_order),('steering_order',steering_order)]:
        if len(order)!=4 or any(type(i) is not int for i in order) or set(order)!={0,1,2,3}:
            raise ValueError(f'{name} must be a permutation of 0,1,2,3')
    for name,signs in [('wheel_signs',wheel_signs),('steering_signs',steering_signs)]:
        if len(signs)!=4 or any(s not in (-1,1) for s in signs):
            raise ValueError(f'{name} must contain four signs (+1 or -1)')
    if len(steering_zero_rad)!=4 or not all(map(math.isfinite,steering_zero_rad)):
        raise ValueError('steering_zero_rad must contain four finite values')
    for name,value in [('max_steering_angle_rad',max_steering_angle_rad),
                       ('steering_tolerance_rad',steering_tolerance_rad)]:
        if not math.isfinite(value) or value<=0:
            raise ValueError(f'{name} must be finite and positive')


class FourWheelSteering:
    def __init__(self, diameter, track, wheelbase, *, max_wheel_speed_radps=10.,
                 wheel_order=(0,1,2,3), steering_order=(0,1,2,3),
                 wheel_signs=(1,1,1,1), steering_signs=(1,1,1,1),
                 steering_zero_rad=(0.,0.,0.,0.), max_steering_angle_rad=math.pi/2,
                 steering_tolerance_rad=.1):
        for name,value in [('wheel diameter',diameter),('track',track),('wheelbase',wheelbase),
                           ('max_wheel_speed_radps',max_wheel_speed_radps)]:
            if not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
        validate_steering_options(wheel_order, steering_order, wheel_signs, steering_signs,
                                  steering_zero_rad, max_steering_angle_rad, steering_tolerance_rad)
        self.radius=diameter/2
        self.positions=((wheelbase/2,track/2),(wheelbase/2,-track/2),
                        (-wheelbase/2,track/2),(-wheelbase/2,-track/2))
        self.limit=max_wheel_speed_radps
        self.wheel_order=tuple(wheel_order)
        self.steering_order=tuple(steering_order)
        self.wheel_signs=tuple(wheel_signs)
        self.steering_signs=tuple(steering_signs)
        self.zero=tuple(steering_zero_rad)
        self.max_angle=max_steering_angle_rad
        self.tolerance=steering_tolerance_rad
        self._angles=[0.]*4

    def solve(self, v, w):
        if not all(map(math.isfinite,(v,w))):
            raise ValueError('body velocity must be finite')
        angles=list(self._angles)
        speeds=[]
        for i,(x,y) in enumerate(self.positions):
            vx,vy=v-w*y,w*x
            speed=math.hypot(vx,vy)/self.radius
            if speed>1e-12:
                angle=math.atan2(vy,vx)
                # Reverse drive is equivalent to a 180 degree steering change.
                if angle>math.pi/2:
                    angle-=math.pi
                    speed=-speed
                elif angle<-math.pi/2:
                    angle+=math.pi
                    speed=-speed
                if abs(angle)>self.max_angle+1e-9:
                    raise ValueError(f'wheel {i} requires steering {angle:.3f} rad, limit {self.max_angle:.3f}')
                angles[i]=angle
            speeds.append(speed)
        scale=min(1.,self.limit/max(max(map(abs,speeds)),1e-12))
        self._angles=angles
        return WheelCommand(
            tuple(speeds[i]*s*scale for i,s in zip(self.wheel_order,self.wheel_signs)),
            tuple(angles[i]*s+z for i,s,z in zip(self.steering_order,self.steering_signs,self.zero)),scale)

    def aligned(self, command, feedback_angles):
        # Compare actual constraint angles, not wrapped headings: the joint must
        # physically arrive before driving. Stop commands retain steering.
        return feedback_angles is not None and all(
            abs(a-b)<=self.tolerance for a,b in zip(command.turn_angles,feedback_angles))


def load_wheel_geometry(path):
    with Path(path).open(encoding='utf-8') as stream:
        capability=yaml.safe_load(stream)['capability']
    return tuple(float(capability[key]) for key in
                 ('wheel_diameter_m','track_width_m','wheelbase_m'))


STEERING_CONFIG_FIELDS=('wheel_order','steering_order','wheel_signs','steering_signs',
                       'steering_zero_rad','max_steering_angle_rad','steering_tolerance_rad')


def steering_options(config):
    return {name:getattr(config,name) for name in STEERING_CONFIG_FIELDS}
