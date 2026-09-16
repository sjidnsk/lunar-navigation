"""Bounded commissioning turn test; never launch beside a navigation TCP client."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from .geometry import rotation_matrix
from .kinematics import FourWheelSteering, load_wheel_geometry, steering_options, STEERING_CONFIG_FIELDS
from .configuration import load_simulation_config, SimulationConfig
from .transport import TcpVehicleTransport


def summarize(rows):
    result = {}
    for phase in dict.fromkeys(row['phase'] for row in rows):
        samples = [row for row in rows if row['phase'] == phase]
        if len(samples) < 2:
            continue
        t = np.array([r['t'] for r in samples])
        rotations = np.array([rotation_matrix(r['orientation_xyzw']) for r in samples])
        # Actor forward is -X, then reflect world Y into ROS. Independent of bridge.
        yaw = np.unwrap(np.arctan2(rotations[:, 1, 0], -rotations[:, 0, 0]))
        angular = np.array([r['angular_velocity'] for r in samples])
        integral = np.sum(angular[:-1] * np.diff(t)[:, None], axis=0)
        delta = float(yaw[-1]-yaw[0])
        result[phase] = dict(samples=len(samples), yaw_delta_deg=math.degrees(delta),
                            angular_mean=angular.mean(axis=0).tolist(),
                            angular_integral=integral.tolist(),
                            yaw_rad_per_raw_y=(delta/integral[1] if abs(integral[1]) > .01 else None),
                            wheel_mean=np.mean([r['wheel_speeds'] for r in samples], axis=0).tolist())
    return result


def run(host, port, vehicle_id, platform_config, output, duration=3., speed=.1, config=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = output/'turn-feedback.json'
    if record.exists():
        raise ValueError('use a new output directory to preserve previous test evidence')
    model = FourWheelSteering(*load_wheel_geometry(platform_config),
                              **(steering_options(load_simulation_config(config)) if config else {
                                  name:SimulationConfig.__dataclass_fields__[name].default for name in STEERING_CONFIG_FIELDS}))
    rows, errors = [], []
    phase = 'connect'
    def receive(item):
        rows.append(dict(t=item.received_monotonic, phase=phase, **asdict(item.feedback)))
    transport = TcpVehicleTransport(host, port, vehicle_id=vehicle_id,
                                   on_feedback=receive, on_error=lambda e:errors.append(str(e)))
    status, error = 'FAILED', None
    try:
        transport.start()
        deadline = time.monotonic()+5
        while not transport.feedback_is_fresh():
            if time.monotonic()>deadline or errors:
                raise RuntimeError('no fresh vehicle feedback: '+str(errors))
            time.sleep(.05)
        start = np.array(rows[-1]['position_m'])
        for phase, seconds, w in [('stationary',2,0), ('positive_turn',duration,speed),
                                   ('stop_1',2,0), ('negative_turn',duration,-speed), ('stop_2',3,0)]:
            print(phase, flush=True)
            deadline = time.monotonic()+seconds
            while time.monotonic()<deadline:
                if not transport.feedback_is_fresh() or errors:
                    raise RuntimeError('feedback lost: '+str(errors))
                if np.linalg.norm(np.array(rows[-1]['position_m'])-start)>.3:
                    raise RuntimeError('translation exceeded 0.3m during turn test')
                command = model.solve(0,w)
                angles = transport.latest_feedback().feedback.turn_angles
                transport.set_wheel_command(command.wheel_speeds if model.aligned(command,angles) else (0.,)*4,
                                            command.turn_angles)
                time.sleep(.05)
        summary = summarize(rows)
        moved = (summary.get('positive_turn', {}).get('yaw_delta_deg', 0.) > 1.
                 and summary.get('negative_turn', {}).get('yaw_delta_deg', 0.) < -1.)
        stopped = max(abs(v) for v in rows[-1]['wheel_speeds']) < .02
        status = 'MOTION_OBSERVED' if moved and stopped else 'NO_MOTION_OR_NOT_STOPPED'
    except BaseException as exc:
        error = str(exc)
    finally:
        phase = 'final_zero'
        transport.set_wheel_command((0.,)*4)
        time.sleep(.3)
        transport.close()
        record.write_text(json.dumps(dict(status=status, error=error, errors=errors,
                          host=host, port=port, vehicle_id=vehicle_id,
                          command_angular_radps=speed, duration_s=duration,
                          summary=summarize(rows), samples=rows), indent=2), encoding='utf-8')
    print(f'{status}: {record}', flush=True)
    return 0 if status == 'MOTION_OBSERVED' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=6668)
    parser.add_argument('--vehicle-id', type=int, default=0)
    parser.add_argument('--platform-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', help='simulation YAML containing wheel/steering calibration')
    args = parser.parse_args()
    raise SystemExit(run(**vars(args)))


if __name__ == '__main__':
    main()
