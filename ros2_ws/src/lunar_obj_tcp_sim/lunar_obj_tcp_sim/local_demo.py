"""Run a local vehicle until the existing ROS operator has finished stopping."""
import argparse
import math
from pathlib import Path
import signal
import subprocess
import threading

from .geometry import TerrainMap
from .configuration import load_simulation_config, SimulationConfig
from .local_vehicle import LocalVehicle
from .kinematics import load_wheel_geometry, steering_options, STEERING_CONFIG_FIELDS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True)
    parser.add_argument('--runner', required=True)
    parser.add_argument('--platform-config', required=True)
    parser.add_argument('--mode', choices=('nav', 'explore'), default='explore')
    parser.add_argument('--port', type=int, default=0, help='local port; 0 selects an available port')
    parser.add_argument('--start-x', type=float)
    parser.add_argument('--start-y', type=float)
    parser.add_argument('--yaw', type=float, default=0., help='initial yaw in radians')
    parser.add_argument('--domain-id', type=int, default=74)
    parser.add_argument('--no-rviz', action='store_true')
    parser.add_argument('--config')
    args = parser.parse_args(argv)
    terrain = TerrainMap(args.map)
    xmin, ymin, xmax, ymax = terrain.metadata['task_bounds_xy']
    x = args.start_x if args.start_x is not None else (xmin+xmax)/2
    y = args.start_y if args.start_y is not None else (ymin+ymax)/2
    z = terrain.height_at(x, y)
    if not all(math.isfinite(v) for v in (x, y, z, args.yaw)):
        parser.error('start pose must have finite coordinates and valid terrain elevation')
    diameter, track, wheelbase = load_wheel_geometry(args.platform_config)
    config = load_simulation_config(args.config) if args.config else None
    vehicle_id = config.vehicle_id if config else 0
    vehicle = LocalVehicle((x, y, z), yaw=args.yaw, radius=diameter/2, track=track, wheelbase=wheelbase,
                           steering_options=steering_options(config) if config else {
                               name:SimulationConfig.__dataclass_fields__[name].default for name in STEERING_CONFIG_FIELDS},
                           port=args.port, height_at=terrain.height_at, vehicle_id=vehicle_id)
    stopping = threading.Event()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: stopping.set())
    child = None
    try:
        command = ['bash', str(Path(args.runner).resolve()), '--host', '127.0.0.1',
                   '--port', str(vehicle.port), '--map', str(Path(args.map).resolve()),
                   '--mode', args.mode, '--domain-id', str(args.domain_id),
                   '--platform-config', str(Path(args.platform_config).resolve())]
        command += ['--no-rviz'] if args.no_rviz else ['--rviz', '--local-rviz']
        if args.config:
            command += ['--config', args.config]
        print(f'LOCAL RViz demo: 127.0.0.1:{vehicle.port}, start=({x:g},{y:g},{z:g}), '
              f'mode={args.mode}, 1x; kinematics and height following, no UE collision physics', flush=True)
        child = subprocess.Popen(command, start_new_session=True)
        while child.poll() is None and not stopping.wait(.1):
            if vehicle.errors:
                print('Local vehicle error: '+vehicle.errors[-1], flush=True)
                stopping.set()
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            # Keep feedback alive while the existing operator cancels and observes stop.
            child.wait()
        return 1 if vehicle.errors else child.returncode
    finally:
        vehicle.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
