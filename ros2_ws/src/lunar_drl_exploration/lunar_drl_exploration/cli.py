"""Stdlib-only CLI bootstrap; spawned ROS workers never import learner modules."""
import argparse
import os
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description='Sparse graph exploration: isolated training and observed-only inference')
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate', 'infer', 'export'):
        cmd = commands.add_parser(name)
        if name != 'export':
            cmd.add_argument('--config', type=Path)
            cmd.add_argument('--sensor-range', type=float)
            cmd.add_argument('--sensor-fov', type=float)
        if name == 'train':
            cmd.add_argument('--resume', action='store_true', help='restore output-dir/resume.pt; start new environment episodes')
            cmd.add_argument('--output-dir')
            cmd.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
            cmd.add_argument('--max-transitions', type=int, help='new admissions this run; bounded finished drain may overshoot')
            cmd.add_argument('--probe', action='store_true', help='explicit finite diagnostic run; keeps full model/batch/physical backend')
            cmd.add_argument('--probe-warmup', type=int)
            cmd.add_argument('--probe-extent', type=float)
            cmd.add_argument('--probe-budget', type=int)
            cmd.add_argument('--domain-base', type=int, default=210)
        elif name == 'evaluate':
            cmd.add_argument('--actor', '--frozen-actor', required=True, type=Path)
            cmd.add_argument('--seeds', '--seed', type=int, nargs='+', default=[2026091501, 2026091502])
            cmd.add_argument('--families', '--family', choices=('moon', 'cave'), nargs='+', default=['moon', 'cave'])
            cmd.add_argument('--extents', '--extent', type=float, nargs='+', default=[40.])
            cmd.add_argument('--budget', type=int, default=512)
            cmd.add_argument('--output', type=Path)
            cmd.add_argument('--domain-base', type=int, default=220)
        elif name == 'infer':
            cmd.add_argument('--actor', required=True, type=Path)
            cmd.add_argument('--task-topic', default='/Car/T4/exploration/task')
        else:
            cmd.add_argument('--checkpoint', required=True, type=Path)
            cmd.add_argument('--output', required=True, type=Path)
    return result


def load_config(args):
    from dataclasses import replace
    from .config import TrainingConfig, config_record, training_config_from_record
    config = TrainingConfig()
    if args.config:
        import yaml
        loaded = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        record = config_record(config)
        for key, value in loaded.items():
            if key not in record: raise ValueError('unknown training setting: ' + key)
            if isinstance(record[key], dict): record[key].update(value)
            else: record[key] = value
        config = training_config_from_record(record)
    if args.sensor_range is not None:
        config = replace(config, sensor=replace(config.sensor, range_m=args.sensor_range))
    if args.sensor_fov is not None:
        config = replace(config, sensor=replace(config.sensor, fov_deg=args.sensor_fov))
    if args.command == 'train':
        overrides = any(getattr(args, key) is not None for key in ('probe_warmup', 'probe_extent', 'probe_budget'))
        if overrides and not args.probe: raise ValueError('probe overrides require --probe')
        if args.max_transitions is not None and args.max_transitions <= 0: raise ValueError('positive --max-transitions required')
        for key in ('probe_extent', 'probe_budget'):
            if getattr(args, key) is not None and getattr(args, key) <= 0: raise ValueError('positive ' + key + ' required')
        if args.probe_warmup is not None: config = replace(config, warmup=args.probe_warmup)
        if args.output_dir: config = replace(config, output_dir=args.output_dir)
    return config


def main(argv=None):
    args = parser().parse_args(argv)
    # Set before config imports NumPy, and before a spawn child imports __main__.
    from .worker import numerical_threads
    numerical_threads(4 if args.command == 'train' else 2)
    if args.command == 'export':
        from .evaluation import export_actor
        return export_actor(args.checkpoint, args.output)
    config = load_config(args)
    if args.command == 'train':
        if args.probe and args.max_transitions is None:
            raise ValueError('--probe requires a finite --max-transitions')
        from .training import run_training
        return run_training(config, resume=args.resume, device=args.device,
            max_transitions=args.max_transitions, domain_base=args.domain_base,
            probe_extent=args.probe_extent, probe_budget=args.probe_budget)
    if args.command == 'evaluate':
        from .evaluation import evaluate
        return evaluate(config, args.actor, seeds=args.seeds, families=args.families,
            extents=args.extents, budget=args.budget, output=args.output, domain_base=args.domain_base)
    from .evaluation import infer
    return infer(config, args.actor, task_topic=args.task_topic)


if __name__ == '__main__':
    main()
