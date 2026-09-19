"""One trainable C=0 policy, evaluated on the same maps at cumulative milestones."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import os


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else {}


def atomic_json(path, value):
    from .evaluation import _atomic_file
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
    _atomic_file(path, lambda stream: stream.write(payload))


def checkpoint_progress(path, config):
    if not path.exists(): return 0, 0
    import torch
    from .checkpoint import TrainingState
    record = torch.load(path, map_location='cpu', weights_only=False)
    TrainingState(record).validate(config)
    return record['schedule']['transitions'], record['schedule']['updates']


def execute(command, log, stop):
    """Forward interrupts to the owning child; retain only one bounded stage log."""
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    def interrupt(signum, frame):
        stop['requested'] = True
        if child.poll() is None: child.send_signal(signal.SIGINT)
    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with log.open('wb') as stream:
            while True:
                block = child.stdout.read1(65536)
                if not block: break
                if stream.tell()+len(block) > 8*1024**2:
                    stream.seek(0); stream.truncate()
                stream.write(block); stream.flush()
                if sys.stdout.isatty():
                    sys.stdout.buffer.write(block); sys.stdout.buffer.flush()
        child.stdout.close()
        return child.wait()
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


def write_report(output, state):
    from .evaluation import _atomic_file
    lines = ['# C=0 探索基线', '', f"状态：{state['status']}；阶段：{state.get('phase', 'pending')}",
        f"代码：`{state['code_revision']}`", '',
        '同一条策略持续训练；各阶段冻结评估，无学习。单种子、小图结果不代表收敛或泛化。', '',
        '| 目标转移 | 实际转移 | 更新 | 场景 | 覆盖率 | 路程 m | 零收益比例 | 最长零收益段 | 两点循环窗口数 | 耗尽 | 截断 | 碰撞 | 错误 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for target in state['plan']['milestones']:
        record = state['records'].get(str(target))
        if record is None:
            lines.append(f'| {target} | 待评估 | | | | | | | | | | | |')
            continue
        for case in record['evaluation']['cases']:
            values = [target, record['transitions'], record['updates'], case['family'],
                f"{case['final_coverage']:.2%}", f"{case['distance_m']:.1f}",
                f"{case['zero_gain_ratio']:.1%}", case['max_zero_gain_run'],
                case['zero_gain_two_point_loop_steps'], case['exhausted'], case['truncated'],
                case['collisions'], case['error'] or '']
            lines.append('| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in values)+' |')
    lines += ['', '两点循环按实际世界坐标、0.05 m 容差、连续四次零收益 A-B-A-B-A 判定；',
        '该指标计满足条件的滑动窗口，不是所有循环动作的总数。必要回退不直接算失败。',
        '预算截断、碰撞、基础设施错误均保留，不能冒充探索耗尽。', '']
    payload = '\n'.join(lines).encode()
    _atomic_file(output/'baseline-report.md', lambda stream: stream.write(payload))


def run_baseline(args):
    import fcntl
    from .cli import load_config
    from .config import config_record, resume_semantics
    from types import SimpleNamespace
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if not args.milestones or args.milestones != sorted(set(args.milestones)) or args.milestones[0] <= 0:
        raise ValueError('milestones must be increasing positive cumulative counts')
    if args.eval_budget <= 0: raise ValueError('positive evaluation budget required')
    config = load_config(SimpleNamespace(config=args.config, command='evaluate', sensor_range=None, sensor_fov=None))
    config = replace(config, output_dir=str(output))
    if config.model.actor_score_bound != 0: raise ValueError('baseline uses C=0 Actor')
    plan = dict(milestones=args.milestones, eval_budget=args.eval_budget,
        families=['moon','cave'], seeds=[2026091901], extents=[40.],
        config=config_record(config), semantics=resume_semantics(config))
    with (output/'.baseline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output/'baseline-state.json'; state = read_json(state_path)
        if state and not args.resume: raise ValueError('baseline already exists; use --resume')
        if state and state['plan'] != plan: raise ValueError('resume requires identical baseline plan')
        if not state and (output/'resume.pt').exists(): raise ValueError('checkpoint belongs to another run')
        revision = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
        state = state or dict(schema='exploration_baseline_v1', plan=plan, records={}, code_revision=revision)
        state.update(status='running', runner_pid=os.getpid(), current_code_revision=revision)
        atomic_json(output/'config.json', config_record(config))
        stop = {'requested': False}
        def save():
            atomic_json(state_path, state); write_report(output, state)
        def command(parts):
            code = execute([sys.executable,'-m','lunar_drl_exploration.cli',*parts],output/'current-stage.log',stop)
            if stop['requested']: raise KeyboardInterrupt
            if code: raise RuntimeError(f'child exit {code}; see current-stage.log')
        try:
            for target in args.milestones:
                if str(target) in state['records']: continue
                count, updates = checkpoint_progress(output/'resume.pt',config)
                if count < target:
                    state['phase'] = f'train_to_{target}'; save()
                    print(f"训练至 {target} 条转移：{output/'current-stage.log'}",flush=True)
                    parts = ['train','--config',str(output/'config.json'),'--device',args.device,
                        '--domain-base',str(args.domain_base),'--max-transitions',str(target-count)]
                    if (output/'resume.pt').exists(): parts += ['--resume']
                    command(parts)
                    final = read_json(output/'run.json').get('observations',{}).get('final',{})
                    if not final.get('stop_reason','').startswith('max new transitions reached'):
                        raise KeyboardInterrupt
                    count, updates = checkpoint_progress(output/'resume.pt',config)
                    if count < target: raise RuntimeError('training stopped below milestone')
                state['phase'] = f'evaluate_{target}'; save()
                print(f"冻结评估：目标 {target}，实际 {count}，更新 {updates}",flush=True)
                command(['export','--checkpoint',str(output/'resume.pt'),'--output',str(output/'actor.pt')])
                command(['evaluate','--config',str(output/'config.json'),'--actor',str(output/'actor.pt'),
                    '--families','moon','cave','--extents','40','--seeds','2026091901',
                    '--budget',str(args.eval_budget),'--domain-base',str(args.domain_base+8),
                    '--output',str(output/'evaluation.json')])
                evaluation = read_json(output/'evaluation.json')
                state['records'][str(target)] = dict(transitions=count,updates=updates,
                    evaluation=evaluation,code_revision=revision)
                save()
            state.update(status='complete',phase='complete');save();return 0
        except KeyboardInterrupt:
            state['status']='interrupted';save();return 130
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise


def main(argv=None):
    from .worker import numerical_threads
    numerical_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/drl_exploration.yaml'))
    parser.add_argument('--output-dir',type=Path,default=Path('training-output/drl-baseline-clearance-v1'))
    parser.add_argument('--milestones',type=int,nargs='+',default=[2000,5000,10000])
    parser.add_argument('--eval-budget',type=int,default=512)
    parser.add_argument('--domain-base',type=int,default=210)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--resume',action='store_true')
    return run_baseline(parser.parse_args(argv))


if __name__=='__main__': sys.exit(main())
