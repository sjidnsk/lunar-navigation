import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def args(tmp_path, resume=False):
    return SimpleNamespace(output_dir=tmp_path/'run', config=None, milestones=[2,5,10],
        eval_budget=512, device='cpu', domain_base=210, resume=resume)


def test_baseline_trains_one_continuing_policy_and_skips_completed_evaluations(tmp_path, monkeypatch):
    from lunar_drl_exploration import baseline_experiment as b
    counts=[0,0];calls=[]
    monkeypatch.setattr(b,'checkpoint_progress',lambda path,config: tuple(counts))
    def execute(command,log,stop):
        calls.append(command)
        root=log.parent
        if 'train' in command:
            delta=int(command[command.index('--max-transitions')+1]);counts[0]+=delta;counts[1]+=1
            (root/'resume.pt').write_bytes(b'fixture')
            b.atomic_json(root/'run.json',{'observations':{'final':{'stop_reason':'max new transitions reached'}}})
        elif 'evaluate' in command:
            b.atomic_json(root/'evaluation.json',{'cases':[dict(family=f,extent_m=40,seed=2026091901,
                final_coverage=.5,distance_m=10,zero_gain_ratio=.2,max_zero_gain_run=2,
                zero_gain_two_point_loop_steps=0,exhausted=False,truncated=True,collisions=0,error=None)
                for f in ('moon','cave')]})
        return 0
    monkeypatch.setattr(b,'execute',execute)
    assert b.run_baseline(args(tmp_path))==0
    trained=[c for c in calls if 'train' in c]
    assert [int(c[c.index('--max-transitions')+1]) for c in trained]==[2,3,5]
    assert ['--resume' in c for c in trained]==[False,True,True]
    evals=[c for c in calls if 'evaluate' in c]
    assert len(evals)==3 and all(c[c.index('--seeds')+1]=='2026091901' for c in evals)
    state=b.read_json(tmp_path/'run'/'baseline-state.json')
    assert state['status']=='complete' and state['records']['10']['transitions']==10
    config=b.read_json(tmp_path/'run'/'config.json')
    assert config['model']['actor_score_bound']==0 and config['learning']['gamma']==.995
    assert config['learning']['target_entropy_factor']==.1
    before=len(calls)
    assert b.run_baseline(args(tmp_path,True))==0 and len(calls)==before
    changed=args(tmp_path,True);changed.eval_budget=128
    with pytest.raises(ValueError,match='plan'):b.run_baseline(changed)


def test_failed_child_cannot_mark_milestone_complete(tmp_path, monkeypatch):
    from lunar_drl_exploration import baseline_experiment as b
    monkeypatch.setattr(b,'checkpoint_progress',lambda path,config:(0,0))
    monkeypatch.setattr(b,'execute',lambda *a: 9)
    with pytest.raises(RuntimeError,match='exit'):b.run_baseline(args(tmp_path))
    state=b.read_json(tmp_path/'run'/'baseline-state.json')
    assert state['status']=='failed' and not state['records']


def test_orderly_training_interrupt_does_not_start_evaluation(tmp_path, monkeypatch):
    from lunar_drl_exploration import baseline_experiment as b
    monkeypatch.setattr(b,'checkpoint_progress',lambda path,config:(0,0))
    def execute(command,log,stop):
        assert 'train' in command
        b.atomic_json(log.parent/'run.json',{'observations':{'final':{'stop_reason':'signal'}}})
        return 0
    monkeypatch.setattr(b,'execute',execute)
    assert b.run_baseline(args(tmp_path))==130
    assert b.read_json(tmp_path/'run'/'baseline-state.json')['status']=='interrupted'


def test_saved_json_config_roundtrips_scientific_notation(tmp_path):
    from lunar_drl_exploration.config import TrainingConfig,config_record
    from lunar_drl_exploration.cli import load_config
    path=tmp_path/'config.json'
    config=TrainingConfig()
    path.write_text(json.dumps(config_record(config)))
    loaded=load_config(SimpleNamespace(config=path,command='evaluate',sensor_range=None,sensor_fov=None))
    assert config_record(loaded)==config_record(config)
