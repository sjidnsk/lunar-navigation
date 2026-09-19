from dataclasses import replace
import numpy as np
import pytest
from lunar_drl_exploration.contracts import Pose, SensorSpec
from lunar_drl_exploration.decision import DecisionCore
from lunar_drl_exploration.config import LearningConfig
from test_task_analysis import snapshot, task


def test_fully_measured_stance_has_only_transit_actions():
    m=np.ones((12,12),np.uint8)
    s=snapshot(m,pose=Pose(5.7,5.7,0))
    o,_=DecisionCore(task(0,0,12,12),SensorSpec(range_m=3)).observe(s)
    assert len(o.goals)>0
    assert np.all(o.action_nodes!=o.current_index)
    assert np.all(np.linalg.norm(o.goals[:,:2]-[5.7,5.7],axis=1)>s.goal_position_tolerance_m)
    assert np.all(o.features[:,3:11]==0)  # transit needs no immediate utility


@pytest.mark.parametrize('yaw', [0., 2*np.pi-0.01])
def test_pending_scan_uses_visibility_and_native_yaw_tolerance(yaw):
    m=np.zeros((15,15),np.uint8); m[4:11,4:11]=1
    s=snapshot(m,pose=Pose(7.5,7.5,yaw))
    o,r=DecisionCore(task(0,0,15,15),SensorSpec(range_m=5)).observe(s)
    assert not r.exhausted
    idx=np.flatnonzero(o.action_nodes==o.current_index)
    assert len(idx)>0
    errors=np.abs(np.arctan2(np.sin(o.action_yaws[idx]-yaw),np.cos(o.action_yaws[idx]-yaw)))
    assert np.all(errors>s.goal_yaw_tolerance_rad)
    headings=np.rint(o.action_yaws[idx]/(np.pi/4)).astype(int)%8
    assert np.all(o.features[o.current_index,3+headings]>0)


def test_discount_and_entropy_defaults_and_range():
    assert LearningConfig().gamma==.995
    assert LearningConfig().target_entropy_factor==.10
    assert LearningConfig(gamma=.999).gamma==.999
    for gamma in (0.,1.,float('nan')):
        with pytest.raises(ValueError): LearningConfig(gamma=gamma)


def test_legacy_actor_cannot_be_relabelled_for_new_actions(tmp_path):
    import torch
    from lunar_drl_exploration.config import ModelConfig
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.runtime import ActorPolicy
    from lunar_drl_exploration.evaluation import export_actor
    actor=SACLearner(ModelConfig(width=16,heads=2,layers=1)).actor_state()
    actor['schema']='task_graph_v1'
    path=tmp_path/'old.pt'; torch.save(actor,path)
    with pytest.raises(ValueError): ActorPolicy.load(path)
    with pytest.raises(ValueError): export_actor(path,tmp_path/'export.pt')
    assert not (tmp_path/'export.pt').exists()


def test_fixed_small_curriculum_keeps_large_counts_small():
    from lunar_drl_exploration.cli import parser, load_config
    from lunar_drl_exploration.training import Curriculum
    c=load_config(parser().parse_args(['train','--config','config/drl_exploration_small.yaml']))
    curriculum=Curriculum(c,np.random.default_rng(3))
    for i in range(30):
        spec=curriculum.next(i%8,100000)
        assert 40<=spec['extent']<=80 and spec['episode_budget']==512
    assert 'small' in str(c.output_dir)
