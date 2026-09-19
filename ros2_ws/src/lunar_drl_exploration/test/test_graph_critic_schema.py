"""Protocol changes are explicit; old production checkpoints are never rewritten."""
import pytest
from lunar_drl_exploration.config import TrainingConfig,resume_semantics


def test_full_resume_identifies_new_geometry_and_privileged_contract():
    semantics=resume_semantics(TrainingConfig())
    assert semantics['observation_schema']=='task_graph_v3'
    assert semantics['action_schema']=='local_metric_pose_v3'
    assert semantics['privileged_schema']=='candidate_truth_v1'
    assert semantics['graph_geometry']=={'coverage_radius_m':2.,'connection_limit_m':8.,'stretch':1.2,
        'base_order':'clearance_desc_yx_v1'}


def test_actor_only_version_rejects_legacy_despite_unchanged_weight_shapes(tmp_path):
    import torch
    from dataclasses import asdict
    from lunar_drl_exploration.model import Actor
    from lunar_drl_exploration.runtime import ActorPolicy
    from lunar_drl_exploration.config import ModelConfig
    config=ModelConfig(width=16,heads=4,layers=1)
    artifact=dict(schema='task_graph_v3',model_config=asdict(config),version=0,state_dict=Actor(config).state_dict())
    path=tmp_path/'actor.pt';torch.save(artifact,path)
    assert ActorPolicy.load(path).actor.config==config
    artifact['schema']='task_graph_v2';torch.save(artifact,path)
    with pytest.raises(ValueError,match='task_graph_v3'):ActorPolicy.load(path)


def test_incomplete_graph_reports_unavailable_not_exhausted(monkeypatch):
    import numpy as np
    from lunar_drl_exploration.decision import DecisionCore
    from lunar_drl_exploration.contracts import SensorSpec
    from test_task_analysis import snapshot,task
    core=DecisionCore(task(0,0,8,8),SensorSpec(range_m=2))
    original=core.builder.build
    def build(*args):
        result=original(*args)
        core.builder.unrepresented_interfaces=np.array([[4,4]])
        return result
    monkeypatch.setattr(core.builder,'build',build)
    measured=np.ones((8,8),np.uint8);measured[4,4]=0
    _,report=core.observe(snapshot(measured))
    assert not report.available and not report.exhausted
    assert report.reason_code=='UNREPRESENTED_TASK_INTERFACES'
