import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from lunar_drl_exploration.contracts import Pose,SensorSpec
from lunar_drl_exploration.ros_messages import NavigationResult
from test_task_analysis import snapshot,task


def test_fresh_inference_import_never_imports_training_modules():
    code='''import sys,importlib.abc
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in {'lunar_drl_exploration.ros_env','lunar_drl_exploration.scene','lunar_drl_exploration.reference','lunar_drl_exploration.sac'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from lunar_drl_exploration.runtime import InferenceRuntime
assert 'torch' not in sys.modules
'''
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable,'-c',code],check=True,env=env)


class Adapter:
    def __init__(self):
        m=np.ones((9,9),np.uint8);m[8,:]=0
        self.snapshot=snapshot(m,pose=Pose(2.5,2.5,0))
        self.velocity=(0.,0.);self.pose=self.snapshot.pose
        self.inflight=False;self.goal=None;self.result=None;self.cancelled=False
    def poll_snapshot(self,*args):return self.snapshot
    def begin_goal(self,goal):self.goal=tuple(goal);self.inflight=True
    def cancel_goal(self):self.cancelled=True
    def poll_goal(self):
        value=self.result
        if value:self.result=None;self.inflight=False
        return value


def test_pause_waits_for_cancel_and_stop_resume_redecides_without_truth():
    from lunar_drl_exploration.runtime import InferenceRuntime
    a=Adapter();r=InferenceRuntime(a,lambda obs:0,SensorSpec(range_m=3))
    r.start(task(0,0,9,9));r.tick()
    assert a.inflight
    original=a.goal
    r.pause();assert a.cancelled and r.state=='PAUSING'
    a.result=NavigationResult(5,'CANCELED',1);a.velocity=(.1,0.)
    r.tick();assert r.state=='PAUSING'
    a.velocity=(0.,0.);r.tick();assert r.state=='PAUSED'
    assert r.status()['reference_area_m2'] is None
    assert 'coverage_ratio' not in r.status()
    r.resume();r.tick();assert a.inflight and a.goal==original
    r.cancel();a.result=NavigationResult(5,'CANCELED',2);r.tick()
    assert r.state=='CANCELED'


def test_actor_artifact_load_never_constructs_critic(tmp_path,monkeypatch):
    import pytest
    torch=pytest.importorskip('torch')
    import lunar_drl_exploration.model as model
    from lunar_drl_exploration.config import ModelConfig
    from lunar_drl_exploration.runtime import ActorPolicy
    cfg=ModelConfig(width=16,heads=2,layers=1)
    actor=model.Actor(cfg)
    def forbidden(*a,**k):raise AssertionError('Critic instantiated')
    monkeypatch.setattr(model,'Critic',forbidden)
    path=tmp_path/'actor.pt'
    torch.save({'schema':'task_graph_v1','model_config':cfg.__dict__,'state_dict':actor.state_dict(),'version':4},path)
    policy=ActorPolicy.load(path)
    assert isinstance(policy.actor,model.Actor)
    torch.save({'schema':'task_graph_v1','model_config':cfg.__dict__,'state_dict':actor.state_dict(),'version':4,'critic':{}},path)
    with pytest.raises(ValueError):ActorPolicy.load(path)


def test_pause_uses_public_controller_stopped_tolerances():
    from lunar_drl_exploration.runtime import InferenceRuntime
    from lunar_pure_wheeled_controller.tracking import TrackingPolicy
    p=TrackingPolicy();a=Adapter();r=InferenceRuntime(a,lambda obs:0)
    r.start(task(0,0,9,9));r.pause()
    a.velocity=(p.stopped_linear_mps,p.stopped_angular_radps)
    r.tick();assert r.state=='PAUSED'
