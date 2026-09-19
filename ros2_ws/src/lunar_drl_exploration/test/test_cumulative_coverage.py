from dataclasses import replace
import numpy as np
import pytest
from lunar_drl_exploration.contracts import Pose,SensorSpec
from lunar_drl_exploration.task_analysis import TaskAnalyzer
from test_task_analysis import snapshot,task


def sequence():
    m=np.zeros((12,12),np.uint8);m[3:9,3:9]=1
    first=snapshot(m,pose=Pose(3.5,3.5,0))
    m[5,5]=0
    second=replace(snapshot(m,pose=first.pose),revision=2)
    m[5,5]=1
    third=replace(snapshot(m,pose=first.pose),revision=3)
    return first,second,third


def test_same_epoch_classification_regression_never_repays_observation():
    a=TaskAnalyzer(task(0,0,12,12),SensorSpec(range_m=3))
    states=sequence();reports=[a.update(s) for s in states]
    assert [r.known_area_m2 for r in reports]==[36,36,36]
    assert [r.new_area_m2 for r in reports]==[36,0,0]
    assert states[1].cell_at(5,5).intrinsic_state==0
    assert states[1].cell_at(5,5).navigation_state==0


def test_spatial_union_counts_new_cell_during_old_regression_not_max_area():
    from lunar_drl_exploration.coverage import CoverageHistory
    h=CoverageHistory(task(0,0,12,12));first,second,_=sequence()
    h.consume(first)
    m=np.zeros((12,12),np.uint8);m[3:9,3:9]=1;m[5,5]=0;m[9,5]=1
    replacement=replace(snapshot(m,pose=first.pose),revision=2)
    h.consume(replacement)
    assert h.known_area_m2==37
    mask=h.mask((0,0,12,12))
    assert mask[5,5] and mask[9,5]
    h.consume(replacement);assert h.known_area_m2==37
    h.consume(replace(second,epoch='new',revision=1));assert h.known_area_m2==35
    assert not h.mask((0,0,12,12))[5,5]


def test_cheap_consume_and_decision_report_share_one_meter_and_projection():
    from lunar_drl_exploration.decision import DecisionCore
    first,second,third=sequence();c=DecisionCore(task(0,0,12,12),SensorSpec(range_m=3))
    c.consume(first);_,r1=c.observe(first)
    c.consume(second);c.consume(third)
    assert c.coverage.known_area_m2==36
    _,r2=c.observe(third)
    assert r2.new_area_m2==0 and r1.known_area_m2==r2.known_area_m2
    assert c.coverage.mask((0,0,12,12)).sum()==36
    other=DecisionCore(task(0,0,12,12),SensorSpec(range_m=3));other.consume(second)
    assert other.coverage.known_area_m2==35


def test_training_projection_inference_pause_epoch_and_new_task_agree():
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.decision import DecisionCore
    from lunar_drl_exploration.reference import CoverageReference
    from lunar_drl_exploration.ros_env import RosExplorationEnv,make_transition
    from lunar_drl_exploration.runtime import InferenceRuntime
    from lunar_drl_exploration.scene import TerrainGrid
    from test_runtime import Adapter
    first,second,third=sequence();t=task(0,0,12,12);sensor=SensorSpec(range_m=3)
    env=RosExplorationEnv(TrainingConfig(),0)
    env.terrain=TerrainGrid.from_heights(np.zeros((12,12),np.float32),1.,(0.,0.),env.config.platform)
    env.reference=CoverageReference.build(env.terrain,first.pose,t,sensor)
    from lunar_drl_exploration.privileged import build_truth, PrivilegedBuilder
    env.privileged_builder=PrivilegedBuilder(env.terrain,build_truth(env.terrain,env.reference),t,sensor)
    env.core=DecisionCore(t,sensor)
    adapter=Adapter();runtime=InferenceRuntime(adapter,lambda obs:0,sensor)
    runtime.start(t);runtime.pause()
    for snap in (first,second,third,third):
        env._apply_known(snap);adapter.snapshot=snap;runtime.tick()
        env.observation,report=env.core.observe(snap)
        assert runtime.state=='PAUSED'
        assert runtime.status()['known_area_m2']==env._known_area==36
        assert np.unpackbits(env._privileged().observed,bitorder='little').sum()==36
        observation,report=env.core.observe(snap)
        assert report.known_area_m2==36
        if snap is first:
            initial_observation=observation;initial_privileged=env._privileged()
            initial_area=env._known_area
    transition=make_transition(initial_observation,0,observation,initial_privileged,env._privileged(),
        initial=(initial_area,0.,0.),final=(env._known_area,2.,.5),completed=False,
        budget_hit=False,episode_id="coverage_fixture",actor_version=7)
    assert transition.parts.new_area_m2==0
    assert transition.reward==pytest.approx(-.02*2/10-.005*.5/np.pi-.001)
    # Same epoch, a FREE->BLOCKED classification still carries observed evidence.
    m=np.zeros((12,12),np.uint8);m[3:9,3:9]=1;m[5,5]=2
    flipped=replace(snapshot(m,b=m,pose=first.pose),revision=4)
    env._apply_known(flipped);assert env._known_area==36
    # New task starts from current native inputs, not previous task's private K.
    runtime.start(t);runtime.pause();adapter.snapshot=second;runtime.tick()
    assert runtime.status()['known_area_m2']==35
    # Epoch and lattice changes independently invalidate old coordinate history.
    changed=replace(second,epoch='replacement',revision=1)
    env._apply_known(changed);assert env._known_area==35
    assert np.unpackbits(env._privileged().observed,bitorder='little').sum()==35
    shifted=replace(changed,origin=(100.,100.,0.),revision=1)
    env._apply_known(shifted);assert env._known_area==0
