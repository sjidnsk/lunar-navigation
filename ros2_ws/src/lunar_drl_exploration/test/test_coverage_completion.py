"""Measured completion must bound unobserved truth, not count frontiers."""
from dataclasses import replace
import numpy as np
import pytest
from lunar_drl_exploration.contracts import Pose, SensorSpec
from lunar_drl_exploration.task_analysis import TaskAnalyzer
from test_task_analysis import snapshot, task


def test_ratio_completion_retains_frontiers_and_actual_observation():
    m=np.ones((10,10),np.uint8);m[5,5]=0
    a=TaskAnalyzer(task(0,0,10,10),SensorSpec(range_m=3),coverage_target=.99)
    r=a.update(snapshot(m))
    assert r.completed and not r.exhausted
    assert len(r.frontier_cells)>0
    assert r.known_area_m2==99 and r.remaining_area_upper_m2==1
    assert r.coverage_lower_bound==pytest.approx(.99)
    assert a.remaining_mask.sum()==1
    assert a.update(snapshot(m)).new_area_m2==0


def test_threshold_and_strict_exhaustion_are_distinct():
    m=np.ones((10,10),np.uint8);m[5,5]=0
    s=snapshot(m);t=task(0,0,10,10);sensor=SensorSpec(range_m=3)
    assert not TaskAnalyzer(t,sensor,coverage_target=.9901).update(s).completed
    assert not TaskAnalyzer(t,sensor).update(s).completed
    r=TaskAnalyzer(t,sensor).update(snapshot(np.ones_like(m)))
    assert r.completed and r.exhausted and r.coverage_lower_bound==1


def test_sealed_unobservable_room_is_excluded_from_remaining_area():
    m=np.ones((15,15),np.uint8);b=m.copy()
    m[4:11,4:11]=2;b[4:11,4:11]=2
    m[6:9,6:9]=0;b[6:9,6:9]=0
    a=TaskAnalyzer(task(0,0,15,15),SensorSpec(range_m=4))
    r=a.update(snapshot(m,b))
    assert r.remaining_area_upper_m2==0 and r.exhausted and r.completed
    assert r.known_area_m2==216 and r.coverage_lower_bound==1
    assert not a.remaining_mask.any()


def test_narrow_door_does_not_hide_large_task_demand():
    m=np.full((20,30),2,np.uint8);m[8:12,2:8]=1
    m[9,8:18]=0;m[5:15,18:28]=0
    a=TaskAnalyzer(task(18,5,28,15),SensorSpec(range_m=3),coverage_target=.99)
    r=a.update(snapshot(m,m,pose=Pose(5.5,9.5,0)))
    assert r.remaining_area_upper_m2==100
    assert r.coverage_lower_bound==0 and not r.completed


def test_task_outside_area_is_not_part_of_remaining_budget():
    m=np.ones((20,20),np.uint8);m[:15,10]=0
    # The single task demand connects to the unknown exterior; do not charge
    # the whole exterior's area or discard the demand for lacking a closed hole.
    a=TaskAnalyzer(task(10,14,11,15),SensorSpec(range_m=3))
    r=a.update(snapshot(m))
    assert r.remaining_area_upper_m2==1 and not r.completed


def test_missing_native_start_does_not_certify_completion():
    s=replace(snapshot(np.ones((5,5),np.uint8)),
              start_connections=np.empty((0,2),int),start_connection_status='INPUT_UNAVAILABLE')
    r=TaskAnalyzer(task(0,0,5,5),SensorSpec(),coverage_target=.99).update(s)
    assert not r.available and not r.completed
    assert r.coverage_lower_bound is None and r.remaining_area_upper_m2 is None


def test_zero_coverable_area_is_exhaustion_without_invented_percentage():
    m=np.ones((15,15),np.uint8);m[4:11,4:11]=2;m[6:9,6:9]=0
    r=TaskAnalyzer(task(6,6,9,9),SensorSpec(range_m=4)).update(snapshot(m,m))
    assert r.completed and r.exhausted and r.remaining_area_upper_m2==0
    assert r.coverage_lower_bound is None


@pytest.mark.parametrize('target',[0.,1.1,float('nan')])
def test_invalid_target_is_rejected_at_configuration(target):
    with pytest.raises(ValueError):TaskAnalyzer(task(0,0,5,5),SensorSpec(),coverage_target=target)


def test_partial_measured_worlds_never_exclude_true_remaining_observations():
    from scipy.ndimage import label
    from lunar_drl_exploration.sensor import visible_cells
    sensor=SensorSpec(range_m=2.8,fov_deg=90)
    for seed in range(80):
        rng=np.random.default_rng(seed)
        truth=np.full((8,9),2,np.uint8)
        truth[1:-1,1:-1]=rng.choice([1,2],(6,7),p=[.7,.3])
        truth[3,3]=1
        labels,_=label(truth==1)
        reachable=labels==labels[3,3]
        poses=np.argwhere(reachable)
        reference=np.zeros_like(reachable);known=reference.copy()
        # Ground truth is enumerated only in the test. Actual K comes from a
        # subset of attainable sensor poses, so K is not arbitrary oracle data.
        for y,x in poses:
            rows,cols=visible_cells(truth,(0.,0.),1.,Pose(x+.5,y+.5,0.),
                                   SensorSpec(range_m=sensor.range_m,fov_deg=360))
            reference[rows,cols]=True
        selected=np.vstack(([3,3],poses[rng.choice(len(poses),min(3,len(poses)),replace=False)]))
        for y,x in selected:
            rows,cols=visible_cells(truth,(0.,0.),1.,Pose(x+.5,y+.5,float(rng.uniform(-np.pi,np.pi))),sensor)
            known[rows,cols]=True
        measured=np.where(known,truth,0).astype(np.uint8)
        a=TaskAnalyzer(task(1,1,8,7),sensor,coverage_target=.99)
        r=a.update(snapshot(measured,measured,pose=Pose(3.5,3.5,0)))
        x0,y0,_,_=a.workspace.bounds
        u=a.remaining_mask[-y0:8-y0,-x0:9-x0]
        mask=np.zeros_like(known);mask[1:7,1:8]=True
        remaining=reference&mask&~known
        assert not np.any(remaining&~u),seed
        true_coverage=(reference&mask&known).sum()/(reference&mask).sum()
        assert r.coverage_lower_bound is None or r.coverage_lower_bound<=true_coverage+1e-12,seed


def test_completed_threshold_is_used_by_observed_only_runtime():
    from lunar_drl_exploration.runtime import InferenceRuntime
    from test_runtime import Adapter
    m=np.ones((10,10),np.uint8);m[5,5]=0
    adapter=Adapter();adapter.snapshot=snapshot(m,pose=Pose(2.5,2.5,0))
    def unexpected_policy(_):raise AssertionError('finished task must not issue another goal')
    runtime=InferenceRuntime(adapter,unexpected_policy,SensorSpec(range_m=3),coverage_target=.99)
    runtime.start(task(0,0,10,10));runtime.tick()
    assert runtime.state=='COVERAGE_REACHED' and not adapter.inflight
    assert runtime.report.completed and not runtime.report.exhausted
    assert runtime.status()['coverage_lower_bound']==pytest.approx(.99)
    assert not runtime.status()['reference_available']


def test_target_is_part_of_full_resume_semantics():
    from lunar_drl_exploration.config import TrainingConfig,resume_semantics
    config=TrainingConfig()
    assert resume_semantics(config)!=resume_semantics(replace(config,coverage_target=.99))


def test_measured_bound_can_finish_without_residual_frontier_witness(monkeypatch):
    import lunar_drl_exploration.task_analysis as analysis
    m=np.ones((10,10),np.uint8);m[5,5]=0
    # Sparse witness generation may be unavailable although the dense measured
    # remaining-area bound is valid. Finishing requires no further graph action.
    monkeypatch.setattr(analysis,'direct_witnesses',
                        lambda _b,_r,targets,_radius:np.full_like(targets,-1))
    r=TaskAnalyzer(task(0,0,10,10),SensorSpec(range_m=3),coverage_target=.99).update(snapshot(m))
    assert r.remaining_area_upper_m2==1 and not len(r.frontier_cells)
    assert r.available and r.completed and not r.exhausted
