"""Owned replay, real transport and crash-consistent continuation."""
from dataclasses import replace
import gc
import multiprocessing as mp
import pickle
import weakref
import numpy as np
import pytest
from lunar_drl_exploration.contracts import (
    DecisionObservation, PrivilegedScene, PrivilegedState, Transition, RewardParts)
from lunar_drl_exploration.replay import ReplayBuffer, dumps_transport, loads_transport


def observation(revision=0):
    return DecisionObservation(np.arange(2), np.array([[0,0],[1,0]]),
        np.zeros((2,19)), np.array([[0,1]]), np.ones(1), 0,
        np.array([[0,0],[2,0],[2,2],[0,2]]), np.zeros(8),
        np.array([0,1]), np.zeros(2), np.array([[0,0,0],[1,0,0]]), 'e', revision)


def scene(key='s'):
    return PrivilegedScene(key, np.array([[0,0],[1,0]]), np.array([[0,1]]),
        np.ones(1), np.array([0,2,4]), np.arange(4), np.array([15], np.uint8),
        (2,2), {'terrain_id':'terrain', 'nested': {'values':[1,2]}})


def transition(obs=None, nxt=None, key='s'):
    obs = observation() if obs is None else obs
    nxt = observation(1) if nxt is None else nxt
    state = PrivilegedState(key, np.array([3], np.uint8))
    return Transition(obs, 0, .1, nxt, state, state, RewardParts(10,0,0),
                      False, False, 'episode', 0)


def pipe_roundtrip(connection):
    value = loads_transport(connection.recv_bytes())
    connection.send_bytes(dumps_transport(value))
    connection.close()


def test_real_spawn_transport_refreezes_arrays_mappingproxy_and_preserves_alias():
    obs = observation()
    # Actual shared array views as well as shared observation objects.
    other = observation(2)
    object.__setattr__(other, 'positions', obs.positions.view())
    payload = ([transition(obs, other), transition(other, obs)], {'s':scene()})
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    worker = ctx.Process(target=pipe_roundtrip, args=(child,)); worker.start(); child.close()
    parent.send_bytes(dumps_transport(payload))
    assert parent.poll(20)
    restored, scenes = loads_transport(parent.recv_bytes())
    worker.join(20); assert worker.exitcode == 0; parent.close()
    assert restored[0].next_observation is restored[1].observation
    assert np.shares_memory(restored[0].observation.positions, restored[0].next_observation.positions)
    for array in (restored[0].observation.features, scenes['s'].reference_indices):
        assert not array.flags.writeable
        with pytest.raises(ValueError): array.setflags(write=True)
    with pytest.raises(TypeError): scenes['s'].generator_descriptor['nested']['values'] = (9,)
    assert scenes['s'].generator_descriptor['nested']['values'] == (1,2)
    with pytest.raises(TypeError): dumps_transport(object())


def test_replay_accounts_shared_roots_and_releases_last_scene_reference():
    a,b,c = observation(0), observation(1), observation(2)
    first, second = transition(a,b), transition(b,c)
    truth = scene(); watched = weakref.ref(truth)
    replay = ReplayBuffer(1000000)
    replay.add(first, {'s':truth})
    replay.add(second, {'s':truth}); shared_bytes = replay.bytes_used
    independent = ReplayBuffer(1000000)
    independent.add(transition(), {'s':scene()}); independent.add(transition(), {'s':scene()})
    assert shared_bytes < independent.bytes_used
    assert replay.scene_refcounts == {'s':2}
    del truth
    replay.evict_oldest()
    assert replay.scene_refcounts == {'s':1} and watched() is not None
    assert replay.bytes_used < shared_bytes  # Retained table capacity remains charged.
    replay.evict_oldest(); gc.collect()
    assert not replay.scenes and replay.bytes_used == 0 and watched() is None


def test_eviction_snapshot_is_bounded_and_has_no_dangling_scene():
    replay = ReplayBuffer(1000000)
    replay.add(transition(), {'s':scene()})
    budget = replay.bytes_used + 2048
    limited = ReplayBuffer(budget)
    for i in range(9): limited.add(transition(key=str(i)), {str(i):scene(str(i))})
    assert limited.bytes_used <= budget and len(limited) == 1
    assert set(limited.scenes) == {'8'}
    before = limited.bytes_used
    with pytest.raises(ValueError, match='scene'): limited.add(transition(key='absent'), {})
    assert limited.bytes_used == before
    saved = limited.snapshot(100000)
    assert len(saved) <= 100000
    restored = ReplayBuffer.from_snapshot(saved, max_bytes=budget)
    assert len(restored) == 1 and set(restored.scenes) == {'8'}
    item = restored.sample(64, np.random.default_rng(1))[0]
    assert item.privileged.scene_id == '8' and not item.observation.features.flags.writeable
    tiny = ReplayBuffer.from_snapshot(limited.snapshot(512), max_bytes=budget)
    assert len(tiny) == 0 and not tiny.scenes
    with pytest.raises(ValueError, match='exceed'): ReplayBuffer(1).add(transition(), {'s':scene()})


def test_static_scene_id_binds_actual_reference_and_evicts_independently():
    from lunar_drl_exploration.scene import TerrainGrid
    from lunar_drl_exploration.reference import CoverageReference
    from lunar_drl_exploration.graph import GraphBuilder
    from lunar_drl_exploration.contracts import Pose, TaskSpec, SensorSpec
    from lunar_drl_exploration.config import load_platform_config
    terrain = TerrainGrid.from_heights(np.zeros((25,25), np.float32),
        resolution_m=1., origin=(0.,0.), platform=load_platform_config())
    truths=[]
    for edge, count in ((10,81),(20,361)):
        task = TaskSpec(str(edge),'map',np.array([[1,1],[edge,1],[edge,edge],[1,edge]]))
        ref = CoverageReference.build(terrain,Pose(5.5,5.5,0),task,SensorSpec(range_m=3))
        truth = GraphBuilder().build_truth(terrain,ref)
        assert truth.scene_id == ref.reference_id
        assert len(truth.reference_indices) == count
        assert truth.generator_descriptor['terrain_id'] == terrain.terrain_id
        truths.append(truth)
    assert truths[0].scene_id != truths[1].scene_id
    replay = ReplayBuffer(1000000)
    for truth in truths:
        t = transition(key=truth.scene_id)
        p = PrivilegedState(truth.scene_id,truth.packed_reference)
        replay.add(replace(t,privileged=p,next_privileged=p), {truth.scene_id:truth})
    assert set(replay.scenes) == {s.scene_id for s in truths}
    for t in replay.sample(64,np.random.default_rng(3)):
        s = replay.scenes[t.privileged.scene_id]
        assert np.unpackbits(t.privileged.observed, bitorder='little').sum() == len(s.reference_indices)
    replay.evict_oldest()
    assert set(replay.scenes) == {truths[1].scene_id}


def test_config_records_are_primitive_torch_free_and_semantic_controls_are_separate():
    import json
    import subprocess
    import sys
    from pathlib import Path
    from lunar_drl_exploration.config import TrainingConfig, config_record, resume_semantics
    from lunar_drl_exploration.sensor import OBSERVATION_MODEL_VERSION
    config = TrainingConfig()
    assert config.sensor.range_m == 10 and config.sensor.fov_deg == 90
    record = config_record(config)
    json.dumps(record, allow_nan=False)
    assert record['platform']['capability']['maximum_forward_speed_mps'] == .2
    assert resume_semantics(config)['observation_model'] == OBSERVATION_MODEL_VERSION
    changed = replace(config, output_dir='another', save_interval_s=10, target_rtf=1,
        learning=replace(config.learning, microbatch_size=32, learning_rate=3e-5,
                         polyak=.1, initial_alpha=2e-5))
    assert resume_semantics(changed) == resume_semantics(config)
    assert resume_semantics(replace(config,sensor=replace(config.sensor,range_m=15))) != resume_semantics(config)
    subprocess.run([sys.executable,'-c',
        f"import sys; sys.path.insert(0,{str(Path(__file__).resolve().parents[1])!r}); "
        "from lunar_drl_exploration.config import TrainingConfig,config_record; "
        "config_record(TrainingConfig()); import sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'lunar_drl_exploration.scene' not in sys.modules"],check=True)


def capture_state(learner, config, replay, schedule, boundary):
    from lunar_drl_exploration.checkpoint import TrainingState
    import torch
    return TrainingState.capture(learner=learner, config=config, replay=replay,
        schedule=schedule, boundary=boundary,
        replay_rng=np.random.default_rng(91), curriculum_rng=np.random.default_rng(17),
        curriculum={'stage':1,'episodes':17,'scene_seed_counter':34},
        collector_state={'policy_rng':torch.Generator().manual_seed(83).get_state(),
                         'actor_version':learner.updates,'next_episode_id':18},
        counters={'episodes_completed':17,'simulated_seconds':42.5})


def test_full_atomic_resume_restores_optimizer_rng_credit_replay_and_compatible_config(tmp_path):
    torch = pytest.importorskip('torch')
    import random
    from lunar_drl_exploration.config import TrainingConfig, ModelConfig
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.checkpoint import CheckpointManager, UpdateBoundary
    config = replace(TrainingConfig(), model=ModelConfig(width=16,heads=2,layers=1),warmup=0)
    learner = SACLearner(config.model,config.learning)
    replay=ReplayBuffer(1000000); replay.add(transition(),{'s':scene()})
    schedule=UpdateSchedule(warmup=0)
    for _ in range(7): schedule.collected()
    boundary=UpdateBoundary()
    with boundary.update():
        learner.update(replay.sample(64,np.random.default_rng(2)),replay.scenes)
        schedule.updated()
    random.seed(6); np.random.seed(7); torch.manual_seed(8)
    state=capture_state(learner,config,replay,schedule,boundary)
    manager=CheckpointManager(tmp_path, interval_s=1800, snapshot_max_bytes=1000000)
    manager.save(state)
    expected=(random.random(),np.random.random(),torch.rand(3))
    changed=replace(config,learning=replace(config.learning,microbatch_size=32,
        learning_rate=3e-5,polyak=.1,initial_alpha=2e-5),output_dir='new',save_interval_s=23,target_rtf=5)
    loaded=manager.load(changed)
    restored=SACLearner(changed.model,changed.learning)
    result=loaded.restore(restored,changed)
    assert result.schedule.transitions == 7 and result.schedule.updates == 1
    assert float(result.schedule.credit) == .75 and result.schedule.inflight == 0
    assert result.curriculum['scene_seed_counter'] == 34 and result.counters['episodes_completed'] == 17
    actual=(random.random(),np.random.random(),torch.rand(3))
    assert actual[:2] == expected[:2]; torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
    np.testing.assert_equal(result.replay_rng.integers(1000,size=10), np.random.default_rng(91).integers(1000,size=10))
    np.testing.assert_equal(result.curriculum_rng.integers(1000,size=10), np.random.default_rng(17).integers(1000,size=10))
    generator=torch.Generator(); generator.set_state(result.collector_state['policy_rng'])
    torch.testing.assert_close(torch.rand(5,generator=generator),torch.rand(5,generator=torch.Generator().manual_seed(83)))
    for name in ('actor','q1','q2','target1','target2'):
        for key,value in getattr(learner,name).state_dict().items():
            torch.testing.assert_close(value,getattr(restored,name).state_dict()[key],rtol=0,atol=0)
    for name in ('actor_optimizer','q1_optimizer','q2_optimizer','alpha_optimizer'):
        before=getattr(learner,name).state_dict(); after=getattr(restored,name).state_dict()
        for index, values in before['state'].items():
            for key,value in values.items(): torch.testing.assert_close(value,after['state'][index][key],rtol=0,atol=0)
        assert after['param_groups'][0]['lr'] == 3e-5
    assert restored.log_alpha.item() == learner.log_alpha.item()
    assert not result.replay.sample(1,np.random.default_rng())[0].observation.features.flags.writeable
    with pytest.raises(ValueError,match='sensor'):
        manager.load(replace(config,sensor=replace(config.sensor,range_m=15)))
    bad=torch.load(tmp_path/'resume.pt',weights_only=False); bad['semantics']['reward']='wrong'
    torch.save(bad,tmp_path/'resume.pt')
    with pytest.raises(ValueError,match='reward'): manager.load(config)
    bad['schema']='old_gru'; torch.save(bad,tmp_path/'resume.pt')
    with pytest.raises(ValueError,match='schema'): manager.load(config)


def test_atomic_interruption_budget_monotonic_and_failed_update_preserve_last_good(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from lunar_drl_exploration.config import TrainingConfig, ModelConfig
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.checkpoint import CheckpointManager, UpdateBoundary
    from lunar_drl_exploration.metrics import ResourceLimitError
    config=replace(TrainingConfig(),model=ModelConfig(width=16,heads=2,layers=1))
    learner=SACLearner(config.model,config.learning); replay=ReplayBuffer(1000000)
    replay.add(transition(),{'s':scene()}); schedule=UpdateSchedule(); schedule.collected(); boundary=UpdateBoundary()
    state=capture_state(learner,config,replay,schedule,boundary)
    now=[10.]
    manager=CheckpointManager(tmp_path,clock=lambda:now[0],snapshot_max_bytes=1000000)
    assert not manager.due
    now[0]=1809; assert not manager.due
    now[0]=1810; assert manager.due
    manager.save(state); good=(tmp_path/'resume.pt').read_bytes()
    assert not manager.due
    def interrupted(*args): raise OSError('interrupted rename')
    import lunar_drl_exploration.checkpoint as checkpoint
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint.os,'replace',interrupted)
        with pytest.raises(OSError): manager.save(state)
    assert (tmp_path/'resume.pt').read_bytes() == good
    assert sorted(p.name for p in tmp_path.iterdir()) == ['resume.pt']
    # Old checkpoint alone fits; simultaneous old+new must stop before exceeding.
    small=CheckpointManager(tmp_path,total_max_bytes=len(good)+100,snapshot_max_bytes=1000000)
    with pytest.raises(ResourceLimitError,match='artifact') as error: small.save(state)
    assert error.value.observed > error.value.limit
    assert (tmp_path/'resume.pt').read_bytes() == good
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= len(good)+100
    with pytest.raises(RuntimeError):
        with boundary.update():
            with torch.no_grad(): next(learner.actor.parameters()).add_(1)
            raise RuntimeError('optimizer failed halfway')
    with pytest.raises(ValueError,match='boundary'): capture_state(learner,config,replay,schedule,boundary)
    assert (tmp_path/'resume.pt').read_bytes() == good


def test_capture_rejects_unadmitted_replay_and_config_counter_mismatch():
    pytest.importorskip('torch')
    from lunar_drl_exploration.config import TrainingConfig,ModelConfig
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.checkpoint import UpdateBoundary
    config=replace(TrainingConfig(),model=ModelConfig(width=16,heads=2,layers=1))
    learner=SACLearner(config.model,config.learning)
    replay=ReplayBuffer(1000000); replay.add(transition(),{'s':scene()})
    with pytest.raises(ValueError,match='admission'):
        capture_state(learner,config,replay,UpdateSchedule(),UpdateBoundary())
    schedule=UpdateSchedule(warmup=0); schedule.collected()
    with pytest.raises(ValueError,match='schedule config'):
        capture_state(learner,config,replay,schedule,UpdateBoundary())


def test_run_manifest_contains_current_hardware_config_revision_and_no_new_admission_gate(tmp_path):
    pytest.importorskip('torch')
    import os,json
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.checkpoint import CheckpointManager
    config=TrainingConfig()
    manager=CheckpointManager(tmp_path)
    record=manager.write_run(config,owned_pids=[os.getpid()],code_revision='test-revision',
        extra={'completed_updates':0,'seed':config.seed})
    saved=json.loads((tmp_path/'run.json').read_text())
    assert saved==record and saved['code_revision']=='test-revision'
    assert saved['hardware']['cpu_count']>0 and saved['hardware']['memory_available_bytes']>0
    assert saved['config']['replay_max_bytes']==8*1024**3
    assert saved['config']['snapshot_max_bytes']==2*1024**3
    assert saved['config']['total_output_max_bytes']==20*1024**3
    assert saved['observations']['seed']==config.seed


def test_sigint_requests_save_but_cannot_capture_inside_optimizer_step(tmp_path):
    pytest.importorskip('torch')
    import signal
    from lunar_drl_exploration.config import TrainingConfig,ModelConfig
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.checkpoint import CheckpointManager,UpdateBoundary
    config=replace(TrainingConfig(),model=ModelConfig(width=16,heads=2,layers=1))
    learner=SACLearner(config.model,config.learning)
    replay=ReplayBuffer(1000000); schedule=UpdateSchedule(); boundary=UpdateBoundary()
    manager=CheckpointManager(tmp_path)
    previous=signal.signal(signal.SIGINT,manager.request_sigint)
    try:
        with boundary.update():
            signal.raise_signal(signal.SIGINT)
            assert manager.save_requested and manager.stop_requested
            assert not (tmp_path/'resume.pt').exists()
            with pytest.raises(ValueError,match='boundary'):
                capture_state(learner,config,replay,schedule,boundary)
        manager.save(capture_state(learner,config,replay,schedule,boundary))
        assert (tmp_path/'resume.pt').exists() and manager.stop_requested
        assert not manager.save_requested
    finally:
        signal.signal(signal.SIGINT,previous)


def test_actor_only_accepts_changed_sensor_context_without_full_resume_gate():
    torch=pytest.importorskip('torch')
    from lunar_drl_exploration.model import Actor
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.config import ModelConfig,LearningConfig,load_platform_config
    from lunar_drl_exploration.contracts import Pose
    cfg=ModelConfig(width=16,heads=2,layers=1)
    publication=SACLearner(cfg,LearningConfig()).actor_state()
    actor=Actor(cfg).eval(); actor.load_state_dict(publication['state_dict'])
    base=observation(); platform=load_platform_config()
    observations=[]
    for distance,fov in ((5,90),(15,90),(10,60),(10,120),(10,180)):
        context=platform.actor_context(Pose(0,0,0),linear_speed_mps=0,
            angular_speed_radps=0,sensor_range_m=distance,sensor_fov_rad=np.deg2rad(fov))
        observations.append(replace(base,context=context))
    with torch.inference_mode():
        outputs=actor(observations)
    assert len(outputs)==5 and all(torch.isfinite(out.probs).all() for out in outputs)


def test_config_roundtrip_json_spawn_record_reconstructs_mappingproxy_without_scene():
    import json
    from lunar_drl_exploration.config import TrainingConfig,config_record,training_config_from_record
    config=TrainingConfig()
    copied=training_config_from_record(json.loads(json.dumps(config_record(config))))
    assert config_record(copied)==config_record(config)
    assert copied.curriculum_extents_m==((40,80),(80,150),(100,300))
    with pytest.raises(TypeError): copied.platform.capability['maximum_forward_speed_mps']=9


def test_shared_array_backing_bytes_are_charged_once_even_for_distinct_views():
    a,b=observation(0),observation(1)
    object.__setattr__(b,'features',a.features.view())
    shared=ReplayBuffer(1000000); shared.add(transition(a,b),{'s':scene()})
    independent=ReplayBuffer(1000000); independent.add(transition(),{'s':scene()})
    assert independent.buffer_bytes-shared.buffer_bytes==152  # 2 nodes * 19 floats * 4 bytes
    restored=ReplayBuffer.from_snapshot(shared.snapshot(1000000),1000000)
    assert restored.buffer_bytes==shared.buffer_bytes


def test_heavily_aliased_distinct_transitions_charge_all_entry_metadata_and_evict():
    import sys
    # Independent lower bound: retained dict tables, entry containers and every
    # newly allocated membership-ID integer, excluding all payloads/ledger rows.
    def owned_metadata(replay):
        seen=set()
        def charge(value):
            if id(value) in seen: return 0
            seen.add(id(value)); return sys.getsizeof(value)
        result=sum(charge(value) for value in (replay._entries,replay._scenes,
                    replay._refs,replay._scene_keys,replay._ledger.items))
        for key,entry in replay._entries.items():
            result+=sum(charge(value) for value in (key,entry,entry[1],entry[2]))
            result+=sum(charge(value) for value in entry[2])
        for keys in replay._scene_keys.values():
            result+=charge(keys)+sum(charge(value) for value in keys)
        return result

    replay=ReplayBuffer(800000)
    original,truth=transition(),scene()
    for _ in range(1000): replay.add(replace(original),{'s':truth})
    assert replay.bytes_used >= owned_metadata(replay)
    assert replay.bytes_used <= replay.max_bytes and len(replay)<1000
    # Capacity left after partial deletion is still charged, not just len(dict).
    while len(replay)>2: replay.evict_oldest()
    assert replay.bytes_used >= owned_metadata(replay)
    assert replay.scene_refcounts == {'s':2}
    shared_bytes=replay.buffer_bytes
    replay.evict_oldest()
    assert replay.buffer_bytes==shared_bytes and replay.scene_refcounts=={'s':1}
    replay.evict_oldest()
    assert replay.bytes_used==0 and replay.buffer_bytes==0 and not replay.scenes


def test_saturated_credit_requires_saved_sample_and_failure_keeps_last_checkpoint(tmp_path):
    torch=pytest.importorskip('torch')
    from lunar_drl_exploration.config import TrainingConfig,ModelConfig
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.checkpoint import CheckpointManager,TrainingState,UpdateBoundary
    from lunar_drl_exploration.metrics import ResourceLimitError
    config=replace(TrainingConfig(),model=ModelConfig(width=16,heads=2,layers=1),warmup=0)
    learner=SACLearner(config.model,config.learning)
    replay=ReplayBuffer(1000000); replay.add(transition(),{'s':scene()})
    schedule=UpdateSchedule(warmup=0)
    for _ in range(128): schedule.collected()
    boundary=UpdateBoundary(); manager=CheckpointManager(tmp_path)
    valid=capture_state(learner,config,replay,schedule,boundary)
    manager.save(valid); original=(tmp_path/'resume.pt').read_bytes()
    tiny=replace(config,snapshot_max_bytes=512)
    with pytest.raises(ResourceLimitError,match='sample'):
        manager.save(capture_state(learner,tiny,replay,schedule,boundary))
    assert schedule.credit==32 and not schedule.can_dispatch()
    assert (tmp_path/'resume.pt').read_bytes()==original
    # A published or manually corrupted payload must not bypass the capture guard.
    bad=TrainingState(dict(valid.record,replay=ReplayBuffer(1000000).snapshot(512)))
    with pytest.raises(ValueError,match='sample'): manager.save(bad)
    assert (tmp_path/'resume.pt').read_bytes()==original
    target=SACLearner(config.model,config.learning)
    before=target.actor_state()
    with pytest.raises(ValueError,match='sample'): bad.restore(target,config)
    for key,value in before['state_dict'].items():
        torch.testing.assert_close(value,target.actor_state()['state_dict'][key],rtol=0,atol=0)
    assert target.updates==0 and not target.actor_optimizer.state
    # A forged nonzero cheap envelope still cannot conceal an actually empty
    # decoded payload during restore. Validation precedes every learner mutation.
    forged=bytearray(bad.record['replay'])
    forged[8:16]=(1).to_bytes(8,'little')
    inconsistent=TrainingState(dict(valid.record,replay=bytes(forged)))
    with pytest.raises(ValueError,match='count'): inconsistent.restore(target,config)
    for key,value in before['state_dict'].items():
        torch.testing.assert_close(value,target.actor_state()['state_dict'][key],rtol=0,atol=0)
    # One complete sample suffices: preserve every credit and draw effective64.
    restored=manager.load(config).restore(target,config)
    assert len(restored.replay)==1 and restored.schedule.credit==32
    batch=restored.replay.sample(64,restored.replay_rng)
    assert len(batch)==64 and all(item is batch[0] for item in batch)
    with UpdateBoundary().update():
        target.update(batch,restored.replay.scenes); restored.schedule.updated()
    assert restored.schedule.credit==31 and restored.schedule.can_dispatch()


def test_one_entry_budget_releases_old_table_capacity_and_preserves_newest():
    original,truth=transition(),scene()
    probe=ReplayBuffer(1000000); probe.add(original,{'s':truth})
    replay=ReplayBuffer(probe.bytes_used+1000)
    for index in range(1000):
        replay.add(replace(original,episode_id=str(index)),{'s':truth})
        assert replay.bytes_used<=replay.max_bytes and len(replay)==1
    assert replay.sample(1,np.random.default_rng(5))[0].episode_id=='999'
    replay.evict_oldest()
    assert replay.bytes_used==0 and replay.scene_refcounts=={}


def test_save_load_inspect_only_envelope_restore_materializes_contracts_once(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from lunar_drl_exploration.config import TrainingConfig,ModelConfig
    from lunar_drl_exploration.sac import SACLearner
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.checkpoint import CheckpointManager,UpdateBoundary
    config=replace(TrainingConfig(),model=ModelConfig(width=16,heads=2,layers=1))
    learner=SACLearner(config.model,config.learning)
    replay=ReplayBuffer(1000000); replay.add(transition(),{'s':scene()})
    schedule=UpdateSchedule(); schedule.collected()
    state=capture_state(learner,config,replay,schedule,UpdateBoundary())
    manager=CheckpointManager(tmp_path)
    # Trace real constructor allocations rather than substituting decoded output.
    constructed=[]; original=DecisionObservation.__post_init__
    def observe_materialization(self):
        constructed.append(self.revision)
        original(self)
    monkeypatch.setattr(DecisionObservation,'__post_init__',observe_materialization)
    manager.save(state); loaded=manager.load(config)
    assert constructed==[], 'save/load must not reconstruct the full replay just to count samples'
    restored=loaded.restore(SACLearner(config.model,config.learning),config)
    assert sorted(constructed)==[0,1] and len(restored.replay)==1
