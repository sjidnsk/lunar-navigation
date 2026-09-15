"""Complete learner-boundary continuation; one atomically replaced local resume."""
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import torch
from .config import config_record, resume_semantics
from .metrics import ResourceLimitError, artifact_bytes, live_hardware
from .replay import ReplayBuffer, dumps_transport, loads_transport
from .schedule import UpdateSchedule

SCHEMA='bounded_training_v1'


class UpdateBoundary:
    """Task8 must wrap update AND schedule.updated in this guard.

    A failed partial optimizer step permanently poisons this run's save boundary.
    Preserve the previous resume; reconstruct from it to continue training.
    """
    def __init__(self):
        self.safe=True; self.failed=False

    @contextmanager
    def update(self):
        if not self.safe or self.failed: raise ValueError('invalid optimizer boundary')
        self.safe=False
        try:
            yield
        except BaseException:
            self.failed=True
            raise
        else:
            self.safe=True

    def require_safe(self):
        if not self.safe or self.failed: raise ValueError('checkpoint requires completed optimizer boundary')


@dataclass
class RestoredTraining:
    replay: ReplayBuffer
    schedule: UpdateSchedule
    replay_rng: np.random.Generator
    curriculum_rng: np.random.Generator
    curriculum: dict
    collector_state: dict
    counters: dict


def _generator(state):
    name=state['bit_generator']
    cls=getattr(np.random,name,None)
    if cls is None: raise ValueError(f'unknown numpy RNG: {name}')
    result=np.random.Generator(cls()); result.bit_generator.state=state
    return result


@dataclass
class TrainingState:
    """Owned immutable-time snapshot, captured after collector acknowledgement barrier."""
    record: dict

    @classmethod
    def capture(cls, *, learner, config, replay, schedule, boundary, replay_rng,
                curriculum_rng, curriculum, collector_state, counters):
        boundary.require_safe()
        if len(replay) > schedule.transitions:
            raise ValueError('replay admission count exceeds learner-owned counter')
        if (schedule.warmup != config.warmup or
                schedule.ratio != type(schedule.ratio)(str(config.update_ratio))):
            raise ValueError('schedule config differs from captured config')
        if learner.updates != schedule.updates:
            raise ValueError('learner/schedule counters disagree at checkpoint boundary')
        if collector_state['actor_version'] > learner.updates:
            raise ValueError('collector Actor version exceeds completed learner updates')
        # Collector must supply its CPU Actor sampling RNG and seed/curriculum
        # counters. Environment-local unfinished motion is intentionally absent.
        if not isinstance(collector_state.get('policy_rng'),torch.Tensor):
            raise ValueError('collector policy_rng tensor is required')
        rng=dict(python=random.getstate(),numpy=dumps_transport(np.random.get_state()),
            torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=[state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [],
            replay=dumps_transport(replay_rng.bit_generator.state),
            curriculum=dumps_transport(curriculum_rng.bit_generator.state))
        return cls(dict(schema=SCHEMA,semantics=resume_semantics(config),config=config_record(config),
            learner=learner.state_dict(),schedule=schedule.state_dict(),rng=rng,
            replay=replay.snapshot(config.snapshot_max_bytes),
            curriculum=dumps_transport(curriculum),collector_state=copy.deepcopy(collector_state),
            counters=dumps_transport(counters)))

    def validate(self, config=None):
        record=self.record
        if record.get('schema') != SCHEMA: raise ValueError('training checkpoint schema mismatch')
        for key in ('semantics','config','learner','schedule','rng','replay',
                    'curriculum','collector_state','counters'):
            if key not in record: raise ValueError(f'checkpoint missing {key}')
        if config is not None:
            expected=resume_semantics(config)
            different=sorted(key for key in set(expected)|set(record['semantics'])
                if expected.get(key)!=record['semantics'].get(key))
            if different: raise ValueError('incompatible full resume semantics: '+', '.join(different))
        schedule=UpdateSchedule.from_state_dict(record['schedule'])
        schedule_config=record['semantics']['schedule']
        if (schedule.warmup != schedule_config['warmup'] or
                schedule.ratio != type(schedule.ratio)(str(schedule_config['ratio']))):
            raise ValueError('checkpoint schedule config disagrees with semantics')
        if record['learner']['model_config'] != record['semantics']['model']:
            raise ValueError('checkpoint model config disagrees with semantics')
        for key, value in record['semantics']['learning'].items():
            if record['learner']['learning_config'].get(key) != value:
                raise ValueError('checkpoint learning config disagrees with semantics')
        if record['learner'].get('updates') != schedule.updates:
            raise ValueError('checkpoint learner/schedule counters disagree')
        if record['collector_state']['actor_version']>schedule.updates:
            raise ValueError('checkpoint collector version exceeds learner')

    def restore(self, learner, config):
        self.validate(config)
        record=self.record
        # Parse owned state first; no environment Scene/TerrainGrid regeneration.
        replay=ReplayBuffer.from_snapshot(record['replay'],config.replay_max_bytes)
        schedule=UpdateSchedule.from_state_dict(record['schedule'])
        # A larger operational credit cap is safe; a smaller one must still cover
        # all saved earned credit (no truncation and no silent credit loss).
        if schedule.credit>config.max_update_credit:
            raise ValueError('requested credit cap is below saved pending credit')
        schedule.max_credit=type(schedule.max_credit)(config.max_update_credit)
        rng=record['rng']
        replay_rng=_generator(loads_transport(rng['replay']))
        curriculum_rng=_generator(loads_transport(rng['curriculum']))
        if rng['torch_cuda'] and learner.device.type=='cuda' and len(rng['torch_cuda'])!=torch.cuda.device_count():
            raise ValueError('CUDA RNG device count mismatch')
        learner.load_state_dict(record['learner'])
        random.setstate(rng['python']); np.random.set_state(loads_transport(rng['numpy']))
        torch.set_rng_state(rng['torch_cpu'])
        if rng['torch_cuda'] and learner.device.type=='cuda':
            torch.cuda.set_rng_state_all(rng['torch_cuda'])
        return RestoredTraining(replay,schedule,replay_rng,curriculum_rng,
            loads_transport(record['curriculum']),copy.deepcopy(record['collector_state']),
            loads_transport(record['counters']))


class _BudgetWriter:
    """Enforce simultaneous old+temporary bytes before each physical write."""
    def __init__(self, stream, existing, limit, free):
        self.stream=stream; self.existing=existing; self.limit=limit; self.free=free
        self.extent=0; self.failure=None

    def write(self, data):
        if self.failure is not None: raise self.failure
        end=self.stream.tell()+len(data)
        extent=max(self.extent,end)
        if self.existing+extent>self.limit:
            self.failure=ResourceLimitError('artifact peak bytes',self.existing+extent,self.limit)
            raise self.failure
        if extent>self.free:
            self.failure=ResourceLimitError('disk temporary bytes',extent,self.free)
            raise self.failure
        result=self.stream.write(data); self.extent=extent
        return result

    def __getattr__(self, name):
        return getattr(self.stream,name)


class CheckpointManager:
    def __init__(self, output_dir, interval_s=1800, *, total_max_bytes=20*1024**3,
                 snapshot_max_bytes=2*1024**3, clock=time.monotonic):
        if interval_s<=0 or total_max_bytes<=0 or snapshot_max_bytes<=0:
            raise ValueError('positive checkpoint budgets required')
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True)
        self.interval_s=interval_s; self.total_max_bytes=total_max_bytes
        self.snapshot_max_bytes=snapshot_max_bytes; self.clock=clock
        self.last_save=clock()
        self.requested_reason=None; self.stop_requested=False

    @property
    def due(self):
        return self.clock()-self.last_save>=self.interval_s

    @property
    def save_requested(self):
        return self.requested_reason is not None or self.due

    def request_save(self, reason, *, stop=False):
        self.requested_reason=str(reason)[:2048]
        self.stop_requested = self.stop_requested or stop

    def request_sigint(self, signum, frame):
        """Install from Task8 main thread; handler sets flags and never saves/raises."""
        self.request_save('SIGINT', stop=True)

    def _atomic(self, name, write):
        existing=artifact_bytes(self.output_dir)
        free=shutil.disk_usage(self.output_dir).free
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w+b',dir=self.output_dir,
                    prefix='.'+name+'.',suffix='.tmp',delete=False) as stream:
                temporary=Path(stream.name)
                writer=_BudgetWriter(stream,existing,self.total_max_bytes,free)
                try:
                    write(writer)
                except BaseException:
                    # Torch ZIP finalization can mask a failing write. Preserve
                    # the first observed budget error instead of its ZIP follow-up.
                    if writer.failure is not None: raise writer.failure
                    raise
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary,self.output_dir/name)
            descriptor=os.open(self.output_dir,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()

    def save(self, state):
        state.validate()
        if len(state.record['replay'])>self.snapshot_max_bytes:
            raise ResourceLimitError('replay snapshot bytes',len(state.record['replay']),self.snapshot_max_bytes)
        self._atomic('resume.pt',lambda stream:torch.save(state.record,stream))
        self.last_save=self.clock()
        self.requested_reason=None
        return self.output_dir/'resume.pt'

    def load(self, config=None):
        # Local artifacts are trusted; Torch itself is explicit about pickle load.
        state=TrainingState(torch.load(self.output_dir/'resume.pt',map_location='cpu',weights_only=False))
        state.validate(config)
        return state

    def write_run(self, config, *, owned_pids, code_revision, extra=None):
        record=dict(schema=SCHEMA,code_revision=code_revision,config=config_record(config),semantics=resume_semantics(config),
            hardware=live_hardware(self.output_dir),owned_pids=sorted(set(owned_pids)),
            observations=extra or {})
        payload=(json.dumps(record,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')
        self._atomic('run.json',lambda stream:stream.write(payload))
        return record
