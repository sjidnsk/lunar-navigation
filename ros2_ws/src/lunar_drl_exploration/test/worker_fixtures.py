"""Controllable external environment double; real worker IPC/lifecycle stays intact."""
import threading
import uuid
from types import SimpleNamespace
import numpy as np
from lunar_drl_exploration.contracts import (
    DecisionObservation, PrivilegedScene, PrivilegedState, PrivilegedActionContext, Transition, RewardParts)


def observation():
    features = np.zeros((2, 19)); features[1, 0] = .1; features[1, 4] = .5
    return DecisionObservation(np.arange(2), [[0, 0], [1, 0]], features,
        [[0, 1]], [1], 0, [[0, 0], [2, 0], [2, 2]], np.ones(8),
        [0, 1], [0, 0], [[0, 0, 0], [1, 0, 0]], 'measured', 1)


class ControlledEnv:
    def __init__(self, config, env_id, **kwargs):
        self.env_id = env_id
        self.stop = threading.Event()
        self.steps = 0
        self.owned_pids = ()
        self.last_execution = None
        self.active = False

    def reset(self, seed, family, extent, *, episode_budget=None):
        self.stop.clear()
        self.steps = 0
        self.seed = seed
        self.budget = episode_budget
        self.episode_id = uuid.uuid4().hex
        self.obs = observation()
        if extent == 1:
            from dataclasses import replace
            self.obs = replace(self.obs, action_nodes=[], action_yaws=[], goals=np.empty((0, 3)))
        self.observation = self.obs
        self.reference = SimpleNamespace(coverage_ratio=lambda bits: 0.)
        positions, inverse = np.unique(self.obs.goals[:, :2], axis=0, return_inverse=True)
        context = PrivilegedActionContext(positions, inverse, self.obs.goals[:,2], np.arange(len(positions)+1),
            np.arange(len(positions)) % 2, np.ones(len(positions)), np.ones(len(self.obs.goals)))
        self.state = PrivilegedState(self.episode_id, [0], context)
        scene = PrivilegedScene(self.episode_id, [[0, 0], [1, 0]], [[0, 1]], [1],
            [0, 1, 2], [0, 1], [3], (1, 2), {})
        self.static_scenes = {scene.scene_id: scene}
        self.episode_metadata = dict(episode_id=self.episode_id, scene_id=scene.scene_id,
            reference_area_m2=2., initial_known_area_m2=0., family=family,
            extent_m=extent, budget=episode_budget, seed=seed)
        self.report = SimpleNamespace(exhausted=extent == 1)
        return self.obs, self.state

    def step(self, action_index, *, actor_version=None):
        self.active = True
        try:
            # Env0 is deliberately slower; seed=-1 blocks until cancellation.
            if self.stop.wait(60 if self.seed == -1 else .12 if self.env_id == 0 else .005):
                class TaskCanceled(RuntimeError): pass
                raise TaskCanceled('controlled cancellation')
            if self.seed == -2:
                raise RuntimeError('controlled infrastructure failure')
            self.steps += 1
            return Transition(self.obs, action_index, -.001, self.obs,
                self.state, self.state, RewardParts(0., 0., 0.), False,
                self.steps >= self.budget, self.episode_id, actor_version)
        finally:
            self.active = False

    def progress(self):
        return dict(steps=self.steps, owned_pids=[], known_area_m2=0., distance_m=0.,
            turn_rad=0., simulation_s=self.steps, wall_s=1., sensor_frames=1,
            max_command_speed_mps=0.)

    def cancel(self):
        self.stop.set()

    def close(self):
        if self.active:
            raise AssertionError('close before joining active step')


class OwnedChildEnv(ControlledEnv):
    def reset(self, *args, **kwargs):
        import sys
        from lunar_drl_exploration.processes import OwnedProcesses
        result = super().reset(*args, **kwargs)
        self.children = OwnedProcesses()
        self.children.spawn([sys.executable, '-c', 'import time; time.sleep(30)'])
        import time
        time.sleep(.2)  # child has installed PDEATHSIG before reset returns
        return result

    def step(self, *args, **kwargs):
        self.children.check()
        return super().step(*args, **kwargs)

    def close(self):
        super().close()
        if hasattr(self, 'children'): self.children.close()


def large_transport_collector(child, record, state, large_kind, entered, sent):
    from lunar_drl_exploration.collector import collector_main
    class LargePipe:
        def send(self, message):
            if message['kind'] == large_kind:
                message = dict(message, transport_probe_padding=b'x' * 1024**2)
                entered.set()
                child.send(message)
                sent.set()
            else: child.send(message)
        def __getattr__(self, name): return getattr(child, name)
    collector_main(LargePipe(), record, state, env_factory='worker_fixtures:ControlledEnv')


def instrumented_batch_collector(child, record, state, sizes, count):
    from lunar_drl_exploration.worker import numerical_threads
    numerical_threads(record['collector_threads'])
    from lunar_drl_exploration.collector import collector_main
    from lunar_drl_exploration.model import Actor
    original = Actor.forward
    def forward(self, observations):
        sizes[count.value] = len(observations); count.value += 1
        return original(self, observations)
    Actor.forward = forward
    collector_main(child, record, state, env_factory='worker_fixtures:ControlledEnv')
