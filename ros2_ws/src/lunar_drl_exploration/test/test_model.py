"""Sparse network contract tests: topology, packing and privileged isolation."""
from dataclasses import replace
import inspect
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from lunar_drl_exploration.contracts import DecisionObservation, PrivilegedScene, PrivilegedState
from lunar_drl_exploration.config import ModelConfig
from lunar_drl_exploration.model import Actor, Critic, PolygonEncoder
from lunar_drl_exploration.batch import pack_privileged

torch.set_num_threads(2)


def observation(n=7, locations=3, seed=1):
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(n, 2)).astype(np.float32)
    features = rng.normal(size=(n, 19)).astype(np.float32)
    features[:, :2] = (positions - positions[0]) / 10
    edges = np.column_stack((np.arange(n-1), np.arange(1, n)))
    nodes = np.repeat(np.arange(min(n, locations)), 8)
    yaws = np.tile(np.arange(8)*np.pi/4, min(n, locations))
    return DecisionObservation(np.arange(n), positions, features, edges,
        np.ones(n-1), 0, np.array([[0,0],[1,0],[1,1],[.4,.3],[0,1]]),
        np.arange(8)/8, nodes, yaws, np.column_stack((positions[nodes], yaws)), 'e', seed)


def scene(n=9):
    return PrivilegedScene('s', np.column_stack((np.arange(n), np.zeros(n))),
        np.column_stack((np.arange(n-1), np.arange(1,n))), np.ones(n-1),
        np.arange(n+1)*2, np.arange(n*2),
        np.packbits(np.ones(n*2, np.uint8), bitorder='little'), (2,n), {})


def state(n=9, full=False):
    return PrivilegedState('s', np.packbits(np.full(n*2, full, np.uint8), bitorder='little'))


def test_actor_packing_permutation_and_joint_argmax():
    torch.manual_seed(4)
    actor = Actor(ModelConfig()).eval()
    observations = [observation(1,1), observation(), observation(21,20)]
    with torch.no_grad():
        packed = actor(observations)
        for obs, out in zip(observations, packed):
            single = actor([obs])[0]
            torch.testing.assert_close(out.logits, single.logits, atol=2e-6, rtol=2e-5)
            assert out.probs.shape == (len(obs.action_nodes),)
            torch.testing.assert_close(out.probs.sum(), torch.tensor(1.))
            assert out.probs.argmax() == out.logits.argmax()
        obs = observations[1]
        order = np.array([3,1,5,0,6,2,4]); inverse = np.argsort(order)
        action_order = np.arange(len(obs.action_nodes))[::-1].copy()
        shuffled = replace(obs, node_ids=obs.node_ids[order], positions=obs.positions[order],
            features=obs.features[order], edges=inverse[obs.edges[::-1]],
            edge_lengths=obs.edge_lengths[::-1], current_index=int(inverse[0]),
            action_nodes=inverse[obs.action_nodes[action_order]],
            action_yaws=obs.action_yaws[action_order], goals=obs.goals[action_order])
        torch.testing.assert_close(actor([shuffled])[0].logits,
            packed[1].logits[torch.tensor(action_order)], atol=2e-6, rtol=2e-5)


def test_polygon_retains_boundary_adjacency_and_cyclic_start_invariance():
    torch.manual_seed(12)
    encoder = PolygonEncoder(128)
    polygon = torch.tensor([[0.,0.],[2.,0.],[1.,.4],[2.,2.],[0.,2.]])
    def encode(p):
        return encoder(torch.cat((p, p.roll(-1,0)),1), torch.zeros(5,dtype=torch.long), 1)
    torch.testing.assert_close(encode(polygon), encode(polygon.roll(2,0)))
    assert not torch.allclose(encode(polygon), encode(polygon[[0,1,3,2,4]]))


def test_critic_uses_current_coverage_and_measured_observation_actor_is_isolated():
    torch.manual_seed(9)
    obs, truth = observation(), scene()
    actor, critic = Actor(ModelConfig()), Critic(ModelConfig())
    assert list(inspect.signature(actor.forward).parameters) == ['observations']
    with torch.no_grad():
        original = actor([obs])[0].logits.clone()
        q0 = critic([obs], [state()], {'s':truth})[0]
        q1 = critic([obs], [state(full=True)], {'s':truth})[0]
        q2 = critic([replace(obs, features=obs.features+0.5)], [state()], {'s':truth})[0]
        assert not torch.allclose(q0,q1)
        assert not torch.allclose(q0,q2)
        torch.testing.assert_close(actor([obs])[0].logits, original, atol=0,rtol=0)
    batch = pack_privileged([obs], [state(full=True)], {'s':truth}, 'cpu')
    np.testing.assert_allclose(batch.features[:, 3].numpy(), 1.)


def test_sparse_attention_never_allocates_node_square():
    from torch.utils._python_dispatch import TorchDispatchMode
    class NoSquare(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for value in torch.utils._pytree.tree_leaves(out):
                if isinstance(value, torch.Tensor):
                    assert sum(d == 257 for d in value.shape) < 2, (func, value.shape)
            return out
    with torch.no_grad(), NoSquare():
        out = Actor(ModelConfig())([observation(257,20)])[0]
    assert len(out.probs) == 160


def test_critic_packs_different_scene_topologies_and_reference_owner_counts():
    torch.manual_seed(2)
    critic = Critic(ModelConfig()).eval()
    observations = [observation(4,2),observation(13,7)]
    truth1 = replace(scene(3), reference_offsets=np.array([0,0,2,6]),
        reference_indices=np.array([0,1,2,3,4,5]))
    truth2 = replace(scene(11),scene_id='other')
    states = [PrivilegedState('s',np.array([0b00101101],np.uint8)),
              replace(state(11,True),scene_id='other')]
    scenes={'s':truth1,'other':truth2}
    packed=pack_privileged(observations,states,scenes,'cpu')
    torch.testing.assert_close(packed.features[:3,3],torch.tensor([0.,.5,.75]))
    with torch.no_grad():
        results=critic(observations,states,scenes)
        for o,s,q in zip(observations,states,results):
            torch.testing.assert_close(q,critic([o],[s],scenes)[0],atol=2e-6,rtol=2e-5)
        order=np.array([2,0,1]); inverse=np.argsort(order)
        owner_chunks=[truth1.reference_indices[truth1.reference_offsets[i]:truth1.reference_offsets[i+1]] for i in order]
        shuffled=replace(truth1,positions=truth1.positions[order],edges=inverse[truth1.edges],
            reference_offsets=np.r_[0,np.cumsum([len(a) for a in owner_chunks])],
            reference_indices=np.concatenate(owner_chunks))
        torch.testing.assert_close(critic([observations[0]],[states[0]],{'s':shuffled})[0],results[0],atol=2e-6,rtol=2e-5)


def test_training_recompute_reduces_saved_activations_without_changing_gradients():
    import copy
    torch.manual_seed(32)
    actor=Actor(ModelConfig()).train()
    direct=copy.deepcopy(actor).eval()
    def run(network):
        sizes=[]
        def save(tensor):
            sizes.append(tensor.numel())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(save,lambda x:x):
            logits=network([observation(129,20)])[0].logits
            logits.square().sum().backward()
        return logits,sum(sizes)
    recomputed,recomputed_size=run(actor)
    normal,normal_size=run(direct)
    torch.testing.assert_close(recomputed,normal)
    for p,q in zip(actor.parameters(),direct.parameters()):
        torch.testing.assert_close(p.grad,q.grad)
    assert recomputed_size < normal_size / 2
