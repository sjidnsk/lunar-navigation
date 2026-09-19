"""Actor score parameterization, diagnostics and continuation contracts."""
from dataclasses import replace
import copy
import math
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from lunar_drl_exploration import model
from lunar_drl_exploration.batch import pack_observations
from lunar_drl_exploration.config import ModelConfig, LearningConfig, TrainingConfig
from lunar_drl_exploration.model import Actor, Critic
from lunar_drl_exploration.sac import SACLearner
from test_model import observation, scene
from test_sac import transitions


@pytest.mark.parametrize('bound', [-1., float('inf'), float('-inf'), float('nan')])
def test_invalid_bound_rejected(bound):
    with pytest.raises(ValueError, match='bound'):
        ModelConfig(actor_score_bound=bound)


def test_score_transform_independent_shift_invariant_equivariant_and_gradchecked():
    scores = torch.tensor([1., 5., 9., -20., 40.], dtype=torch.double, requires_grad=True)
    groups = torch.tensor([0, 0, 0, 2, 2])  # Also exercises an empty state.
    transform = model.bound_actor_scores
    actual = transform(scores, groups, 3, 10.)
    expected = torch.tensor([-.3799489622552249, 0., .3799489622552249,
                             -.9950547536867305, .9950547536867305], dtype=torch.double) * 10
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(transform(scores[:3], groups[:3], 1, 10.), actual[:3])
    torch.testing.assert_close(transform(scores + torch.tensor([100., 100., 100., -80., -80.]),
                                        groups, 3, 10.), actual)
    order = torch.tensor([4, 1, 3, 0, 2])
    torch.testing.assert_close(transform(scores[order], groups[order], 3, 10.), actual[order])
    assert torch.autograd.gradcheck(lambda z: transform(z, groups, 3, 10.), (scores,))
    gradient = torch.autograd.grad(actual[0], scores)[0]
    assert gradient[1] != 0  # The state mean must remain attached to autograd.
    torch.testing.assert_close(gradient[:3].sum(), torch.tensor(0., dtype=torch.double), atol=1e-15, rtol=0)
    assert transform(scores, groups, 3, 0.) is scores
    assert transform(scores[:0], groups[:0], 1, 10.).numel() == 0
    torch.testing.assert_close(transform(scores[:1], groups[:1], 1, 10.), torch.zeros(1, dtype=torch.double))


def test_actor_packed_and_per_observation_paths_bound_only_actor_with_same_weights():
    config = ModelConfig(width=16, heads=2, layers=1)
    original = Actor(config).eval()
    bounded = Actor(replace(config, actor_score_bound=10.)).eval()
    bounded.load_state_dict(original.state_dict(), strict=True)
    # Expose a nontrivial bounded transform without replacing any production path.
    with torch.no_grad():
        original.head[-1].weight.mul_(300)
        bounded.load_state_dict(original.state_dict())
    obs = observation()
    single = replace(obs, action_nodes=obs.action_nodes[:1], action_yaws=obs.action_yaws[:1], goals=obs.goals[:1])
    empty = replace(obs, action_nodes=np.empty(0, int), action_yaws=np.empty(0), goals=np.empty((0, 3)))
    observations = [obs, single, empty, observation(11, 5)]
    batch = pack_observations(observations, 'cpu')
    with torch.no_grad():
        raw = original.forward_packed(batch)
        actual = bounded.forward_packed(batch)
        torch.testing.assert_close(raw.logits, raw.raw_logits, rtol=0, atol=0)
        torch.testing.assert_close(actual.raw_logits, raw.logits)
        assert actual.logits.abs().max() <= 10.
        for source, result, packed in zip(original(observations), bounded(observations),
                                           actual.probs.split(batch.action_counts)):
            if not len(source.logits):
                assert result.probs.numel() == 0
                continue
            expected = (10 * torch.tanh((source.logits - source.logits.mean()) / 10)).softmax(0)
            torch.testing.assert_close(result.probs, expected)
            torch.testing.assert_close(result.probs, packed)
            assert result.probs.argmax() == source.probs.argmax()
    critic = Critic(config)
    other = Critic(replace(config, actor_score_bound=10.))
    other.load_state_dict(critic.state_dict(), strict=True)
    from test_model import state
    torch.testing.assert_close(critic([obs], [state(obs=obs)], {'s': scene()})[0],
                               other([obs], [state(obs=obs)], {'s': scene()})[0])


def test_bounded_distribution_used_by_targets_export_and_runtime(tmp_path):
    from lunar_drl_exploration.runtime import ActorPolicy
    from lunar_drl_exploration.evaluation import export_actor
    learner = SACLearner(ModelConfig(width=16, heads=2, layers=1, actor_score_bound=10.))
    with torch.no_grad():
        learner.actor.head[-1].weight.mul_(300)
    samples = transitions()[1:3]
    actual = learner.td_targets(samples, {'s': scene()})
    observations = [t.next_observation for t in samples]
    states = [t.next_privileged for t in samples]
    with torch.no_grad():
        policies = learner.actor(observations)
        qa = learner.target1(observations, states, {'s': scene()})
        qb = learner.target2(observations, states, {'s': scene()})
        expected = torch.stack([t.reward + learner.learning_config.gamma *
            (p.probs * (torch.minimum(a, b) - learner.alpha * p.log_probs)).sum()
            for t, p, a, b in zip(samples, policies, qa, qb)])
        torch.testing.assert_close(actual, expected)
    source, output = tmp_path / 'source.pt', tmp_path / 'actor.pt'
    torch.save(learner.actor_state(), source)
    export_actor(source, output)
    loaded = ActorPolicy.load(output)
    assert loaded.actor.config.actor_score_bound == 10.
    for obs, policy in zip(observations, policies):
        torch.testing.assert_close(loaded.actor([obs])[0].probs, policy.probs)
        assert loaded(obs) == int(policy.probs.argmax())


def test_legacy_c0_learner_and_full_resume_normalize_only_missing_bound(tmp_path):
    from lunar_drl_exploration.checkpoint import UpdateBoundary
    from lunar_drl_exploration.schedule import UpdateSchedule
    from lunar_drl_exploration.replay import ReplayBuffer
    from lunar_drl_exploration.evaluation import export_actor
    from test_replay_checkpoint import capture_state
    config = replace(TrainingConfig(), model=ModelConfig(width=16, heads=2, layers=1))
    learner = SACLearner(config.model)
    state = capture_state(learner, config, ReplayBuffer(1000000),
                          UpdateSchedule(warmup=config.warmup), UpdateBoundary())
    for record in (state.record['learner']['model_config'], state.record['semantics']['model'],
                   state.record['config']['model']):
        record.pop('actor_score_bound')
    state.validate(config)
    state.restore(SACLearner(config.model), config)
    bounded = replace(config, model=replace(config.model, actor_score_bound=10.))
    with pytest.raises(ValueError, match='model'):
        state.validate(bounded)
    with pytest.raises(ValueError, match='model'):
        SACLearner(bounded.model).load_state_dict(state.record['learner'])
    malformed = copy.deepcopy(state.record['learner'])
    malformed['model_config'].pop('layers')
    with pytest.raises(ValueError, match='model'):
        learner.load_state_dict(malformed)
    source, output = tmp_path / 'legacy.pt', tmp_path / 'actor.pt'
    torch.save(state.record, source)
    export_actor(source, output)
    assert torch.load(output, weights_only=True)['model_config']['actor_score_bound'] == 0.


def test_sac_metrics_decompose_joint_entropy_and_keep_microbatch_gradients():
    config = ModelConfig(width=16, heads=2, layers=1, actor_score_bound=10.)
    a = SACLearner(config, LearningConfig(microbatch_size=64))
    b = SACLearner(config, LearningConfig(microbatch_size=16))
    with torch.no_grad():
        a.actor.head[-1].weight.zero_()
        a.actor.head[-1].bias.fill_(1000.)  # Shift alone must not saturate.
    b.load_state_dict(a.state_dict())
    samples = transitions()
    ma, mb = a.update(samples, {'s': scene()}), b.update(samples, {'s': scene()})
    position_entropy = np.mean([math.log(1 + i % 3) for i in range(64)])
    assert ma['position_entropy'] == pytest.approx(position_entropy, abs=1e-6)
    assert ma['heading_conditional_entropy'] == pytest.approx(math.log(8), abs=1e-6)
    assert ma['position_max_probability'] == pytest.approx(np.mean([1 / (1 + i % 3) for i in range(64)]))
    assert ma['raw_score_span'] == 0.
    assert ma['score_saturation_fraction'] == 0.
    assert ma['raw_score_gradient_abs_mean'] > 0
    assert ma['raw_score_gradient_abs_max'] >= ma['raw_score_gradient_abs_mean']
    for key in ma:
        assert math.isfinite(ma[key])
        assert ma[key] == pytest.approx(mb[key], rel=2e-4, abs=2e-6)
    for p, q in zip(a.actor.parameters(), b.actor.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)


def test_extreme_scores_are_finite_and_saturation_diagnostics_use_centered_logits():
    from lunar_drl_exploration.sac import policy_diagnostics
    obs = observation(2, 2)
    obs = replace(obs, action_nodes=obs.action_nodes[[0, 8]],
                  action_yaws=obs.action_yaws[[0, 8]], goals=obs.goals[[0, 8]])
    batch = pack_observations([obs], 'cpu')
    raw = torch.tensor([-1e20, 1e20], requires_grad=True)
    logits = model.bound_actor_scores(raw, batch.action_groups, 1, 10.)
    logp = logits.log_softmax(0)
    policy = model.PolicyOutput(logits, logp.exp(), logp, raw)
    (-logp[0]).backward()
    assert torch.isfinite(logits).all() and torch.isfinite(logp).all() and torch.isfinite(raw.grad).all()
    metrics = policy_diagnostics(policy, batch, 10.)
    assert metrics[4] == 1.  # Both centered scores saturate at opposite bounds.
    assert metrics[3] == raw.detach().diff()[0]
    assert metrics[5] == 0. and metrics[6] == 0.
    assert metrics[1] == 0.  # One heading per position.
    assert policy_diagnostics(policy, batch, 0.)[4] == 0.


def test_bounded_nonuniform_actor_loss_matches_direct_frozen_distribution():
    config = ModelConfig(width=16, heads=2, layers=1, actor_score_bound=10.)
    learner = SACLearner(config, LearningConfig(microbatch_size=64))
    with torch.no_grad():
        learner.actor.head[-1].weight.mul_(300)
    before = copy.deepcopy(learner.actor).eval()
    samples, scenes = transitions(), {'s': scene()}
    metrics = learner.update(samples, scenes)
    observations = [t.observation for t in samples]
    states = [t.privileged for t in samples]
    with torch.no_grad():
        policies = before(observations)
        q1 = learner.q1(observations, states, scenes)
        q2 = learner.q2(observations, states, scenes)
        expected = torch.stack([(p.probs * (learner.learning_config.initial_alpha * p.log_probs -
            torch.minimum(a, b))).sum() for p, a, b in zip(policies, q1, q2)]).mean()
    assert metrics['actor_loss'] == pytest.approx(float(expected), abs=2e-6, rel=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for synchronization audit')
def test_bounded_cuda_update_has_no_scalar_device_synchronization():
    from torch.utils._python_dispatch import TorchDispatchMode
    class NoScalarDeviceSync(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                assert not args[0].is_cuda, 'per-action CUDA scalar read'
            return func(*args, **(kwargs or {}))
    learner = SACLearner(ModelConfig(width=16, heads=2, layers=1, actor_score_bound=10.), device='cuda')
    with NoScalarDeviceSync():
        metrics = learner.update(transitions(), {'s': scene()})
    assert all(math.isfinite(value) for value in metrics.values())
