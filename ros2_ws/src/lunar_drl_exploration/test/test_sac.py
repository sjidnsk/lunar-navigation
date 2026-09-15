"""Numerical SAC invariants use full action expectations and independent weights."""
from dataclasses import replace
import copy
import math
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from lunar_drl_exploration.config import ModelConfig, LearningConfig
from lunar_drl_exploration.contracts import Transition, RewardParts
from lunar_drl_exploration.sac import SACLearner, soft_values, policy_loss, temperature_loss
from test_model import observation, scene, state


def transitions():
    result=[]
    for i in range(64):
        obs = observation(3+i%4, 1+i%3, seed=i+1)
        next_obs = observation(3+(i+1)%4, 1+(i+1)%3, seed=i+2)
        terminal = i%5==0
        if terminal:
            next_obs=replace(next_obs, action_nodes=np.empty(0,int),
                action_yaws=np.empty(0),goals=np.empty((0,3)))
        result.append(Transition(obs, i%len(obs.goals), (i-30)/100., next_obs,
            state(),state(full=i%2==0),RewardParts(0,0,0),terminal,i%7==0 and not terminal,'e',0))
    return result


def test_enumerated_math_and_temperature_direction():
    logp=torch.tensor([.25,.75,.1,.2,.7]).log().requires_grad_()
    probs=logp.exp()
    q1=torch.tensor([2.,4.,1.,2.,3.],requires_grad=True)
    q2=torch.tensor([3.,1.,2.,1.,4.],requires_grad=True)
    groups=torch.tensor([0,0,1,1,1])
    expected=torch.tensor([.25*2+.75*1,.1*1+.2*1+.7*3])
    entropy=torch.tensor([-(.25*math.log(.25)+.75*math.log(.75)),
        -(.1*math.log(.1)+.2*math.log(.2)+.7*math.log(.7))])
    torch.testing.assert_close(soft_values(probs,logp,q1,q2,.1,groups,2),expected+.1*entropy)
    loss=policy_loss(probs,logp,q1,q2,.1,groups,2)
    torch.testing.assert_close(loss,(-expected-.1*entropy).mean())
    loss.backward()
    assert q1.grad is None and q2.grad is None and logp.grad is not None
    for measured, sign in ((1.,1), (0.,-1)):
        logalpha=torch.tensor(math.log(5e-5),requires_grad=True)
        h=torch.tensor([measured],requires_grad=True)
        temperature_loss(logalpha,h,torch.tensor([.1])).backward()
        assert logalpha.grad*sign > 0 and h.grad is None


def test_terminal_skips_empty_actions_truncated_bootstraps_and_target_detaches():
    learner=SACLearner(ModelConfig(),LearningConfig(),device='cpu')
    samples=transitions()[:3]
    samples[1]=replace(samples[1],truncated=True)
    actual=learner.td_targets(samples,{'s':scene()})
    assert not actual.requires_grad
    with torch.no_grad():
        obs=[t.next_observation for t in samples[1:]]
        states=[t.next_privileged for t in samples[1:]]
        policies=learner.actor(obs)
        q1=learner.target1(obs,states,{'s':scene()}); q2=learner.target2(obs,states,{'s':scene()})
        expected=[torch.tensor(samples[0].reward)]
        for t,p,a,b in zip(samples[1:],policies,q1,q2):
            expected.append(t.reward+(p.probs*(torch.minimum(a,b)-learner.alpha*p.log_probs)).sum())
    torch.testing.assert_close(actual,torch.stack(expected))


def test_micro64_and_16x4_match_steps_restore_publication_and_independence():
    torch.manual_seed(31)
    a=SACLearner(ModelConfig(),LearningConfig(microbatch_size=64),device='cpu')
    b=SACLearner(ModelConfig(),LearningConfig(microbatch_size=16),device='cpu')
    b.load_state_dict(a.state_dict())
    # Eval uses the identical deterministic network without activation recomputation.
    for network in (a.actor, a.q1, a.q2):
        network.eval()
    storages=[{p.data_ptr() for p in m.parameters()} for m in
        (a.actor,a.q1,a.q2,a.target1,a.target2)]
    for i, left in enumerate(storages):
        for right in storages[i+1:]: assert left.isdisjoint(right)
    before=copy.deepcopy(a.state_dict())
    publication=a.actor_state()
    assert publication['version']==0 and all(v.device.type=='cpu' for v in publication['state_dict'].values())
    samples=transitions(); scenes={'s':scene()}
    ma=a.update(samples,scenes); mb=b.update(samples,scenes)
    assert a.updates==b.updates==1
    assert a.alpha < a.learning_config.initial_alpha
    for key in ('critic1_loss','critic2_loss','actor_loss','alpha_loss','entropy','target_mean'):
        assert ma[key]==pytest.approx(mb[key],abs=2e-6,rel=2e-5)
    for name in ('actor','q1','q2','target1','target2'):
        for key,value in getattr(a,name).state_dict().items():
            torch.testing.assert_close(value,getattr(b,name).state_dict()[key],atol=2e-6,rtol=2e-5)
    for net in (a.q1,a.q2,a.target1,a.target2):
        assert all(p.grad is None for p in net.parameters())
    for opt in (a.actor_optimizer,a.q1_optimizer,a.q2_optimizer,a.alpha_optimizer):
        assert all(s['step']==1 for s in opt.state.values())
    for key,value in a.target1.state_dict().items():
        torch.testing.assert_close(value,before['target1'][key]*.995+a.q1.state_dict()[key]*.005)
    assert a.actor_state()['version']==1
    assert any(not torch.equal(publication['state_dict'][k],v) for k,v in a.actor.state_dict().items())
    import io
    stream=io.BytesIO()
    torch.save(a.state_dict(),stream)
    stream.seek(0)
    checkpoint=torch.load(stream,map_location='cpu',weights_only=True)
    c=SACLearner(ModelConfig(),LearningConfig(),device='cpu'); c.load_state_dict(checkpoint)
    assert c.updates==1
    a.update(samples,scenes); c.update(samples,scenes)
    for name in ('actor','q1','q2','target1','target2'):
        for key,value in getattr(a,name).state_dict().items():
            torch.testing.assert_close(value,getattr(c,name).state_dict()[key],atol=3e-6,rtol=3e-5)
    assert c.log_alpha==a.log_alpha
    with pytest.raises(ValueError): c.update(samples[:16],scenes)


def test_temperature_caps_trainable_value_without_windup():
    learner=SACLearner(ModelConfig(),LearningConfig(),device='cpu')
    with torch.no_grad(): learner.log_alpha.fill_(math.log(1e-4)+.1)
    learner.update(transitions(),{'s':scene()})
    assert learner.log_alpha <= math.log(1e-4)
    assert learner.alpha <= 1.000001e-4


def test_saved_state_owns_tensors_and_rejects_old_schema():
    learner=SACLearner(ModelConfig(),LearningConfig(),device='cpu')
    saved=learner.state_dict()
    first=next(iter(saved['actor']))
    snapshot=saved['actor'][first].clone()
    with torch.no_grad(): next(learner.actor.parameters()).add_(1)
    torch.testing.assert_close(saved['actor'][first],snapshot,atol=0,rtol=0)
    with pytest.raises(ValueError): learner.load_state_dict(dict(saved,schema='old_gru'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for synchronization audit')
def test_cuda_complete_update_avoids_scalar_device_synchronization():
    from torch.utils._python_dispatch import TorchDispatchMode
    class NoScalarDeviceSync(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                assert not args[0].is_cuda, 'per-transition CUDA scalar read'
            return func(*args,**(kwargs or {}))
    learner=SACLearner(ModelConfig(),LearningConfig(),device='cuda')
    with NoScalarDeviceSync():
        metrics=learner.update(transitions(),{'s':scene()})
    assert metrics['updates']==1 and all(math.isfinite(v) for v in metrics.values())
    restored=SACLearner(ModelConfig(),LearningConfig(),device='cpu')
    restored.load_state_dict(learner.state_dict())
    assert restored.updates==1
    gpu_publication=learner.actor_state()['state_dict']
    for key,value in restored.actor_state()['state_dict'].items():
        torch.testing.assert_close(value,gpu_publication[key])
    for optimizer in (restored.actor_optimizer,restored.q1_optimizer,
                      restored.q2_optimizer,restored.alpha_optimizer):
        assert all(state['exp_avg'].device.type=='cpu' for state in optimizer.state.values())


def test_restore_keeps_adam_history_and_uses_requested_learning_rate_and_polyak():
    torch.manual_seed(45)
    original = SACLearner(ModelConfig(), LearningConfig(), device='cpu')
    samples, scenes = transitions(), {'s': scene()}
    original.update(samples, scenes)
    saved = original.state_dict()
    requested = LearningConfig(microbatch_size=32, learning_rate=3e-5,
                               polyak=0.1, initial_alpha=2e-5)
    restored = SACLearner(ModelConfig(), requested, device='cpu')
    restored.load_state_dict(saved)
    assert restored.updates == 1
    torch.testing.assert_close(restored.log_alpha, saved['log_alpha'], atol=0, rtol=0)
    assert restored.alpha != requested.initial_alpha
    for name in ('actor', 'q1', 'q2', 'target1', 'target2'):
        for key, value in getattr(restored, name).state_dict().items():
            torch.testing.assert_close(value, saved[name][key], atol=0, rtol=0)
    optimizer_names = ('actor_optimizer', 'q1_optimizer', 'q2_optimizer', 'alpha_optimizer')
    parameters_before = {}
    for name in optimizer_names:
        optimizer = getattr(restored, name)
        state = optimizer.state_dict()
        assert all(group['lr'] == requested.learning_rate for group in state['param_groups'])
        for index, values in saved[name]['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(state['state'][index][key], value, atol=0, rtol=0)
        parameters_before[name] = optimizer.param_groups[0]['params'][0].detach().clone()
    targets_before = {name: copy.deepcopy(getattr(restored, name).state_dict())
                      for name in ('target1', 'target2')}
    restored.update(samples, scenes)
    assert restored.updates == 2
    for name in optimizer_names:
        optimizer = getattr(restored, name)
        assert all(value['step'] == 2 for value in optimizer.state.values())
        group = optimizer.param_groups[0]
        parameter = group['params'][0]
        state = optimizer.state[parameter]
        # Independent Adam equation verifies the requested lr actually controls
        # the next update, with the continued (not reset) second-step moments.
        beta1, beta2 = group['betas']
        mean = state['exp_avg'] / (1 - beta1 ** 2)
        variance = state['exp_avg_sq'] / (1 - beta2 ** 2)
        expected = parameters_before[name] - requested.learning_rate * mean / (
            variance.sqrt() + group['eps'])
        torch.testing.assert_close(parameter, expected, atol=1e-7, rtol=1e-6)
    for target, source in (('target1', 'q1'), ('target2', 'q2')):
        for key, value in getattr(restored, target).state_dict().items():
            expected = targets_before[target][key] * 0.9 + getattr(restored, source).state_dict()[key] * 0.1
            torch.testing.assert_close(value, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('field,value', [('batch_size', 32), ('gamma', 0.99),
    ('target_entropy_factor', 0.02), ('maximum_alpha', 2e-4)])
def test_restore_still_rejects_changed_experience_objective(field, value):
    learner = SACLearner(ModelConfig(), LearningConfig(), device='cpu')
    saved = learner.state_dict()
    saved['learning_config'][field] = value
    with pytest.raises(ValueError, match='incompatible SAC semantics'):
        learner.load_state_dict(saved)


def test_restore_still_rejects_changed_model_shape():
    learner = SACLearner(ModelConfig(), LearningConfig(), device='cpu')
    saved = learner.state_dict()
    saved['model_config']['width'] = 64
    with pytest.raises(ValueError, match='incompatible learner model schema'):
        learner.load_state_dict(saved)
