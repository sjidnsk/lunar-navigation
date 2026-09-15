"""Exact enumerated discrete SAC with one optimizer step per effective batch."""
import copy
from dataclasses import asdict
import math
import numpy as np
import torch
from .batch import pack_observations, pack_privileged
from .config import ModelConfig, LearningConfig
from .model import Actor, Critic, segment_sum


def soft_values(probs, log_probs, q1, q2, alpha, groups, count):
    return segment_sum(probs * (torch.minimum(q1, q2) - alpha * log_probs), groups, count)


def policy_loss(probs, log_probs, q1, q2, alpha, groups, count):
    return -soft_values(probs, log_probs, q1.detach(), q2.detach(),
                        alpha, groups, count).mean()


def temperature_loss(log_alpha, entropy, target_entropy):
    # Gradient descent reduces alpha when entropy exceeds its target.
    return (log_alpha * (entropy - target_entropy).detach()).mean()


class SACLearner:
    """Learner owns only networks/optimizers. Scheduler, replay and RNG are external.

    ``actor_state()`` returns an owned CPU snapshot and completed-update version.
    The scheduler chooses the 16-update publication interval. ``state_dict()``
    returns an owned, Torch-serializable full learner snapshot. Restore uses the
    current microbatch/rate/Polyak controls; initial alpha is new-training-only.
    Model and experience-objective settings must remain compatible.
    """
    def __init__(self, model_config=ModelConfig(), learning_config=LearningConfig(), *, device='cpu'):
        self.model_config, self.learning_config = model_config, learning_config
        self.device = torch.device(device)
        self.actor = Actor(model_config).to(self.device)
        self.q1, self.q2 = (Critic(model_config).to(self.device) for _ in range(2))
        self.target1, self.target2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.target1.requires_grad_(False)
        self.target2.requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(learning_config.initial_alpha), device=self.device,
                                      requires_grad=True)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_config.learning_rate)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=learning_config.learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=learning_config.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_config.learning_rate)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.detach().exp()

    @torch.no_grad()
    def td_targets(self, transitions, scenes):
        targets = torch.tensor([t.reward for t in transitions], device=self.device)
        active = [i for i, t in enumerate(transitions) if not t.terminated]
        if active:
            observations = [transitions[i].next_observation for i in active]
            if any(len(o.action_nodes) == 0 for o in observations):
                raise ValueError("nonterminal successor requires valid actions")
            states = [transitions[i].next_privileged for i in active]
            batch = pack_observations(observations, self.device)
            truth = pack_privileged(observations, states, scenes, self.device)
            policy = self.actor.forward_packed(batch)
            values = soft_values(policy.probs, policy.log_probs,
                self.target1.forward_packed(batch, truth), self.target2.forward_packed(batch, truth),
                self.alpha, batch.action_groups, batch.size)
            targets[torch.tensor(active, device=self.device)] += self.learning_config.gamma * values
        return targets

    def update(self, transitions, scenes):
        config = self.learning_config
        if len(transitions) != config.batch_size:
            raise ValueError("a complete update requires 64 transitions")
        if any(not 0 <= t.action < len(t.observation.action_nodes) for t in transitions):
            raise ValueError("replay action must index the frozen valid action list")
        chunks = [transitions[i:i+config.microbatch_size]
                  for i in range(0,len(transitions),config.microbatch_size)]
        # Six scalar metrics stay on-device until the single final transfer.
        metrics = torch.zeros(6, device=self.device)
        self.q1_optimizer.zero_grad(set_to_none=True)
        self.q2_optimizer.zero_grad(set_to_none=True)
        for chunk in chunks:
            weight = len(chunk) / len(transitions)
            targets = self.td_targets(chunk, scenes)
            batch = pack_observations([t.observation for t in chunk], self.device)
            truth = pack_privileged([t.observation for t in chunk], [t.privileged for t in chunk], scenes, self.device)
            indices = torch.tensor(np.cumsum((0,) + batch.action_counts[:-1]) +
                [t.action for t in chunk], device=self.device)
            # Independent networks permit immediate backward/release, without
            # retaining both critics' activations or changing optimizer cadence.
            for i, critic in enumerate((self.q1, self.q2)):
                selected = critic.forward_packed(batch, truth)[indices]
                loss = (selected - targets).square().mean()
                (loss * weight).backward()
                metrics[i] += loss.detach() * weight
                del selected, loss
            metrics[5] += targets.mean() * weight
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        # Clear critic grads before the actor phase: detached Q cannot pollute them.
        self.q1_optimizer.zero_grad(set_to_none=True)
        self.q2_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        entropy_gap = torch.zeros((), device=self.device)
        for chunk in chunks:
            weight = len(chunk) / len(transitions)
            batch = pack_observations([t.observation for t in chunk], self.device)
            truth = pack_privileged([t.observation for t in chunk], [t.privileged for t in chunk], scenes, self.device)
            with torch.no_grad():
                q1, q2 = self.q1.forward_packed(batch, truth), self.q2.forward_packed(batch, truth)
            policy = self.actor.forward_packed(batch)
            loss = policy_loss(policy.probs, policy.log_probs, q1, q2, self.alpha,
                               batch.action_groups, batch.size)
            (loss * weight).backward()
            entropy = -segment_sum(policy.probs.detach() * policy.log_probs.detach(), batch.action_groups, batch.size)
            target_entropy = torch.tensor(np.log(batch.action_counts) * config.target_entropy_factor,
                                          dtype=entropy.dtype, device=self.device)
            entropy_gap += (entropy-target_entropy).mean() * weight
            metrics[2] += loss.detach() * weight
            metrics[4] += entropy.mean() * weight
        self.actor_optimizer.step()
        alpha_loss = temperature_loss(self.log_alpha, entropy_gap, 0.0)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        metrics[3] = alpha_loss.detach()
        with torch.no_grad():
            self.log_alpha.clamp_(max=math.log(config.maximum_alpha))
            for target, source in ((self.target1,self.q1),(self.target2,self.q2)):
                for target_p, source_p in zip(target.parameters(), source.parameters()):
                    target_p.lerp_(source_p, config.polyak)
        self.updates += 1
        values = torch.cat((metrics, self.alpha.reshape(1))).detach().cpu().tolist()
        return dict(zip(('critic1_loss','critic2_loss','actor_loss','alpha_loss','entropy','target_mean','alpha'),values),
                    updates=self.updates, batch_size=len(transitions))

    def actor_state(self):
        return {'schema':'task_graph_v1', 'model_config':asdict(self.model_config),
                'version':self.updates,
                'state_dict':{k:v.detach().cpu().clone() for k,v in self.actor.state_dict().items()}}

    def state_dict(self):
        result = {'schema':'sparse_graph_sac_v1', 'model_config':asdict(self.model_config),
                  'learning_config':asdict(self.learning_config), 'updates':self.updates,
                  'log_alpha':self.log_alpha.detach().clone()}
        for name in ('actor','q1','q2','target1','target2','actor_optimizer',
                     'q1_optimizer','q2_optimizer','alpha_optimizer'):
            result[name] = getattr(self,name).state_dict()
        return copy.deepcopy(result)

    def load_state_dict(self, state):
        if state['schema'] != 'sparse_graph_sac_v1' or state['model_config'] != asdict(self.model_config):
            raise ValueError("incompatible learner model schema")
        saved, current = dict(state['learning_config']), asdict(self.learning_config)
        for name in ('microbatch_size', 'learning_rate', 'polyak', 'initial_alpha'):
            saved.pop(name)
            current.pop(name)
        if saved != current:
            raise ValueError("incompatible SAC semantics")
        # Optimizer load can retain references to input tensors; own the checkpoint.
        state = copy.deepcopy(state)
        for name in ('actor','q1','q2','target1','target2','actor_optimizer',
                     'q1_optimizer','q2_optimizer','alpha_optimizer'):
            getattr(self,name).load_state_dict(state[name])
        # Loading Adam restores parameter-group metadata as well as history.
        # Preserve saved moments/steps, but apply the caller's continuation rate.
        for optimizer in (self.actor_optimizer, self.q1_optimizer,
                          self.q2_optimizer, self.alpha_optimizer):
            for group in optimizer.param_groups:
                group['lr'] = self.learning_config.learning_rate
        with torch.no_grad(): self.log_alpha.copy_(state['log_alpha'])
        self.updates = int(state['updates'])
