"""Non-recurrent sparse graph actor and independent asymmetric critic.

All attention is edge-index or a single current query over graph nodes. Ragged
valid actions use segments, never padded logits or node-square attention.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .batch import pack_observations, pack_privileged
from .config import ModelConfig


def segment_sum(values, groups, count):
    output = values.new_zeros((count,) + values.shape[1:])
    return output.index_add_(0, groups, values)


def segment_log_softmax(logits, groups, count):
    indices = groups.reshape((-1,) + (1,) * (logits.ndim - 1)).expand_as(logits)
    maxima = logits.new_full((count,) + logits.shape[1:], -torch.inf)
    maxima.scatter_reduce_(0, indices, logits.detach(), reduce='amax', include_self=True)
    shifted = logits - maxima[groups]
    total = segment_sum(shifted.exp(), groups, count)
    return shifted - total[groups].log()


class SparseAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dim = config.heads, config.width // config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, config.width * 3)
        self.edge_bias = nn.Linear(1, config.heads, bias=False)
        self.project = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, config.width * 2), nn.GELU(),
                                nn.Linear(config.width * 2, config.width))

    def forward(self, x, graph):
        q, k, v = self.qkv(self.norm1(x)).reshape(-1, 3, self.heads, self.dim).unbind(1)
        source, target = graph.edges
        scores = (q[target] * k[source]).sum(-1) / math.sqrt(self.dim)
        scores = scores + self.edge_bias(graph.lengths[:, None])
        weights = segment_log_softmax(scores, target, x.shape[0]).exp()
        message = segment_sum(weights[..., None] * v[source], target, x.shape[0])
        x = x + self.project(message.flatten(1))
        return x + self.ff(self.norm2(x))


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input = nn.Linear(input_dim, config.width)
        self.layers = nn.ModuleList([SparseAttentionLayer(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)

    def forward(self, graph):
        x = self.input(graph.features)
        for layer in self.layers:
            if self.training and torch.is_grad_enabled():
                # Deterministic layer recomputation bounds activation residency;
                # inference and detached target/Q passes incur no recomputation.
                x = checkpoint(layer, x, graph, use_reentrant=False, preserve_rng_state=False)
            else:
                x = layer(x, graph)
        return self.norm(x)


class CurrentRead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dim = config.heads, config.width // config.heads
        self.query = nn.Linear(config.width, config.width)
        self.key_value = nn.Linear(config.width, config.width * 2)
        self.project = nn.Linear(config.width, config.width)

    def forward(self, query, nodes, groups):
        q = self.query(query).reshape(-1, self.heads, self.dim)
        k, v = self.key_value(nodes).reshape(-1, 2, self.heads, self.dim).unbind(1)
        scores = (q[groups] * k).sum(-1) / math.sqrt(self.dim)
        weights = segment_log_softmax(scores, groups, len(query)).exp()
        return self.project(segment_sum(weights[..., None] * v, groups, len(query)).flatten(1))


class PolygonEncoder(nn.Module):
    """Pool learned ordered boundary-edge features, retaining corner adjacency."""
    def __init__(self, width):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, edges, groups, count):
        values = segment_sum(self.edge(edges), groups, count)
        counts = segment_sum(edges.new_ones((len(edges), 1)), groups, count)
        return values / counts.clamp_min(1)


class MeasuredEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.graph = GraphEncoder(19, config)
        self.read = CurrentRead(config)
        self.polygon = PolygonEncoder(width)
        self.platform = nn.Linear(8, width)
        self.context = nn.Sequential(nn.Linear(width * 4, width), nn.GELU())
        self.action = nn.Sequential(nn.Linear(6, width), nn.GELU())

    def forward(self, batch):
        nodes = self.graph(batch.graph)
        current = nodes[batch.graph.current]
        context = self.context(torch.cat((current,
            self.read(current, nodes, batch.graph.groups),
            self.polygon(batch.polygon_edges, batch.polygon_groups, batch.size),
            self.platform(batch.context)), dim=-1))
        actions = torch.cat((nodes[batch.action_nodes], self.action(batch.action_features),
                             context[batch.action_groups]), dim=-1)
        return actions, current


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    log_probs: torch.Tensor


class Actor(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = MeasuredEncoder(config)
        self.head = nn.Sequential(nn.Linear(config.width * 3, config.width), nn.GELU(),
                                  nn.Linear(config.width, 1))

    def forward_packed(self, batch):
        actions, _ = self.encoder(batch)
        logits = self.head(actions).squeeze(-1)
        log_probs = segment_log_softmax(logits, batch.action_groups, batch.size)
        return PolicyOutput(logits, log_probs.exp(), log_probs)

    def forward(self, observations):
        batch = pack_observations(observations, next(self.parameters()).device)
        out = self.forward_packed(batch)
        return [PolicyOutput(*parts) for parts in zip(
            out.logits.split(batch.action_counts), out.probs.split(batch.action_counts),
            out.log_probs.split(batch.action_counts))]


class Critic(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.encoder = MeasuredEncoder(config)
        self.truth = GraphEncoder(4, config)
        self.truth_read = CurrentRead(config)
        self.head = nn.Sequential(nn.Linear(config.width * 4, config.width), nn.GELU(),
                                  nn.Linear(config.width, 1))

    def forward_packed(self, batch, privileged):
        actions, current = self.encoder(batch)
        truth = self.truth_read(current, self.truth(privileged), privileged.groups)
        return self.head(torch.cat((actions, truth[batch.action_groups]), dim=-1)).squeeze(-1)

    def forward(self, observations, states, scene_registry):
        device = next(self.parameters()).device
        batch = pack_observations(observations, device)
        truth = pack_privileged(observations, states, scene_registry, device)
        return self.forward_packed(batch, truth).split(batch.action_counts)
