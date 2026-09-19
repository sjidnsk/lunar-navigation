"""Disconnected sparse graph packing. All ragged sizes stay in CPU metadata.

Scene objects stay scene-owned. Only compact features/topology are copied into
an ephemeral batch; no scene or GPU graph cache survives replay eviction.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class GraphBatch:
    features: torch.Tensor
    edges: torch.Tensor
    lengths: torch.Tensor
    groups: torch.Tensor
    counts: tuple
    current: torch.Tensor | None = None


@dataclass
class ObservationBatch:
    graph: GraphBatch
    polygon_edges: torch.Tensor
    polygon_groups: torch.Tensor
    context: torch.Tensor
    action_nodes: torch.Tensor
    action_features: torch.Tensor
    action_groups: torch.Tensor
    action_counts: tuple

    @property
    def size(self):
        return len(self.action_counts)


@dataclass
class PrivilegedBatch:
    graph: GraphBatch
    query_positions: torch.Tensor
    query_groups: torch.Tensor
    support_query: torch.Tensor
    support_nodes: torch.Tensor
    support_geometry: torch.Tensor
    action_queries: torch.Tensor
    gains: torch.Tensor


def _tensor(arrays, device, dtype=torch.float32):
    return torch.as_tensor(np.concatenate(arrays), dtype=dtype, device=device)


def _graph(features, edges, lengths, currents, device):
    counts = tuple(len(f) for f in features)
    offsets = np.cumsum((0,) + counts[:-1])
    edge_arrays, length_arrays = [], []
    for count, offset, pairs, distance in zip(counts, offsets, edges, lengths):
        # Inputs are undirected physical edges. Add both directions and self
        # loops, so isolated nodes always have a well-defined neighborhood.
        pairs = np.asarray(pairs).reshape(-1, 2) + offset
        self_nodes = np.arange(count) + offset
        edge_arrays.append(np.concatenate((pairs, pairs[:, ::-1],
            np.column_stack((self_nodes, self_nodes)))))
        length_arrays.append(np.concatenate((distance, distance, np.zeros(count))) / 10.)
    return GraphBatch(_tensor(features, device),
        _tensor(edge_arrays, device, torch.long).T.contiguous(),
        _tensor(length_arrays, device),
        torch.as_tensor(np.repeat(np.arange(len(counts)), counts), dtype=torch.long, device=device),
        counts, None if currents is None else torch.as_tensor(
            offsets + currents, dtype=torch.long, device=device))


def pack_observations(observations, device):
    if not observations:
        raise ValueError("at least one decision observation required")
    graph = _graph([o.features for o in observations], [o.edges for o in observations],
        [o.edge_lengths for o in observations], [o.current_index for o in observations], device)
    offsets = np.cumsum((0,) + graph.counts[:-1])
    polygons = [np.concatenate((o.polygon, np.roll(o.polygon, -1, axis=0)), axis=1)
                for o in observations]
    action_counts = tuple(len(o.action_nodes) for o in observations)
    action_features = []
    for o in observations:
        heading = np.rint(o.action_yaws / (np.pi / 4)).astype(np.int64) % 8
        # Explicit position, world heading and direction-specific gain/history.
        action_features.append(np.column_stack((o.features[o.action_nodes, :2],
            np.cos(o.action_yaws), np.sin(o.action_yaws),
            o.features[o.action_nodes, 3 + heading], o.features[o.action_nodes, 11 + heading])))
    return ObservationBatch(graph, _tensor(polygons, device),
        torch.as_tensor(np.repeat(np.arange(len(observations)), [len(p) for p in polygons]),
                        dtype=torch.long, device=device),
        torch.as_tensor(np.stack([o.context for o in observations]), device=device),
        _tensor([o.action_nodes + offset for o, offset in zip(observations, offsets)], device, torch.long),
        _tensor(action_features, device),
        torch.as_tensor(np.repeat(np.arange(len(observations)), action_counts), dtype=torch.long, device=device),
        action_counts)


def pack_privileged(observations, states, scene_registry, device):
    if len(observations) != len(states):
        raise ValueError("one privileged state per observation required")
    features, edges, lengths = [], [], []
    query_positions, query_groups = [], []
    support_query, support_nodes, support_geometry = [], [], []
    action_queries, gains = [], []
    node_offset = query_offset = 0
    for group, (observation, state) in enumerate(zip(observations, states)):
        scene = scene_registry[state.scene_id]
        context = state.actions
        if context is None:
            raise ValueError("candidate action context required for critic packing")
        if len(context.action_positions) != len(observation.action_nodes):
            raise ValueError("candidate actions must align with measured actions")
        if (len(observation.goals) and not np.array_equal(
                context.positions[context.action_positions], observation.goals[:, :2])):
            raise ValueError("candidate positions must exactly match frozen action goals")
        if not np.array_equal(context.action_yaws, observation.goals[:, 2]):
            raise ValueError("candidate action yaws must exactly match frozen action goals")
        if np.any(context.support_indices >= len(scene.positions)):
            raise ValueError("candidate support index is outside truth graph")
        counts = np.diff(scene.reference_offsets)
        # Gather only reference bits, without unpacking the full hidden terrain.
        indices = scene.reference_indices
        bits = ((state.observed[indices // 8] >> (indices % 8)) & 1).astype(np.float32)
        sums = np.diff(np.r_[0., np.cumsum(bits, dtype=np.float64)][scene.reference_offsets])
        fractions = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        origin = observation.positions[observation.current_index]
        features.append(np.column_stack(((scene.positions - origin) / 10.,
            np.log1p(counts), fractions)).astype(np.float32))
        edges.append(scene.edges)
        lengths.append(scene.edge_lengths)
        query_positions.append(((context.positions - origin) / 10.).astype(np.float32))
        query_groups.append(np.full(len(context.positions), group, np.int64))
        repeated = np.repeat(np.arange(len(context.positions)), np.diff(context.support_offsets))
        support_query.append(repeated + query_offset)
        support_nodes.append(context.support_indices + node_offset)
        delta = (scene.positions[context.support_indices] - context.positions[repeated]) / 10.
        support_geometry.append(np.column_stack((delta, context.support_distances / 10.)).astype(np.float32))
        action_queries.append(context.action_positions + query_offset)
        gains.append(context.gains / 100.)
        node_offset += len(scene.positions)
        query_offset += len(context.positions)
    graph = _graph(features, edges, lengths, None, device)
    def cat(values, *, dtype=torch.float32, columns=None):
        if values and sum(len(v) for v in values):
            return _tensor(values, device, dtype)
        shape = (0, columns) if columns else (0,)
        return torch.empty(shape, dtype=dtype, device=device)
    return PrivilegedBatch(graph, cat(query_positions, columns=2), cat(query_groups, dtype=torch.long),
        cat(support_query, dtype=torch.long), cat(support_nodes, dtype=torch.long),
        cat(support_geometry, columns=3), cat(action_queries, dtype=torch.long), cat(gains))
