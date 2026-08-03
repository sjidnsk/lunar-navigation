"""Stage 4 严格 FP32 PPO update。"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..policy.cross_attention import PolicyForwardOutput, batch_policy_observations, recompute_action_log_probs
from .rollout import ROLLOUT_ENV_COUNT, RolloutBatch, RolloutContractError


class PPOTrainingError(RuntimeError):
    """PPO 数学、batch 或 optimizer 状态不满足冻结合同。"""


@dataclass(frozen=True, slots=True)
class PPOLossTerms:
    ratio: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    frontier_entropy: torch.Tensor
    total_loss: torch.Tensor
    approx_kl: torch.Tensor


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    update_step: int
    initial_ratio_max_abs_error: float
    policy_loss: float
    value_loss: float
    frontier_entropy_mean: float
    frontier_entropy_min: float
    frontier_entropy_by_valid_candidate_count: Mapping[int, float]
    theta_kappa_mean: float
    theta_kappa_max: float
    approx_kl: float
    optimizer_steps: int
    planned_optimizer_steps: int
    effective_minibatch_sizes: tuple[int, ...]
    physical_microbatch_sizes: tuple[int, ...]
    early_stopped: bool
    early_stop_epoch: int | None
    early_stop_minibatch: int | None
    skipped_minibatch_count: int
    grad_pre_clip_norm_max: float
    grad_post_clip_norm_max: float
    parameter_change_l2: float
    policy_state_sha256_before: str
    policy_state_sha256_after: str
    math_evidence: Mapping[str, object]


def snapshot_hash_list_sha256(snapshot_hashes: tuple[str, ...] | list[str]) -> str:
    values = list(snapshot_hashes)
    if not values or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise PPOTrainingError("snapshot hash evidence is invalid")
    payload = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seal_math_evidence(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = dict(value)
    if "evidence_sha256" in payload:
        raise PPOTrainingError("math evidence cannot be sealed twice")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return MappingProxyType(
        {**payload, "evidence_sha256": hashlib.sha256(encoded).hexdigest()}
    )


def validate_ppo_math_evidence(
    value: object,
    *,
    expected_policy_state_sha256: str,
    expected_snapshot_hashes: Sequence[str],
    expected_sample_count: int,
    expected_optimizer_steps: int,
) -> dict[str, object]:
    boolean_fields = (
        "observation_finite",
        "action_finite",
        "old_logprob_finite",
        "old_value_finite",
        "advantage_finite",
        "return_finite",
        "new_logprob_finite",
        "new_value_finite",
        "ratio_finite",
        "loss_finite",
        "kl_finite",
        "grad_finite",
    )
    zero_count_fields = (
        "mask_violation_count",
        "snapshot_mismatch_count",
        "stale_policy_transition_count",
    )
    count_fields = (
        "sample_count",
        "initial_forward_sample_count",
        "forward_sample_count",
        "loss_sample_count",
        "gradient_step_count",
    )
    error_fields = (
        "old_joint_logprob_factorization_max_abs_error",
        "joint_logprob_factorization_max_abs_error",
    )
    required = {
        "schema_version",
        "snapshot_list_sha256",
        "batch_policy_state_sha256",
        "observed_compute_dtypes",
        "evidence_sha256",
        *boolean_fields,
        *zero_count_fields,
        *count_fields,
        *error_fields,
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise PPOTrainingError("PPO math evidence schema drifted")
    evidence = dict(value)
    digest = evidence.pop("evidence_sha256")
    encoded = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_digest = hashlib.sha256(encoded).hexdigest()
    if digest != expected_digest:
        raise PPOTrainingError("PPO math evidence seal drifted")
    snapshot_hashes = tuple(expected_snapshot_hashes)
    if (
        evidence.get("schema_version") != "ppo_update_math_evidence/v1"
        or evidence.get("batch_policy_state_sha256")
        != expected_policy_state_sha256
        or evidence.get("snapshot_list_sha256")
        != snapshot_hash_list_sha256(snapshot_hashes)
        or evidence.get("observed_compute_dtypes") != ["float32"]
        or type(expected_sample_count) is not int
        or expected_sample_count <= 0
        or type(expected_optimizer_steps) is not int
        or expected_optimizer_steps <= 0
    ):
        raise PPOTrainingError("PPO math evidence binding drifted")
    if any(evidence[field] is not True for field in boolean_fields):
        raise PPOTrainingError("PPO math evidence detected nonfinite values")
    if any(evidence[field] != 0 for field in zero_count_fields):
        raise PPOTrainingError("PPO math evidence detected contract violations")
    if any(type(evidence[field]) is not int or int(evidence[field]) <= 0 for field in count_fields):
        raise PPOTrainingError("PPO math evidence count drifted")
    if (
        evidence["sample_count"] != expected_sample_count
        or evidence["initial_forward_sample_count"] != expected_sample_count
        or int(evidence["forward_sample_count"])
        != int(evidence["initial_forward_sample_count"])
        + int(evidence["loss_sample_count"])
        or evidence["gradient_step_count"] != expected_optimizer_steps
    ):
        raise PPOTrainingError("PPO math evidence coverage drifted")
    for field in error_fields:
        candidate = evidence[field]
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, (int, float))
            or not math.isfinite(float(candidate))
            or not 0.0 <= float(candidate) <= 1.0e-6
        ):
            raise PPOTrainingError("PPO joint logprob evidence drifted")
    return {**evidence, "evidence_sha256": digest}


def _audit_rollout_batch_inputs(
    batch: RolloutBatch,
    *,
    current_policy_state_sha256: str,
) -> dict[str, object]:
    transitions = batch.transitions
    observation_finite = True
    action_finite = True
    old_logprob_finite = True
    old_value_finite = True
    mask_violation_count = 0
    snapshot_mismatch_count = 0
    stale_policy_transition_count = 0
    observed_dtypes: set[str] = set()
    old_joint_error = 0.0

    for index, transition in enumerate(transitions):
        actual_snapshot_sha256 = hashlib.sha256(transition.snapshot_bytes).hexdigest()
        if (
            actual_snapshot_sha256 != transition.candidate_snapshot_sha256
            or batch.snapshot_hashes[index] != transition.candidate_snapshot_sha256
        ):
            snapshot_mismatch_count += 1
            continue
        try:
            snapshot = transition.snapshot
        except RolloutContractError:
            snapshot_mismatch_count += 1
            continue
        for array in snapshot.array_fields()[:-1]:
            observed_dtypes.add(str(array.dtype))
            observation_finite = observation_finite and bool(np.isfinite(array).all())
        selected = transition.selected_frontier_index
        if (
            type(selected) is not int
            or not 0 <= selected < snapshot.candidate_mask.size
            or not bool(snapshot.candidate_mask[selected])
        ):
            mask_violation_count += 1
        action_finite = action_finite and math.isfinite(float(transition.selected_theta))
        old_logprob_finite = old_logprob_finite and all(
            math.isfinite(float(value))
            for value in (
                transition.old_log_prob_frontier,
                transition.old_log_prob_theta,
                transition.old_log_prob_total,
            )
        )
        old_value_finite = old_value_finite and math.isfinite(
            float(transition.old_value)
        )
        old_joint_error = max(
            old_joint_error,
            abs(
                float(transition.old_log_prob_total)
                - float(transition.old_log_prob_frontier)
                - float(transition.old_log_prob_theta)
            ),
        )
        stale_policy_transition_count += int(
            transition.policy_state_sha256 != current_policy_state_sha256
        )

    for array in (
        batch.rewards,
        batch.raw_advantages,
        batch.normalized_advantages,
        batch.returns,
        batch.old_log_prob_frontier,
        batch.old_log_prob_theta,
        batch.old_log_prob_total,
        batch.old_values,
        batch.selected_thetas,
    ):
        observed_dtypes.add(str(array.dtype))
    evidence = {
        "schema_version": "ppo_update_math_evidence/v1",
        "sample_count": batch.size,
        "snapshot_list_sha256": snapshot_hash_list_sha256(batch.snapshot_hashes),
        "batch_policy_state_sha256": batch.policy_state_sha256,
        "observation_finite": observation_finite,
        "action_finite": action_finite,
        "old_logprob_finite": old_logprob_finite,
        "old_value_finite": old_value_finite,
        "advantage_finite": bool(
            np.isfinite(batch.raw_advantages).all()
            and np.isfinite(batch.normalized_advantages).all()
        ),
        "return_finite": bool(np.isfinite(batch.returns).all()),
        "mask_violation_count": mask_violation_count,
        "snapshot_mismatch_count": snapshot_mismatch_count,
        "stale_policy_transition_count": stale_policy_transition_count,
        "old_joint_logprob_factorization_max_abs_error": old_joint_error,
        "observed_compute_dtypes": sorted(observed_dtypes),
    }
    if (
        not all(
            evidence[name] is True
            for name in (
                "observation_finite",
                "action_finite",
                "old_logprob_finite",
                "old_value_finite",
                "advantage_finite",
                "return_finite",
            )
        )
        or any(
            evidence[name] != 0
            for name in (
                "mask_violation_count",
                "snapshot_mismatch_count",
                "stale_policy_transition_count",
            )
        )
        or evidence["observed_compute_dtypes"] != ["float32"]
    ):
        raise PPOTrainingError("rollout batch math evidence failed")
    return evidence


def _record_forward_math_evidence(
    tracker: dict[str, object],
    *,
    output: PolicyForwardOutput,
    evaluation: object,
    sample_count: int,
    initial_pass: bool,
    terms: PPOLossTerms | None = None,
) -> None:
    log_prob_frontier = getattr(evaluation, "log_prob_frontier", None)
    log_prob_theta = getattr(evaluation, "log_prob_theta", None)
    log_prob_total = getattr(evaluation, "log_prob_total", None)
    logprob_tensors = (log_prob_frontier, log_prob_theta, log_prob_total)
    if any(not isinstance(value, torch.Tensor) for value in logprob_tensors):
        raise PPOTrainingError("action logprob evidence is unavailable")
    typed_logprobs = tuple(logprob_tensors)  # type: ignore[assignment]
    observed_dtypes = tracker["observed_compute_dtypes"]
    assert isinstance(observed_dtypes, set)
    for tensor in (*typed_logprobs, output.value):
        observed_dtypes.add(str(tensor.dtype).removeprefix("torch."))
    tracker["new_logprob_finite"] = bool(tracker["new_logprob_finite"]) and all(
        bool(torch.isfinite(tensor).all()) for tensor in typed_logprobs
    )
    tracker["new_value_finite"] = bool(tracker["new_value_finite"]) and bool(
        torch.isfinite(output.value).all()
    )
    factorization_error = (
        typed_logprobs[2] - (typed_logprobs[0] + typed_logprobs[1])
    ).abs()
    tracker["joint_logprob_factorization_max_abs_error"] = max(
        float(tracker["joint_logprob_factorization_max_abs_error"]),
        float(factorization_error.max().detach().cpu()),
    )
    tracker["forward_sample_count"] = int(tracker["forward_sample_count"]) + sample_count
    if initial_pass:
        tracker["initial_forward_sample_count"] = (
            int(tracker["initial_forward_sample_count"]) + sample_count
        )
    if terms is None:
        return
    term_tensors = (
        terms.ratio,
        terms.policy_loss,
        terms.value_loss,
        terms.frontier_entropy,
        terms.total_loss,
        terms.approx_kl,
    )
    for tensor in term_tensors:
        observed_dtypes.add(str(tensor.dtype).removeprefix("torch."))
    tracker["ratio_finite"] = bool(tracker["ratio_finite"]) and bool(
        torch.isfinite(terms.ratio).all()
    )
    tracker["loss_finite"] = bool(tracker["loss_finite"]) and all(
        bool(torch.isfinite(tensor).all()) for tensor in term_tensors[1:5]
    )
    tracker["kl_finite"] = bool(tracker["kl_finite"]) and bool(
        torch.isfinite(terms.approx_kl).all()
    )
    tracker["loss_sample_count"] = int(tracker["loss_sample_count"]) + sample_count


def _record_gradient_math_evidence(
    tracker: dict[str, object],
    parameters,
) -> None:
    gradients = tuple(
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    )
    observed_dtypes = tracker["observed_compute_dtypes"]
    assert isinstance(observed_dtypes, set)
    if not gradients:
        tracker["grad_finite"] = False
        return
    for gradient in gradients:
        observed_dtypes.add(str(gradient.dtype).removeprefix("torch."))
    tracker["grad_finite"] = bool(tracker["grad_finite"]) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    tracker["gradient_step_count"] = int(tracker["gradient_step_count"]) + 1


def compute_ppo_loss_terms(
    *,
    new_log_prob_total: torch.Tensor,
    old_log_prob_total: torch.Tensor,
    normalized_advantage: torch.Tensor,
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    frontier_entropy: torch.Tensor,
) -> PPOLossTerms:
    """计算 joint-action clipped PPO 与 clipped value 的 batch mean。"""

    tensors = (
        new_log_prob_total,
        old_log_prob_total,
        normalized_advantage,
        new_value,
        old_value,
        returns,
        frontier_entropy,
    )
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in tensors):
        raise PPOTrainingError("PPO loss inputs must be FP32 tensors")
    shape = new_log_prob_total.shape
    if len(shape) != 1 or shape[0] == 0 or any(value.shape != shape for value in tensors):
        raise PPOTrainingError("PPO loss input shapes must match [N]")
    devices = {value.device for value in tensors}
    if len(devices) != 1:
        raise PPOTrainingError("PPO loss inputs must share one device")
    device = new_log_prob_total.device
    try:
        autocast_enabled = torch.is_autocast_enabled(device.type)
    except TypeError:  # pragma: no cover - older supported torch compatibility
        autocast_enabled = torch.is_autocast_enabled()
    if autocast_enabled:
        raise PPOTrainingError("AMP/autocast is forbidden for PPO")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise PPOTrainingError("PPO loss inputs must be finite")

    ratio = torch.exp(new_log_prob_total - old_log_prob_total)
    unclipped = ratio * normalized_advantage
    clipped = torch.clamp(ratio, 0.8, 1.2) * normalized_advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean(dtype=torch.float32)
    value_clipped = old_value + torch.clamp(new_value - old_value, -0.2, 0.2)
    value_loss = 0.5 * torch.maximum(
        (new_value - returns).square(),
        (value_clipped - returns).square(),
    ).mean(dtype=torch.float32)
    entropy_mean = frontier_entropy.mean(dtype=torch.float32)
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_mean
    approx_kl = (old_log_prob_total - new_log_prob_total).mean(dtype=torch.float32)
    outputs = (ratio, policy_loss, value_loss, entropy_mean, total_loss, approx_kl)
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in outputs):
        raise PPOTrainingError("PPO loss outputs must be finite FP32")
    return PPOLossTerms(
        ratio=ratio,
        policy_loss=policy_loss,
        value_loss=value_loss,
        frontier_entropy=entropy_mean,
        total_loss=total_loss,
        approx_kl=approx_kl,
    )


def policy_state_sha256(policy: nn.Module) -> str:
    """对 state_dict 的名称、dtype、shape 和原始 bytes 做稳定 SHA-256。"""

    if not isinstance(policy, nn.Module):
        raise PPOTrainingError("policy hash requires a torch module")
    digest = hashlib.sha256()
    state = policy.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise PPOTrainingError("policy state contains a non-tensor value")
        contiguous = tensor.detach().contiguous().cpu()
        raw = contiguous.numpy().tobytes(order="C")
        for value in (
            name.encode("utf-8"),
            str(contiguous.dtype).encode("ascii"),
            np.asarray(tuple(contiguous.shape), dtype="<i8").tobytes(),
            raw,
        ):
            digest.update(struct.pack("<Q", len(value)))
            digest.update(value)
    return digest.hexdigest()


def physical_microbatch_slices(sample_count: int) -> tuple[tuple[int, int], ...]:
    if type(sample_count) is not int or sample_count <= 0:
        raise PPOTrainingError("physical microbatch sample count must be positive")
    return tuple(
        (start, min(start + 32, sample_count))
        for start in range(0, sample_count, 32)
    )


class PPOTrainer:
    """固定 v1 超参数、只消费保存 snapshot 的单次 PPO updater。"""

    PPO_EPOCHS = 4
    EFFECTIVE_MINIBATCH_SIZE = 256
    PHYSICAL_MICROBATCH_SIZE = 32
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.03

    def __init__(
        self,
        policy: nn.Module,
        *,
        device: torch.device | str,
        optimizer: torch.optim.Optimizer | None = None,
        shuffle_seed: int = 0,
    ) -> None:
        if not isinstance(policy, nn.Module):
            raise PPOTrainingError("PPOTrainer requires a torch policy module")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise PPOTrainingError("CUDA was requested but unavailable; CPU fallback is forbidden")
        if self.device.type not in {"cpu", "cuda"}:
            raise PPOTrainingError("PPOTrainer supports only CPU or CUDA")
        self.policy = policy.to(self.device)
        if any(parameter.dtype != torch.float32 for parameter in self.policy.parameters()):
            raise PPOTrainingError("PPO policy parameters must remain FP32")
        if type(shuffle_seed) is not int:
            raise PPOTrainingError("shuffle_seed must be an integer")
        self.shuffle_seed = shuffle_seed
        self.optimizer = optimizer or torch.optim.AdamW(
            self.policy.parameters(),
            lr=3e-4,
            eps=1e-5,
            weight_decay=1e-4,
        )
        self._validate_optimizer()
        self.update_step = 0

    def restore_update_step(self, update_step: int) -> None:
        """只允许把 fresh trainer 绑定到已验证 checkpoint step。"""

        if self.update_step != 0:
            raise PPOTrainingError("trainer update step can only be restored once")
        if type(update_step) is not int or update_step <= 0:
            raise PPOTrainingError(
                "restored trainer update step must be a positive integer"
            )
        self.update_step = update_step

    def update(
        self,
        batch: RolloutBatch,
        *,
        resource_guard: Callable[[str], None] | None = None,
    ) -> PPOUpdateMetrics:
        if not isinstance(batch, RolloutBatch):
            raise PPOTrainingError("PPOTrainer.update requires a RolloutBatch")
        if resource_guard is not None and not callable(resource_guard):
            raise PPOTrainingError("resource_guard must be callable")

        def poll(boundary: str) -> None:
            if resource_guard is not None:
                resource_guard(boundary)

        poll("ppo:before-update")
        policy_hash_before = policy_state_sha256(self.policy)
        try:
            batch.claim_for_update(policy_hash_before)
        except RolloutContractError:
            raise
        try:
            input_math_evidence = _audit_rollout_batch_inputs(
                batch,
                current_policy_state_sha256=policy_hash_before,
            )
        except BaseException:
            if batch.claimed:
                batch.invalidate()
            raise
        math_tracker: dict[str, object] = {
            "new_logprob_finite": True,
            "new_value_finite": True,
            "ratio_finite": True,
            "loss_finite": True,
            "kl_finite": True,
            "grad_finite": True,
            "joint_logprob_factorization_max_abs_error": 0.0,
            "initial_forward_sample_count": 0,
            "forward_sample_count": 0,
            "loss_sample_count": 0,
            "gradient_step_count": 0,
            "observed_compute_dtypes": set(
                input_math_evidence["observed_compute_dtypes"]
            ),
        }
        parameters_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.policy.named_parameters()
        }
        try:
            self.policy.train()
            initial_error = self._initial_ratio_error(
                batch,
                math_tracker,
                resource_guard=resource_guard,
            )
            tolerance = 1e-5 if self.device.type == "cuda" else 1e-6
            if initial_error > tolerance:
                raise PPOTrainingError(
                    f"initial ratio mismatch: max_abs_error={initial_error:.9g}"
                )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.shuffle_seed + self.update_step)
            total_size = batch.size
            planned_steps = self.PPO_EPOCHS * math.ceil(
                total_size / self.EFFECTIVE_MINIBATCH_SIZE
            )
            optimizer_steps = 0
            effective_sizes: list[int] = []
            physical_sizes: list[int] = []
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropy_sum = 0.0
            entropy_count = 0
            entropy_min = math.inf
            entropy_by_count: dict[int, list[float]] = {}
            kappa_sum = 0.0
            kappa_count = 0
            kappa_max = -math.inf
            grad_pre_max = 0.0
            grad_post_max = 0.0
            approx_kl = 0.0
            early_stopped = False
            early_stop_epoch: int | None = None
            early_stop_minibatch: int | None = None
            skipped_minibatches = 0

            for epoch in range(self.PPO_EPOCHS):
                permutation = torch.randperm(total_size, generator=generator).numpy()
                minibatch_count = math.ceil(
                    total_size / self.EFFECTIVE_MINIBATCH_SIZE
                )
                for minibatch_index in range(minibatch_count):
                    start = minibatch_index * self.EFFECTIVE_MINIBATCH_SIZE
                    stop = min(start + self.EFFECTIVE_MINIBATCH_SIZE, total_size)
                    effective_indices = np.asarray(
                        permutation[start:stop], dtype=np.int64
                    )
                    effective_size = int(effective_indices.size)
                    self.optimizer.zero_grad(set_to_none=True)
                    kl_sum = 0.0
                    local_policy_sum = 0.0
                    local_value_sum = 0.0
                    local_entropy_values: list[tuple[int, float]] = []
                    local_kappas: list[float] = []

                    for micro_start, micro_stop in physical_microbatch_slices(
                        effective_size
                    ):
                        poll("ppo:before-physical-microbatch")
                        indices = effective_indices[micro_start:micro_stop]
                        physical_size = int(indices.size)
                        physical_sizes.append(physical_size)
                        output, evaluation, policy_batch = self._evaluate(
                            batch,
                            indices,
                        )
                        old_total = self._batch_tensor(
                            batch.old_log_prob_total,
                            indices,
                            dtype=torch.float32,
                        )
                        terms = compute_ppo_loss_terms(
                            new_log_prob_total=evaluation.log_prob_total,
                            old_log_prob_total=old_total,
                            normalized_advantage=self._batch_tensor(
                                batch.normalized_advantages,
                                indices,
                                dtype=torch.float32,
                            ),
                            new_value=output.value,
                            old_value=self._batch_tensor(
                                batch.old_values,
                                indices,
                                dtype=torch.float32,
                            ),
                            returns=self._batch_tensor(
                                batch.returns,
                                indices,
                                dtype=torch.float32,
                            ),
                            frontier_entropy=evaluation.frontier_entropy,
                        )
                        _record_forward_math_evidence(
                            math_tracker,
                            output=output,
                            evaluation=evaluation,
                            sample_count=physical_size,
                            initial_pass=False,
                            terms=terms,
                        )
                        weight = physical_size / effective_size
                        (terms.total_loss * weight).backward()
                        kl_sum += float(
                            (old_total - evaluation.log_prob_total)
                            .detach()
                            .sum(dtype=torch.float32)
                            .cpu()
                        )
                        local_policy_sum += float(terms.policy_loss.detach().cpu()) * physical_size
                        local_value_sum += float(terms.value_loss.detach().cpu()) * physical_size
                        valid_counts = (
                            policy_batch.candidate_mask.sum(dim=1).detach().cpu().tolist()
                        )
                        entropies = evaluation.frontier_entropy.detach().cpu().tolist()
                        local_entropy_values.extend(
                            (int(count), float(entropy))
                            for count, entropy in zip(valid_counts, entropies, strict=True)
                        )
                        selected = self._batch_tensor(
                            batch.selected_frontier_indices,
                            indices,
                            dtype=torch.int64,
                        )
                        selected_kappa = output.theta_kappa.gather(
                            1, selected.unsqueeze(1)
                        ).squeeze(1)
                        local_kappas.extend(
                            float(value) for value in selected_kappa.detach().cpu().tolist()
                        )
                        poll("ppo:after-physical-microbatch")

                    approx_kl = kl_sum / effective_size
                    if not math.isfinite(approx_kl):
                        raise PPOTrainingError("approx_kl is non-finite")
                    linear_minibatch = epoch * minibatch_count + minibatch_index
                    if approx_kl > self.TARGET_KL:
                        self.optimizer.zero_grad(set_to_none=True)
                        early_stopped = True
                        early_stop_epoch = epoch
                        early_stop_minibatch = minibatch_index
                        skipped_minibatches = planned_steps - linear_minibatch
                        break

                    pre_clip = _gradient_norm(self.policy.parameters())
                    _record_gradient_math_evidence(
                        math_tracker,
                        self.policy.parameters(),
                    )
                    if not math.isfinite(pre_clip):
                        raise PPOTrainingError("pre-clip gradient norm is non-finite")
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.MAX_GRAD_NORM,
                        error_if_nonfinite=True,
                    )
                    post_clip = _gradient_norm(self.policy.parameters())
                    if not math.isfinite(post_clip) or post_clip > 0.500001:
                        raise PPOTrainingError("post-clip gradient norm exceeds 0.500001")
                    poll("ppo:before-optimizer-step")
                    self.optimizer.step()
                    poll("ppo:after-optimizer-step")
                    optimizer_steps += 1
                    effective_sizes.append(effective_size)
                    policy_losses.append(local_policy_sum / effective_size)
                    value_losses.append(local_value_sum / effective_size)
                    grad_pre_max = max(grad_pre_max, pre_clip)
                    grad_post_max = max(grad_post_max, post_clip)
                    for count, entropy in local_entropy_values:
                        entropy_sum += entropy
                        entropy_count += 1
                        entropy_min = min(entropy_min, entropy)
                        aggregate = entropy_by_count.setdefault(count, [0.0, 0.0])
                        aggregate[0] += entropy
                        aggregate[1] += 1.0
                    for value in local_kappas:
                        kappa_sum += value
                        kappa_count += 1
                        kappa_max = max(kappa_max, value)
                if early_stopped:
                    break

            parameter_change_l2 = _parameter_change_l2(
                self.policy,
                parameters_before,
            )
            if optimizer_steps == 0 or parameter_change_l2 <= 0.0:
                raise PPOTrainingError("PPO update changed no trainable parameter")
            if not all(
                math.isfinite(value)
                for value in (
                    parameter_change_l2,
                    grad_pre_max,
                    grad_post_max,
                    entropy_sum,
                    kappa_sum,
                )
            ):
                raise PPOTrainingError("PPO update metrics are non-finite")
            policy_hash_after = policy_state_sha256(self.policy)
            if policy_hash_after == policy_hash_before:
                raise PPOTrainingError("PPO policy hash did not change")
            self.update_step += 1
            observed_compute_dtypes = math_tracker["observed_compute_dtypes"]
            assert isinstance(observed_compute_dtypes, set)
            math_evidence_payload = {
                **input_math_evidence,
                "new_logprob_finite": bool(math_tracker["new_logprob_finite"]),
                "new_value_finite": bool(math_tracker["new_value_finite"]),
                "ratio_finite": bool(math_tracker["ratio_finite"]),
                "loss_finite": bool(math_tracker["loss_finite"]),
                "kl_finite": bool(math_tracker["kl_finite"]),
                "grad_finite": bool(math_tracker["grad_finite"]),
                "joint_logprob_factorization_max_abs_error": float(
                    math_tracker["joint_logprob_factorization_max_abs_error"]
                ),
                "initial_forward_sample_count": int(
                    math_tracker["initial_forward_sample_count"]
                ),
                "forward_sample_count": int(math_tracker["forward_sample_count"]),
                "loss_sample_count": int(math_tracker["loss_sample_count"]),
                "gradient_step_count": int(math_tracker["gradient_step_count"]),
                "observed_compute_dtypes": sorted(observed_compute_dtypes),
            }
            if (
                any(
                    math_evidence_payload[name] is not True
                    for name in (
                        "new_logprob_finite",
                        "new_value_finite",
                        "ratio_finite",
                        "loss_finite",
                        "kl_finite",
                        "grad_finite",
                    )
                )
                or math_evidence_payload["initial_forward_sample_count"] != batch.size
                or int(math_evidence_payload["forward_sample_count"])
                != int(math_evidence_payload["initial_forward_sample_count"])
                + int(math_evidence_payload["loss_sample_count"])
                or int(math_evidence_payload["loss_sample_count"]) <= 0
                or math_evidence_payload["gradient_step_count"] != optimizer_steps
                or math_evidence_payload["observed_compute_dtypes"] != ["float32"]
            ):
                raise PPOTrainingError("PPO runtime math evidence failed")
            math_evidence = _seal_math_evidence(math_evidence_payload)
            metrics = PPOUpdateMetrics(
                update_step=self.update_step,
                initial_ratio_max_abs_error=initial_error,
                policy_loss=sum(policy_losses) / len(policy_losses),
                value_loss=sum(value_losses) / len(value_losses),
                frontier_entropy_mean=entropy_sum / entropy_count,
                frontier_entropy_min=entropy_min,
                frontier_entropy_by_valid_candidate_count=MappingProxyType(
                    {
                        count: values[0] / values[1]
                        for count, values in sorted(entropy_by_count.items())
                    }
                ),
                theta_kappa_mean=kappa_sum / kappa_count,
                theta_kappa_max=kappa_max,
                approx_kl=approx_kl,
                optimizer_steps=optimizer_steps,
                planned_optimizer_steps=planned_steps,
                effective_minibatch_sizes=tuple(effective_sizes),
                physical_microbatch_sizes=tuple(physical_sizes),
                early_stopped=early_stopped,
                early_stop_epoch=early_stop_epoch,
                early_stop_minibatch=early_stop_minibatch,
                skipped_minibatch_count=skipped_minibatches,
                grad_pre_clip_norm_max=grad_pre_max,
                grad_post_clip_norm_max=grad_post_max,
                parameter_change_l2=parameter_change_l2,
                policy_state_sha256_before=policy_hash_before,
                policy_state_sha256_after=policy_hash_after,
                math_evidence=math_evidence,
            )
            batch.mark_consumed()
            poll("ppo:after-update")
            return metrics
        except BaseException as exc:
            if batch.claimed:
                batch.invalidate()
            if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
                raise PPOTrainingError("CUDA OOM during PPO update") from exc
            raise

    def _initial_ratio_error(
        self,
        batch: RolloutBatch,
        math_tracker: dict[str, object],
        *,
        resource_guard: Callable[[str], None] | None = None,
    ) -> float:
        max_error = 0.0
        with torch.no_grad():
            for start in range(0, batch.size, ROLLOUT_ENV_COUNT):
                stop = min(start + ROLLOUT_ENV_COUNT, batch.size)
                if resource_guard is not None:
                    resource_guard("ppo:before-initial-microbatch")
                indices = np.arange(start, stop, dtype=np.int64)
                output, evaluation, _ = self._evaluate(batch, indices)
                _record_forward_math_evidence(
                    math_tracker,
                    output=output,
                    evaluation=evaluation,
                    sample_count=int(indices.size),
                    initial_pass=True,
                )
                old_frontier = self._batch_tensor(
                    batch.old_log_prob_frontier,
                    indices,
                    dtype=torch.float32,
                )
                old_theta = self._batch_tensor(
                    batch.old_log_prob_theta,
                    indices,
                    dtype=torch.float32,
                )
                old_total = self._batch_tensor(
                    batch.old_log_prob_total,
                    indices,
                    dtype=torch.float32,
                )
                differences = (
                    (evaluation.log_prob_frontier - old_frontier).abs(),
                    (evaluation.log_prob_theta - old_theta).abs(),
                    (evaluation.log_prob_total - old_total).abs(),
                    (torch.exp(evaluation.log_prob_total - old_total) - 1.0).abs(),
                )
                max_error = max(
                    max_error,
                    *(float(value.max().cpu()) for value in differences),
                )
                if resource_guard is not None:
                    resource_guard("ppo:after-initial-microbatch")
        if not math.isfinite(max_error):
            raise PPOTrainingError("initial ratio is non-finite")
        return max_error

    def _evaluate(
        self,
        batch: RolloutBatch,
        indices: np.ndarray,
    ) -> tuple[PolicyForwardOutput, object, object]:
        observations = batch.policy_observations(indices)
        policy_batch = batch_policy_observations(observations, device=self.device)
        output = self.policy(policy_batch)
        if not isinstance(output, PolicyForwardOutput):
            raise PPOTrainingError("policy must return PolicyForwardOutput")
        selected = self._batch_tensor(
            batch.selected_frontier_indices,
            indices,
            dtype=torch.int64,
        )
        theta = self._batch_tensor(
            batch.selected_thetas,
            indices,
            dtype=torch.float32,
        )
        evaluation = recompute_action_log_probs(
            output,
            policy_batch.candidate_mask,
            selected,
            theta,
        )
        return output, evaluation, policy_batch

    def _batch_tensor(
        self,
        matrix: np.ndarray,
        indices: np.ndarray,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        flat = matrix.reshape(-1)
        selected = np.asarray(flat[indices]).copy()
        return torch.as_tensor(selected, dtype=dtype, device=self.device)

    def _validate_optimizer(self) -> None:
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise PPOTrainingError("Stage 4 optimizer must be AdamW")
        for group in self.optimizer.param_groups:
            if (
                group.get("lr") != 3e-4
                or group.get("eps") != 1e-5
                or group.get("weight_decay") != 1e-4
            ):
                raise PPOTrainingError("Stage 4 AdamW hyperparameters drifted")


def _gradient_norm(parameters) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            return math.inf
        total += float(gradient.double().square().sum().cpu())
    return math.sqrt(total) if found else 0.0


def _parameter_change_l2(
    policy: nn.Module,
    before: Mapping[str, torch.Tensor],
) -> float:
    total = 0.0
    for name, parameter in policy.named_parameters():
        if name not in before:
            raise PPOTrainingError("policy parameter set changed during update")
        difference = parameter.detach().cpu().double() - before[name].double()
        total += float(difference.square().sum())
    if set(before) != {name for name, _ in policy.named_parameters()}:
        raise PPOTrainingError("policy parameter set changed during update")
    return math.sqrt(total)


__all__ = [
    "PPOLossTerms",
    "PPOTrainer",
    "PPOTrainingError",
    "PPOUpdateMetrics",
    "compute_ppo_loss_terms",
    "physical_microbatch_slices",
    "policy_state_sha256",
    "snapshot_hash_list_sha256",
    "validate_ppo_math_evidence",
]
