"""Stage 3 无状态 cross-attention frontier policy。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import nn

from .observation import PolicyObservation


NETWORK_ARCHITECTURE_VERSION: Final = "cross_attention_frontier_policy/v1"
NETWORK_MEMORY_MODE: Final = "stateless_observation_only/v1"
TOKEN_DIM: Final = 128
MAP_POOL_SHAPE: Final = (16, 16)
GLOBAL_TOKEN_COUNT: Final = 256
LOCAL_TOKEN_COUNT: Final = 256
CONTEXT_TOKEN_COUNT: Final = 513
ATTENTION_HEADS: Final = 4
CROSS_ATTENTION_LAYERS: Final = 2
FFN_HIDDEN_DIM: Final = 512
ACTION_HIDDEN_DIM: Final = 128
DROPOUT: Final = 0.0
INVALID_LOGIT_VALUE: Final = -1.0e9
INITIAL_KAPPA_RAW: Final = math.log(math.expm1(0.1 - 1e-3))


class PolicyInputError(ValueError):
    """输入未满足 Stage 3 observed-only tensor 合同。"""


class PolicyActionError(ValueError):
    """动作或分布输入违反 Stage 3 合同。"""


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    prior_channels: torch.Tensor
    coverage_summary: torch.Tensor
    local_crop: torch.Tensor
    frontier_features: torch.Tensor
    pose_features: torch.Tensor
    candidate_mask: torch.Tensor

    @property
    def candidate_valid_mask(self) -> torch.Tensor:
        return self.candidate_mask


@dataclass(frozen=True, slots=True)
class PolicyForwardOutput:
    frontier_logits: torch.Tensor
    theta_mu_sin_raw: torch.Tensor
    theta_mu_cos_raw: torch.Tensor
    theta_kappa_raw: torch.Tensor
    theta_mu: torch.Tensor
    theta_kappa: torch.Tensor
    value: torch.Tensor
    global_map_tokens: torch.Tensor
    local_map_tokens: torch.Tensor
    pose_token: torch.Tensor
    context_tokens: torch.Tensor
    refined_frontier_tokens: torch.Tensor
    action_hidden: torch.Tensor


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    log_prob_frontier: torch.Tensor
    log_prob_theta: torch.Tensor
    log_prob_total: torch.Tensor
    frontier_entropy: torch.Tensor


@dataclass(frozen=True, slots=True)
class ActionSample:
    selected_frontier_index: torch.Tensor
    selected_theta: torch.Tensor
    log_prob_frontier: torch.Tensor
    log_prob_theta: torch.Tensor
    log_prob_total: torch.Tensor
    frontier_entropy: torch.Tensor
    value: torch.Tensor


def batch_policy_observations(
    observations: tuple[PolicyObservation, ...] | list[PolicyObservation],
    *,
    device: torch.device | str | None = None,
) -> PolicyBatch:
    """验证并批处理一个或多个 Stage 2 `PolicyObservation`。"""

    rows = tuple(observations)
    if not rows:
        raise PolicyInputError("at least one PolicyObservation is required")
    validated = tuple(_validated_observation_arrays(row) for row in rows)
    prior_shape = validated[0][0].shape
    local_shape = validated[0][2].shape
    if any(
        arrays[0].shape != prior_shape
        or arrays[1].shape[1:] != prior_shape[1:]
        or arrays[2].shape != local_shape
        for arrays in validated
    ):
        raise PolicyInputError("batch map spatial shapes must match")

    max_candidates = max(arrays[3].shape[0] for arrays in validated)
    frontier_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for arrays in validated:
        frontier = np.zeros((max_candidates, 22), dtype=np.float32)
        mask = np.zeros(max_candidates, dtype=bool)
        count = arrays[3].shape[0]
        frontier[:count] = arrays[3]
        mask[:count] = arrays[5]
        frontier_rows.append(frontier)
        mask_rows.append(mask)

    target = torch.device(device) if device is not None else torch.device("cpu")

    def float_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.float32).copy()).to(target)

    return PolicyBatch(
        prior_channels=float_tensor(np.stack([row[0] for row in validated])),
        coverage_summary=float_tensor(np.stack([row[1] for row in validated])),
        local_crop=float_tensor(np.stack([row[2] for row in validated])),
        frontier_features=float_tensor(np.stack(frontier_rows)),
        pose_features=float_tensor(np.stack([row[4] for row in validated])),
        candidate_mask=torch.from_numpy(np.stack(mask_rows).copy()).to(target),
    )


def _validated_observation_arrays(
    observation: PolicyObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(observation, PolicyObservation):
        raise PolicyInputError("batch rows must be PolicyObservation objects")
    prior = np.asarray(observation.prior_channels)
    coverage = np.asarray(observation.coverage_summary)
    local = np.asarray(observation.local_crop)
    frontier = np.asarray(observation.frontier_features)
    pose = np.asarray(observation.pose_features)
    mask = np.asarray(observation.candidate_mask)
    if prior.ndim != 3 or prior.shape[0] != 7:
        raise PolicyInputError("prior_channels shape must be [7,H,W]")
    if coverage.ndim != 3 or coverage.shape[0] != 8 or coverage.shape[1:] != prior.shape[1:]:
        raise PolicyInputError("coverage_summary shape must be [8,H,W] matching prior")
    if local.ndim != 3 or local.shape[0] != 8:
        raise PolicyInputError("local_crop shape must be [8,H,W]")
    if frontier.ndim != 2 or frontier.shape[1:] != (22,) or frontier.shape[0] == 0:
        raise PolicyInputError("frontier_features shape must be [M,22] with M positive")
    if pose.shape != (6,):
        raise PolicyInputError("pose_features shape must be [6]")
    if mask.shape != (frontier.shape[0],):
        raise PolicyInputError("candidate_mask shape must match frontier_features rows")
    if mask.dtype != np.bool_:
        raise PolicyInputError("candidate_mask must use boolean dtype")
    if not bool(mask.any()):
        raise PolicyInputError("each batch row must contain a valid candidate")
    float_arrays = (prior, coverage, local, frontier, pose)
    if any(not np.issubdtype(array.dtype, np.floating) for array in float_arrays):
        raise PolicyInputError("policy float inputs must use floating dtypes")
    if any(not np.isfinite(array).all() for array in float_arrays):
        raise PolicyInputError("policy float inputs must be finite")
    return tuple(np.asarray(array, dtype=np.float32) for array in float_arrays) + (mask,)


class MapEncoder(nn.Module):
    """用 GroupNorm 编码地图，并固定池化为 16x16 token 网格。"""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, TOKEN_DIM, kernel_size=3, padding=1),
            nn.GroupNorm(16, TOKEN_DIM),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(MAP_POOL_SHAPE)
        self.register_buffer(
            "position_encoding",
            _map_position_encoding(),
            persistent=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature_map = self.pool(self.features(value))
        tokens = feature_map.flatten(2).transpose(1, 2)
        return tokens + self.position_encoding.to(dtype=tokens.dtype)


class CrossAttentionBlock(nn.Module):
    """仅允许 frontier query 读取 context key/value。"""

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            TOKEN_DIM,
            ATTENTION_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(TOKEN_DIM)
        self.feed_forward = nn.Sequential(
            nn.Linear(TOKEN_DIM, FFN_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(FFN_HIDDEN_DIM, TOKEN_DIM),
        )
        self.feed_forward_norm = nn.LayerNorm(TOKEN_DIM)

    def forward(
        self,
        frontier_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            query=frontier_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=False,
        )
        refined = self.attention_norm(frontier_tokens + attended)
        return self.feed_forward_norm(refined + self.feed_forward(refined))


class CrossAttentionFrontierPolicy(nn.Module):
    """固定 v1 模块图；行为由后续 TDD 周期逐步接入。"""

    def __init__(self) -> None:
        super().__init__()
        self.global_encoder = MapEncoder(15)
        self.local_encoder = MapEncoder(8)
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, TOKEN_DIM),
            nn.LayerNorm(TOKEN_DIM),
            nn.GELU(),
            nn.Linear(TOKEN_DIM, TOKEN_DIM),
            nn.LayerNorm(TOKEN_DIM),
        )
        self.frontier_encoder = nn.Sequential(
            nn.Linear(22, TOKEN_DIM),
            nn.LayerNorm(TOKEN_DIM),
            nn.GELU(),
            nn.Linear(TOKEN_DIM, TOKEN_DIM),
            nn.LayerNorm(TOKEN_DIM),
        )
        self.frontier_position_encoder = nn.Sequential(
            nn.Linear(4, TOKEN_DIM),
            nn.LayerNorm(TOKEN_DIM),
        )
        self.cross_attention_blocks = nn.ModuleList(
            CrossAttentionBlock() for _ in range(CROSS_ATTENTION_LAYERS)
        )
        self.action_output_mlp = nn.Sequential(
            nn.Linear(TOKEN_DIM * 2 + 22, ACTION_HIDDEN_DIM),
            nn.LayerNorm(ACTION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(ACTION_HIDDEN_DIM, ACTION_HIDDEN_DIM),
            nn.LayerNorm(ACTION_HIDDEN_DIM),
            nn.GELU(),
        )
        self.frontier_logit_head = nn.Linear(ACTION_HIDDEN_DIM, 1)
        self.theta_sin_head = nn.Linear(ACTION_HIDDEN_DIM, 1)
        self.theta_cos_head = nn.Linear(ACTION_HIDDEN_DIM, 1)
        self.theta_kappa_head = nn.Linear(ACTION_HIDDEN_DIM, 1)
        self.value_mlp = nn.Sequential(
            nn.Linear(TOKEN_DIM * 3, ACTION_HIDDEN_DIM),
            nn.LayerNorm(ACTION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(ACTION_HIDDEN_DIM, 1),
        )
        self._initialize_action_heads()

    def _initialize_action_heads(self) -> None:
        nn.init.uniform_(self.frontier_logit_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.frontier_logit_head.bias)
        for head in (self.theta_sin_head, self.theta_cos_head, self.theta_kappa_head):
            nn.init.zeros_(head.weight)
        nn.init.zeros_(self.theta_sin_head.bias)
        nn.init.ones_(self.theta_cos_head.bias)
        nn.init.constant_(self.theta_kappa_head.bias, INITIAL_KAPPA_RAW)

    @staticmethod
    def architecture_metadata() -> dict[str, object]:
        return {
            "network_architecture_version": NETWORK_ARCHITECTURE_VERSION,
            "network_memory_mode": NETWORK_MEMORY_MODE,
            "prior_channels": 7,
            "coverage_summary_channels": 8,
            "global_encoder_input_channels": 15,
            "local_crop_channels": 8,
            "frontier_feature_dim": 22,
            "pose_feature_dim": 6,
            "map_pool_shape": list(MAP_POOL_SHAPE),
            "global_token_count": GLOBAL_TOKEN_COUNT,
            "local_token_count": LOCAL_TOKEN_COUNT,
            "pose_token_count": 1,
            "context_token_count": CONTEXT_TOKEN_COUNT,
            "token_dim": TOKEN_DIM,
            "cross_attention_layers": CROSS_ATTENTION_LAYERS,
            "attention_heads": ATTENTION_HEADS,
            "ffn_hidden_dim": FFN_HIDDEN_DIM,
            "action_hidden_dim": ACTION_HIDDEN_DIM,
            "dropout": DROPOUT,
            "cross_attention_direction": (
                "frontier_queries_to_context_keys_values/v1"
            ),
        }

    def forward(self, observation: PolicyBatch) -> PolicyForwardOutput:
        if not isinstance(observation, PolicyBatch):
            raise PolicyInputError("forward requires a validated PolicyBatch")
        model_device = next(self.parameters()).device
        _require_fp32_model_execution(self, model_device)
        _validate_policy_batch(observation, expected_device=model_device)
        global_input = torch.cat(
            (observation.prior_channels, observation.coverage_summary),
            dim=1,
        )
        global_tokens = self.global_encoder(global_input)
        local_tokens = self.local_encoder(observation.local_crop)
        pose_token = self.pose_encoder(observation.pose_features).unsqueeze(1)
        context_tokens = torch.cat((global_tokens, local_tokens, pose_token), dim=1)

        valid_mask = observation.candidate_mask.unsqueeze(-1)
        masked_frontier_features = torch.where(
            valid_mask,
            observation.frontier_features,
            torch.zeros_like(observation.frontier_features),
        )
        frontier_tokens = self.frontier_encoder(masked_frontier_features)
        frontier_position = masked_frontier_features[..., (0, 1, 3, 4)]
        frontier_tokens = frontier_tokens + self.frontier_position_encoder(
            frontier_position
        )
        refined = frontier_tokens
        for block in self.cross_attention_blocks:
            refined = block(refined, context_tokens)

        action_hidden = self.action_output_mlp(
            torch.cat(
                (frontier_tokens, refined, masked_frontier_features),
                dim=-1,
            )
        )
        frontier_logits_raw = self.frontier_logit_head(action_hidden).squeeze(-1)
        theta_mu_sin_unmasked = self.theta_sin_head(action_hidden).squeeze(-1)
        theta_mu_cos_unmasked = self.theta_cos_head(action_hidden).squeeze(-1)
        theta_kappa_unmasked = self.theta_kappa_head(action_hidden).squeeze(-1)
        frontier_logits = torch.where(
            observation.candidate_mask,
            frontier_logits_raw,
            torch.full_like(frontier_logits_raw, INVALID_LOGIT_VALUE),
        )
        theta_mu_sin_raw = torch.where(
            observation.candidate_mask,
            theta_mu_sin_unmasked,
            torch.zeros_like(theta_mu_sin_unmasked),
        )
        theta_mu_cos_raw = torch.where(
            observation.candidate_mask,
            theta_mu_cos_unmasked,
            torch.ones_like(theta_mu_cos_unmasked),
        )
        theta_kappa_raw = torch.where(
            observation.candidate_mask,
            theta_kappa_unmasked,
            torch.full_like(theta_kappa_unmasked, INITIAL_KAPPA_RAW),
        )
        theta_mu = _safe_theta_mu(
            raw_sin=theta_mu_sin_raw,
            raw_cos=theta_mu_cos_raw,
            recommended_sin=masked_frontier_features[..., 14],
            recommended_cos=masked_frontier_features[..., 15],
            candidate_mask=observation.candidate_mask,
        )
        theta_kappa = torch.clamp(
            torch.nn.functional.softplus(theta_kappa_raw) + 1e-3,
            min=1e-3,
            max=20.0,
        )
        candidate_mean, candidate_max = _masked_mean_max(
            refined,
            observation.candidate_mask,
        )
        context_mean = context_tokens.mean(dim=1)
        value = self.value_mlp(
            torch.cat((candidate_mean, candidate_max, context_mean), dim=-1)
        ).squeeze(-1)
        return PolicyForwardOutput(
            frontier_logits=frontier_logits,
            theta_mu_sin_raw=theta_mu_sin_raw,
            theta_mu_cos_raw=theta_mu_cos_raw,
            theta_kappa_raw=theta_kappa_raw,
            theta_mu=theta_mu,
            theta_kappa=theta_kappa,
            value=value,
            global_map_tokens=global_tokens,
            local_map_tokens=local_tokens,
            pose_token=pose_token,
            context_tokens=context_tokens,
            refined_frontier_tokens=refined,
            action_hidden=action_hidden,
        )


def _normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(theta + torch.pi, 2.0 * torch.pi) - torch.pi


def normalize_theta(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor) or not theta.is_floating_point():
        raise PolicyActionError("theta must be a floating tensor")
    if theta.dtype != torch.float32:
        raise PolicyActionError("theta operations require FP32")
    if not bool(torch.isfinite(theta).all()):
        raise PolicyActionError("theta must be finite")
    return _normalize_angle(theta)


def masked_frontier_probabilities(
    frontier_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    _require_autocast_disabled(frontier_logits.device, PolicyActionError)
    masked_logits = _validated_masked_logits(frontier_logits, candidate_mask)
    probabilities = torch.softmax(masked_logits, dim=-1)
    if probabilities.dtype != torch.float32 or not bool(torch.isfinite(probabilities).all()):
        raise PolicyActionError("masked frontier probabilities are not finite FP32")
    return probabilities


def sample_action(
    output: PolicyForwardOutput,
    mask: torch.Tensor,
    deterministic: bool,
) -> ActionSample:
    _require_autocast_disabled(mask.device, PolicyActionError)
    _validate_policy_output(output, mask)
    if not isinstance(deterministic, bool):
        raise PolicyActionError("deterministic must be boolean")
    masked_logits = _validated_masked_logits(output.frontier_logits, mask)
    frontier_distribution = torch.distributions.Categorical(logits=masked_logits)
    if deterministic:
        selected_index = masked_logits.argmax(dim=-1)
    else:
        selected_index = frontier_distribution.sample()
    selected_mu = _gather_candidate(output.theta_mu, selected_index)
    selected_kappa = _gather_candidate(output.theta_kappa, selected_index)
    theta_distribution = torch.distributions.VonMises(selected_mu, selected_kappa)
    if deterministic:
        selected_theta = normalize_theta(selected_mu)
    else:
        selected_theta = normalize_theta(theta_distribution.sample().to(torch.float32))
    evaluation = recompute_action_log_probs(
        output,
        mask,
        selected_index,
        selected_theta,
    )
    sample = ActionSample(
        selected_frontier_index=selected_index,
        selected_theta=selected_theta,
        log_prob_frontier=evaluation.log_prob_frontier,
        log_prob_theta=evaluation.log_prob_theta,
        log_prob_total=evaluation.log_prob_total,
        frontier_entropy=evaluation.frontier_entropy,
        value=output.value,
    )
    _require_finite_action_sample(sample)
    return sample


def recompute_action_log_probs(
    output: PolicyForwardOutput,
    mask: torch.Tensor,
    selected_frontier_index: torch.Tensor,
    selected_theta: torch.Tensor,
) -> ActionEvaluation:
    _require_autocast_disabled(mask.device, PolicyActionError)
    _validate_policy_output(output, mask)
    batch_size, candidate_count = mask.shape
    if not isinstance(selected_frontier_index, torch.Tensor) or selected_frontier_index.dtype != torch.int64:
        raise PolicyActionError("selected frontier indices must be int64 tensors")
    if selected_frontier_index.shape != (batch_size,) or selected_frontier_index.device != mask.device:
        raise PolicyActionError("selected frontier index shape or device is invalid")
    if bool((selected_frontier_index < 0).any()):
        raise PolicyActionError("selected frontier index is negative")
    if bool((selected_frontier_index >= candidate_count).any()):
        raise PolicyActionError("selected frontier index is out of range")
    selected_valid = mask.gather(1, selected_frontier_index.unsqueeze(1)).squeeze(1)
    if not bool(selected_valid.all()):
        raise PolicyActionError("selected frontier index is masked")
    if not isinstance(selected_theta, torch.Tensor) or selected_theta.dtype != torch.float32:
        raise PolicyActionError("selected theta must be an FP32 tensor")
    if selected_theta.shape != (batch_size,) or selected_theta.device != mask.device:
        raise PolicyActionError("selected theta shape or device is invalid")
    normalized_theta = normalize_theta(selected_theta)

    masked_logits = _validated_masked_logits(output.frontier_logits, mask)
    frontier_distribution = torch.distributions.Categorical(logits=masked_logits)
    selected_mu = _gather_candidate(output.theta_mu, selected_frontier_index)
    selected_kappa = _gather_candidate(output.theta_kappa, selected_frontier_index)
    theta_distribution = torch.distributions.VonMises(selected_mu, selected_kappa)
    log_prob_frontier = frontier_distribution.log_prob(selected_frontier_index).to(torch.float32)
    log_prob_theta = theta_distribution.log_prob(normalized_theta).to(torch.float32)
    log_prob_total = log_prob_frontier + log_prob_theta
    frontier_entropy = frontier_distribution.entropy().to(torch.float32)
    evaluation = ActionEvaluation(
        log_prob_frontier=log_prob_frontier,
        log_prob_theta=log_prob_theta,
        log_prob_total=log_prob_total,
        frontier_entropy=frontier_entropy,
    )
    if any(
        tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
        for tensor in (
            evaluation.log_prob_frontier,
            evaluation.log_prob_theta,
            evaluation.log_prob_total,
            evaluation.frontier_entropy,
        )
    ):
        raise PolicyActionError("action log probabilities or entropy are not finite FP32")
    return evaluation


def _validated_masked_logits(
    frontier_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(frontier_logits, torch.Tensor) or frontier_logits.dtype != torch.float32:
        raise PolicyActionError("frontier logits must be FP32")
    if not isinstance(candidate_mask, torch.Tensor) or candidate_mask.dtype != torch.bool:
        raise PolicyActionError("candidate mask must be boolean")
    if frontier_logits.ndim != 2 or candidate_mask.shape != frontier_logits.shape:
        raise PolicyActionError("frontier logits and candidate mask shapes must match [B,M]")
    if frontier_logits.device != candidate_mask.device:
        raise PolicyActionError("frontier logits and candidate mask devices must match")
    if not bool(candidate_mask.any(dim=1).all()):
        raise PolicyActionError("each distribution row must contain a valid candidate")
    if not bool(torch.isfinite(frontier_logits).all()):
        raise PolicyActionError("frontier logits must be finite")
    return torch.where(
        candidate_mask,
        frontier_logits,
        torch.full_like(frontier_logits, float("-inf")),
    )


def _validate_policy_output(
    output: PolicyForwardOutput,
    mask: torch.Tensor,
) -> None:
    if not isinstance(output, PolicyForwardOutput):
        raise PolicyActionError("output must be PolicyForwardOutput")
    _validated_masked_logits(output.frontier_logits, mask)
    expected = output.frontier_logits.shape
    for name in (
        "theta_mu_sin_raw",
        "theta_mu_cos_raw",
        "theta_kappa_raw",
        "theta_mu",
        "theta_kappa",
    ):
        value = getattr(output, name)
        if value.shape != expected or value.dtype != torch.float32 or value.device != mask.device:
            raise PolicyActionError(f"{name} shape, dtype, or device is invalid")
        if not bool(torch.isfinite(value).all()):
            raise PolicyActionError(f"{name} must be finite")
    if output.value.shape != (expected[0],) or output.value.dtype != torch.float32 or output.value.device != mask.device:
        raise PolicyActionError("value shape, dtype, or device is invalid")
    if not bool(torch.isfinite(output.value).all()):
        raise PolicyActionError("value must be finite")
    if bool((output.theta_kappa < 1e-3).any()) or bool((output.theta_kappa > 20.0).any()):
        raise PolicyActionError("theta kappa is out of bounds")


def _gather_candidate(value: torch.Tensor, selected_index: torch.Tensor) -> torch.Tensor:
    return value.gather(1, selected_index.unsqueeze(1)).squeeze(1)


def _require_finite_action_sample(sample: ActionSample) -> None:
    if any(
        tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
        for tensor in (
            sample.selected_theta,
            sample.log_prob_frontier,
            sample.log_prob_theta,
            sample.log_prob_total,
            sample.frontier_entropy,
            sample.value,
        )
    ):
        raise PolicyActionError("sampled action contains non-finite or non-FP32 values")


def _safe_theta_mu(
    *,
    raw_sin: torch.Tensor,
    raw_cos: torch.Tensor,
    recommended_sin: torch.Tensor,
    recommended_cos: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    threshold = 1e-6
    threshold_squared = threshold * threshold

    raw_norm_squared = raw_sin.square() + raw_cos.square()
    raw_denominator = torch.sqrt(raw_norm_squared.clamp_min(threshold_squared))
    raw_unit_sin = raw_sin / raw_denominator
    raw_unit_cos = raw_cos / raw_denominator

    recommended_norm_squared = recommended_sin.square() + recommended_cos.square()
    recommended_denominator = torch.sqrt(
        recommended_norm_squared.clamp_min(threshold_squared)
    )
    recommended_unit_sin = recommended_sin / recommended_denominator
    recommended_unit_cos = recommended_cos / recommended_denominator
    recommended_valid = (
        (recommended_norm_squared > threshold_squared) & candidate_mask
    )
    fallback_sin = torch.where(
        recommended_valid,
        recommended_unit_sin,
        torch.zeros_like(recommended_sin),
    )
    fallback_cos = torch.where(
        recommended_valid,
        recommended_unit_cos,
        torch.ones_like(recommended_cos),
    )
    raw_valid = (raw_norm_squared > threshold_squared) & candidate_mask
    unit_sin = torch.where(raw_valid, raw_unit_sin, fallback_sin)
    unit_cos = torch.where(raw_valid, raw_unit_cos, fallback_cos)
    return _normalize_angle(torch.atan2(unit_sin, unit_cos))


def _validate_policy_batch(
    batch: PolicyBatch,
    *,
    expected_device: torch.device,
) -> None:
    named_float_tensors = {
        "prior_channels": batch.prior_channels,
        "coverage_summary": batch.coverage_summary,
        "local_crop": batch.local_crop,
        "frontier_features": batch.frontier_features,
        "pose_features": batch.pose_features,
    }
    if any(not isinstance(value, torch.Tensor) for value in named_float_tensors.values()) or not isinstance(
        batch.candidate_mask, torch.Tensor
    ):
        raise PolicyInputError("PolicyBatch fields must all be torch tensors")
    if any(value.dtype != torch.float32 for value in named_float_tensors.values()):
        raise PolicyInputError("PolicyBatch float tensors must be FP32")
    if batch.candidate_mask.dtype != torch.bool:
        raise PolicyInputError("candidate_mask must use boolean dtype")
    devices = {value.device for value in named_float_tensors.values()} | {
        batch.candidate_mask.device
    }
    if devices != {expected_device}:
        raise PolicyInputError("PolicyBatch tensors and model must use one device")

    prior = batch.prior_channels
    coverage = batch.coverage_summary
    local = batch.local_crop
    frontier = batch.frontier_features
    pose = batch.pose_features
    mask = batch.candidate_mask
    if prior.ndim != 4 or prior.shape[0] == 0 or prior.shape[1] != 7:
        raise PolicyInputError("prior_channels shape must be [B,7,H,W]")
    batch_size = prior.shape[0]
    if coverage.shape != (batch_size, 8, prior.shape[2], prior.shape[3]):
        raise PolicyInputError("coverage_summary shape must be [B,8,H,W] matching prior")
    if local.ndim != 4 or local.shape[0] != batch_size or local.shape[1] != 8:
        raise PolicyInputError("local_crop shape must be [B,8,H,W]")
    if frontier.ndim != 3 or frontier.shape[0] != batch_size or frontier.shape[1] == 0 or frontier.shape[2] != 22:
        raise PolicyInputError("frontier_features shape must be [B,M,22]")
    if pose.shape != (batch_size, 6):
        raise PolicyInputError("pose_features shape must be [B,6]")
    if mask.shape != frontier.shape[:2]:
        raise PolicyInputError("candidate_mask shape must be [B,M]")
    if not bool(mask.any(dim=1).all()):
        raise PolicyInputError("each batch row must contain a valid candidate")
    if any(not bool(torch.isfinite(value).all()) for value in named_float_tensors.values()):
        raise PolicyInputError("PolicyBatch float tensors must be finite")


def _require_fp32_model_execution(
    model: nn.Module,
    device: torch.device,
) -> None:
    _require_autocast_disabled(device, PolicyInputError)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise PolicyInputError("model parameters must remain FP32")


def _require_autocast_disabled(
    device: torch.device,
    error_type: type[ValueError],
) -> None:
    try:
        enabled = torch.is_autocast_enabled(device.type)
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        enabled = torch.is_autocast_enabled()
        if device.type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            enabled = bool(torch.is_autocast_cpu_enabled())
    if enabled:
        raise error_type("AMP/autocast is forbidden for Stage 3 FP32 execution")


def _masked_mean_max(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1)
    mean = torch.where(expanded, tokens, torch.zeros_like(tokens)).sum(dim=1) / count
    maximum = tokens.masked_fill(~expanded, torch.finfo(tokens.dtype).min).amax(dim=1)
    return mean, maximum


def _map_position_encoding() -> torch.Tensor:
    quarter = TOKEN_DIM // 4
    coordinates = torch.linspace(-1.0, 1.0, MAP_POOL_SHAPE[0])
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequencies = torch.exp(
        torch.arange(quarter, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0)) / max(quarter - 1, 1))
    )
    x_phase = x.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * frequencies.reshape(1, -1)
    encoding = torch.cat(
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()),
        dim=-1,
    )
    return encoding.unsqueeze(0)
