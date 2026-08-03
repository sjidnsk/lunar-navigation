"""Stateless cross-attention policy backbone."""
from __future__ import annotations
import math
from typing import Final
import torch
from torch import nn
TOKEN_DIM: Final = 128
MAP_POOL_SHAPE: Final = (16, 16)
ATTENTION_HEADS: Final = 4
CROSS_ATTENTION_LAYERS: Final = 2
FFN_HIDDEN_DIM: Final = 512
ACTION_HIDDEN_DIM: Final = 128
DROPOUT: Final = 0.0
INVALID_LOGIT_VALUE: Final = -1000000000.0
INITIAL_KAPPA_RAW: Final = math.log(math.expm1(0.1 - 0.001))

class PolicyActionError(ValueError):
    """动作或分布输入违反共享策略合同。"""

class MapEncoder(nn.Module):
    """用 GroupNorm 编码地图，并固定池化为 16x16 token 网格。"""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.features = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.GELU(), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, TOKEN_DIM, kernel_size=3, padding=1), nn.GroupNorm(16, TOKEN_DIM), nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d(MAP_POOL_SHAPE)
        self.register_buffer('position_encoding', _map_position_encoding(), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature_map = self.pool(self.features(value))
        tokens = feature_map.flatten(2).transpose(1, 2)
        return tokens + self.position_encoding.to(dtype=tokens.dtype)

class CrossAttentionBlock(nn.Module):
    """仅允许 frontier query 读取 context key/value。"""

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(TOKEN_DIM, ATTENTION_HEADS, dropout=DROPOUT, batch_first=True)
        self.attention_norm = nn.LayerNorm(TOKEN_DIM)
        self.feed_forward = nn.Sequential(nn.Linear(TOKEN_DIM, FFN_HIDDEN_DIM), nn.GELU(), nn.Linear(FFN_HIDDEN_DIM, TOKEN_DIM))
        self.feed_forward_norm = nn.LayerNorm(TOKEN_DIM)

    def forward(self, frontier_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        (attended, _) = self.attention(query=frontier_tokens, key=context_tokens, value=context_tokens, need_weights=False)
        refined = self.attention_norm(frontier_tokens + attended)
        return self.feed_forward_norm(refined + self.feed_forward(refined))

def _normalize_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(theta + torch.pi, 2.0 * torch.pi) - torch.pi

def normalize_theta(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor) or not theta.is_floating_point():
        raise PolicyActionError('theta must be a floating tensor')
    if theta.dtype != torch.float32:
        raise PolicyActionError('theta operations require FP32')
    if not bool(torch.isfinite(theta).all()):
        raise PolicyActionError('theta must be finite')
    return _normalize_angle(theta)

def _validated_masked_logits(frontier_logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(frontier_logits, torch.Tensor) or frontier_logits.dtype != torch.float32:
        raise PolicyActionError('frontier logits must be FP32')
    if not isinstance(candidate_mask, torch.Tensor) or candidate_mask.dtype != torch.bool:
        raise PolicyActionError('candidate mask must be boolean')
    if frontier_logits.ndim != 2 or candidate_mask.shape != frontier_logits.shape:
        raise PolicyActionError('frontier logits and candidate mask shapes must match [B,M]')
    if frontier_logits.device != candidate_mask.device:
        raise PolicyActionError('frontier logits and candidate mask devices must match')
    if not bool(candidate_mask.any(dim=1).all()):
        raise PolicyActionError('each distribution row must contain a valid candidate')
    if not bool(torch.isfinite(frontier_logits).all()):
        raise PolicyActionError('frontier logits must be finite')
    return torch.where(candidate_mask, frontier_logits, torch.full_like(frontier_logits, float('-inf')))

def _gather_candidate(value: torch.Tensor, selected_index: torch.Tensor) -> torch.Tensor:
    return value.gather(1, selected_index.unsqueeze(1)).squeeze(1)

def _safe_theta_mu(*, raw_sin: torch.Tensor, raw_cos: torch.Tensor, recommended_sin: torch.Tensor, recommended_cos: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    threshold = 1e-06
    threshold_squared = threshold * threshold
    raw_norm_squared = raw_sin.square() + raw_cos.square()
    raw_denominator = torch.sqrt(raw_norm_squared.clamp_min(threshold_squared))
    raw_unit_sin = raw_sin / raw_denominator
    raw_unit_cos = raw_cos / raw_denominator
    recommended_norm_squared = recommended_sin.square() + recommended_cos.square()
    recommended_denominator = torch.sqrt(recommended_norm_squared.clamp_min(threshold_squared))
    recommended_unit_sin = recommended_sin / recommended_denominator
    recommended_unit_cos = recommended_cos / recommended_denominator
    recommended_valid = (recommended_norm_squared > threshold_squared) & candidate_mask
    fallback_sin = torch.where(recommended_valid, recommended_unit_sin, torch.zeros_like(recommended_sin))
    fallback_cos = torch.where(recommended_valid, recommended_unit_cos, torch.ones_like(recommended_cos))
    raw_valid = (raw_norm_squared > threshold_squared) & candidate_mask
    unit_sin = torch.where(raw_valid, raw_unit_sin, fallback_sin)
    unit_cos = torch.where(raw_valid, raw_unit_cos, fallback_cos)
    return _normalize_angle(torch.atan2(unit_sin, unit_cos))

def _masked_mean_max(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1)
    mean = torch.where(expanded, tokens, torch.zeros_like(tokens)).sum(dim=1) / count
    maximum = tokens.masked_fill(~expanded, torch.finfo(tokens.dtype).min).amax(dim=1)
    return (mean, maximum)

def _map_position_encoding() -> torch.Tensor:
    quarter = TOKEN_DIM // 4
    coordinates = torch.linspace(-1.0, 1.0, MAP_POOL_SHAPE[0])
    (y, x) = torch.meshgrid(coordinates, coordinates, indexing='ij')
    frequencies = torch.exp(torch.arange(quarter, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / max(quarter - 1, 1)))
    x_phase = x.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * frequencies.reshape(1, -1)
    encoding = torch.cat((x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1)
    return encoding.unsqueeze(0)
