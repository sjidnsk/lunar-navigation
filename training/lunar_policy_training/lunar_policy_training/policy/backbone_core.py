"""Stateless cross-attention policy backbone."""
from __future__ import annotations
import math
from typing import Final
import torch
from torch import nn
TOKEN_DIM: Final = 128
ATTENTION_HEADS: Final = 4
CROSS_ATTENTION_LAYERS: Final = 2
FFN_HIDDEN_DIM: Final = 512
ACTION_HIDDEN_DIM: Final = 128
DROPOUT: Final = 0.0
INVALID_LOGIT_VALUE: Final = -1000000000.0
INITIAL_KAPPA_RAW: Final = math.log(math.expm1(1.0))

class PolicyActionError(ValueError):
    """动作或分布输入违反共享策略合同。"""

class MapEncoder(nn.Module):
    """Encode maps into the contract-selected fixed token grid."""

    def __init__(
        self,
        input_channels: int,
        input_grid: tuple[int, int],
        output_grid: tuple[int, int],
    ) -> None:
        super().__init__()
        if (
            not isinstance(input_grid, tuple)
            or len(input_grid) != 2
            or any(type(size) is not int or size <= 0 for size in input_grid)
            or not isinstance(output_grid, tuple)
            or len(output_grid) != 2
            or any(type(size) is not int or size <= 0 for size in output_grid)
        ):
            raise ValueError("input_grid and output_grid must be positive pairs")
        self.input_channels = input_channels
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.features = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.GELU(), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, TOKEN_DIM, kernel_size=3, padding=1), nn.GroupNorm(16, TOKEN_DIM), nn.GELU())
        feature_grid = tuple((size + 1) // 2 for size in input_grid)
        if any(
            feature_size < output_size
            or feature_size % output_size != 0
            for feature_size, output_size in zip(feature_grid, output_grid)
        ):
            raise ValueError("fixed feature grid must divide into output_grid")
        kernel = tuple(
            feature_size // output_size
            for feature_size, output_size in zip(feature_grid, output_grid)
        )
        self.pool = (
            nn.Identity()
            if kernel == (1, 1)
            else nn.AvgPool2d(kernel_size=kernel, stride=kernel)
        )
        self.register_buffer('position_encoding', _map_position_encoding(output_grid), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-2:]) != self.input_grid:
            raise ValueError("map tensor differs from the fixed encoder grid")
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

def _masked_mean_max(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1)
    mean = torch.where(expanded, tokens, torch.zeros_like(tokens)).sum(dim=1) / count
    maximum = tokens.masked_fill(~expanded, torch.finfo(tokens.dtype).min).amax(dim=1)
    return (mean, maximum)

def _map_position_encoding(output_grid: tuple[int, int]) -> torch.Tensor:
    quarter = TOKEN_DIM // 4
    y_coordinates = torch.linspace(-1.0, 1.0, output_grid[0])
    x_coordinates = torch.linspace(-1.0, 1.0, output_grid[1])
    (y, x) = torch.meshgrid(y_coordinates, x_coordinates, indexing='ij')
    frequencies = torch.exp(torch.arange(quarter, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / max(quarter - 1, 1)))
    x_phase = x.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * frequencies.reshape(1, -1)
    encoding = torch.cat((x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1)
    return encoding.unsqueeze(0)
