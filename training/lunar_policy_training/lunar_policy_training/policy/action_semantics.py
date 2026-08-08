"""Platform-derived action semantics shared by training and planner adapters."""

from __future__ import annotations

import math
from typing import Protocol

import torch

from lunar_model_contract.observation import PLATFORM_CONTEXTS


class GoalThetaFields(Protocol):
    yaw_rad: float | None
    yaw_tolerance_rad: float


def theta_action_mask(platform_context: torch.Tensor) -> torch.Tensor:
    """Return rows where absolute terminal body yaw is an active action."""
    if (
        not isinstance(platform_context, torch.Tensor)
        or platform_context.dtype != torch.float32
        or platform_context.ndim != 2
        or platform_context.shape[0] < 1
        or platform_context.shape[1] != 3
    ):
        raise ValueError("platform_context must be float32 [B,3]")
    valid_rows = torch.logical_and(
        torch.logical_and(
            torch.isfinite(platform_context),
            torch.logical_or(
                platform_context == 0.0,
                platform_context == 1.0,
            ),
        ).all(dim=1),
        platform_context.sum(dim=1) == 1.0,
    )
    if not bool(valid_rows.all()):
        raise ValueError("platform_context must be finite one-hot")
    return platform_context[:, 2] == 0.0


def apply_goal_theta(
    goal: GoalThetaFields,
    platform_type: str,
    theta_rad: float,
) -> None:
    """Apply terminal yaw only for platforms whose capability makes it active."""
    if platform_type not in PLATFORM_CONTEXTS:
        raise ValueError("platform type must be WHEELED, LEGGED, or HOPPER")
    if (
        not isinstance(theta_rad, (int, float))
        or isinstance(theta_rad, bool)
        or not math.isfinite(float(theta_rad))
    ):
        raise ValueError("theta must be finite")
    if platform_type == "HOPPER":
        goal.yaw_rad = None
        goal.yaw_tolerance_rad = 0.0
    else:
        goal.yaw_rad = float(theta_rad)
        goal.yaw_tolerance_rad = math.pi / 24.0


__all__ = ["GoalThetaFields", "apply_goal_theta", "theta_action_mask"]
