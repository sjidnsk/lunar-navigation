"""Observed-only baseline candidate selection mathematical core."""
from __future__ import annotations
import math
from typing import Final
import numpy as np
BASELINE_METHODS: Final = ('random_valid_frontier', 'nearest_frontier', 'max_potential_gain_frontier', 'gain_over_cost_frontier')
ALL_METHODS: Final = (*BASELINE_METHODS, 'ppo_policy')
_DISTANCE_FEATURE: Final = 2
_POTENTIAL_GAIN_FEATURE: Final = 5
_RECOMMENDED_THETA_SIN_FEATURE: Final = 14
_RECOMMENDED_THETA_COS_FEATURE: Final = 15
_REACHABLE_COST_FEATURE: Final = 18
_MIN_DIRECTION_NORM: Final = 1e-12

class BaselineSelectionError(ValueError):
    """A baseline decision input violates the fail-closed contract."""

class NoCandidateAction(BaselineSelectionError):
    """No action may be fabricated when the shared valid set is empty."""

def reconstruct_recommended_theta(feature_row: np.ndarray) -> float:
    """Recover the exact candidate direction from its frozen sine/cosine pair."""
    row = np.asarray(feature_row)
    if row.ndim != 1 or row.size <= _RECOMMENDED_THETA_COS_FEATURE:
        raise BaselineSelectionError('recommended theta feature row is incomplete')
    sine = float(row[_RECOMMENDED_THETA_SIN_FEATURE])
    cosine = float(row[_RECOMMENDED_THETA_COS_FEATURE])
    if not math.isfinite(sine) or not math.isfinite(cosine) or math.hypot(sine, cosine) <= _MIN_DIRECTION_NORM:
        raise BaselineSelectionError('recommended theta is non-finite or near-zero')
    return math.atan2(sine, cosine)

def validate_selected_index(index: int, candidate_mask: np.ndarray) -> int:
    """Reject negative, masked, boolean, and out-of-range candidate indices."""
    mask = np.asarray(candidate_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise BaselineSelectionError('candidate index mask must be one-dimensional boolean')
    if type(index) is not int or not 0 <= index < mask.size or (not bool(mask[index])):
        raise BaselineSelectionError('selected candidate index is invalid')
    return index
