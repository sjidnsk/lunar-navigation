"""Measured upper bound on remaining task observations; no truth inputs.

Callers supply an optimistic set of future native stances, including unknown
transit. UNKNOWN B transmits, so new consistent evidence can only remove rays.
The union uses exactly the sensor's directed finite-beam relation.
"""
import math
import numpy as np
from scipy.ndimage import maximum_filter
import lunar_drl_terrain_native as native
from .sensor import optical_candidate_mask


def remaining_demands(intrinsic, pending, possible_stances, range_cells):
    """Return task-only demand mask U visible from some possible future stance."""
    d=math.ceil(range_cells)
    candidate=maximum_filter(possible_stances,size=2*d+1,mode='constant',cval=0)
    candidate &= pending
    candidate &= optical_candidate_mask(intrinsic,possible_stances)
    visible=np.zeros(pending.shape,np.uint8)
    native.visible_union(intrinsic,np.asarray(possible_stances,np.uint8),
                         np.asarray(candidate,np.uint8),float(range_cells),visible)
    return visible.astype(bool)
