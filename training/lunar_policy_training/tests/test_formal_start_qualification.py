from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lunar_policy_training.environment.formal_start_qualification import (
    FormalStartQualification,
    _physical_position_at_cell,
)
from lunar_policy_training.environment.platform_reachability import (
    PHYSICAL_PROJECTION_SCHEMA,
    HopperOpportunityAuthority,
    PhysicalReachabilityResult,
)

def test_formal_start_qualification_requires_a_nonempty_physical_universe() -> None:
    qualification = FormalStartQualification((12, 34), 7)

    assert qualification.cell == (12, 34)
    assert qualification.initial_candidate_count == 7

    with pytest.raises(ValueError, match="invalid"):
        FormalStartQualification((12, 34), 0)


def test_hopper_start_position_comes_from_certified_landing_authority() -> None:
    direct = np.asarray([[False, True], [False, False]], dtype=np.bool_)
    certified = np.asarray([[True, True], [False, False]], dtype=np.bool_)
    start_position = (1.0, 2.0, 3.0)
    target_position = (5.0, 6.0, 7.0)
    authority = HopperOpportunityAuthority(
        bridge=object(),
        context=SimpleNamespace(
            direct=np.ascontiguousarray(np.flipud(direct)),
            algorithm_id="test/hopper-opportunity/v1",
        ),
        certified_mask=certified,
        certified_positions_m=np.asarray(
            (start_position, target_position), dtype=np.float64
        ),
    )
    physical = PhysicalReachabilityResult(
        platform_type="HOPPER",
        physical_observation_pose_mask=direct,
        observation_positions_m=np.asarray(
            (target_position,), dtype=np.float64
        ),
        physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
        physical_reachability_algorithm_id="test/hopper-reachability/v1",
        physical_evidence_algorithm_id="test/hopper-evidence/v1",
        physical_safe_pose_count=2,
        physically_reachable_pose_count=1,
        hopper_opportunity_authority=authority,
    )

    assert _physical_position_at_cell(physical, (0, 0)) == start_position
