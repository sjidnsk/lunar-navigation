from __future__ import annotations

import pytest

from lunar_policy_training.environment.formal_start_qualification import (
    FormalStartQualification,
)

def test_formal_start_qualification_requires_a_nonempty_physical_universe() -> None:
    qualification = FormalStartQualification((12, 34), 7)

    assert qualification.cell == (12, 34)
    assert qualification.initial_candidate_count == 7

    with pytest.raises(ValueError, match="invalid"):
        FormalStartQualification((12, 34), 0)
