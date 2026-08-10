from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lunar_policy_training.environment.coverability import (
    build_coverable_detail_mask,
    build_mission_target_detail_mask,
    CoverabilityError,
    IneligibleReason,
    PlatformCoverability,
    classify_ineligibility,
    mask_sha256,
    pack_detail_mask,
    unpack_detail_mask,
)
from lunar_policy_training.training_semantics import (
    FORMAL_TRAINING_SEMANTICS_VERSION,
    TRAINING_SEMANTICS_VERSION,
)


_V6 = (
    "lunar-training-semantics/"
    "sensor-30m-360-platform-coverable-detail95-unbounded-per-platform-subset/v6"
)


def test_formal_training_semantics_names_platform_coverable_v6() -> None:
    """Would fail if a new run retained the old full-ROI/common-subset identity."""
    assert FORMAL_TRAINING_SEMANTICS_VERSION == _V6
    assert TRAINING_SEMANTICS_VERSION == _V6


def test_detail_mask_pack_is_row_major_and_rejects_nonzero_padding() -> None:
    """Would fail if a cache could reinterpret bit order or hide cells in padding."""
    mask = np.asarray(
        [
            [True, False, True],
            [False, True, False],
            [True, True, False],
        ],
        dtype=np.bool_,
    )

    packed = pack_detail_mask(mask)

    assert packed.dtype == np.uint8
    assert packed.flags.c_contiguous
    assert packed.tolist() == [0b10101011, 0b00000000]
    assert np.array_equal(unpack_detail_mask(packed, mask.shape), mask)

    corrupted = packed.copy()
    corrupted[-1] |= np.uint8(1)
    with pytest.raises(CoverabilityError, match="padding"):
        unpack_detail_mask(corrupted, mask.shape)


def _eligible_coverability() -> PlatformCoverability:
    reachable = np.asarray(
        [[True, False], [True, True]],
        dtype=np.bool_,
    )
    detail = np.asarray(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, True],
            [False, False, True, True],
        ],
        dtype=np.bool_,
    )
    ratio = np.asarray(
        [[0.75, 0.0], [0.5, 1.0]],
        dtype=np.float32,
    )
    return PlatformCoverability(
        platform_type="WHEELED",
        qualified_start_cell=(0, 0),
        reachable_pose_mask=reachable,
        coverable_detail_shape=detail.shape,
        coverable_detail_bits=pack_detail_mask(detail),
        coverable_ratio=ratio,
        mission_target_detail_cell_count=9,
        coverable_detail_cell_count=9,
        mission_coverable_fraction=1.0,
        initial_coverable_fraction=3.0 / 9.0,
        initial_candidate_count=2,
        reachability_algorithm_id="ground-start-component/v1",
        visibility_algorithm_id="two-dimensional-detail-los/v1",
        reachable_mask_sha256=mask_sha256(reachable),
        coverable_mask_sha256=mask_sha256(detail),
        exact=True,
        eligible=True,
        ineligible_reason=None,
    )


def test_platform_coverability_validates_exact_masks_counts_hashes_and_ratio() -> None:
    """Would fail if cache metadata could disagree with its exact detail bits."""
    value = _eligible_coverability()

    assert value.coverable_detail_cell_count == 9
    assert value.reachable_pose_mask.dtype == np.bool_
    assert value.coverable_detail_mask.tolist() == [
        [True, True, False, False],
        [True, False, False, False],
        [True, True, True, True],
        [False, False, True, True],
    ]

    with pytest.raises(CoverabilityError, match="coverable mask hash"):
        replace(value, coverable_mask_sha256="f" * 64)
    with pytest.raises(CoverabilityError, match="coverable ratio"):
        replace(value, coverable_ratio=np.zeros((2, 2), np.float32))
    with pytest.raises(CoverabilityError, match="exact"):
        replace(value, exact=False)


@pytest.mark.parametrize(
    ("start", "target", "coverable", "initial", "candidates", "expected"),
    [
        (None, 100, 100, 0.10, 1, IneligibleReason.UNSAFE_START),
        ((1, 1), 0, 0, 0.0, 0, IneligibleReason.ZERO_MISSION_TARGET),
        (
            (1, 1),
            100,
            94,
            0.10,
            1,
            IneligibleReason.MISSION_COVERABLE_BELOW_95,
        ),
        (
            (1, 1),
            100,
            95,
            0.95,
            0,
            IneligibleReason.INITIAL_ALREADY_SUCCESS,
        ),
        ((1, 1), 100, 95, 0.10, 0, IneligibleReason.NO_INITIAL_CANDIDATE),
        ((1, 1), 100, 95, 0.10, 1, None),
    ],
)
def test_ineligibility_uses_one_stable_first_failure(
    start: tuple[int, int] | None,
    target: int,
    coverable: int,
    initial: float,
    candidates: int,
    expected: IneligibleReason | None,
) -> None:
    """Would fail if scheduling could change reason by evaluating gates out of order."""
    assert classify_ineligibility(
        qualified_start_cell=start,
        mission_target_detail_cell_count=target,
        coverable_detail_cell_count=coverable,
        initial_coverable_fraction=initial,
        initial_candidate_count=candidates,
    ) is expected


def test_platform_coverability_rejects_eligibility_that_disagrees_with_gates() -> None:
    """Would fail if a manifest could label a below-threshold task eligible."""
    value = _eligible_coverability()

    with pytest.raises(CoverabilityError, match="eligibility"):
        replace(
            value,
            mission_target_detail_cell_count=10,
            mission_coverable_fraction=0.9,
        )
    with pytest.raises(CoverabilityError, match="ineligible reason"):
        replace(
            value,
            initial_candidate_count=0,
            eligible=False,
            ineligible_reason=IneligibleReason.UNSAFE_START,
        )


def test_mission_target_excludes_obstacle_forbidden_invalid_and_intrinsic_faults() -> None:
    """Would fail if observed evidence and success-target semantics were conflated."""
    inside_roi = np.ones((3, 4), dtype=np.bool_)
    valid = np.ones((3, 4), dtype=np.bool_)
    forbidden = np.zeros((3, 4), dtype=np.float32)
    obstacle = np.zeros((3, 4), dtype=np.float32)
    intrinsic = np.ones((3, 4), dtype=np.bool_)
    inside_roi[0, 0] = False
    valid[0, 1] = False
    forbidden[0, 2] = 0.01
    obstacle[0, 3] = 0.01
    intrinsic[1, 0] = False

    target = build_mission_target_detail_mask(
        inside_mission_roi=inside_roi,
        detail_valid=valid,
        forbidden_ratio=forbidden,
        physical_obstacle_ratio=obstacle,
        intrinsic_terrain_feasible=intrinsic,
    )

    assert target.dtype == np.bool_
    assert target.flags.c_contiguous
    assert not target[0].any()
    assert not target[1, 0]
    assert target[1, 1:].all()
    assert target[2].all()


def test_coverable_union_keeps_visible_nonoccupiable_cell_and_excludes_island() -> None:
    """Would fail if coverability meant occupiable cells instead of visible targets."""
    target = np.ones((6, 6), dtype=np.bool_)
    reachable = np.zeros((3, 3), dtype=np.bool_)
    reachable[0, 0] = True
    visible = np.zeros((6, 6), dtype=np.bool_)
    visible[:4, :4] = True
    visible[2, 2] = True  # target need not itself be a reachable pose
    visible[5, 5] = False  # enclosed/invisible free island
    calls: list[tuple[int, int]] = []

    def reveal(
        truth_obstacle_ratio: np.ndarray,
        pose_cell: tuple[int, int],
    ) -> np.ndarray:
        assert truth_obstacle_ratio.shape == target.shape
        calls.append(pose_cell)
        return visible.copy()

    result = build_coverable_detail_mask(
        mission_target_detail_mask=target,
        truth_obstacle_ratio=np.zeros(target.shape, dtype=np.float32),
        reachable_pose_mask=reachable,
        reveal_from_pose=reveal,
    )

    assert calls == [(1, 1)]
    assert result[2, 2]
    assert not result[5, 5]


def test_coverable_union_intersects_los_with_target_and_is_repeat_identical() -> None:
    """Would fail if blocked/obstacle cells leaked into the exact packed denominator."""
    target = np.ones((4, 4), dtype=np.bool_)
    target[1, 2] = False
    reachable = np.asarray([[True, False], [False, True]], dtype=np.bool_)

    def reveal(
        _truth_obstacle_ratio: np.ndarray,
        pose_cell: tuple[int, int],
    ) -> np.ndarray:
        output = np.zeros((4, 4), dtype=np.bool_)
        if pose_cell == (1, 1):
            output[0:2, 0:3] = True
        elif pose_cell == (3, 3):
            output[2:4, 2:4] = True
        return output

    arguments = {
        "mission_target_detail_mask": target,
        "truth_obstacle_ratio": np.zeros(target.shape, dtype=np.float32),
        "reachable_pose_mask": reachable,
        "reveal_from_pose": reveal,
    }
    first = build_coverable_detail_mask(**arguments)
    second = build_coverable_detail_mask(**arguments)

    assert not first[1, 2]
    assert np.array_equal(first, second)
    assert np.array_equal(pack_detail_mask(first), pack_detail_mask(second))
    assert mask_sha256(first) == mask_sha256(second)
