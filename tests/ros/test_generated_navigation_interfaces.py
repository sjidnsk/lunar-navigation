"""Checks for generated provisional lunar navigation Python messages."""

import pytest
from lunar_navigation_msgs.msg import (
    ExplorationTask,
    LocalizationStatus,
    ScienceTargetRegion,
)


def test_localization_constants_and_fields():
    assert LocalizationStatus.UNKNOWN == 0
    assert LocalizationStatus.VALID == 1
    assert LocalizationStatus.DEGRADED == 2
    assert LocalizationStatus.INVALID == 3
    assert LocalizationStatus.RELOCALIZING == 4
    assert LocalizationStatus.get_fields_and_field_types() == {
        "header": "std_msgs/Header",
        "status": "uint8",
    }


def test_exploration_state_zero_is_not_defined_and_regions_are_bounded():
    assert (
        ExplorationTask.ACTIVE,
        ExplorationTask.PAUSED,
        ExplorationTask.CANCELED,
    ) == (1, 2, 3)
    assert ExplorationTask.get_fields_and_field_types()["science_regions"] == (
        "sequence<lunar_navigation_msgs/ScienceTargetRegion, 64>"
    )
    with pytest.raises(AssertionError):
        ExplorationTask(science_regions=[ScienceTargetRegion() for _ in range(65)])
