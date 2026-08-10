from pathlib import Path

from lunar_external_adapter.profile import load_interface_profile
from lunar_exploration_policy.launch_support import consumer_remappings


ROOT = Path(__file__).resolve().parents[4]


def test_default_profile_connects_converters_and_direct_provider_topics() -> None:
    profile = load_interface_profile(
        ROOT / "ros2_ws/src/lunar_navigation_config/config/interface_profiles/default.yaml"
    )

    remappings = dict(consumer_remappings(profile))

    assert remappings["/environment/map_global"] == "/environment/map_global"
    assert remappings["/localization/status"] == "/lunar/input/localization_status"
    assert remappings["/mission/exploration_task"] == "/lunar/input/exploration_task"
    assert remappings["/execution/motion_feedback"] == "/lunar/input/motion_execution_feedback"
