"""从一个完整接口 profile 生成 planner/policy 的固定输入重映射。"""

from __future__ import annotations

from lunar_external_adapter.profile import InterfaceProfile


_CONSUMER_TOPICS = {
    "map_global": "/environment/map_global",
    "map_local": "/environment/map_local",
    "odometry": "/localization/odometry",
    "localization_status": "/localization/status",
    "exploration_task": "/mission/exploration_task",
    "motion_execution_feedback": "/execution/motion_feedback",
    "tf": "/tf",
}


def consumer_remappings(profile: InterfaceProfile) -> tuple[tuple[str, str], ...]:
    """标准消息直达 provider；自定义消息接显式 converter 的输出。"""
    if not isinstance(profile, InterfaceProfile):
        raise TypeError("profile must use InterfaceProfile")
    result = []
    for name in sorted(_CONSUMER_TOPICS):
        channel = profile.channels[name]
        target = (
            channel.input_topic
            if channel.mode == "direct_remap"
            else channel.output_topic
        )
        result.append((_CONSUMER_TOPICS[name], target))
    return tuple(result)


__all__ = ["consumer_remappings"]
