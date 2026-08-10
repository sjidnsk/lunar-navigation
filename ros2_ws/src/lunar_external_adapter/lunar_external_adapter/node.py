"""ROS 2 lifecycle node that runs only explicit message converters."""

from __future__ import annotations

from typing import Any

import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosidl_runtime_py.utilities import get_message

from .conversions import ConversionError
from .profile import (
    DEFAULT_RUNTIME_PATH,
    load_interface_profile,
    resolve_converter,
)


class ExternalAdapter(LifecycleNode):
    """Convert provisional messages into the stable internal boundary."""

    def __init__(self) -> None:
        super().__init__("lunar_external_adapter")
        self.declare_parameter(
            "interface_profile_file", str(DEFAULT_RUNTIME_PATH)
        )
        self._publishers: list[Any] = []
        self._subscriptions: list[Any] = []

    @staticmethod
    def _channel_qos(channel_name: str) -> QoSProfile:
        """返回固定内部 QoS；外部项目 YAML 不得改变 DDS 语义。"""
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        if channel_name == "exploration_task":
            qos.depth = 1
            qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        return qos

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        del state
        try:
            profile_path = self.get_parameter(
                "interface_profile_file"
            ).get_parameter_value().string_value
            profile = load_interface_profile(profile_path)
            for channel_name, channel in profile.channels.items():
                qos = self._channel_qos(channel_name)
                input_type = get_message(channel.input_type)
                output_type = get_message(channel.output_type)
                if channel.mode == "direct_remap":
                    continue
                converter = resolve_converter(channel.converter or "")
                publisher = self.create_lifecycle_publisher(
                    output_type, channel.output_topic, qos
                )

                def callback(
                    message: Any,
                    *,
                    name: str = channel_name,
                    convert=converter,
                    expected_type=output_type,
                    output=publisher,
                ) -> None:
                    try:
                        converted = convert(message)
                        if not isinstance(converted, expected_type):
                            raise ConversionError(
                                "converter returned an unexpected ROS "
                                "message type"
                            )
                        output.publish(converted)
                    except Exception as error:
                        header = getattr(message, "header", None)
                        stamp = getattr(header, "stamp", None)
                        self.get_logger().warning(
                            f"channel={name} "
                            "reason_code=EXTERNAL_MESSAGE_REJECTED "
                            f"stamp={getattr(stamp, 'sec', 'unknown')}."
                            f"{getattr(stamp, 'nanosec', 'unknown')} "
                            f"detail={error}"
                        )

                subscription = self.create_subscription(
                    input_type, channel.input_topic, callback, qos
                )
                self._publishers.append(publisher)
                self._subscriptions.append(subscription)
            return TransitionCallbackReturn.SUCCESS
        except Exception as error:
            self.get_logger().error(f"adapter configure failed: {error}")
            self._destroy_runtime_entities()
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        del state
        self._destroy_runtime_entities()
        return TransitionCallbackReturn.SUCCESS

    def _destroy_runtime_entities(self) -> None:
        for subscription in self._subscriptions:
            self.destroy_subscription(subscription)
        for publisher in self._publishers:
            self.destroy_publisher(publisher)
        self._subscriptions.clear()
        self._publishers.clear()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ExternalAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
