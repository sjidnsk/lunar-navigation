from luna_t3_map_adapter.t3_map_adapter_node import shutdown_if_running


def test_shutdown_helper_does_not_repeat_ros_shutdown() -> None:
    calls: list[str] = []

    shutdown_if_running(lambda: False, lambda: calls.append("shutdown"))
    assert calls == []

    shutdown_if_running(lambda: True, lambda: calls.append("shutdown"))
    assert calls == ["shutdown"]
