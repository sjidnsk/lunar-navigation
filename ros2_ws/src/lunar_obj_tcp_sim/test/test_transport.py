import socket
import struct
import threading
import time

import pytest

import lunar_obj_tcp_sim.transport as transport_module
from lunar_obj_tcp_sim.transport import TcpVehicleTransport, read_first_feedback


def feedback_bytes(position=(250.0, -125.0, 40.0)):
    values = (
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        *position,
        0.0, 0.0, 0.0, 1.0,
        0.2, 0.0, 0.0,
        0.0, 0.0, 0.1,
    )
    return struct.pack("<4I",0x4C494441,4,92,0x4C494441^4^92)+struct.pack("<II21f",0,0xEB93EB93,*values)


def listening_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server, server.getsockname()[1]


def command_wheels(packet):
    return struct.unpack("<4f", packet[24:40])


def test_duplex_transport_parses_fragmented_feedback_and_zeros_stale_command():
    server, port = listening_socket()
    packets = []
    first_motion = threading.Event()
    stopped = threading.Event()

    def serve():
        connection, _ = server.accept()
        connection.settimeout(1.0)
        wire = b"junk" + feedback_bytes()
        connection.sendall(wire[:5])
        connection.sendall(wire[5:37])
        connection.sendall(wire[37:])
        buffered = bytearray()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                buffered.extend(connection.recv(4096))
            except socket.timeout:
                break
            while len(buffered) >= 57:
                packets.append(bytes(buffered[:57]))
                del buffered[:57]
            if any(command_wheels(item) != (0.,)*4 for item in packets):
                first_motion.set()
            if first_motion.is_set() and packets and command_wheels(packets[-1]) == (0.,)*4:
                stopped.set()
                break
        connection.close()
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    transport = TcpVehicleTransport(
        "127.0.0.1", port, command_timeout_s=0.08, feedback_timeout_s=0.5, send_rate_hz=50.0
    )
    transport.start()
    deadline = time.monotonic() + 1.0
    while transport.latest_feedback() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    received = transport.latest_feedback()
    # Start the command lifetime after connection/feedback, then stop refreshing
    # only once the receiver has observed motion. Startup scheduling is not the
    # 80 ms command timeout under test.
    deadline = time.monotonic() + 1.0
    while not first_motion.is_set() and time.monotonic() < deadline:
        transport.set_wheel_command((1., 2., 1., 2.), (.2, .3, -.2, -.3))
        first_motion.wait(.01)
    stopped.wait(1.0)
    transport.close()
    thread.join(timeout=1.0)

    assert received is not None
    assert received.feedback.position_m == pytest.approx((250., -125., 40.))
    assert any(command_wheels(item) == pytest.approx((1.0, 2.0, 1.0, 2.0)) for item in packets)
    nonzero_index = next(i for i, item in enumerate(packets) if command_wheels(item) != (0.0,) * 4)
    assert all(struct.unpack_from("<4f", p, 40) == pytest.approx((.2,.3,-.2,-.3)) for p in packets[nonzero_index:])
    assert stopped.is_set()
    assert any(command_wheels(item) == (0.0,) * 4 for item in packets[nonzero_index + 1 :])


def test_feedback_freshness_expires_without_restamping_receive_time():
    server, port = listening_socket()
    packets = []

    def serve():
        connection, _ = server.accept()
        connection.sendall(feedback_bytes())
        connection.settimeout(0.2)
        buffered = bytearray()
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            try:
                buffered.extend(connection.recv(4096))
            except socket.timeout:
                break
            while len(buffered) >= 57:
                packets.append(bytes(buffered[:57]))
                del buffered[:57]
        connection.close()
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    transport = TcpVehicleTransport(
        "127.0.0.1", port, command_timeout_s=0.5, feedback_timeout_s=0.05, send_rate_hz=50.0
    )
    transport.set_wheel_command((1.0, 1.0, 1.0, 1.0))
    transport.start()
    deadline = time.monotonic() + 1.0
    while transport.latest_feedback() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    first = transport.latest_feedback()
    time.sleep(0.07)

    assert first is not None
    assert transport.latest_feedback() is first
    assert transport.feedback_is_fresh() is False
    thread.join(timeout=1.0)
    nonzero_index = next(i for i, item in enumerate(packets) if command_wheels(item) == (1.0,) * 4)
    assert any(command_wheels(item) == (0.0,) * 4 for item in packets[nonzero_index + 1 :])
    transport.close()


def test_read_first_feedback_receives_only_and_returns_position():
    server, port = listening_socket()
    peer_received = []

    def serve():
        connection, _ = server.accept()
        connection.sendall(feedback_bytes((1000.0, 2000.0, -300.0)))
        connection.settimeout(0.1)
        try:
            peer_received.append(connection.recv(57))
        except socket.timeout:
            peer_received.append(b"")
        connection.close()
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    feedback = read_first_feedback("127.0.0.1", port, 1.0)
    thread.join(timeout=1.0)

    assert feedback.position_m == pytest.approx((1000.0, 2000.0, -300.0))
    assert peer_received == [b""]


def test_transport_uses_only_latest_feedback_from_a_coalesced_backlog():
    server, port = listening_socket()
    callbacks = []

    def serve():
        connection, _ = server.accept()
        connection.sendall(feedback_bytes((100.0, 0.0, 0.0)) + feedback_bytes((200.0, 0.0, 0.0)))
        time.sleep(0.05)
        connection.close()
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    transport = TcpVehicleTransport("127.0.0.1", port, on_feedback=callbacks.append)
    transport.start()
    deadline = time.monotonic() + 1.0
    while not callbacks and time.monotonic() < deadline:
        time.sleep(0.005)
    transport.close()
    thread.join(timeout=1.0)

    assert len(callbacks) == 1
    assert callbacks[0].feedback.position_m == pytest.approx((200.0, 0.0, 0.0))


def test_shutdown_barrier_makes_zero_the_final_complete_command():
    server, port = listening_socket()
    packets = []
    callback_count = 0
    second_callback_entered = threading.Event()
    release_callback = threading.Event()

    def on_feedback(_feedback):
        nonlocal callback_count
        callback_count += 1
        if callback_count == 2:
            second_callback_entered.set()
            release_callback.wait(timeout=1.0)

    def serve():
        connection, _ = server.accept()
        connection.sendall(feedback_bytes())
        buffered = bytearray()
        connection.settimeout(1.0)
        while not any(command_wheels(item) == (1.0,) * 4 for item in packets):
            buffered.extend(connection.recv(4096))
            while len(buffered) >= 57:
                packets.append(bytes(buffered[:57]))
                del buffered[:57]
        connection.sendall(feedback_bytes((300.0, 0.0, 0.0)))
        while True:
            data = connection.recv(4096)
            if not data:
                break
            buffered.extend(data)
            while len(buffered) >= 57:
                packets.append(bytes(buffered[:57]))
                del buffered[:57]
        connection.close()
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    transport = TcpVehicleTransport("127.0.0.1", port, send_rate_hz=100.0, on_feedback=on_feedback)
    transport.set_wheel_command((1.0,) * 4)
    transport.start()
    assert second_callback_entered.wait(timeout=1.0)
    closer = threading.Thread(target=transport.close)
    closer.start()
    release_callback.set()
    closer.join(timeout=1.0)
    thread.join(timeout=1.0)

    assert not closer.is_alive()
    assert len(packets) >= 2
    assert command_wheels(packets[-1]) == (0.0,) * 4
    assert all(len(packet) == 57 for packet in packets)


def test_write_all_retains_partial_offset_across_would_block(monkeypatch):
    class PartialSocket:
        def __init__(self):
            self.calls = 0
            self.output = bytearray()

        def send(self, data):
            self.calls += 1
            if self.calls == 1:
                count = 7
            elif self.calls == 2:
                raise BlockingIOError()
            else:
                count = len(data)
            self.output.extend(bytes(data[:count]))
            return count

    connection = PartialSocket()
    monkeypatch.setattr(
        transport_module.select,
        "select",
        lambda _read, write, _error, _timeout: ([], write, []),
    )
    first = bytes(range(57))
    second = bytes(reversed(range(57)))

    transport_module._write_all(connection, first, time.monotonic() + 1.0)
    transport_module._write_all(connection, second, time.monotonic() + 1.0)

    assert bytes(connection.output) == first + second


def test_fatal_write_failure_closes_connection_and_surfaces_error(monkeypatch):
    class FatalSocket:
        def __init__(self):
            self.closed = False

        def setblocking(self, _blocking):
            pass

        def send(self, _data):
            raise ConnectionResetError("forced write failure")

        def shutdown(self, _how):
            pass

        def close(self):
            self.closed = True

    connection = FatalSocket()
    errors = []
    monkeypatch.setattr(
        transport_module.socket, "create_connection", lambda *_args, **_kwargs: connection
    )
    monkeypatch.setattr(
        transport_module.select,
        "select",
        lambda _read, write, _error, _timeout: ([], write, []),
    )
    transport = TcpVehicleTransport("127.0.0.1", 6668, on_error=errors.append)
    transport.start()
    deadline = time.monotonic() + 1.0
    while not errors and time.monotonic() < deadline:
        time.sleep(0.005)

    assert connection.closed is True
    assert len(errors) == 1
    assert "forced write failure" in str(errors[0])
