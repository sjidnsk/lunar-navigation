"""Bounded single-connection TCP transport for the Unreal vehicle."""

from dataclasses import dataclass
import select
import socket
import threading
import time
from typing import Callable, Optional, Sequence

from .protocol import Feedback, FeedbackParser, encode_command


@dataclass(frozen=True)
class ReceivedFeedback:
    feedback: Feedback
    received_monotonic: float


def _write_all(connection: socket.socket, data: bytes, deadline: float) -> None:
    """Write one complete frame without losing the offset on backpressure."""
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out writing TCP command frame")
        _, writable, _ = select.select([], [connection], [], remaining)
        if not writable:
            raise TimeoutError("timed out writing TCP command frame")
        try:
            written = connection.send(view[offset:])
        except BlockingIOError:
            continue
        if written == 0:
            raise ConnectionError("TCP connection closed while writing command frame")
        offset += written


class TcpVehicleTransport:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        vehicle_id: int = 0,
        command_timeout_s: float = 0.5,
        feedback_timeout_s: float = 0.5,
        send_rate_hz: float = 20.0,
        connect_timeout_s: float = 5.0,
        write_timeout_s: float = 0.5,
        on_feedback: Optional[Callable[[ReceivedFeedback], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        if (
            command_timeout_s <= 0
            or feedback_timeout_s <= 0
            or send_rate_hz <= 0
            or write_timeout_s <= 0
        ):
            raise ValueError("timeouts and send rate must be positive")
        encode_command((0.,)*4, vehicle_id=vehicle_id)
        self._vehicle_id = vehicle_id
        self._host = host
        self._port = int(port)
        self._command_timeout_s = command_timeout_s
        self._feedback_timeout_s = feedback_timeout_s
        self._send_period_s = 1.0 / send_rate_hz
        self._connect_timeout_s = connect_timeout_s
        self._write_timeout_s = write_timeout_s
        self._on_feedback = on_feedback
        self._on_error = on_error
        self._lock = threading.Lock()
        self._command = (0.0, 0.0, 0.0, 0.0)
        self._turn_angles = (0.0,) * 4
        self._command_time = 0.0
        self._latest: ReceivedFeedback | None = None
        self._socket: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("transport already started")
        connection = socket.create_connection(
            (self._host, self._port), timeout=self._connect_timeout_s
        )
        connection.setblocking(False)
        self._socket = connection
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="tcp-vehicle-transport", daemon=True
        )
        self._thread.start()

    def set_wheel_command(self, wheel_speeds: Sequence[float], turn_angles=None) -> None:
        command = tuple(float(value) for value in wheel_speeds)
        if len(command) != 4:
            raise ValueError("wheel command must contain exactly four speeds")
        # encode_command provides the finite-value validation at this boundary.
        with self._lock:
            angles = self._turn_angles if turn_angles is None else tuple(turn_angles)
            encode_command(command, angles, vehicle_id=self._vehicle_id)
            self._turn_angles = angles
            self._command = command  # type: ignore[assignment]
            self._command_time = time.monotonic()

    def latest_feedback(self) -> ReceivedFeedback | None:
        with self._lock:
            return self._latest

    def feedback_is_fresh(self) -> bool:
        latest = self.latest_feedback()
        return (
            latest is not None
            and time.monotonic() - latest.received_monotonic
            <= self._feedback_timeout_s
        )

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        if thread is not threading.current_thread():
            # The worker is the sole writer. Its finally block sends the final
            # zero frame and closes the socket before this barrier returns.
            thread.join()

    def _run(self) -> None:
        parser = FeedbackParser(self._vehicle_id)
        next_send = time.monotonic()
        connection = self._socket
        assert connection is not None
        failure: BaseException | None = None
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                wait_s = max(0.0, min(0.05, next_send - now))
                readable, _, _ = select.select([connection], [], [], wait_s)
                if readable:
                    data = connection.recv(65536)
                    if not data:
                        break
                    parsed = parser.feed(data)
                    if parsed:
                        feedback = parsed[-1]
                        received = ReceivedFeedback(feedback, time.monotonic())
                        with self._lock:
                            self._latest = received
                        if self._on_feedback is not None:
                            self._on_feedback(received)
                if self._stop.is_set():
                    break
                now = time.monotonic()
                if now >= next_send:
                    with self._lock:
                        command = self._command
                        angles = self._turn_angles
                        command_time = self._command_time
                        latest = self._latest
                    feedback_fresh = (
                        latest is not None
                        and now - latest.received_monotonic <= self._feedback_timeout_s
                    )
                    if now - command_time > self._command_timeout_s or not feedback_fresh:
                        command = (0.0, 0.0, 0.0, 0.0)
                    _write_all(
                        connection,
                        encode_command(command, angles, vehicle_id=self._vehicle_id),
                        now + self._write_timeout_s,
                    )
                    next_send = now + self._send_period_s
        except (OSError, ValueError, TimeoutError, ConnectionError) as error:
            failure = error
        finally:
            if self._stop.is_set() and failure is None:
                try:
                    _write_all(
                        connection,
                        encode_command((0.0, 0.0, 0.0, 0.0), self._turn_angles, vehicle_id=self._vehicle_id),
                        time.monotonic() + self._write_timeout_s,
                    )
                except (OSError, TimeoutError, ConnectionError) as error:
                    failure = error
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
            with self._lock:
                self._socket = None
                self._thread = None
            if failure is not None and self._on_error is not None:
                self._on_error(failure)

    def __enter__(self) -> "TcpVehicleTransport":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def read_first_feedback(host: str, port: int, timeout: float, vehicle_id: int = 0) -> Feedback:
    """Receive one valid feedback body without transmitting a vehicle command."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    deadline = time.monotonic() + timeout
    parser = FeedbackParser(vehicle_id)
    with socket.create_connection((host, int(port)), timeout=timeout) as connection:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for valid vehicle feedback")
            connection.settimeout(remaining)
            try:
                data = connection.recv(65536)
            except socket.timeout as error:
                raise TimeoutError("timed out waiting for valid vehicle feedback") from error
            if not data:
                raise ConnectionError("connection closed before valid vehicle feedback")
            parsed = parser.feed(data)
            if parsed:
                return parsed[-1]
