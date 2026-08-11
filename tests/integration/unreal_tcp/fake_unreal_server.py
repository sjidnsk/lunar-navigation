#!/usr/bin/env python3
"""Repository-local lunar-unreal-tcp/v1 server used only for ROS-side tests."""

from __future__ import annotations

import argparse
import json
import select
import socket
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass


MAGIC = b"LNT1"
HEADER_SIZE = 48
PROTOCOL_VERSION = 1
MAXIMUM_BODY = 8 * 1024 * 1024

HELLO = 0x0001
HELLO_ACK = 0x0002
CONTROL = 0x0010
CONTROL_ACK = 0x0011
ROBOT_STATE = 0x0020
LOCAL_ELEVATION_MAP = 0x0021
MOTION_REFERENCE = 0x0030
EXECUTION_FEEDBACK = 0x0031
HEARTBEAT = 0x0040

HEADER = struct.Struct("<4sHHHHQqIIIII")
MAP_CELL_COUNT = 320 * 320
MAP_PAYLOAD = struct.pack("<f", 0.0) * MAP_CELL_COUNT + b"\xff" * (
    MAP_CELL_COUNT // 8
)
IDENTITY = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


@dataclass(frozen=True)
class Frame:
    message_type: int
    sequence: int
    simulation_time_ns: int
    metadata: dict
    payload: bytes = b""


def encode_frame(frame: Frame) -> bytes:
    metadata = json.dumps(
        frame.metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    flags = 0x01 | (0x02 if frame.payload else 0)
    body = metadata + frame.payload
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
    header = HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        HEADER_SIZE,
        frame.message_type,
        flags,
        frame.sequence,
        frame.simulation_time_ns,
        len(metadata),
        len(frame.payload),
        body_crc,
        0,
        0,
    )
    header_crc = zlib.crc32(header) & 0xFFFFFFFF
    header = bytearray(header)
    struct.pack_into("<I", header, 40, header_crc)
    return bytes(header) + body


def decode_available(buffer: bytearray) -> list[Frame]:
    frames: list[Frame] = []
    while len(buffer) >= HEADER_SIZE:
        fields = HEADER.unpack_from(buffer)
        (
            magic,
            version,
            header_size,
            message_type,
            flags,
            sequence,
            simulation_time_ns,
            metadata_size,
            payload_size,
            body_crc,
            header_crc,
            reserved,
        ) = fields
        if magic != MAGIC or version != 1 or header_size != 48:
            raise ValueError("INVALID_HEADER")
        if flags not in (1, 3) or reserved != 0:
            raise ValueError("INVALID_FLAGS_OR_RESERVED")
        body_size = metadata_size + payload_size
        if metadata_size > 65_536 or body_size > MAXIMUM_BODY:
            raise ValueError("BODY_LIMIT")
        if len(buffer) < HEADER_SIZE + body_size:
            break
        header_copy = bytearray(buffer[:HEADER_SIZE])
        struct.pack_into("<I", header_copy, 40, 0)
        if zlib.crc32(header_copy) & 0xFFFFFFFF != header_crc:
            raise ValueError("HEADER_CRC")
        body = bytes(buffer[HEADER_SIZE : HEADER_SIZE + body_size])
        if zlib.crc32(body) & 0xFFFFFFFF != body_crc:
            raise ValueError("BODY_CRC")
        metadata = json.loads(body[:metadata_size].decode("utf-8"))
        frames.append(
            Frame(
                message_type=message_type,
                sequence=sequence,
                simulation_time_ns=simulation_time_ns,
                metadata=metadata,
                payload=body[metadata_size:],
            )
        )
        del buffer[: HEADER_SIZE + body_size]
    return frames


class FakeUnrealServer:
    def __init__(self, host: str, port: int, control_port: int) -> None:
        self.host = host
        self.port = port
        self.control_port = control_port
        self._stop = threading.Event()
        self._disconnect = threading.Event()
        self._wrong_session = threading.Event()
        self._wrong_plan = threading.Event()
        self._complete_active = threading.Event()
        self._pause_streams = threading.Event()
        self._skew_once = threading.Event()
        self._connection_count = 0
        self._lock = threading.Lock()

    def run(self) -> None:
        control = threading.Thread(target=self._control_loop, daemon=True)
        control.start()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(1)
            listener.settimeout(0.2)
            while not self._stop.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    connection.settimeout(0.05)
                    with self._lock:
                        self._connection_count += 1
                    try:
                        self._serve_session(connection)
                    except (ConnectionError, OSError, ValueError):
                        pass
                self._disconnect.clear()
                self._wrong_plan.clear()
                self._complete_active.clear()
                self._pause_streams.clear()
                self._skew_once.clear()
        self._stop.set()
        control.join(timeout=1.0)

    def _control_loop(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
            control.bind((self.host, self.control_port))
            control.settimeout(0.2)
            while not self._stop.is_set():
                try:
                    command = control.recv(128).decode("ascii").strip().upper()
                except socket.timeout:
                    continue
                if command == "WRONG_SESSION":
                    self._wrong_session.set()
                elif command == "WRONG_PLAN":
                    self._wrong_plan.set()
                elif command == "COMPLETE_ACTIVE":
                    self._complete_active.set()
                elif command == "STALE":
                    self._pause_streams.set()
                elif command == "SKEW":
                    self._skew_once.set()
                elif command == "DISCONNECT":
                    self._disconnect.set()
                elif command == "STOP":
                    self._stop.set()
                    self._disconnect.set()

    def _serve_session(self, connection: socket.socket) -> None:
        session_id = str(uuid.uuid4())
        receive_buffer = bytearray()
        incoming_sequence = 0
        outgoing_sequence = 0
        simulation_time_ns = 1_000_000_000
        handshake_deadline = time.monotonic() + 5.0
        while time.monotonic() < handshake_deadline:
            try:
                chunk = connection.recv(65_536)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("CLIENT_CLOSED_DURING_HELLO")
            receive_buffer.extend(chunk)
            frames = decode_available(receive_buffer)
            if not frames:
                continue
            hello = frames[0]
            if hello.message_type != HELLO or hello.sequence != 1:
                raise ValueError("HELLO_REQUIRED")
            incoming_sequence = hello.sequence
            break
        else:
            raise ConnectionError("HELLO_TIMEOUT")

        outgoing_sequence += 1
        self._send(
            connection,
            Frame(
                HELLO_ACK,
                outgoing_sequence,
                simulation_time_ns,
                {
                    "session_id": session_id,
                    "scene_id": "fake-flat-observed-scene",
                    "robot_id": "wheeled-1",
                    "engine_version": "5.0.1",
                    "agx_plugin_version": "UNKNOWN",
                    "coordinate_convention": "UE_NATIVE",
                    "length_unit_to_m": 0.01,
                    "handedness": "LEFT",
                    "up_axis": "Z",
                    "T_base_link_from_unreal_root": IDENTITY,
                    "T_base_footprint_from_base_link": IDENTITY,
                    "local_map_width": 320,
                    "local_map_height": 320,
                    "resolution_native_cm": 20.0,
                    "server_nonce": "fake-server",
                    "maximum_body_bytes": MAXIMUM_BODY,
                    "calibration_hash": "sha256:unreal-tcp-wheeled-v1",
                },
            ),
        )

        robot_position_cm = [0.0, 0.0, 0.0]
        next_state = 0.0
        next_map = 0.0
        next_heartbeat = 0.0
        feedback_queue: list[tuple[float, str, int, str]] = []
        active_plan_id: str | None = None
        receive_buffer.clear()
        while not self._stop.is_set() and not self._disconnect.is_set():
            now = time.monotonic()
            if self._skew_once.is_set():
                self._skew_once.clear()
                simulation_time_ns += 250_000_000
                outgoing_sequence += 1
                self._send(
                    connection,
                    self._robot_state(
                        outgoing_sequence,
                        simulation_time_ns,
                        session_id,
                        robot_position_cm,
                    ),
                )
                self._pause_streams.set()
            if not self._pause_streams.is_set() and now >= next_state:
                simulation_time_ns += 50_000_000
                outgoing_sequence += 1
                self._send(
                    connection,
                    self._robot_state(
                        outgoing_sequence,
                        simulation_time_ns,
                        session_id,
                        robot_position_cm,
                    ),
                )
                next_state = now + 0.05
            if not self._pause_streams.is_set() and now >= next_map:
                outgoing_sequence += 1
                self._send(
                    connection,
                    self._local_map(
                        outgoing_sequence, simulation_time_ns, session_id
                    ),
                )
                next_map = now + 0.2
            if now >= next_heartbeat:
                outgoing_sequence += 1
                heartbeat_session = session_id
                if self._wrong_session.is_set():
                    heartbeat_session = "00000000-0000-4000-8000-000000000000"
                    self._wrong_session.clear()
                    self._pause_streams.set()
                self._send(
                    connection,
                    Frame(
                        HEARTBEAT,
                        outgoing_sequence,
                        simulation_time_ns,
                        {
                            "session_id": heartbeat_session,
                            "last_received_sequence": incoming_sequence,
                        },
                    ),
                )
                next_heartbeat = now + 1.0

            due = [item for item in feedback_queue if item[0] <= now]
            feedback_queue = [item for item in feedback_queue if item[0] > now]
            for _, plan_id, feedback_sequence, state in due:
                outgoing_sequence += 1
                self._send(
                    connection,
                    self._feedback(
                        outgoing_sequence,
                        simulation_time_ns,
                        session_id,
                        plan_id,
                        feedback_sequence,
                        state,
                    ),
                )
                if state == "SEGMENT_COMPLETE":
                    active_plan_id = None

            if self._complete_active.is_set() and active_plan_id is not None:
                self._complete_active.clear()
                feedback_queue.append(
                    (now + 0.10, active_plan_id, 3, "SEGMENT_COMPLETE")
                )

            if self._wrong_plan.is_set() and active_plan_id is not None:
                self._wrong_plan.clear()
                feedback_queue.clear()
                outgoing_sequence += 1
                self._send(
                    connection,
                    self._feedback(
                        outgoing_sequence,
                        simulation_time_ns,
                        session_id,
                        "00000000-0000-4000-8000-000000000000",
                        3,
                        "EXECUTING",
                    ),
                )
                active_plan_id = None
                self._pause_streams.set()

            readable, _, _ = select.select([connection], [], [], 0.01)
            if not readable:
                continue
            try:
                chunk = connection.recv(1_048_576)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("CLIENT_CLOSED")
            receive_buffer.extend(chunk)
            for frame in decode_available(receive_buffer):
                incoming_sequence = frame.sequence
                if frame.message_type == CONTROL:
                    outgoing_sequence += 1
                    self._send(
                        connection,
                        Frame(
                            CONTROL_ACK,
                            outgoing_sequence,
                            simulation_time_ns,
                            {
                                "session_id": session_id,
                                "command_id": frame.metadata["command_id"],
                                "status": "OK",
                                "reason_code": "",
                            },
                        ),
                    )
                elif frame.message_type == MOTION_REFERENCE:
                    plan_id = frame.metadata["plan_id"]
                    active_plan_id = plan_id
                    feedback_queue.extend(
                        [
                            (now + 0.02, plan_id, 1, "ACCEPTED"),
                            (now + 0.04, plan_id, 2, "EXECUTING"),
                        ]
                    )

    @staticmethod
    def _send(connection: socket.socket, frame: Frame) -> None:
        connection.sendall(encode_frame(frame))

    @staticmethod
    def _robot_state(
        sequence: int,
        simulation_time_ns: int,
        session_id: str,
        position_cm: list[float],
    ) -> Frame:
        return Frame(
            ROBOT_STATE,
            sequence,
            simulation_time_ns,
            {
                "session_id": session_id,
                "position_cm": position_cm,
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity_cmps": [0.0, 0.0, 0.0],
                "angular_velocity_radps": [0.0, 0.0, 0.0],
                "control_state": "READY",
                "simulation_covariance_profile": "deterministic",
            },
        )

    @staticmethod
    def _local_map(sequence: int, simulation_time_ns: int, session_id: str) -> Frame:
        pose = {
            "position_cm": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        return Frame(
            LOCAL_ELEVATION_MAP,
            sequence,
            simulation_time_ns,
            {
                "session_id": session_id,
                "width": 320,
                "height": 320,
                "resolution_cm": 20.0,
                "cell_zero_center_world_cm": [-3200.0, 3200.0, 0.0],
                "u_axis_world": [1.0, 0.0, 0.0],
                "v_axis_world": [0.0, -1.0, 0.0],
                "sensor_pose_world": pose,
                "robot_pose_world": pose,
                "elevation_encoding": "float32_le",
                "valid_encoding": "bitset_lsb0",
            },
            MAP_PAYLOAD,
        )

    @staticmethod
    def _feedback(
        sequence: int,
        simulation_time_ns: int,
        session_id: str,
        plan_id: str,
        feedback_sequence: int,
        state: str,
    ) -> Frame:
        return Frame(
            EXECUTION_FEEDBACK,
            sequence,
            simulation_time_ns,
            {
                "session_id": session_id,
                "platform_type": "WHEELED",
                "plan_id": plan_id,
                "segment_id": plan_id,
                "sequence": feedback_sequence,
                "state": state,
                "reason_code": "" if state != "CANCELED" else "USER_HOLD",
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--control-port", type=int, required=True)
    args = parser.parse_args()
    FakeUnrealServer(args.host, args.port, args.control_port).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
