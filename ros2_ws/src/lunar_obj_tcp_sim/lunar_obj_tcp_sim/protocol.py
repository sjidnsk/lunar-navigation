"""Wire protocol shared by the TCP transport and ROS bridge."""

from dataclasses import dataclass
import math
import struct
from typing import Iterable, Sequence


COMMAND_MAGIC = 0x4C494441
COMMAND_ID = 3
CONTROL_HEAD = 0xEB92EB92
FEEDBACK_HEAD = 0xEB93EB93
CONTROL_PAYLOAD_SIZE = 41
COMMAND_PACKET_SIZE = 57
FEEDBACK_SIZE = 108

_FEEDBACK_SIGNATURE = struct.pack("<I", COMMAND_MAGIC)
_FEEDBACK_STRUCT = struct.Struct("<I21f")


@dataclass(frozen=True)
class Feedback:
    wheel_speeds: tuple[float, float, float, float]
    turn_angles: tuple[float, float, float, float]
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    vehicle_id: int = 0


def _four_finite(values: Iterable[float], name: str) -> tuple[float, float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result  # type: ignore[return-value]


def encode_command(
    wheel_speeds: Sequence[float],
    turn_angles: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    *, vehicle_id: int = 0,
) -> bytes:
    """Encode TCP header + FVehicleControlPacket (57 bytes)."""
    if not isinstance(vehicle_id, int) or not 0 <= vehicle_id <= 0xffffffff:
        raise ValueError("vehicle_id must be a uint32")
    speeds = _four_finite(wheel_speeds, "wheel_speeds")
    angles = _four_finite(turn_angles, "turn_angles")
    values = speeds + angles
    # Match UTCPComponent::ValidateControlData (not the unused Head-inclusive helper).
    checksum_sum = sum(
        struct.unpack("<I", struct.pack("<f", value))[0] for value in values
    )
    payload = struct.pack("<I", vehicle_id) + struct.pack("<I8fB", CONTROL_HEAD, *values, checksum_sum & 0xFF)
    header_checksum = COMMAND_MAGIC ^ COMMAND_ID ^ CONTROL_PAYLOAD_SIZE
    return struct.pack(
        "<4I", COMMAND_MAGIC, COMMAND_ID, CONTROL_PAYLOAD_SIZE, header_checksum
    ) + payload


class FeedbackParser:
    """Persistent parser for fragmented, coalesced and noise-prefixed TCP data."""

    def __init__(self, vehicle_id: int = 0) -> None:
        self.vehicle_id = vehicle_id
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[Feedback]:
        self._buffer.extend(data)
        parsed: list[Feedback] = []
        while True:
            start = self._buffer.find(_FEEDBACK_SIGNATURE)
            if start < 0:
                # Three trailing bytes are enough to preserve any split signature.
                del self._buffer[:-3]
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 16:
                break
            magic, command, size, checksum = struct.unpack_from('<4I', self._buffer)
            if command != 4 or size != 92 or checksum != magic ^ command ^ size:
                del self._buffer[:1]
                continue
            if len(self._buffer) < FEEDBACK_SIZE:
                break
            candidate = bytes(self._buffer[20:FEEDBACK_SIZE])
            vehicle_id = struct.unpack_from('<I', self._buffer, 16)[0]
            del self._buffer[:FEEDBACK_SIZE]
            if vehicle_id != self.vehicle_id:
                continue
            feedback = _decode_feedback(candidate, vehicle_id)
            if feedback is not None:
                parsed.append(feedback)
        return parsed


def _decode_feedback(data: bytes, vehicle_id: int) -> Feedback | None:
    unpacked = _FEEDBACK_STRUCT.unpack(data)
    if unpacked[0] != FEEDBACK_HEAD:
        return None
    values = unpacked[1:]
    if not all(math.isfinite(value) for value in values):
        return None
    orientation = tuple(values[11:15])
    if sum(value * value for value in orientation) <= 1.0e-12:
        return None
    return Feedback(
        wheel_speeds=tuple(values[0:4]),  # type: ignore[arg-type]
        turn_angles=tuple(values[4:8]),  # type: ignore[arg-type]
        position_m=tuple(values[8:11]),  # type: ignore[arg-type]
        orientation_xyzw=orientation,  # type: ignore[arg-type]
        linear_velocity=tuple(values[15:18]),  # type: ignore[arg-type]
        vehicle_id=vehicle_id,
        angular_velocity=tuple(values[18:21]),  # type: ignore[arg-type]
    )
