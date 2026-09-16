import math
import struct

import pytest

from lunar_obj_tcp_sim.protocol import (
    FeedbackParser,
    encode_command,
)


def feedback_bytes(*, position=(123.0, -456.0, 78.0), orientation=(0.1, 0.2, 0.3, 0.9)):
    values = (
        1.0, 2.0, 3.0, 4.0,
        0.0, 0.0, 0.0, 0.0,
        *position,
        *orientation,
        5.0, 6.0, 7.0,
        0.4, 0.5, 0.6,
    )
    return struct.pack("<4I",0x4C494441,4,92,0x4C494441^4^92)+struct.pack("<II21f",0,0xEB93EB93,*values)


def test_encode_command_matches_cpp_packed_layout_and_checksums():
    packet = encode_command((1.0, -2.0, 3.5, -4.25))

    assert len(packet) == 57
    assert struct.unpack("<4I", packet[:16]) == (
        0x4C494441,
        3,
        41,
        0x4C494441 ^ 3 ^ 41,
    )
    head, *values = struct.unpack("<I8fB", packet[20:])
    assert head == 0xEB92EB92
    assert values[:-1] == [1.0, -2.0, 3.5, -4.25, 0.0, 0.0, 0.0, 0.0]
    # UTCPComponent::ValidateControlData sums only the eight floats, not Head.
    assert values[-1] == 0


def test_feedback_parser_preserves_fragment_and_returns_direct_cpp_units():
    parser = FeedbackParser()
    wire = b"noise\x93" + feedback_bytes()

    assert parser.feed(wire[:11]) == []
    assert parser.feed(wire[11:50]) == []
    feedback = parser.feed(wire[50:])

    assert len(feedback) == 1
    assert feedback[0].position_m == pytest.approx((123., -456., 78.))
    assert feedback[0].orientation_xyzw == pytest.approx((0.1, 0.2, 0.3, 0.9))
    assert feedback[0].linear_velocity == pytest.approx((5.0, 6.0, 7.0))
    assert feedback[0].angular_velocity == pytest.approx((0.4, 0.5, 0.6))


def test_feedback_parser_handles_coalesced_frames_and_recovers_after_invalid_frame():
    parser = FeedbackParser()
    invalid = feedback_bytes(orientation=(0.0, 0.0, 0.0, 0.0))
    valid1 = feedback_bytes(position=(100.0, 200.0, 300.0))
    valid2 = feedback_bytes(position=(-100.0, -200.0, -300.0))

    feedback = parser.feed(invalid + valid1 + valid2)

    assert [item.position_m for item in feedback] == [
        pytest.approx((100.0, 200.0, 300.0)),
        pytest.approx((-100.0, -200.0, -300.0)),
    ]


def test_feedback_parser_rejects_nonfinite_values():
    parser = FeedbackParser()
    assert parser.feed(feedback_bytes(position=(math.nan, 0.0, 0.0))) == []


def test_vehicle_id_envelope_and_filter_do_not_accept_another_vehicle():
    command = encode_command((0.,)*4, vehicle_id=42)
    assert struct.unpack_from('<I',command,16)[0] == 42
    parser = FeedbackParser(vehicle_id=42)
    wrong = feedback_bytes()
    correct = bytearray(wrong)
    struct.pack_into('<I',correct,16,42)
    assert parser.feed(wrong) == []
    parsed = parser.feed(correct)
    assert len(parsed)==1 and parsed[0].vehicle_id==42
    assert parsed[0].position_m[0]==123.


def test_header_corruption_recovers_without_accepting_body_as_raw_feedback():
    wrong = bytearray(feedback_bytes())
    wrong[12] ^= 1
    parser = FeedbackParser()
    assert parser.feed(wrong)==[]
    assert len(parser.feed(feedback_bytes()))==1


def test_receiver_checksum_uses_low_bytes_of_float_words_only():
    packet = encode_command((.123, -.456, .789, -.321), (.013, -.027, .039, -.051))
    # Independent receiver rule: little-endian low bytes at payload word offsets.
    assert packet[-1] == sum(packet[i] for i in range(24, 56, 4)) % 256
    assert encode_command((0.,)*4)[-1] == 0
