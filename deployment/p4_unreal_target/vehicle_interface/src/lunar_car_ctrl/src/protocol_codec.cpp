/**
 * @file protocol_codec.cpp
 * @brief 通信层 — 协议常量与字节序编解码实现
 */

#include "lunar_car_ctrl/protocol_codec.hpp"

#include <cstring>

namespace lunar_car_ctrl
{

ProtocolCodec::ProtocolCodec(bool big_endian)
: big_endian_(big_endian)
{
}

/** @brief 从字节流读取 uint32 (根据字节序配置) */
uint32_t ProtocolCodec::readU32(const uint8_t * p) const
{
    if (big_endian_) {
        return (static_cast<uint32_t>(p[0]) << 24)
             | (static_cast<uint32_t>(p[1]) << 16)
             | (static_cast<uint32_t>(p[2]) << 8)
             |  static_cast<uint32_t>(p[3]);
    }
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

/** @brief 从字节流读取 float32 (根据字节序配置) */
float ProtocolCodec::readFloat(const uint8_t * p) const
{
    uint32_t v = readU32(p);
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

/** @brief 从字节流读取 float64 (根据字节序配置) */
double ProtocolCodec::readDouble(const uint8_t * p) const
{
    uint64_t v;
    if (big_endian_) {
        v = (static_cast<uint64_t>(p[0]) << 56)
          | (static_cast<uint64_t>(p[1]) << 48)
          | (static_cast<uint64_t>(p[2]) << 40)
          | (static_cast<uint64_t>(p[3]) << 32)
          | (static_cast<uint64_t>(p[4]) << 24)
          | (static_cast<uint64_t>(p[5]) << 16)
          | (static_cast<uint64_t>(p[6]) << 8)
          |  static_cast<uint64_t>(p[7]);
    } else {
        v = static_cast<uint64_t>(p[0])
          | (static_cast<uint64_t>(p[1]) << 8)
          | (static_cast<uint64_t>(p[2]) << 16)
          | (static_cast<uint64_t>(p[3]) << 24)
          | (static_cast<uint64_t>(p[4]) << 32)
          | (static_cast<uint64_t>(p[5]) << 40)
          | (static_cast<uint64_t>(p[6]) << 48)
          | (static_cast<uint64_t>(p[7]) << 56);
    }
    double d;
    std::memcpy(&d, &v, sizeof(d));
    return d;
}

/** @brief 将 uint32 写入缓冲区 (根据字节序配置) */
void ProtocolCodec::pushU32(std::vector<uint8_t> & buf, uint32_t v) const
{
    if (big_endian_) {
        buf.push_back((v >> 24) & 0xFF);
        buf.push_back((v >> 16) & 0xFF);
        buf.push_back((v >> 8) & 0xFF);
        buf.push_back(v & 0xFF);
    } else {
        buf.push_back(v & 0xFF);
        buf.push_back((v >> 8) & 0xFF);
        buf.push_back((v >> 16) & 0xFF);
        buf.push_back((v >> 24) & 0xFF);
    }
}

/** @brief 将 float 写入缓冲区 (根据字节序配置) */
void ProtocolCodec::pushFloat(std::vector<uint8_t> & buf, float f) const
{
    uint32_t v;
    std::memcpy(&v, &f, sizeof(v));
    pushU32(buf, v);
}

std::vector<uint8_t> ProtocolCodec::buildDynamicsPayload(
    float ws0, float ws1, float ws2, float ws3,
    float ta0, float ta1, float ta2, float ta3) const
{
    std::vector<uint8_t> payload;
    payload.reserve(DYNAMICS_PAYLOAD_SIZE);

    // 控制头标识
    pushU32(payload, HEAD_CONTROL);

    // WheelSpeed 0~3 (rad/s)
    pushFloat(payload, ws0);
    pushFloat(payload, ws1);
    pushFloat(payload, ws2);
    pushFloat(payload, ws3);

    // TurnAngle 0~3 (rad)
    pushFloat(payload, ta0);
    pushFloat(payload, ta1);
    pushFloat(payload, ta2);
    pushFloat(payload, ta3);

    // Checksum: 所有 uint32 字段 (含 float 的位模式) 求和，取低 8 位
    uint32_t sum = HEAD_CONTROL;
    auto addFloat = [&sum](float f) {
        uint32_t v;
        std::memcpy(&v, &f, sizeof(v));
        sum += v;
    };
    addFloat(ws0); addFloat(ws1); addFloat(ws2); addFloat(ws3);
    addFloat(ta0); addFloat(ta1); addFloat(ta2); addFloat(ta3);
    payload.push_back(static_cast<uint8_t>(sum));

    return payload;
}

} // namespace lunar_car_ctrl
