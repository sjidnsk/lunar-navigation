/**
 * @file protocol_codec.hpp
 * @brief 通信层 — 协议常量与字节序编解码
 *
 * 协议约定 (与车端一致):
 * - 包头: MAGIC(4) + CMD(4) + PayloadSize(4) + Checksum(4) = 16 字节
 *   Checksum = MAGIC ^ CMD ^ PayloadSize
 * - 控制 payload: HEAD_CONTROL(4) + 4×WheelSpeed(16) + 4×TurnAngle(16)
 *   + Checksum(1) = 37 字节
 * - 遥测 payload: HEAD_FEEDBACK(4) + 4×WheelSpeed(16) + 4×TurnAngle(16)
 *   + 3×Position(12) + 4×Rotation(16) + 3×LinearVel(12) + 3×AngularVel(12)
 *   = 88 字节 (全部 float32)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lunar_car_ctrl
{

// ============================================================
//  协议常量
// ============================================================
static constexpr uint32_t MAGIC              = 0x4C494441;  ///< 包头魔数 "LIDA"
static constexpr uint32_t CMD_DYNAMICS       = 3;           ///< 命令类型: 动力学控制
static constexpr uint32_t HEAD_FEEDBACK      = 0xEB93EB93;  ///< 遥测反馈头标识
static constexpr uint32_t HEAD_CONTROL       = 0xEB92EB92;  ///< 控制指令头标识

// ============ 动力学控制 payload 大小 ============
// Head(4) + 4*WheelSpeed(16) + 4*TurnAngle(16) + Checksum(1) = 37
static constexpr size_t DYNAMICS_PAYLOAD_SIZE = 37;

// ============ 遥测数据最小大小 (与车端反馈结构一致) ============
// Head(4) + 4 wheel_speed(16) + 4 turn_angle(16) + 3 pos(12)
// + 4 quat(16) + 3 linear_vel(12) + 3 angular_vel(12) = 88
static constexpr size_t TELEMETRY_MIN_SIZE = 88;

/**
 * @brief 字节序感知的协议编解码器
 *
 * 负责协议数据的序列化/反序列化, 字节序由构造参数指定
 * (车端字节序需与配置一致, 否则解析错乱)
 */
class ProtocolCodec
{
public:
    explicit ProtocolCodec(bool big_endian = false);

    // ---- 反序列化 ----
    uint32_t readU32(const uint8_t * p) const;       ///< 从字节流读取 uint32
    float readFloat(const uint8_t * p) const;        ///< 从字节流读取 float32
    double readDouble(const uint8_t * p) const;      ///< 从字节流读取 float64

    // ---- 序列化 ----
    void pushU32(std::vector<uint8_t> & buf, uint32_t v) const;   ///< 写入 uint32
    void pushFloat(std::vector<uint8_t> & buf, float f) const;    ///< 写入 float32

    /**
     * @brief 构建动力学控制 payload
     * @param ws0~ws3 四轮轮速 (FL/FR/RL/RR, rad/s)
     * @param ta0~ta3 四轮转角 (FL/FR/RL/RR, rad)
     * @return HEAD(4) + 4×WheelSpeed(16) + 4×TurnAngle(16) + Checksum(1) = 37 字节
     *         Checksum: 所有 uint32 字段 (含 float 位模式) 求和取低 8 位
     */
    std::vector<uint8_t> buildDynamicsPayload(
        float ws0, float ws1, float ws2, float ws3,
        float ta0, float ta1, float ta2, float ta3) const;

private:
    bool big_endian_;    ///< true=大端, false=小端
};

} // namespace lunar_car_ctrl
