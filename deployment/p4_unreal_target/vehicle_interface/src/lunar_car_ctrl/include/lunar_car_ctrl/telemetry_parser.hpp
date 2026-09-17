/**
 * @file telemetry_parser.hpp
 * @brief 功能层 — 遥测数据解析与左手系→右手系转换
 *
 * 遥测 payload 布局 (全部 float32, 共 88 字节):
 *   HEAD(4) + 4×WheelSpeed(16) + 4×TurnAngle(16)
 *   + 3×Position(12) + 4×Rotation(16)
 *   + 3×LinearVel(12) + 3×AngularVel(12)
 *
 * 车端原始数据为左手系:
 *   位置单位 m, 轮速单位 rad/s
 *   转换要求: 位置 + 四元数 + 四轮转角 转换为右手系;
 *             车轮速度为标量, 不做坐标系转换
 */

#pragma once

#include <vector>
#include <cstdint>

#include <rclcpp/rclcpp.hpp>
#include "lunar_car_ctrl/protocol_codec.hpp"
#include "lunar_car_ctrl/msg/vehicle_telemetry.hpp"

namespace lunar_car_ctrl
{

/**
 * @brief 解析遥测 payload 为 VehicleTelemetry 消息
 *
 * @param payload  遥测 payload (以 HEAD_FEEDBACK 开头, ≥ 88 字节)
 * @param codec    字节序编解码器
 * @param stamp    消息时间戳
 * @param out      输出: 解析后的遥测消息 (含左手系原始字段与右手系转换字段)
 * @return true 解析成功; false 数据不足
 */
bool parseTelemetryPayload(const std::vector<uint8_t> & payload,
                           const ProtocolCodec & codec,
                           const rclcpp::Time & stamp,
                           lunar_car_ctrl::msg::VehicleTelemetry & out);

} // namespace lunar_car_ctrl
