/**
 * @file telemetry_parser.cpp
 * @brief 功能层 — 遥测数据解析与左手系→右手系转换实现
 */

#include "lunar_car_ctrl/telemetry_parser.hpp"
#include "lunar_car_ctrl/coord_transform.hpp"

namespace lunar_car_ctrl
{

bool parseTelemetryPayload(const std::vector<uint8_t> & payload,
                           const ProtocolCodec & codec,
                           const rclcpp::Time & stamp,
                           lunar_car_ctrl::msg::VehicleTelemetry & out)
{
    if (payload.size() < TELEMETRY_MIN_SIZE) {
        return false;
    }

    const uint8_t * p = payload.data();
    size_t offset = 4; // 跳过 HEAD_FEEDBACK

    auto & msg = out;
    msg.header.stamp = stamp;
    msg.header.frame_id = "base_link";

    // ---- 四轮轮速 (4 × float32 = 16), 单位 rad/s ----
    // 车轮速度为标量，不做坐标系转换，直接读取
    msg.wheel_speed_fl = codec.readFloat(p + offset); offset += 4;
    msg.wheel_speed_fr = codec.readFloat(p + offset); offset += 4;
    msg.wheel_speed_rl = codec.readFloat(p + offset); offset += 4;
    msg.wheel_speed_rr = codec.readFloat(p + offset); offset += 4;

    // ---- 四轮转向角 (4 × float32 = 16) ----
    // 左手系→右手系: 绕 Z 轴平面角取反 (θ_rh = -θ_lh)
    msg.steer_angle_fl = coord::transformSteerAngle(codec.readFloat(p + offset)); offset += 4;
    msg.steer_angle_fr = coord::transformSteerAngle(codec.readFloat(p + offset)); offset += 4;
    msg.steer_angle_rl = coord::transformSteerAngle(codec.readFloat(p + offset)); offset += 4;
    msg.steer_angle_rr = coord::transformSteerAngle(codec.readFloat(p + offset)); offset += 4;

    // ---- 位置 (3 × float32 = 12), 单位 m ----
    // 左手系原始值存入 _lh 字段，再调用转换函数得到右手系值
    msg.x_lh = static_cast<double>(codec.readFloat(p + offset)); offset += 4;
    msg.y_lh = static_cast<double>(codec.readFloat(p + offset)); offset += 4;
    msg.z_lh = static_cast<double>(codec.readFloat(p + offset)); offset += 4;
    msg.x = msg.x_lh;
    msg.y = msg.y_lh;
    msg.z = msg.z_lh;
    coord::transformPositionLHtoRH(msg.x, msg.y, msg.z);

    // ---- 四元数 (4 × float32 = 16) ----
    // 左手系原始值存入 _lh 字段，再调用转换函数得到右手系值
    msg.qx_lh = codec.readFloat(p + offset); offset += 4;
    msg.qy_lh = codec.readFloat(p + offset); offset += 4;
    msg.qz_lh = codec.readFloat(p + offset); offset += 4;
    msg.qw_lh = codec.readFloat(p + offset); offset += 4;
    msg.qx = msg.qx_lh;
    msg.qy = msg.qy_lh;
    msg.qz = msg.qz_lh;
    msg.qw = msg.qw_lh;
    coord::transformQuaternionLHtoRH(msg.qx, msg.qy, msg.qz, msg.qw);

    // ---- 线速度 (3 × float32 = 12), 单位 m/s ----
    // 向量变换: Y 轴取反
    msg.vx = static_cast<double>(codec.readFloat(p + offset)); offset += 4;
    msg.vy = -static_cast<double>(codec.readFloat(p + offset)); offset += 4;
    msg.vz = static_cast<double>(codec.readFloat(p + offset)); offset += 4;

    // ---- 角速度 (3 × float32 = 12), 单位 rad/s ----
    // 伪向量变换: wx / wz 取反
    msg.wx = codec.readFloat(p + offset); offset += 4;
    msg.wy = codec.readFloat(p + offset); offset += 4;
    msg.wz = codec.readFloat(p + offset); offset += 4;
    coord::transformAngularVelocityLHtoRH(msg.wx, msg.wy, msg.wz);

    return true;
}

} // namespace lunar_car_ctrl
