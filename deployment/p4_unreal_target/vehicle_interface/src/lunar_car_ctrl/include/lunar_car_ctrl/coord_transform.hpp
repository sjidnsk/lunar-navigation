/**
 * @file coord_transform.hpp
 * @brief 功能层 — 左手系/右手系坐标变换
 *
 * 约定:
 *   车端左手系 (LH): X 前进, Y 向右, Z 向上
 *   内部右手系 (RH): X 前进, Y 向左, Z 向上
 *   变换矩阵: T = diag(1, -1, 1) (Y 轴取反)
 *
 * 变换规则:
 *   - 位置/线速度 (向量): Y 取反
 *   - 四元数: qx / qz 取反
 *   - 角速度 (伪向量): ω_rh = det(T)·T·ω_lh, wx / wz 取反
 *   - 转向角 (绕Z轴平面角): 取反 (LH↔RH 双向对称)
 *   - 轮速 (标量): 不做坐标系转换
 */

#pragma once

namespace lunar_car_ctrl
{
namespace coord
{

/** @brief 位置左手系→右手系: 向量变换, Y 轴取反 */
inline void transformPositionLHtoRH(double & x, double & y, double & z)
{
    // x_rh = x_lh, y_rh = -y_lh, z_rh = z_lh
    (void)x;
    y = -y;
    (void)z;
}

/** @brief 四元数左手系→右手系: 绕被取反轴的旋转分量取反 (qx / qz 取反) */
inline void transformQuaternionLHtoRH(double & qx, double & qy, double & qz, double & qw)
{
    // qx_rh = -qx_lh, qy_rh = qy_lh, qz_rh = -qz_lh, qw_rh = qw_lh
    qx = -qx;
    (void)qy;
    qz = -qz;
    (void)qw;
}

/** @brief 角速度左手系→右手系: 伪向量变换, wx / wz 取反 */
inline void transformAngularVelocityLHtoRH(double & wx, double & wy, double & wz)
{
    // wx_rh = -wx_lh, wy_rh = wy_lh, wz_rh = -wz_lh
    wx = -wx;
    (void)wy;
    wz = -wz;
}

/** @brief 转向角左手系↔右手系转换: 取反 (双向对称, θ' = -θ) */
inline double transformSteerAngle(double angle)
{
    return -angle;
}

/**
 * @brief 四轮速度/角度右手系→左手系转换
 *
 * 轮速 (标量): 不变; 转角 (绕Z轴平面角): 取反
 * @param wheel_speed_rh 右手系四轮轮速 (rad/s, FL/FR/RL/RR)
 * @param steer_angle_rh 右手系四轮转角 (rad,  FL/FR/RL/RR)
 * @param wheel_speed_lh 输出: 左手系四轮轮速 (float, 发送精度)
 * @param steer_angle_lh 输出: 左手系四轮转角 (float, 发送精度)
 */
inline void convertWheelsRHtoLH(const double wheel_speed_rh[4],
                                const double steer_angle_rh[4],
                                float wheel_speed_lh[4],
                                float steer_angle_lh[4])
{
    for (int i = 0; i < 4; ++i) {
        // 轮速为标量，左手系与右手系相同
        wheel_speed_lh[i] = static_cast<float>(wheel_speed_rh[i]);
        // 转角: Y 轴取反导致角度符号翻转, θ_lh = -θ_rh
        steer_angle_lh[i] = static_cast<float>(transformSteerAngle(steer_angle_rh[i]));
    }
}

} // namespace coord
} // namespace lunar_car_ctrl
