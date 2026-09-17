/**
 * @file kinematics.hpp
 * @brief 功能层 — 四驱四转向 (4W4S) 运动学
 *
 * 在右手系 (X 前进, Y 向左) 下将车体速度/曲率分解为四轮轮速与转角:
 *   基于刚体速度叠加: v_wheel = v_body + ω × r
 *   车轮位置: FL(+L/2,+W/2), FR(+L/2,-W/2), RL(-L/2,+W/2), RR(-L/2,-W/2)
 *   轮速单位: rad/s (轮线速度按车轮半径换算: ω = v / r)
 */

#pragma once

namespace lunar_car_ctrl
{

/** @brief 车辆物理参数 */
struct VehicleParams
{
    double wheel_base = 0.815;          ///< 轴距 (前后轮距离, m)
    double track_width = 0.622;         ///< 轮距 (左右轮距离, m)
    double wheel_radius = 0.12;         ///< 车轮半径 (m), 轮速↔线速度换算
    double max_wheel_speed = 10.0;      ///< 单轮最大轮速 (rad/s)
    double spin_max_linear_speed = 0.1; ///< 原地转弯 (仅角速度) 时车轮最大线速度 (m/s)
};

/**
 * @brief 将车体线速度和角速度分解为四轮轮速 + 转角 (右手系)
 *
 * 各轮速度:
 *   vx_i = linear_x - angular_z * y_i
 *   vy_i = angular_z * x_i
 * 轮速 = 速度模长 / 车轮半径 (限幅), 转角 = atan2(vy, vx)
 * 速度方向与前进方向相反时: 轮速取反并翻转转角 (等效同一轮姿态)
 * 原地转弯限制: 仅角速度 (线速度为零) 时, 车轮线速度不超过
 *   spin_max_linear_speed (等价于对角速度限幅: |ω| ≤ v_max / R,
 *   R 为车体中心到车轮的距离)
 *
 * @param params      车辆物理参数
 * @param linear_x    车体线速度 (m/s)
 * @param angular_z   车体角速度 (rad/s)
 * @param wheel_speed 输出: 四轮轮速 (rad/s, FL/FR/RL/RR)
 * @param steer_angle 输出: 四轮转角 (rad, 右手系, FL/FR/RL/RR)
 */
void computeWheelCommands(const VehicleParams & params,
                          double linear_x, double angular_z,
                          double wheel_speed[4], double steer_angle[4]);

/**
 * @brief 将车体速度 (m/s) 和路径曲率 (1/m) 分解为四轮轮速 + 转角
 *
 * 曲率 κ 与角速度关系: ω = v × κ, 内部委托 computeWheelCommands
 */
void computeWheelCommandsFromCurvature(const VehicleParams & params,
                                       double velocity, double curvature,
                                       double wheel_speed[4], double steer_angle[4]);

} // namespace lunar_car_ctrl
