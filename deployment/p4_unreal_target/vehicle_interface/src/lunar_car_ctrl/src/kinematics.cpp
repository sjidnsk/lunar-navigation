/**
 * @file kinematics.cpp
 * @brief 功能层 — 四驱四转向 (4W4S) 运动学实现
 */

#include "lunar_car_ctrl/kinematics.hpp"

#include <algorithm>
#include <cmath>

namespace lunar_car_ctrl
{

void computeWheelCommands(const VehicleParams & params,
                          double linear_x, double angular_z,
                          double wheel_speed[4], double steer_angle[4])
{
    // 基于刚体速度叠加: v_wheel = v_body + ω × r
    //
    // 车轮位置 (右手系, Y 向左):
    //   FL = (+L/2, +W/2)  左轮 Y 为正
    //   FR = (+L/2, -W/2)  右轮 Y 为负
    //   RL = (-L/2, +W/2)
    //   RR = (-L/2, -W/2)
    //
    // 各轮速度:
    //   vx_i = linear_x - angular_z * y_i
    //   vy_i = angular_z * x_i

    double half_L = params.wheel_base * 0.5;
    double half_W = params.track_width * 0.5;

    // ---- 原地转弯限制: 仅角速度 (线速度为零) 时, 限制车轮线速度 ----
    // 纯旋转时各轮线速度 = |ω| × R (R 为车体中心到车轮的距离),
    // 限制车轮线速度 ≤ spin_max_linear_speed 等价于 |ω| ≤ v_max / R
    double angular = angular_z;
    if (std::abs(linear_x) < 1e-9 && std::abs(angular) > 1e-9
        && params.spin_max_linear_speed > 0.0) {
        double R = std::hypot(half_L, half_W);
        double max_omega = params.spin_max_linear_speed / R;
        angular = std::clamp(angular, -max_omega, max_omega);
    }

    // 右手系各轮速度向量 (使用限幅后的角速度)
    const double vx[4] = {
        linear_x - angular *  half_W,   // FL
        linear_x - angular * -half_W,   // FR
        linear_x - angular *  half_W,   // RL
        linear_x - angular * -half_W    // RR
    };
    const double vy[4] = {
         angular *  half_L,             // FL
         angular *  half_L,             // FR
         angular * -half_L,             // RL
         angular * -half_L              // RR
    };

    // 轮速限幅 (rad/s)
    double max_speed = params.max_wheel_speed;
    if (max_speed <= 0.0) max_speed = 1e9;

    // 车轮半径换算: 轮线速度 (m/s) / 半径 (m) = 轮速 (rad/s)
    double inv_radius = 1.0 / params.wheel_radius;
    if (std::abs(params.wheel_radius) < 1e-9) inv_radius = 1.0;

    for (int i = 0; i < 4; ++i) {
        // 速度模长 (轮线速度, m/s) 和方向角 (右手系转角)
        double speed = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
        double angle = std::atan2(vy[i], vx[i]);
        // 速度方向与前进方向相反时: 轮速取反并翻转转角 (等效同一轮姿态)
        if (vx[i] < 0.0) {
            speed = -speed;
            angle += (angle >= 0.0) ? -M_PI : M_PI;
        }
        // 换算为轮速 (rad/s) 并限幅
        wheel_speed[i] = std::clamp(speed * inv_radius, -max_speed, max_speed);
        steer_angle[i] = angle;
    }
}

void computeWheelCommandsFromCurvature(const VehicleParams & params,
                                       double velocity, double curvature,
                                       double wheel_speed[4], double steer_angle[4])
{
    // 曲率 κ 与角速度关系: ω = v × κ
    computeWheelCommands(params, velocity, velocity * curvature,
                         wheel_speed, steer_angle);
}

} // namespace lunar_car_ctrl
