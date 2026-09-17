/**
 * @file main.hpp
 * @brief 月球车遥测与控制 ROS2 节点 (调度层)
 *
 * 分层架构:
 *   - 通信层: protocol_codec (协议编解码) + tcp_client (TCP 连接与收发)
 *   - 功能层: coord_transform (坐标变换) + kinematics (4W4S 运动学)
 *             + telemetry_parser (遥测解析)
 *   - 调度层: 本节点 — ROS2 接口、控制模式状态机、定时器与线程编排
 *
 * 处理流程:
 *   1. 建立 TCP 连接, 持续接收车端遥测数据
 *   2. 遥测数据左手系→右手系转换 (位置/四元数/转角; 轮速不转)
 *   3. 速度 + 曲率分解到四个车轮 (右手系)
 *   4. 四轮指令右手系→左手系转换
 *   5. 通过网络发送到车端
 *
 * 控制模式 (无直接角度模式):
 *   - 停车模式 (park): 发送车速/转角均为 0 的指令
 *   - 自动驾驶模式 (auto): 5Hz 发送控制指令,
 *     1s 未收到 cmd_vel 自动切换到停车模式
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include "lunar_car_ctrl/msg/vehicle_telemetry.hpp"
#include "lunar_car_ctrl/msg/wheel_commands.hpp"

#include "lunar_car_ctrl/tcp_client.hpp"
#include "lunar_car_ctrl/kinematics.hpp"

namespace lunar_car_ctrl
{

/**
 * @brief 控制模式 (只有两种, 无直接角度模式)
 */
enum class ControlMode : int
{
    PARK = 0,   ///< 停车模式: 发送车速/转角均为 0 的指令 (5Hz)
    AUTO = 1    ///< 自动驾驶模式: 5Hz 发送控制指令, 1s 未收到 cmd_vel 自动切换为停车模式
};

/**
 * @brief 月球车控制节点 (调度层)
 *
 * 架构:
 *   - 独立接收线程 (receiveLoop): 持续接收车端遥测反馈并发布
 *   - 定时器回调 (onControlTimer): 按控制模式周期性发送控制指令
 *   - 订阅回调 (onCmdVel / onSetMode): 缓存指令与切换模式
 *
 * Topic:
 *   订阅: /car/cmd_vel (geometry_msgs/Twist) — 速度控制指令
 *         /car/set_mode (std_msgs/String) — 控制模式切换 ("park"/"auto")
 *   发布: /car/telemetry (VehicleTelemetry) — 解析后的遥测数据 (右手系)
 *         /car/odom (nav_msgs/Odometry) — 标准里程计
 *         /car/pose (geometry_msgs/PoseStamped) — 位姿
 *         /car/wheel_commands (WheelCommands) — 实际发送的四轮指令 (左手系)
 */
class LunarCarNode : public rclcpp::Node
{
public:
    explicit LunarCarNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~LunarCarNode() override;

private:
    // ========================================================
    //  可配置参数
    // ========================================================
    VehicleParams vehicle_params_;     ///< 车辆物理参数 (轴距/轮距/轮半径/最大轮速)
    double control_period_;            ///< 控制定时器周期 (s, 20Hz 基准, 两种模式均按 4 分频为 5Hz 发送)
    double cmd_timeout_;               ///< cmd_vel 超时阈值 (s, 自动驾驶模式超时未收到则自动切换停车模式)
    double max_linear_speed_;          ///< 车辆最大线速度 (m/s, 对 cmd_vel 线速度限幅)
    double spin_prep_time_;            ///< 原地转弯预置时间 (s, 先发转角/轮速 0 指令等车轮到位)
    double steer_step_deg_;            ///< 转角每发送周期最大增量 (度, 平滑转向, 0=不限制)

    // ========================================================
    //  通信层
    // ========================================================
    std::unique_ptr<LunarTcpClient> client_;     ///< 车端 TCP 客户端
    std::atomic<bool> running_;                  ///< 节点运行标志
    std::thread recv_thread_;                    ///< 遥测接收线程

    // ========================================================
    //  ROS2 订阅者
    // ========================================================
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;       ///< /car/cmd_vel
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_sub_;              ///< /car/set_mode

    // ========================================================
    //  ROS2 发布者
    // ========================================================
    rclcpp::Publisher<lunar_car_ctrl::msg::VehicleTelemetry>::SharedPtr telemetry_pub_;  ///< 遥测数据
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;                     ///< 标准里程计
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;             ///< 位姿
    rclcpp::Publisher<lunar_car_ctrl::msg::WheelCommands>::SharedPtr wheel_cmd_pub_;     ///< 实际发送的四轮指令 (左手系)

    // ========================================================
    //  周期控制与指令缓存
    // ========================================================
    rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr path_request_pub_;
    rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr path_response_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr connection_pub_;
    std::atomic<uint32_t> latest_request_id_{0};
    std::atomic<bool> has_path_request_{false};
    void onPathResponse(const std_msgs::msg::UInt8MultiArray::SharedPtr msg);
    void publishConnection(bool connected);

    rclcpp::TimerBase::SharedPtr control_timer_;           ///< 控制定时器 (20Hz 基准)
    geometry_msgs::msg::Twist latest_cmd_;                 ///< 缓存的最新速度指令
    rclcpp::Time last_cmd_time_{0, 0, RCL_ROS_TIME};       ///< 最近一次收到 cmd_vel 的时间 (与 cmd_mutex_ 共用)
    std::mutex cmd_mutex_;                                 ///< 指令读写互斥锁

    // ========================================================
    //  控制模式状态机 (停车 / 自动驾驶)
    // ========================================================
    std::atomic<ControlMode> mode_{ControlMode::PARK};     ///< 当前控制模式 (线程安全, 默认停车)
    int auto_send_tick_{0};                                 ///< 5Hz 分频计数器 (20Hz/4, 停车/自动驾驶模式共用)
    int spin_prep_dir_{0};                                  ///< 原地转弯预置方向 (+1/-1, 0=未预置)
    rclcpp::Time spin_prep_start_{0, 0, RCL_ROS_TIME};      ///< 原地转弯预置阶段起始时间
    double current_steer_lh_[4]{0.0, 0.0, 0.0, 0.0};        ///< 当前已发送转角 (左手系, rad, 平滑转向递增基准)

    // ========================================================
    //  回调函数
    // ========================================================

    /** @brief 速度指令订阅回调 — 缓存最新指令并记录时间戳 (不直接发送) */
    void onCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg);

    /** @brief 控制模式切换回调 — 解析 "park"/"auto" 并更新 mode_ */
    void onSetMode(const std_msgs::msg::String::SharedPtr msg);

    /** @brief 解析控制模式字符串 ("park"/"auto") */
    static bool parseControlMode(const std::string & name, ControlMode & mode);

    /** @brief 发送停车指令 (四轮速度/转角均为 0) */
    void sendParkCommand();

    /** @brief 控制定时器回调 — 按当前模式读取缓存指令并发送控制包 */
    void onControlTimer();

    // ========================================================
    //  遥测接收与发布 (调度)
    // ========================================================

    /** @brief 接收线程主循环 — 接收遥测包, 解析并发布, 断线自动重连 */
    void receiveLoop();

    /** @brief 断开后延时重连 (3s) */
    void reconnect();

    /** @brief 解析遥测 payload 并发布所有相关消息 */
    void parseAndPublish(const std::vector<uint8_t> & payload);

    /** @brief 根据 VehicleTelemetry 发布 Odometry + PoseStamped */
    void publishOdomAndPose(const lunar_car_ctrl::msg::VehicleTelemetry & tel);

    // ========================================================
    //  控制指令发送 (调度: 运动学 + 坐标变换 + 通信层)
    // ========================================================

    /** @brief 将右手系四轮指令转换为左手系并通过网络发送 */
    void sendWheelCommands(const double wheel_speed_rh[4], const double steer_angle_rh[4]);
};

} // namespace lunar_car_ctrl
