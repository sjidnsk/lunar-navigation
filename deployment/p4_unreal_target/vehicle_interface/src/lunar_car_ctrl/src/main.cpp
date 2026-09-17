#include <cmath>
/**
 * @file main.cpp
 * @brief 月球车遥测与控制节点实现 (调度层)
 *
 * 调度层职责:
 *   - ROS2 参数/话题/定时器管理
 *   - 控制模式状态机 (停车 / 自动驾驶)
 *   - 遥测接收线程编排 (接收→解析→发布, 断线重连)
 *   - 控制指令编排 (缓存→运动学分解→坐标变换→发送)
 *
 * 通信与功能实现分别位于:
 *   通信层: protocol_codec / tcp_client
 *   功能层: coord_transform / kinematics / telemetry_parser
 */

#include "lunar_car_ctrl/main.hpp"

#include "lunar_car_ctrl/coord_transform.hpp"
#include "lunar_car_ctrl/telemetry_parser.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace std::chrono_literals;

namespace lunar_car_ctrl
{

// ============================================================
//  构造函数 — 初始化参数、通信层、发布/订阅者、定时器
// ============================================================
LunarCarNode::LunarCarNode(const rclcpp::NodeOptions & options)
: Node("lunar_car_node", options)
, running_(false)
{
    // ---- 声明参数 (均可通过 launch 或 ros2 param set 覆盖) ----
    std::string tcp_host = this->declare_parameter<std::string>("tcp_host", "192.168.1.22");
    int tcp_port = this->declare_parameter<int>("tcp_port", 6668);
    vehicle_params_.wheel_base = this->declare_parameter<double>("wheel_base", 0.815);
    vehicle_params_.track_width = this->declare_parameter<double>("track_width", 0.622);
    vehicle_params_.wheel_radius = this->declare_parameter<double>("wheel_radius", 0.12);
    vehicle_params_.max_wheel_speed = this->declare_parameter<double>("max_wheel_speed", 10.0);
    vehicle_params_.spin_max_linear_speed =
        this->declare_parameter<double>("spin_max_linear_speed", 0.1);
    control_period_ = this->declare_parameter<double>("control_period", 0.05);  // 20Hz
    cmd_timeout_ = this->declare_parameter<double>("cmd_timeout", 1.0);
    max_linear_speed_ = this->declare_parameter<double>("max_linear_speed", 0.1);
    spin_prep_time_ = this->declare_parameter<double>("spin_prep_time", 3.0);
    steer_step_deg_ = this->declare_parameter<double>("steer_step_deg", 5.0);
    bool big_endian = this->declare_parameter<bool>("big_endian", false);

    // ---- 初始控制模式: "park" / "auto" (默认停车模式) ----
    std::string init_mode = this->declare_parameter<std::string>("control_mode", "park");
    ControlMode parsed_mode;
    if (parseControlMode(init_mode, parsed_mode)) {
        mode_ = parsed_mode;
    } else {
        RCLCPP_WARN(this->get_logger(), "无效初始控制模式 '%s'，使用默认 park", init_mode.c_str());
    }

    // ---- 创建通信层: 车端 TCP 客户端 ----
    client_ = std::make_unique<LunarTcpClient>(tcp_host, tcp_port, big_endian,
                                               this->get_logger());

    // ---- 创建发布者 ----
    telemetry_pub_ = this->create_publisher<lunar_car_ctrl::msg::VehicleTelemetry>(
        "/car/telemetry", 10);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/car/odom", 10);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/car/pose", 10);
    wheel_cmd_pub_ = this->create_publisher<lunar_car_ctrl::msg::WheelCommands>(
        "/car/wheel_commands", 10);

    // Private P4 extension: raw little-endian path payloads only; no planner here.
    if (big_endian) throw std::runtime_error("P4 UE path protocol requires little endian");
    path_request_pub_ = create_publisher<std_msgs::msg::UInt8MultiArray>("/car/ue_path_request", 10);
    connection_pub_ = create_publisher<std_msgs::msg::Bool>("/car/ue_connected", rclcpp::QoS(1).transient_local());
    path_response_sub_ = create_subscription<std_msgs::msg::UInt8MultiArray>(
        "/car/ue_path_response", 10,
        std::bind(&LunarCarNode::onPathResponse, this, std::placeholders::_1));

    // ---- 创建订阅者: 速度控制指令 ----
    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/car/cmd_vel", 10,
        std::bind(&LunarCarNode::onCmdVel, this, std::placeholders::_1));

    // ---- 创建订阅者: 控制模式切换 ----
    mode_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/car/set_mode", 10,
        std::bind(&LunarCarNode::onSetMode, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
        "月球车控制节点启动 - %s:%d (轴距=%.3f, 轮距=%.3f, 轮半径=%.3f, "
        "控制周期=%.1fms, 字节序=%s, 模式=%s)",
        tcp_host.c_str(), tcp_port, vehicle_params_.wheel_base,
        vehicle_params_.track_width, vehicle_params_.wheel_radius,
        control_period_ * 1000.0, big_endian ? "大端" : "小端", init_mode.c_str());

    // ---- 建立 TCP 连接: 初始连接失败时节点退出 ----
    if (!client_->connect()) {
        RCLCPP_ERROR(this->get_logger(), "无法连接车端服务器，节点退出");
        throw std::runtime_error("无法连接车端服务器");
    }

    // ---- 连接成功后才启动遥测接收线程与控制定时器 ----
    publishConnection(true);
    running_ = true;
    recv_thread_ = std::thread(&LunarCarNode::receiveLoop, this);

    // ---- 创建周期控制定时器 (默认 20Hz) ----
    auto period = std::chrono::duration<double>(control_period_);
    control_timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&LunarCarNode::onControlTimer, this));
}

void LunarCarNode::publishConnection(bool connected)
{
    std_msgs::msg::Bool message; message.data = connected;
    connection_pub_->publish(message);
}

void LunarCarNode::onPathResponse(const std_msgs::msg::UInt8MultiArray::SharedPtr msg)
{
    const auto &data = msg->data;
    if (data.size() < 12 || !has_path_request_) return;
    const auto id = client_->codec().readU32(data.data());
    const auto count = client_->codec().readU32(data.data()+8);
    if (id != latest_request_id_ || count > 8192 || data.size() != 12 + size_t(count)*12 ||
        data[4] > 2 || ((data[4] == 0) != (count > 0))) return;
    for (size_t i = 12; i < data.size(); i += 4)
        if (!std::isfinite(client_->codec().readFloat(data.data()+i))) return;
    if (client_->sendPacket(8, data))
        RCLCPP_INFO(get_logger(), "UE path response %u status=%u points=%u", id, data[4], count);
}

// ============ 析构函数 ============
LunarCarNode::~LunarCarNode()
{
    running_ = false;
    if (recv_thread_.joinable()) {
        recv_thread_.join();
    }
    client_->disconnect();
}

// ============================================================
//  参数与模式管理 (参数仅在启动时读取, 不支持动态调整)
// ============================================================

// ============ 速度指令回调 ============
void LunarCarNode::onCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    latest_cmd_ = *msg;
    last_cmd_time_ = this->now();
}

// ============ 控制模式字符串解析 ============
bool LunarCarNode::parseControlMode(const std::string & name, ControlMode & mode)
{
    if (name == "park") {
        mode = ControlMode::PARK;
    } else if (name == "auto") {
        mode = ControlMode::AUTO;
    } else {
        return false;
    }
    return true;
}

// ============ 控制模式切换回调 ============
void LunarCarNode::onSetMode(const std_msgs::msg::String::SharedPtr msg)
{
    ControlMode new_mode;
    if (!parseControlMode(msg->data, new_mode)) {
        RCLCPP_WARN(this->get_logger(),
            "未知控制模式 '%s' (可选: park / auto)", msg->data.c_str());
        return;
    }
    if (new_mode != mode_.load()) {
        mode_ = new_mode;
        auto_send_tick_ = 0;
        spin_prep_dir_ = 0;   // 模式切换后重置原地转弯预置状态
        RCLCPP_INFO(this->get_logger(), "控制模式切换: %s",
            (new_mode == ControlMode::PARK) ? "停车模式" : "自动驾驶模式");
    }
}

// ============================================================
//  控制指令调度 (定时器)
// ============================================================

// ============ 发送停车指令 ============
void LunarCarNode::sendParkCommand()
{
    // 停车模式: 四轮速度和转角均为 0
    static const double zeros[4] = {0.0, 0.0, 0.0, 0.0};
    sendWheelCommands(zeros, zeros);
}

/** @brief 定时器回调: 按当前控制模式读取缓存指令，经运动学分解后发送 */
void LunarCarNode::onControlTimer()
{
    // 未连接时静默跳过，避免刷屏警告
    if (!client_->isConnected()) {
        return;
    }

    // ---- 两种模式统一: 20Hz 定时器分频为 5Hz 发送控制指令 ----
    auto_send_tick_ = (auto_send_tick_ + 1) % 4;
    if (auto_send_tick_ != 0) {
        return;
    }

    // ---- 停车模式: 发送车速/转角均为 0 的指令 ----
    if (mode_.load() == ControlMode::PARK) {
        sendParkCommand();
        return;
    }

    // ---- 自动驾驶模式: 超时未收到订阅的 cmd_vel 时, 自动切换到停车模式 ----
    geometry_msgs::msg::Twist cmd;
    bool cmd_fresh = false;
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        cmd = latest_cmd_;
        cmd_fresh = (this->now() - last_cmd_time_).seconds() < cmd_timeout_;
    }
    if (!cmd_fresh) {
        mode_ = ControlMode::PARK;
        auto_send_tick_ = 0;
        spin_prep_dir_ = 0;
        RCLCPP_WARN(this->get_logger(),
            "自动驾驶模式: %.1fs 未收到 /car/cmd_vel，自动切换到停车模式", cmd_timeout_);
        sendParkCommand();
        return;
    }

    // 速度 + 曲率分解到四轮 (cmd 已在上方读取, 右手系运动学)
    double v = cmd.linear.x;    // 车体线速度 (m/s)
    double w = cmd.angular.z;   // 车体角速度 (rad/s)

    // 车辆线速度限幅: 不超过 max_linear_speed (默认 0.1 m/s)
    if (max_linear_speed_ > 0.0) {
        v = std::clamp(v, -max_linear_speed_, max_linear_speed_);
    }

    double ws_rh[4];
    double ta_rh[4];

    // ---- 原地转弯预置: 先发转角 (轮速 0) 指令等车轮到位, 再发送轮速 ----
    // 避免车轮未转到目标角度就开始转动 (原地转弯目标转角约 ±52.6°)
    bool is_spin = (std::abs(v) < 1e-9 && std::abs(w) > 1e-9);
    if (is_spin) {
        int spin_dir = (w > 0.0) ? 1 : -1;
        // 进入原地转弯或旋转方向翻转: 重新开始预置计时
        if (spin_prep_dir_ != spin_dir) {
            spin_prep_dir_ = spin_dir;
            spin_prep_start_ = this->now();
        }
        // 分解目标转角 (运动学内部的原地转弯限幅对 w 同样生效)
        computeWheelCommands(vehicle_params_, 0.0, w, ws_rh, ta_rh);
        if ((this->now() - spin_prep_start_).seconds() < spin_prep_time_) {
            // 预置阶段: 轮速置 0, 仅发送转角指令, 等待转向执行器到位
            for (int i = 0; i < 4; ++i) ws_rh[i] = 0.0;
        }
        sendWheelCommands(ws_rh, ta_rh);
        return;
    }
    spin_prep_dir_ = 0;   // 退出原地转弯, 重置预置状态

    if (std::abs(v) > 1e-9) {
        // 曲率 κ = ω / v, 使用速度+曲率分解
        computeWheelCommandsFromCurvature(vehicle_params_, v, w / v, ws_rh, ta_rh);
    } else {
        // 线速度与角速度均为零: 发送零指令保持停车状态 (限幅等边界)
        computeWheelCommands(vehicle_params_, 0.0, w, ws_rh, ta_rh);
    }

    // 右手系→左手系转换并通过网络发送
    sendWheelCommands(ws_rh, ta_rh);
}

// ============================================================
//  控制指令发送: 右手系→左手系 + 通信层发送
// ============================================================

void LunarCarNode::sendWheelCommands(const double wheel_speed_rh[4],
                                     const double steer_angle_rh[4])
{
    // 右手系 → 左手系 (轮速不变, 转角取反)
    float ws_lh[4];
    float ta_lh[4];
    coord::convertWheelsRHtoLH(wheel_speed_rh, steer_angle_rh, ws_lh, ta_lh);

    // ---- 平滑转向: 结合当前已发送角度, 每周期递增逼近目标转角 ----
    // 每发送周期转角变化不超过 steer_step_deg (默认 5°), 避免转角突变
    if (steer_step_deg_ > 0.0) {
        const double max_step = steer_step_deg_ * M_PI / 180.0;
        for (int i = 0; i < 4; ++i) {
            double delta = std::clamp(static_cast<double>(ta_lh[i]) - current_steer_lh_[i],
                                      -max_step, max_step);
            current_steer_lh_[i] += delta;
            ta_lh[i] = static_cast<float>(current_steer_lh_[i]);
        }
    } else {
        // 不限制: 直接同步当前角度状态为目标值
        for (int i = 0; i < 4; ++i) {
            current_steer_lh_[i] = static_cast<double>(ta_lh[i]);
        }
    }

    // 发布实际发送的四轮指令 (左手系), 供 GUI 等上位机显示
    lunar_car_ctrl::msg::WheelCommands cmd_msg;
    cmd_msg.header.stamp = this->now();
    cmd_msg.header.frame_id = "base_link";
    for (int i = 0; i < 4; ++i) {
        cmd_msg.wheel_speed[i] = static_cast<double>(ws_lh[i]);
        cmd_msg.steer_angle[i] = static_cast<double>(ta_lh[i]);
    }
    wheel_cmd_pub_->publish(cmd_msg);

    // 构建 payload 并通过网络发送到车端
    auto payload = client_->codec().buildDynamicsPayload(
        ws_lh[0], ws_lh[1], ws_lh[2], ws_lh[3],
        ta_lh[0], ta_lh[1], ta_lh[2], ta_lh[3]);
    if (!client_->sendPacket(CMD_DYNAMICS, payload)) {
        RCLCPP_WARN(this->get_logger(), "发送控制指令失败");
    }
}

// ============================================================
//  遥测接收调度
//  协议适配:
//    车端发送遥测反馈时可能不带标准包头包装 (无 MAGIC/CMD/Size/Checksum)，
//    直接发送 payload，首 4 字节为 HEAD_FEEDBACK(0xEB93EB93)。
//    因此先 peek 前 4 字节判断格式:
//      - MAGIC(0x4C494441)         → 标准包头协议，走 recvPacket 完整解析
//      - HEAD_FEEDBACK(0xEB93EB93) → 裸遥测 payload，直接读取并解析
// ============================================================

// ============ 断开后延时重连 (3s) ============
void LunarCarNode::reconnect()
{
    has_path_request_ = false;
    publishConnection(false);
    client_->disconnect();
    for (int i = 0; i < 30 && running_; ++i) {
        std::this_thread::sleep_for(100ms);
    }
    if (running_) {
        if (client_->connect()) publishConnection(true);
    }
}

void LunarCarNode::receiveLoop()
{
    RCLCPP_INFO(this->get_logger(), "遥测接收线程已启动");

    while (running_) {
        // ---- 先 peek 前 4 字节，判断数据包格式 ----
        uint8_t peek_buf[4];
        ssize_t peeked = client_->peekHeader(peek_buf);

        if (peeked <= 0) {
            // 连接断开或超时
            if (running_) {
                RCLCPP_WARN(this->get_logger(), "接收数据失败，尝试重连...");
                reconnect();
            }
            continue;
        }
        if (peeked < 4) {
            // 数据不足，等待更多数据到达
            continue;
        }

        uint32_t first4 = client_->codec().readU32(peek_buf);

        if (first4 == MAGIC) {
            // ---- 标准包头协议: MAGIC(4)+CMD(4)+PayloadSize(4)+Checksum(4) + payload ----
            uint32_t command = 0;
            std::vector<uint8_t> payload;
            if (!client_->recvPacket(command, payload, running_)) {
                if (running_) {
                    RCLCPP_WARN(this->get_logger(), "包头解析失败，尝试重连...");
                    reconnect();
                }
                continue;
            }

            if (command == 7) {
                if (payload.size() != 16) {
                    RCLCPP_WARN(get_logger(), "Invalid UE path request length: %zu", payload.size());
                    continue;
                }
                bool finite = true;
                for (size_t i = 4; i < 16; i += 4)
                    finite = finite && std::isfinite(client_->codec().readFloat(payload.data()+i));
                latest_request_id_ = client_->codec().readU32(payload.data());
                has_path_request_ = true;
                if (!finite) {
                    auto failure = std::make_shared<std_msgs::msg::UInt8MultiArray>();
                    client_->codec().pushU32(failure->data, latest_request_id_);
                    failure->data.insert(failure->data.end(), {2,0,0,0,0,0,0,0});
                    onPathResponse(failure);
                    std_msgs::msg::UInt8MultiArray invalid; invalid.data = payload;
                    path_request_pub_->publish(invalid);
                    continue;
                }
                std_msgs::msg::UInt8MultiArray request;
                request.data = payload;
                path_request_pub_->publish(request);
                RCLCPP_INFO(get_logger(), "UE path request %u", latest_request_id_.load());
                continue;
            }
            if (command != 4) continue;
            // 检查是否为遥测反馈包
            if (payload.size() >= 4) {
                uint32_t head = client_->codec().readU32(payload.data());
                if (head == HEAD_FEEDBACK) {
                    parseAndPublish(payload);
                } else {
                    RCLCPP_DEBUG(this->get_logger(),
                        "收到非反馈数据, head=0x%08X, cmd=%u, size=%zu",
                        head, command, payload.size());
                }
            }
        } else if (first4 == HEAD_FEEDBACK) {
            // ---- 裸遥测反馈: 车端直接发送 payload，无包头包装 ----
            std::vector<uint8_t> payload;
            if (!client_->recvExact(payload, TELEMETRY_MIN_SIZE, running_)) {
                if (running_) {
                    RCLCPP_WARN(this->get_logger(), "接收遥测数据失败，尝试重连...");
                    reconnect();
                }
                continue;
            }
            parseAndPublish(payload);
        } else {
            // 未知格式，丢弃 1 字节后重新同步
            client_->discardOneByte();
            RCLCPP_DEBUG(this->get_logger(),
                "未知数据头: 0x%08X, 丢弃 1 字节重新同步", first4);
        }
    }

    RCLCPP_INFO(this->get_logger(), "遥测接收线程退出");
}

// ============ 解析遥测并发布 ============
void LunarCarNode::parseAndPublish(const std::vector<uint8_t> & payload)
{
    // 解析遥测数据 (左手系→右手系) 并发布
    lunar_car_ctrl::msg::VehicleTelemetry msg;
    if (!parseTelemetryPayload(payload, client_->codec(), this->now(), msg)) {
        RCLCPP_DEBUG(this->get_logger(),
            "遥测数据不足: %zu / %zu 字节", payload.size(), TELEMETRY_MIN_SIZE);
        return;
    }
    telemetry_pub_->publish(msg);

    // 发布标准 ROS 消息 (Odometry + PoseStamped, 使用右手系坐标)
    publishOdomAndPose(msg);
}

// ============================================================
//  发布标准 ROS 里程计和位姿消息
// ============================================================

void LunarCarNode::publishOdomAndPose(const lunar_car_ctrl::msg::VehicleTelemetry & tel)
{
    auto stamp = tel.header.stamp;

    // ---- PoseStamped ----
    auto pose_msg = geometry_msgs::msg::PoseStamped();
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "odom";
    pose_msg.pose.position.x = tel.x;
    pose_msg.pose.position.y = tel.y;
    pose_msg.pose.position.z = tel.z;
    pose_msg.pose.orientation.x = tel.qx;
    pose_msg.pose.orientation.y = tel.qy;
    pose_msg.pose.orientation.z = tel.qz;
    pose_msg.pose.orientation.w = tel.qw;
    pose_pub_->publish(pose_msg);

    // ---- Odometry ----
    auto odom_msg = nav_msgs::msg::Odometry();
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_link";

    // 位姿
    odom_msg.pose.pose = pose_msg.pose;

    // 速度 (车体坐标系, 右手系)
    odom_msg.twist.twist.linear.x = tel.vx;
    odom_msg.twist.twist.linear.y = tel.vy;
    odom_msg.twist.twist.linear.z = tel.vz;
    odom_msg.twist.twist.angular.x = tel.wx;
    odom_msg.twist.twist.angular.y = tel.wy;
    odom_msg.twist.twist.angular.z = tel.wz;

    odom_pub_->publish(odom_msg);
}

} // namespace lunar_car_ctrl

// ============================================================
//  main 入口
// ============================================================

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<lunar_car_ctrl::LunarCarNode> node;
    try {
        node = std::make_shared<lunar_car_ctrl::LunarCarNode>();
    } catch (const std::exception & e) {
        // 初始连接失败等致命错误: 节点退出, 返回非 0 状态码 (便于脚本/launch 检测)
        fprintf(stderr, "[lunar_car_node] 初始化失败: %s\n", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
