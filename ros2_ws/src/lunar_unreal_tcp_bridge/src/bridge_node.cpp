#include "lunar_unreal_tcp_bridge/bridge_node.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_navigation_msgs/msg/localization_status.hpp>
#include <lunar_navigation_msgs/msg/motion_execution_feedback.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_unreal_tcp_bridge/coordinate_transform.hpp"
#include "lunar_unreal_tcp_bridge/ros_conversions.hpp"

namespace lunar::unreal_tcp {
namespace {

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using SteadyTime = std::chrono::steady_clock::time_point;

constexpr std::chrono::milliseconds kDrainPeriod{10};
constexpr std::chrono::milliseconds kWatchdogPeriod{100};
constexpr std::size_t kMaximumQueuedEvents = 66U;

[[nodiscard]] std::vector<double> Identity4() {
  return {
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
  };
}

[[nodiscard]] std::vector<double> DefaultCovariance(
    const double diagonal) {
  std::vector<double> result(36U, 0.0);
  for (std::size_t index = 0U; index < 6U; ++index) {
    result[index * 6U + index] = diagonal;
  }
  return result;
}

[[nodiscard]] Eigen::Matrix3d Matrix3(
    const std::vector<double>& values) {
  if (values.size() != 9U) {
    throw std::invalid_argument("BASIS_SIZE_INVALID");
  }
  Eigen::Matrix3d result;
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      result(static_cast<Eigen::Index>(row),
             static_cast<Eigen::Index>(column)) = values[row * 3U + column];
    }
  }
  return result;
}

[[nodiscard]] Eigen::Matrix4d Matrix4(
    const std::vector<double>& values) {
  if (values.size() != 16U) {
    throw std::invalid_argument("CALIBRATION_SIZE_INVALID");
  }
  Eigen::Matrix4d result;
  for (std::size_t row = 0U; row < 4U; ++row) {
    for (std::size_t column = 0U; column < 4U; ++column) {
      result(static_cast<Eigen::Index>(row),
             static_cast<Eigen::Index>(column)) = values[row * 4U + column];
    }
  }
  return result;
}

[[nodiscard]] std::array<double, 36> CovarianceArray(
    const std::vector<double>& values) {
  if (values.size() != 36U) {
    throw std::invalid_argument("COVARIANCE_SIZE_INVALID");
  }
  std::array<double, 36> result{};
  std::copy(values.begin(), values.end(), result.begin());
  return result;
}

[[nodiscard]] builtin_interfaces::msg::Time Stamp(
    const std::int64_t nanoseconds) {
  builtin_interfaces::msg::Time result;
  if (nanoseconds > 0) {
    result.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
    result.nanosec = static_cast<std::uint32_t>(
        nanoseconds % 1'000'000'000LL);
  }
  return result;
}

[[nodiscard]] std::string StateName(const SessionState state) {
  switch (state) {
    case SessionState::kDisconnected:
      return "DISCONNECTED";
    case SessionState::kHandshaking:
      return "HANDSHAKING";
    case SessionState::kSyncing:
      return "SYNCING";
    case SessionState::kReady:
      return "READY";
    case SessionState::kExecuting:
      return "EXECUTING";
    case SessionState::kHold:
      return "HOLD";
  }
  return "UNKNOWN";
}

}  // namespace

class BridgeNode::Impl final {
 public:
  Impl(
      BridgeNode& node, BridgeNodeDependencies dependencies)
      : node_(node),
        dependencies_(std::move(dependencies)),
        now_(dependencies_.steady_now
                 ? dependencies_.steady_now
                 : [] { return std::chrono::steady_clock::now(); }) {
    DeclareParameters();
    parameter_callback_ = node_.add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>&) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = !configured_;
          result.reason = configured_
              ? "BRIDGE_PARAMETERS_FROZEN_AFTER_CONFIGURE"
              : "ACCEPTED";
          return result;
        });
  }

  ~Impl() {
    if (transport_) {
      transport_->SetCallbacks({});
      transport_->Stop();
    }
  }

  CallbackReturn Configure() {
    try {
      const auto basis =
          node_.get_parameter("basis_map_from_unreal").as_double_array();
      const auto origin = node_.get_parameter(
          "unreal_world_origin_in_map_m").as_double_array();
      expected_base_link_ = node_.get_parameter(
          "base_link_from_unreal_root").as_double_array();
      expected_base_footprint_ = node_.get_parameter(
          "base_footprint_from_base_link").as_double_array();
      if (origin.size() != 3U) {
        throw std::invalid_argument("ORIGIN_SIZE_INVALID");
      }
      CoordinateTransformConfig transform_config;
      transform_config.basis_map_from_unreal = Matrix3(basis);
      transform_config.unreal_world_origin_in_map_m = {
          origin[0], origin[1], origin[2]};
      transform_config.length_unit_to_m =
          node_.get_parameter("length_unit_to_m").as_double();
      transform_config.base_link_from_unreal_root =
          Matrix4(expected_base_link_);
      transform_config.base_footprint_from_base_link =
          Matrix4(expected_base_footprint_);
      transform_config.calibration_hash =
          node_.get_parameter("calibration_hash").as_string();
      transform_.emplace(
          std::move(transform_config),
          node_.get_parameter("calibration_hash").as_string());
      covariance_profile_ = SimulationCovarianceProfile{
          .pose = CovarianceArray(node_.get_parameter(
              "simulation_pose_covariance").as_double_array()),
          .twist = CovarianceArray(node_.get_parameter(
              "simulation_twist_covariance").as_double_array()),
      };
      heartbeat_timeout_ = std::chrono::seconds{
          PositiveWholeSeconds("heartbeat_timeout_s")};
      freshness_timeout_ = std::chrono::duration_cast<
          std::chrono::steady_clock::duration>(
          std::chrono::duration<double>{
              PositiveDouble("freshness_timeout_s")});
      maximum_snapshot_skew_ns_ = static_cast<std::int64_t>(
          PositiveDouble("maximum_snapshot_skew_s") * 1.0e9);

      transport_ = dependencies_.transport;
      if (!transport_) {
        const std::int64_t port =
            node_.get_parameter("server_port").as_int();
        if (port <= 0 || port > std::numeric_limits<std::uint16_t>::max()) {
          throw std::invalid_argument("SERVER_PORT_INVALID");
        }
        transport_ = std::make_shared<TcpClient>(TcpClientConfig{
            .server_host =
                node_.get_parameter("server_host").as_string(),
            .server_port = static_cast<std::uint16_t>(port),
        });
      }
      transport_->SetCallbacks(TransportCallbacks{
          .on_connected = [this] { Enqueue(Event::Connected(now_())); },
          .on_frame = [this](Frame frame) {
            Enqueue(Event::Received(std::move(frame), now_()));
          },
          .on_disconnected = [this](std::string reason) {
            Enqueue(Event::Disconnected(std::move(reason), now_()));
          },
      });
      CreateRosEntities();
      configured_ = true;
      last_reason_ = "CONFIGURED";
      return CallbackReturn::SUCCESS;
    } catch (const std::exception& error) {
      last_reason_ = std::string{"CONFIGURE_FAILED:"} + error.what();
      CleanupEntities();
      return CallbackReturn::FAILURE;
    }
  }

  CallbackReturn Activate() {
    if (!configured_ || !transport_) {
      last_reason_ = "ACTIVATE_NOT_CONFIGURED";
      return CallbackReturn::FAILURE;
    }
    ActivatePublishers();
    active_ = true;
    drain_timer_->reset();
    watchdog_timer_->reset();
    transport_->Start();
    last_reason_ = "ACTIVE";
    PublishStatus();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Deactivate(std::string reason) {
    if (active_) {
      ForceHold(std::move(reason));
    }
    active_ = false;
    if (drain_timer_) {
      drain_timer_->cancel();
    }
    if (watchdog_timer_) {
      watchdog_timer_->cancel();
    }
    if (transport_) {
      transport_->Stop();
    }
    DeactivatePublishers();
    session_.Reset();
    session_started_at_.reset();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Cleanup() {
    if (active_) {
      (void)Deactivate("LIFECYCLE_CLEANUP");
    }
    if (transport_) {
      transport_->SetCallbacks({});
      transport_->Stop();
    }
    CleanupEntities();
    transform_.reset();
    transport_.reset();
    configured_ = false;
    session_.Reset();
    session_started_at_.reset();
    return CallbackReturn::SUCCESS;
  }

  void DrainEvents() {
    std::deque<Event> events;
    {
      std::scoped_lock lock{events_mutex_};
      events.swap(events_);
    }
    for (Event& event : events) {
      if (!active_) {
        continue;
      }
      if (event.kind == EventKind::kConnected) {
        HandleConnected();
      } else if (event.kind == EventKind::kDisconnected) {
        HandleDisconnected(std::move(event.reason));
      } else if (event.kind == EventKind::kFatal) {
        ForceHold(std::move(event.reason));
      } else if (event.frame.has_value()) {
        HandleFrame(std::move(*event.frame), event.received_at);
      }
    }
  }

  void CheckWatchdogs() {
    if (!active_ || !session_.session().has_value() ||
        session_.state() == SessionState::kHold) {
      return;
    }
    const SteadyTime now = now_();
    if (const auto heartbeat =
            session_.CheckHeartbeat(now, heartbeat_timeout_)) {
      ForceHold(*heartbeat);
      return;
    }
    if ((last_state_received_.has_value() &&
         now - *last_state_received_ > freshness_timeout_) ||
        (last_map_received_.has_value() &&
         now - *last_map_received_ > freshness_timeout_) ||
        (session_started_at_.has_value() &&
         now - *session_started_at_ > freshness_timeout_ &&
         (!last_state_received_.has_value() ||
          !last_map_received_.has_value()))) {
      ForceHold("STATE_OR_MAP_STALE");
    }
  }

  void ReceiveMotionReference(
      const lunar_planning_msgs::msg::MotionReference& message) {
    if (!active_ || !transport_ || !transport_->connected() ||
        session_.state() != SessionState::kReady ||
        !session_.session().has_value() || !transform_.has_value()) {
      ForceHold("REFERENCE_SESSION_NOT_READY");
      return;
    }
    if (active_plan_id_.has_value()) {
      ForceHold("REFERENCE_ALREADY_ACTIVE");
      return;
    }
    const std::uint64_t reference_sequence =
        session_.next_outgoing_sequence();
    auto converted = ConvertMotionReference(
        message, *transform_, *session_.session(), reference_sequence);
    if (!converted.ok()) {
      ForceHold(converted.reason_code);
      return;
    }
    Frame hold = BuildControlFrame("HOLD");
    QueuePushResult sent = transport_->Send(
        std::move(*converted.value), std::move(hold));
    if (!sent.accepted) {
      ForceHold(sent.reason_code, false);
      return;
    }
    active_plan_id_ = message.plan_id;
    active_feedback_sequence_ = 0U;
    last_reason_ = "REFERENCE_SENT";
    PublishStatus();
  }

  [[nodiscard]] SessionState session_state() const noexcept {
    return session_.state();
  }

  [[nodiscard]] std::string last_reason() const {
    return last_reason_;
  }

  [[nodiscard]] bool configured() const noexcept { return configured_; }

 private:
  enum class EventKind : std::uint8_t {
    kConnected,
    kFrame,
    kDisconnected,
    kFatal,
  };

  struct Event final {
    EventKind kind{EventKind::kConnected};
    std::optional<Frame> frame;
    std::string reason;
    SteadyTime received_at{};

    static Event Connected(const SteadyTime now) {
      return Event{.kind = EventKind::kConnected, .received_at = now};
    }
    static Event Received(Frame frame, const SteadyTime now) {
      return Event{
          .kind = EventKind::kFrame,
          .frame = std::move(frame),
          .received_at = now,
      };
    }
    static Event Disconnected(std::string reason, const SteadyTime now) {
      return Event{
          .kind = EventKind::kDisconnected,
          .reason = std::move(reason),
          .received_at = now,
      };
    }
    static Event Fatal(std::string reason, const SteadyTime now) {
      return Event{
          .kind = EventKind::kFatal,
          .reason = std::move(reason),
          .received_at = now,
      };
    }
  };

  void DeclareParameters() {
    node_.declare_parameter<std::string>("server_host", "127.0.0.1");
    node_.declare_parameter<std::int64_t>("server_port", 47001);
    node_.declare_parameter<std::vector<double>>(
        "basis_map_from_unreal",
        {1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0});
    node_.declare_parameter<std::vector<double>>(
        "unreal_world_origin_in_map_m", {0.0, 0.0, 0.0});
    node_.declare_parameter<double>("length_unit_to_m", 0.01);
    node_.declare_parameter<std::vector<double>>(
        "base_link_from_unreal_root", Identity4());
    node_.declare_parameter<std::vector<double>>(
        "base_footprint_from_base_link", Identity4());
    node_.declare_parameter<std::string>("calibration_hash", "");
    node_.declare_parameter<std::vector<double>>(
        "simulation_pose_covariance", DefaultCovariance(0.01));
    node_.declare_parameter<std::vector<double>>(
        "simulation_twist_covariance", DefaultCovariance(0.01));
    node_.declare_parameter<double>("heartbeat_timeout_s", 3.0);
    node_.declare_parameter<double>("freshness_timeout_s", 1.0);
    node_.declare_parameter<double>("maximum_snapshot_skew_s", 0.2);
  }

  [[nodiscard]] double PositiveDouble(const std::string& name) const {
    const double value = node_.get_parameter(name).as_double();
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::invalid_argument(name + "_INVALID");
    }
    return value;
  }

  [[nodiscard]] std::int64_t PositiveWholeSeconds(
      const std::string& name) const {
    const double value = PositiveDouble(name);
    if (std::trunc(value) != value ||
        value > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument(name + "_INVALID");
    }
    return static_cast<std::int64_t>(value);
  }

  void CreateRosEntities() {
    clock_publisher_ = node_.create_publisher<rosgraph_msgs::msg::Clock>(
        "/clock", rclcpp::QoS{10}.best_effort());
    observed_map_publisher_ =
        node_.create_publisher<grid_map_msgs::msg::GridMap>(
            "/lunar/unreal/observed_elevation", rclcpp::QoS{5}.reliable());
    odometry_publisher_ = node_.create_publisher<nav_msgs::msg::Odometry>(
        "/localization/odometry", rclcpp::QoS{10}.reliable());
    wheeled_odometry_publisher_ =
        node_.create_publisher<nav_msgs::msg::Odometry>(
            "/lunar/unreal/wheeled_odometry", rclcpp::QoS{10}.reliable());
    localization_status_publisher_ = node_.create_publisher<
        lunar_navigation_msgs::msg::LocalizationStatus>(
        "/localization/status", rclcpp::QoS{10}.reliable());
    tf_publisher_ = node_.create_publisher<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::QoS{20}.reliable());
    tf_static_publisher_ = node_.create_publisher<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::QoS{1}.reliable().transient_local());
    feedback_publisher_ = node_.create_publisher<
        lunar_navigation_msgs::msg::MotionExecutionFeedback>(
        "/execution/motion_feedback", rclcpp::QoS{10}.reliable());
    status_publisher_ = node_.create_publisher<
        diagnostic_msgs::msg::DiagnosticArray>(
        "/lunar/unreal/status", rclcpp::QoS{10}.reliable());

    reference_subscription_ = node_.create_subscription<
        lunar_planning_msgs::msg::MotionReference>(
        "/lunar/motion_reference", rclcpp::QoS{10}.reliable(),
        [this](
            const lunar_planning_msgs::msg::MotionReference::SharedPtr message) {
          ReceiveMotionReference(*message);
        });
    start_service_ = CreateControlService("start", "START");
    hold_service_ = CreateControlService("hold", "HOLD");
    resume_service_ = CreateControlService("resume", "RESUME");
    reset_service_ = CreateControlService(
        "reset_session", "RESET_SESSION");
    drain_timer_ = node_.create_wall_timer(
        kDrainPeriod, [this] { DrainEvents(); });
    watchdog_timer_ = node_.create_wall_timer(
        kWatchdogPeriod, [this] { CheckWatchdogs(); });
    drain_timer_->cancel();
    watchdog_timer_->cancel();
  }

  [[nodiscard]] rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr
  CreateControlService(std::string name, std::string command) {
    return node_.create_service<std_srvs::srv::Trigger>(
        "/lunar/unreal/" + name,
        [this, command = std::move(command)](
            const std_srvs::srv::Trigger::Request::SharedPtr,
            std_srvs::srv::Trigger::Response::SharedPtr response) {
          if (!active_ || !transport_ || !transport_->connected() ||
              !session_.session().has_value()) {
            response->success = false;
            response->message = "SESSION_NOT_READY";
            return;
          }
          Frame frame = BuildControlFrame(command);
          const QueuePushResult result =
              transport_->Send(frame, command == "HOLD"
                  ? std::optional<Frame>{frame}
                  : std::nullopt);
          response->success = result.accepted;
          response->message = result.reason_code;
          if (result.accepted && command == "RESET_SESSION") {
            session_.Reset();
            active_plan_id_.reset();
          }
        });
  }

  void ActivatePublishers() {
    clock_publisher_->on_activate();
    observed_map_publisher_->on_activate();
    odometry_publisher_->on_activate();
    wheeled_odometry_publisher_->on_activate();
    localization_status_publisher_->on_activate();
    tf_publisher_->on_activate();
    tf_static_publisher_->on_activate();
    feedback_publisher_->on_activate();
    status_publisher_->on_activate();
  }

  void DeactivatePublishers() {
    if (!configured_) {
      return;
    }
    clock_publisher_->on_deactivate();
    observed_map_publisher_->on_deactivate();
    odometry_publisher_->on_deactivate();
    wheeled_odometry_publisher_->on_deactivate();
    localization_status_publisher_->on_deactivate();
    tf_publisher_->on_deactivate();
    tf_static_publisher_->on_deactivate();
    feedback_publisher_->on_deactivate();
    status_publisher_->on_deactivate();
  }

  void CleanupEntities() {
    drain_timer_.reset();
    watchdog_timer_.reset();
    reference_subscription_.reset();
    start_service_.reset();
    hold_service_.reset();
    resume_service_.reset();
    reset_service_.reset();
    clock_publisher_.reset();
    observed_map_publisher_.reset();
    odometry_publisher_.reset();
    wheeled_odometry_publisher_.reset();
    localization_status_publisher_.reset();
    tf_publisher_.reset();
    tf_static_publisher_.reset();
    feedback_publisher_.reset();
    status_publisher_.reset();
  }

  void Enqueue(Event event) {
    std::scoped_lock lock{events_mutex_};
    if (event.kind == EventKind::kFrame && event.frame.has_value() &&
        (event.frame->header.message_type == MessageType::kRobotState ||
         event.frame->header.message_type ==
             MessageType::kLocalElevationMap)) {
      const MessageType type = event.frame->header.message_type;
      const auto existing = std::find_if(
          events_.rbegin(), events_.rend(), [type](const Event& queued) {
            return queued.kind == EventKind::kFrame &&
                queued.frame.has_value() &&
                queued.frame->header.message_type == type;
          });
      if (existing != events_.rend()) {
        *existing = std::move(event);
        return;
      }
    }
    if (events_.size() >= kMaximumQueuedEvents) {
      events_.clear();
      events_.push_back(Event::Fatal(
          "INBOUND_RELIABLE_QUEUE_EXHAUSTED", event.received_at));
      return;
    }
    events_.push_back(std::move(event));
  }

  void HandleConnected() {
    session_.Reset();
    active_plan_id_.reset();
    active_feedback_sequence_ = 0U;
    last_state_received_.reset();
    last_map_received_.reset();
    session_started_at_.reset();
    last_state_simulation_ns_.reset();
    last_map_simulation_ns_.reset();
    QueuePushResult result = transport_->Send(session_.BuildHello());
    if (!result.accepted) {
      ForceHold(result.reason_code, false);
      return;
    }
    last_reason_ = "HANDSHAKING";
    PublishStatus();
  }

  void HandleDisconnected(std::string reason) {
    session_.Reset();
    session_started_at_.reset();
    active_plan_id_.reset();
    last_reason_ = reason.empty() ? "TCP_DISCONNECTED" : std::move(reason);
    PublishLocalization(false, builtin_interfaces::msg::Time{});
    PublishStatus();
  }

  [[nodiscard]] bool CalibrationMatches(const FrozenSession& frozen) const {
    if (!transform_.has_value() ||
        frozen.calibration_hash != transform_->calibration_hash() ||
        std::abs(frozen.length_unit_to_m -
                 transform_->length_unit_to_m()) > 1.0e-12 ||
        frozen.coordinate_convention != "UE_NATIVE" ||
        frozen.handedness != "LEFT" || frozen.up_axis != "Z") {
      return false;
    }
    const auto matches = [](const std::array<double, 16>& actual,
                            const std::vector<double>& expected) {
      return expected.size() == actual.size() &&
          std::equal(actual.begin(), actual.end(), expected.begin(),
              [](const double left, const double right) {
                return std::abs(left - right) <= 1.0e-12;
              });
    };
    return matches(frozen.base_link_from_unreal_root, expected_base_link_) &&
        matches(
            frozen.base_footprint_from_base_link,
            expected_base_footprint_);
  }

  void HandleFrame(Frame frame, const SteadyTime received_at) {
    if (frame.header.message_type == MessageType::kHelloAck) {
      const auto accepted = session_.AcceptHelloAck(frame, received_at);
      if (!accepted.ok()) {
        ForceHold(accepted.reason_code, false);
        return;
      }
      if (!CalibrationMatches(*session_.session())) {
        ForceHold("CALIBRATION_MISMATCH");
        return;
      }
      session_started_at_ = received_at;
      last_reason_ = "SYNCING";
      PublishClock(frame.header.simulation_time_ns);
      PublishStatus();
      return;
    }
    const SessionAcceptResult accepted =
        session_.AcceptIncoming(frame, received_at);
    if (!accepted.ok()) {
      ForceHold(accepted.reason_code);
      return;
    }
    PublishClock(frame.header.simulation_time_ns);
    switch (frame.header.message_type) {
      case MessageType::kRobotState:
        HandleRobotState(frame, received_at);
        break;
      case MessageType::kLocalElevationMap:
        HandleMap(frame, received_at);
        break;
      case MessageType::kExecutionFeedback:
        HandleFeedback(frame);
        break;
      case MessageType::kError:
        ForceHold(frame.metadata.at("reason_code").get<std::string>());
        return;
      default:
        break;
    }
    if (session_.state() == SessionState::kReady) {
      last_reason_ = "READY";
      PublishStatus();
    }
  }

  void HandleRobotState(const Frame& frame, const SteadyTime received_at) {
    auto converted = ConvertRobotState(
        frame, *transform_, covariance_profile_);
    if (!converted.ok()) {
      ForceHold(converted.reason_code);
      return;
    }
    last_state_received_ = received_at;
    last_state_simulation_ns_ = frame.header.simulation_time_ns;
    if (active_) {
      odometry_publisher_->publish(converted.value->external_odometry);
      wheeled_odometry_publisher_->publish(
          converted.value->wheeled_odometry);
      tf_publisher_->publish(converted.value->dynamic_tf);
      tf_static_publisher_->publish(converted.value->static_tf);
      PublishLocalization(true, converted.value->external_odometry.header.stamp);
    }
    CheckSnapshotSkew();
  }

  void HandleMap(const Frame& frame, const SteadyTime received_at) {
    auto converted = ConvertObservedElevation(frame, *transform_);
    if (!converted.ok()) {
      ForceHold(converted.reason_code);
      return;
    }
    last_map_received_ = received_at;
    last_map_simulation_ns_ = frame.header.simulation_time_ns;
    if (active_) {
      observed_map_publisher_->publish(std::move(*converted.value));
    }
    CheckSnapshotSkew();
  }

  void HandleFeedback(const Frame& frame) {
    const std::string plan_id =
        frame.metadata.at("plan_id").get<std::string>();
    const std::uint64_t feedback_sequence =
        frame.metadata.at("sequence").get<std::uint64_t>();
    if (!active_plan_id_.has_value() ||
        plan_id != *active_plan_id_ ||
        feedback_sequence != active_feedback_sequence_ + 1U) {
      ForceHold("EXECUTION_FEEDBACK_IDENTITY_INVALID");
      return;
    }
    auto converted = ConvertExecutionFeedback(
        frame, session_.session()->session_id);
    if (!converted.ok()) {
      ForceHold(converted.reason_code);
      return;
    }
    active_feedback_sequence_ = feedback_sequence;
    const std::uint8_t state = converted.value->state;
    if (active_) {
      feedback_publisher_->publish(*converted.value);
    }
    using Feedback = lunar_navigation_msgs::msg::MotionExecutionFeedback;
    if (state == Feedback::SEGMENT_COMPLETE) {
      active_plan_id_.reset();
      active_feedback_sequence_ = 0U;
    } else if (state == Feedback::FAILED || state == Feedback::CANCELED) {
      ForceHold(converted.value->reason_code);
    }
  }

  void CheckSnapshotSkew() {
    if (last_state_simulation_ns_.has_value() &&
        last_map_simulation_ns_.has_value() &&
        std::llabs(*last_state_simulation_ns_ - *last_map_simulation_ns_) >
            maximum_snapshot_skew_ns_) {
      ForceHold("STATE_MAP_SNAPSHOT_SKEW");
    }
  }

  [[nodiscard]] Frame BuildControlFrame(const std::string_view command) {
    Frame frame;
    frame.header.message_type = MessageType::kControl;
    frame.header.sequence = session_.next_outgoing_sequence();
    frame.metadata = {
        {"session_id", session_.session()->session_id},
        {"command", command},
        {"command_id", std::string{command} + "-" +
            std::to_string(frame.header.sequence)},
    };
    return frame;
  }

  void ForceHold(std::string reason, const bool send_control = true) {
    const bool already_same =
        session_.state() == SessionState::kHold && last_reason_ == reason;
    session_.ForceHold();
    active_plan_id_.reset();
    active_feedback_sequence_ = 0U;
    last_reason_ = std::move(reason);
    if (send_control && !already_same && transport_ &&
        transport_->connected() && session_.session().has_value()) {
      Frame hold = BuildControlFrame("HOLD");
      const QueuePushResult sent = transport_->Send(hold, hold);
      if (!sent.accepted) {
        last_reason_ = sent.reason_code;
      }
    }
    PublishLocalization(false, builtin_interfaces::msg::Time{});
    PublishStatus();
  }

  void PublishClock(const std::int64_t simulation_time_ns) {
    if (!active_ || simulation_time_ns <= 0) {
      return;
    }
    rosgraph_msgs::msg::Clock clock;
    clock.clock = Stamp(simulation_time_ns);
    clock_publisher_->publish(clock);
  }

  void PublishLocalization(
      const bool valid, builtin_interfaces::msg::Time stamp) {
    if (!active_ || !localization_status_publisher_ ||
        !localization_status_publisher_->is_activated()) {
      return;
    }
    if (stamp.sec == 0 && stamp.nanosec == 0U) {
      if (last_state_simulation_ns_.has_value()) {
        stamp = Stamp(*last_state_simulation_ns_);
      } else {
        return;
      }
    }
    lunar_navigation_msgs::msg::LocalizationStatus status;
    status.header.stamp = stamp;
    status.header.frame_id = "odom";
    status.status = valid ? status.VALID : status.INVALID;
    localization_status_publisher_->publish(status);
  }

  void PublishStatus() {
    if (!active_ || !status_publisher_ ||
        !status_publisher_->is_activated()) {
      return;
    }
    diagnostic_msgs::msg::DiagnosticArray array;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "lunar_unreal_tcp_bridge";
    status.hardware_id = session_.session().has_value()
        ? session_.session()->robot_id
        : "unconnected";
    status.level = session_.state() == SessionState::kHold ||
            session_.state() == SessionState::kDisconnected
        ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
        : diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = StateName(session_.state()) + ":" + last_reason_;
    array.status.push_back(std::move(status));
    status_publisher_->publish(array);
  }

  BridgeNode& node_;
  BridgeNodeDependencies dependencies_;
  std::function<SteadyTime()> now_;
  std::shared_ptr<SessionTransport> transport_;
  SessionProtocol session_;
  std::optional<CoordinateTransform> transform_;
  SimulationCovarianceProfile covariance_profile_;
  std::vector<double> expected_base_link_;
  std::vector<double> expected_base_footprint_;
  std::chrono::seconds heartbeat_timeout_{3};
  std::chrono::steady_clock::duration freshness_timeout_{
      std::chrono::seconds{1}};
  std::int64_t maximum_snapshot_skew_ns_{200'000'000LL};
  bool configured_{};
  bool active_{};
  std::string last_reason_{"UNCONFIGURED"};
  std::optional<std::string> active_plan_id_;
  std::uint64_t active_feedback_sequence_{};
  std::optional<SteadyTime> last_state_received_;
  std::optional<SteadyTime> last_map_received_;
  std::optional<SteadyTime> session_started_at_;
  std::optional<std::int64_t> last_state_simulation_ns_;
  std::optional<std::int64_t> last_map_simulation_ns_;
  std::mutex events_mutex_;
  std::deque<Event> events_;

  rclcpp_lifecycle::LifecyclePublisher<rosgraph_msgs::msg::Clock>::SharedPtr
      clock_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<grid_map_msgs::msg::GridMap>::SharedPtr
      observed_map_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr
      odometry_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr
      wheeled_odometry_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<
      lunar_navigation_msgs::msg::LocalizationStatus>::SharedPtr
      localization_status_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<tf2_msgs::msg::TFMessage>::SharedPtr
      tf_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<tf2_msgs::msg::TFMessage>::SharedPtr
      tf_static_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<
      lunar_navigation_msgs::msg::MotionExecutionFeedback>::SharedPtr
      feedback_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<
      diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      status_publisher_;
  rclcpp::Subscription<
      lunar_planning_msgs::msg::MotionReference>::SharedPtr
      reference_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr hold_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr resume_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  rclcpp::TimerBase::SharedPtr drain_timer_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameter_callback_;
};

BridgeNode::BridgeNode(
    const rclcpp::NodeOptions& options,
    BridgeNodeDependencies dependencies)
    : rclcpp_lifecycle::LifecycleNode(
          "lunar_unreal_tcp_bridge", options),
      impl_(std::make_unique<Impl>(*this, std::move(dependencies))) {}

BridgeNode::~BridgeNode() = default;

BridgeNode::CallbackReturn BridgeNode::on_configure(
    const rclcpp_lifecycle::State&) {
  return impl_->Configure();
}

BridgeNode::CallbackReturn BridgeNode::on_activate(
    const rclcpp_lifecycle::State&) {
  return impl_->Activate();
}

BridgeNode::CallbackReturn BridgeNode::on_deactivate(
    const rclcpp_lifecycle::State&) {
  return impl_->Deactivate("LIFECYCLE_DEACTIVATED");
}

BridgeNode::CallbackReturn BridgeNode::on_cleanup(
    const rclcpp_lifecycle::State&) {
  return impl_->Cleanup();
}

BridgeNode::CallbackReturn BridgeNode::on_shutdown(
    const rclcpp_lifecycle::State&) {
  return impl_->Deactivate("LIFECYCLE_SHUTDOWN");
}

BridgeNode::CallbackReturn BridgeNode::on_error(
    const rclcpp_lifecycle::State&) {
  (void)impl_->Deactivate("LIFECYCLE_ERROR");
  return CallbackReturn::SUCCESS;
}

void BridgeNode::DrainEventsForTesting() {
  impl_->DrainEvents();
}

void BridgeNode::CheckWatchdogsForTesting() {
  impl_->CheckWatchdogs();
}

void BridgeNode::ReceiveMotionReferenceForTesting(
    const lunar_planning_msgs::msg::MotionReference& message) {
  impl_->ReceiveMotionReference(message);
}

SessionState BridgeNode::session_state_for_testing() const noexcept {
  return impl_->session_state();
}

std::string BridgeNode::last_reason_for_testing() const {
  return impl_->last_reason();
}

bool BridgeNode::configured_for_testing() const noexcept {
  return impl_->configured();
}

}  // namespace lunar::unreal_tcp
