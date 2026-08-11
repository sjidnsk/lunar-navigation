#include "lunar_observed_map/observed_map_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/qos.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "lunar_observed_map/grid_map_products.hpp"
#include "lunar_observed_map/sparse_observed_map.hpp"

namespace lunar::observed_map {
namespace {

using CallbackReturn = ObservedMapNode::CallbackReturn;

std::optional<std::int64_t> Nanoseconds(
    const builtin_interfaces::msg::Time& stamp) noexcept {
  if (stamp.sec < 0 || stamp.nanosec >= 1'000'000'000U ||
      (stamp.sec == 0 && stamp.nanosec == 0U)) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
      static_cast<std::int64_t>(stamp.nanosec);
}

bool IdentityQuaternion(
    const geometry_msgs::msg::Quaternion& quaternion) noexcept {
  constexpr double tolerance = 1.0e-6;
  return std::isfinite(quaternion.x) &&
      std::isfinite(quaternion.y) &&
      std::isfinite(quaternion.z) &&
      std::isfinite(quaternion.w) &&
      std::abs(quaternion.x) <= tolerance &&
      std::abs(quaternion.y) <= tolerance &&
      std::abs(quaternion.z) <= tolerance &&
      std::abs(std::abs(quaternion.w) - 1.0) <= tolerance;
}

std::optional<std::size_t> CellDimension(
    const double length, const double resolution) noexcept {
  if (!std::isfinite(length) || !std::isfinite(resolution) ||
      length <= 0.0 || resolution <= 0.0) {
    return std::nullopt;
  }
  const double value = length / resolution;
  const double rounded = std::round(value);
  if (std::abs(value - rounded) > 1.0e-6 || rounded < 1.0 ||
      rounded > static_cast<double>(
          std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(rounded);
}

struct LayerLayout final {
  bool row_major{};
  std::size_t offset{};
};

std::optional<LayerLayout> ValidateLayout(
    const std_msgs::msg::Float32MultiArray& layer,
    const std::size_t width, const std::size_t height) noexcept {
  if (layer.layout.dim.size() != 2U ||
      width > std::numeric_limits<std::size_t>::max() / height) {
    return std::nullopt;
  }
  const auto& outer = layer.layout.dim[0];
  const auto& inner = layer.layout.dim[1];
  bool row_major = false;
  if (outer.label == "column_index" && inner.label == "row_index") {
    if (outer.size != height || inner.size != width) {
      return std::nullopt;
    }
  } else if (
      outer.label == "row_index" && inner.label == "column_index") {
    row_major = true;
    if (outer.size != width || inner.size != height) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  const std::size_t count = width * height;
  if (outer.stride != count || inner.stride != inner.size ||
      layer.layout.data_offset > layer.data.size() ||
      layer.data.size() - layer.layout.data_offset != count) {
    return std::nullopt;
  }
  return LayerLayout{
      .row_major = row_major,
      .offset = layer.layout.data_offset,
  };
}

std::vector<float> DecodeLayer(
    const std_msgs::msg::Float32MultiArray& layer,
    const LayerLayout layout, const std::size_t width,
    const std::size_t height, const std::size_t outer_start,
    const std::size_t inner_start) {
  std::vector<float> output(width * height);
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t unwrapped_row = width - 1U - x;
      const std::size_t unwrapped_column = height - 1U - y;
      const std::size_t physical_row =
          (unwrapped_row + outer_start) % width;
      const std::size_t physical_column =
          (unwrapped_column + inner_start) % height;
      const std::size_t source = layout.row_major
          ? physical_row * height + physical_column
          : physical_column * width + physical_row;
      output[y * width + x] = layer.data[layout.offset + source];
    }
  }
  return output;
}

std::string StateName(const ObservedMapNodeState state) {
  switch (state) {
    case ObservedMapNodeState::kUnconfigured:
      return "UNCONFIGURED";
    case ObservedMapNodeState::kInactive:
      return "INACTIVE";
    case ObservedMapNodeState::kSyncing:
      return "SYNCING";
    case ObservedMapNodeState::kReady:
      return "READY";
    case ObservedMapNodeState::kError:
      return "ERROR";
  }
  return "UNKNOWN";
}

}  // namespace

class ObservedMapNode::Impl final {
 public:
  explicit Impl(ObservedMapNode& node) : node_(node) {
    DeclareParameters();
    parameter_callback_ = node_.add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>&) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = !configured_;
          result.reason = configured_
              ? "MAP_PARAMETERS_FROZEN_AFTER_CONFIGURE"
              : "ACCEPTED";
          return result;
        });
  }

  CallbackReturn Configure() {
    try {
      ValidateFrozenParameters();
      const std::size_t maximum_tiles = PositiveSize("maximum_tiles");
      sparse_map_ = std::make_unique<SparseObservedMap>(maximum_tiles);
      products_ = std::make_unique<GridMapProducts>(
          GridMapProductConfig{
              .local_axis_cells = PositiveSize("local_axis_cells"),
              .target_axis_cells = PositiveSize("target_axis_cells"),
              .maximum_axis_cells = PositiveSize("maximum_axis_cells"),
              .maximum_total_cells =
                  PositiveSize("maximum_total_cells"),
              .maximum_global_axis_m =
                  PositiveDouble("maximum_global_axis_m"),
              .planning_boundary_cells =
                  NonNegativeSize("planning_boundary_cells"),
          },
          ObstacleClassifier{ObstacleClassifierConfig{
              .maximum_local_obstacle_relief_m = PositiveDouble(
                  "maximum_local_obstacle_relief_m"),
              .resolution_m = kL0ResolutionM,
          }});
      maximum_snapshot_skew_ns_ = static_cast<std::int64_t>(
          PositiveDouble("maximum_snapshot_skew_s") * 1.0e9);
      CreateRosEntities();
      const std::string initial_session =
          node_.get_parameter("initial_session_id").as_string();
      sparse_map_->ResetSession(initial_session);
      configured_ = true;
      state_ = ObservedMapNodeState::kInactive;
      last_reason_ = "CONFIGURED";
      return CallbackReturn::SUCCESS;
    } catch (const std::exception& error) {
      last_reason_ = error.what();
      CleanupEntities();
      sparse_map_.reset();
      products_.reset();
      configured_ = false;
      state_ = ObservedMapNodeState::kUnconfigured;
      return CallbackReturn::FAILURE;
    }
  }

  CallbackReturn Activate() {
    if (!configured_ || !sparse_map_ || !products_) {
      return CallbackReturn::FAILURE;
    }
    active_ = true;
    local_publisher_->on_activate();
    global_publisher_->on_activate();
    status_publisher_->on_activate();
    state_ = ObservedMapNodeState::kSyncing;
    last_reason_ = "WAITING_FOR_MAP_AND_ODOMETRY";
    TryGenerate();
    PublishStatus();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Deactivate() {
    active_ = false;
    if (local_publisher_) {
      local_publisher_->on_deactivate();
    }
    if (global_publisher_) {
      global_publisher_->on_deactivate();
    }
    if (status_publisher_) {
      status_publisher_->on_deactivate();
    }
    state_ = ObservedMapNodeState::kInactive;
    last_reason_ = "DEACTIVATED";
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn Cleanup() {
    if (active_) {
      (void)Deactivate();
    }
    CleanupEntities();
    sparse_map_.reset();
    products_.reset();
    ClearSnapshot();
    configured_ = false;
    state_ = ObservedMapNodeState::kUnconfigured;
    last_reason_ = "CLEANED_UP";
    return CallbackReturn::SUCCESS;
  }

  void ReceiveObservedMap(const grid_map_msgs::msg::GridMap& message) {
    if (!active_) {
      return;
    }
    if (!sparse_map_ || sparse_map_->session_id().empty()) {
      SetReason("MAP_SESSION_NOT_READY", ObservedMapNodeState::kSyncing);
      return;
    }
    const auto patch = ConvertObservedMap(message);
    if (!patch.has_value()) {
      SetReason("OBSERVED_MAP_INVALID", ObservedMapNodeState::kError);
      return;
    }
    const FuseResult result = sparse_map_->Fuse(*patch);
    if (!result.accepted) {
      SetReason(
          result.reason_code,
          last_global_map_.has_value()
              ? ObservedMapNodeState::kReady
              : ObservedMapNodeState::kError);
      return;
    }
    last_map_time_ns_ = patch->simulation_time_ns;
    TryGenerate();
  }

  void ReceiveOdometry(const nav_msgs::msg::Odometry& message) {
    if (!active_) {
      return;
    }
    if (!ValidOdometry(message)) {
      SetReason("WHEELED_ODOMETRY_INVALID", ObservedMapNodeState::kError);
      return;
    }
    last_odometry_ = message;
    last_odometry_time_ns_ = *Nanoseconds(message.header.stamp);
    if (!last_generated_observation_time_ns_.has_value() ||
        last_generated_observation_time_ns_ != last_map_time_ns_) {
      TryGenerate();
    }
  }

  void ReceiveGoal(const geometry_msgs::msg::PoseStamped& message) {
    if (!active_) {
      return;
    }
    if (message.header.frame_id != "map" ||
        !Nanoseconds(message.header.stamp).has_value() ||
        !std::isfinite(message.pose.position.x) ||
        !std::isfinite(message.pose.position.y) ||
        !std::isfinite(message.pose.position.z) ||
        !IdentityQuaternion(message.pose.orientation)) {
      SetReason("GOAL_INVALID", ObservedMapNodeState::kError);
      return;
    }
    goal_ = message;
    goal_time_ns_ = *Nanoseconds(message.header.stamp);
    TryGenerate();
  }

  void ReceiveBridgeStatus(
      const diagnostic_msgs::msg::DiagnosticArray& message) {
    for (const auto& status : message.status) {
      if (status.name != "lunar_unreal_tcp_bridge") {
        continue;
      }
      std::string session_id;
      std::string session_state;
      for (const auto& value : status.values) {
        if (value.key == "session_id") {
          session_id = value.value;
        } else if (value.key == "session_state") {
          session_state = value.value;
        }
      }
      if (session_id.empty() || session_state == "DISCONNECTED" ||
          session_state == "HANDSHAKING") {
        if (sparse_map_ && !sparse_map_->session_id().empty()) {
          ResetSession({});
        }
      } else if (sparse_map_ && session_id != sparse_map_->session_id()) {
        ResetSession(std::move(session_id));
      }
      return;
    }
  }

  FuseResult FusePatch(const ObservedPatch& patch) {
    if (!active_ || !sparse_map_) {
      return FuseResult{
          .accepted = false,
          .reason_code = "MAP_NODE_NOT_ACTIVE",
      };
    }
    const FuseResult result = sparse_map_->Fuse(patch);
    if (!result.accepted) {
      SetReason(
          result.reason_code,
          last_global_map_.has_value()
              ? ObservedMapNodeState::kReady
              : ObservedMapNodeState::kError);
      return result;
    }
    if (!l0_origin_map_.has_value()) {
      l0_origin_map_ = MapPoint2D{};
    }
    last_map_time_ns_ = patch.simulation_time_ns;
    TryGenerate();
    return result;
  }

  void ResetSession(std::string session_id) {
    if (!sparse_map_) {
      return;
    }
    sparse_map_->ResetSession(std::move(session_id));
    ClearSnapshot();
    RecreateProductPublishers();
    state_ = ObservedMapNodeState::kSyncing;
    last_reason_ = "SESSION_RESET";
    PublishStatus();
  }

  [[nodiscard]] std::optional<grid_map_msgs::msg::GridMap> last_local()
      const {
    return last_local_map_;
  }

  [[nodiscard]] std::optional<grid_map_msgs::msg::GridMap> last_global()
      const {
    return last_global_map_;
  }

  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }

  [[nodiscard]] ObservedMapNodeState state() const noexcept {
    return state_;
  }

  [[nodiscard]] std::string reason() const { return last_reason_; }

 private:
  void DeclareParameters() {
    node_.declare_parameter<std::string>("initial_session_id", "");
    node_.declare_parameter<double>("base_resolution_m", 0.2);
    node_.declare_parameter<std::int64_t>("local_axis_cells", 320);
    node_.declare_parameter<std::int64_t>("tile_side_cells", 256);
    node_.declare_parameter<std::int64_t>("maximum_tiles", 512);
    node_.declare_parameter<std::int64_t>("target_axis_cells", 256);
    node_.declare_parameter<std::int64_t>("maximum_axis_cells", 4096);
    node_.declare_parameter<std::int64_t>(
        "maximum_total_cells", 1'048'576);
    node_.declare_parameter<double>("maximum_global_axis_m", 1024.0);
    node_.declare_parameter<std::int64_t>("planning_boundary_cells", 2);
    node_.declare_parameter<double>("maximum_snapshot_skew_s", 0.2);
    node_.declare_parameter<double>(
        "maximum_local_obstacle_relief_m", 0.2);
  }

  [[nodiscard]] double PositiveDouble(const std::string& name) const {
    const double value = node_.get_parameter(name).as_double();
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::invalid_argument(name + "_INVALID");
    }
    return value;
  }

  [[nodiscard]] std::size_t PositiveSize(const std::string& name) const {
    const std::int64_t value = node_.get_parameter(name).as_int();
    if (value <= 0 || static_cast<std::uint64_t>(value) >
                          std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(name + "_INVALID");
    }
    return static_cast<std::size_t>(value);
  }

  [[nodiscard]] std::size_t NonNegativeSize(const std::string& name) const {
    const std::int64_t value = node_.get_parameter(name).as_int();
    if (value < 0 || static_cast<std::uint64_t>(value) >
                         std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(name + "_INVALID");
    }
    return static_cast<std::size_t>(value);
  }

  void ValidateFrozenParameters() const {
    constexpr double tolerance = 1.0e-12;
    if (std::abs(PositiveDouble("base_resolution_m") -
                 kL0ResolutionM) > tolerance ||
        PositiveSize("local_axis_cells") != 320U ||
        PositiveSize("tile_side_cells") != kTileSideCells ||
        PositiveSize("maximum_tiles") != kMaximumTilesPerSession ||
        PositiveSize("target_axis_cells") != 256U ||
        PositiveSize("maximum_axis_cells") != 4096U ||
        PositiveSize("maximum_total_cells") != 1'048'576U ||
        std::abs(PositiveDouble("maximum_global_axis_m") - 1024.0) >
            tolerance ||
        std::abs(PositiveDouble("maximum_snapshot_skew_s") - 0.2) >
            tolerance ||
        std::abs(PositiveDouble("maximum_local_obstacle_relief_m") - 0.2) >
            tolerance) {
      throw std::invalid_argument("FROZEN_MAP_PARAMETER_MISMATCH");
    }
    (void)NonNegativeSize("planning_boundary_cells");
  }

  void CreateProductPublishers() {
    const rclcpp::QoS map_qos = rclcpp::QoS{1}.reliable().transient_local();
    local_publisher_ =
        node_.create_publisher<grid_map_msgs::msg::GridMap>(
            "/environment/map_local", map_qos);
    global_publisher_ =
        node_.create_publisher<grid_map_msgs::msg::GridMap>(
            "/environment/map_global", map_qos);
  }

  void CreateRosEntities() {
    CreateProductPublishers();
    status_publisher_ = node_.create_publisher<
        diagnostic_msgs::msg::DiagnosticArray>(
        "/lunar/observed_map/status", rclcpp::QoS{10}.reliable());
    observed_subscription_ = node_.create_subscription<
        grid_map_msgs::msg::GridMap>(
        "/lunar/unreal/observed_elevation", rclcpp::QoS{1}.reliable(),
        [this](const grid_map_msgs::msg::GridMap::SharedPtr message) {
          ReceiveObservedMap(*message);
        });
    odometry_subscription_ = node_.create_subscription<nav_msgs::msg::Odometry>(
        "/lunar/unreal/wheeled_odometry", rclcpp::QoS{10}.reliable(),
        [this](const nav_msgs::msg::Odometry::SharedPtr message) {
          ReceiveOdometry(*message);
        });
    goal_subscription_ = node_.create_subscription<
        geometry_msgs::msg::PoseStamped>(
        "/goal_pose", rclcpp::QoS{1}.reliable().durability_volatile(),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
          ReceiveGoal(*message);
        });
    bridge_status_subscription_ = node_.create_subscription<
        diagnostic_msgs::msg::DiagnosticArray>(
        "/lunar/unreal/status", rclcpp::QoS{10}.reliable(),
        [this](
            const diagnostic_msgs::msg::DiagnosticArray::SharedPtr message) {
          ReceiveBridgeStatus(*message);
        });
  }

  void CleanupEntities() {
    local_publisher_.reset();
    global_publisher_.reset();
    status_publisher_.reset();
    observed_subscription_.reset();
    odometry_subscription_.reset();
    goal_subscription_.reset();
    bridge_status_subscription_.reset();
  }

  void RecreateProductPublishers() {
    if (local_publisher_ && local_publisher_->is_activated()) {
      local_publisher_->on_deactivate();
    }
    if (global_publisher_ && global_publisher_->is_activated()) {
      global_publisher_->on_deactivate();
    }
    local_publisher_.reset();
    global_publisher_.reset();
    if (!configured_) {
      return;
    }
    CreateProductPublishers();
    if (active_) {
      local_publisher_->on_activate();
      global_publisher_->on_activate();
    }
  }

  [[nodiscard]] std::optional<ObservedPatch> ConvertObservedMap(
      const grid_map_msgs::msg::GridMap& message) {
    if (message.header.frame_id != "map" ||
        !Nanoseconds(message.header.stamp).has_value() ||
        std::abs(message.info.resolution - kL0ResolutionM) > 1.0e-9 ||
        !std::isfinite(message.info.pose.position.x) ||
        !std::isfinite(message.info.pose.position.y) ||
        !std::isfinite(message.info.pose.position.z) ||
        !IdentityQuaternion(message.info.pose.orientation)) {
      return std::nullopt;
    }
    const auto width =
        CellDimension(message.info.length_x, message.info.resolution);
    const auto height =
        CellDimension(message.info.length_y, message.info.resolution);
    if (!width || !height || *width != 320U || *height != 320U ||
        message.outer_start_index >= *width ||
        message.inner_start_index >= *height ||
        message.layers.size() != message.data.size()) {
      return std::nullopt;
    }
    std::optional<std::size_t> elevation_index;
    std::optional<std::size_t> valid_index;
    for (std::size_t index = 0U; index < message.layers.size(); ++index) {
      if (message.layers[index] == "elevation" && !elevation_index) {
        elevation_index = index;
      } else if (message.layers[index] == "valid_mask" && !valid_index) {
        valid_index = index;
      } else {
        return std::nullopt;
      }
    }
    if (!elevation_index || !valid_index) {
      return std::nullopt;
    }
    const auto elevation_layout = ValidateLayout(
        message.data[*elevation_index], *width, *height);
    const auto valid_layout = ValidateLayout(
        message.data[*valid_index], *width, *height);
    if (!elevation_layout || !valid_layout) {
      return std::nullopt;
    }
    const std::vector<float> elevation = DecodeLayer(
        message.data[*elevation_index], *elevation_layout,
        *width, *height, message.outer_start_index,
        message.inner_start_index);
    const std::vector<float> valid = DecodeLayer(
        message.data[*valid_index], *valid_layout,
        *width, *height, message.outer_start_index,
        message.inner_start_index);
    const MapPoint2D patch_origin{
        message.info.pose.position.x - message.info.length_x * 0.5,
        message.info.pose.position.y - message.info.length_y * 0.5,
    };
    const bool freeze_origin = !l0_origin_map_.has_value();
    const MapPoint2D session_origin =
        freeze_origin ? patch_origin : *l0_origin_map_;
    const double cell_zero_x =
        (patch_origin.x_m - session_origin.x_m) / kL0ResolutionM;
    const double cell_zero_y =
        (patch_origin.y_m - session_origin.y_m) / kL0ResolutionM;
    const double rounded_x = std::round(cell_zero_x);
    const double rounded_y = std::round(cell_zero_y);
    if (std::abs(cell_zero_x - rounded_x) > 1.0e-6 ||
        std::abs(cell_zero_y - rounded_y) > 1.0e-6 ||
        rounded_x < static_cast<double>(
            std::numeric_limits<std::int64_t>::min()) ||
        rounded_x > static_cast<double>(
            std::numeric_limits<std::int64_t>::max()) ||
        rounded_y < static_cast<double>(
            std::numeric_limits<std::int64_t>::min()) ||
        rounded_y > static_cast<double>(
            std::numeric_limits<std::int64_t>::max())) {
      return std::nullopt;
    }
    ObservedPatch patch;
    patch.session_id = sparse_map_->session_id();
    patch.simulation_time_ns = *Nanoseconds(message.header.stamp);
    patch.cell_zero = {
        static_cast<std::int64_t>(rounded_x),
        static_cast<std::int64_t>(rounded_y)};
    patch.width = *width;
    patch.height = *height;
    patch.elevation_m.resize(elevation.size(), 0.0);
    patch.valid.resize(valid.size(), 0U);
    for (std::size_t index = 0U; index < elevation.size(); ++index) {
      if (!std::isfinite(valid[index]) ||
          (valid[index] != 0.0F && valid[index] != 1.0F) ||
          !std::isfinite(elevation[index]) ||
          (valid[index] == 0.0F && elevation[index] != 0.0F)) {
        return std::nullopt;
      }
      patch.valid[index] = static_cast<std::uint8_t>(valid[index]);
      if (patch.valid[index] != 0U) {
        patch.elevation_m[index] = elevation[index];
      }
    }
    if (freeze_origin) {
      l0_origin_map_ = session_origin;
    }
    return patch;
  }

  [[nodiscard]] bool ValidOdometry(
      const nav_msgs::msg::Odometry& message) const noexcept {
    return message.header.frame_id == "odom" &&
        message.child_frame_id == "base_footprint" &&
        Nanoseconds(message.header.stamp).has_value() &&
        std::isfinite(message.pose.pose.position.x) &&
        std::isfinite(message.pose.pose.position.y) &&
        std::isfinite(message.pose.pose.position.z) &&
        IdentityQuaternion(message.pose.pose.orientation);
  }

  void TryGenerate() {
    if (!active_ || !sparse_map_ || !products_ ||
        sparse_map_->session_id().empty() ||
        !l0_origin_map_.has_value() ||
        !last_map_time_ns_.has_value() ||
        !last_odometry_time_ns_.has_value() || !last_odometry_.has_value()) {
      return;
    }
    if (std::llabs(*last_map_time_ns_ - *last_odometry_time_ns_) >
        maximum_snapshot_skew_ns_) {
      SetReason(
          "MAP_ODOMETRY_SNAPSHOT_SKEW",
          last_global_map_.has_value()
              ? ObservedMapNodeState::kReady
              : ObservedMapNodeState::kSyncing);
      return;
    }
    const std::int64_t snapshot_time =
        std::max(*last_map_time_ns_, *last_odometry_time_ns_);
    if (goal_time_ns_.has_value() && snapshot_time < *goal_time_ns_) {
      SetReason(
          "WAITING_FOR_GOAL_SNAPSHOT",
          last_global_map_.has_value()
              ? ObservedMapNodeState::kReady
              : ObservedMapNodeState::kSyncing);
      return;
    }
    ProductRequest request{
        .robot_map = {
            last_odometry_->pose.pose.position.x,
            last_odometry_->pose.pose.position.y},
        .robot_odom = {
            last_odometry_->pose.pose.position.x,
            last_odometry_->pose.pose.position.y},
        .l0_origin_map = *l0_origin_map_,
        .goal_map = std::nullopt,
        .simulation_time_ns = snapshot_time,
    };
    if (goal_.has_value()) {
      request.goal_map = MapPoint2D{
          goal_->pose.position.x, goal_->pose.position.y};
    }
    const MapProductResult local = products_->BuildLocal(*sparse_map_, request);
    const MapProductResult global =
        products_->BuildGlobal(*sparse_map_, request);
    if (!local.ok() || !global.ok()) {
      SetReason(
          local.ok() ? global.reason_code : local.reason_code,
          last_global_map_.has_value()
              ? ObservedMapNodeState::kReady
              : ObservedMapNodeState::kSyncing);
      return;
    }
    last_local_map_ = *local.message;
    last_global_map_ = *global.message;
    last_generated_observation_time_ns_ = last_map_time_ns_;
    ++generation_;
    state_ = ObservedMapNodeState::kReady;
    last_reason_ = "READY";
    if (local_publisher_ && local_publisher_->is_activated()) {
      local_publisher_->publish(*last_local_map_);
      global_publisher_->publish(*last_global_map_);
    }
    PublishStatus();
  }

  void ClearSnapshot() {
    last_map_time_ns_.reset();
    last_generated_observation_time_ns_.reset();
    last_odometry_time_ns_.reset();
    last_odometry_.reset();
    goal_.reset();
    goal_time_ns_.reset();
    last_local_map_.reset();
    last_global_map_.reset();
    l0_origin_map_.reset();
    generation_ = 0U;
  }

  void SetReason(
      std::string reason, const ObservedMapNodeState state) {
    state_ = state;
    last_reason_ = std::move(reason);
    PublishStatus();
  }

  void PublishStatus() {
    if (!active_ || !status_publisher_ ||
        !status_publisher_->is_activated()) {
      return;
    }
    diagnostic_msgs::msg::DiagnosticArray array;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "lunar_observed_map";
    status.hardware_id = sparse_map_ ? sparse_map_->session_id() : "";
    status.level = state_ == ObservedMapNodeState::kError
        ? diagnostic_msgs::msg::DiagnosticStatus::ERROR
        : (state_ == ObservedMapNodeState::kReady
               ? diagnostic_msgs::msg::DiagnosticStatus::OK
               : diagnostic_msgs::msg::DiagnosticStatus::WARN);
    status.message = StateName(state_) + ":" + last_reason_;
    const auto key_value = [](std::string key, std::string value) {
      diagnostic_msgs::msg::KeyValue output;
      output.key = std::move(key);
      output.value = std::move(value);
      return output;
    };
    status.values = {
        key_value("state", StateName(state_)),
        key_value("reason_code", last_reason_),
        key_value("generation", std::to_string(generation_)),
        key_value(
            "session_id", sparse_map_ ? sparse_map_->session_id() : ""),
    };
    array.status.push_back(std::move(status));
    status_publisher_->publish(array);
  }

  ObservedMapNode& node_;
  std::unique_ptr<SparseObservedMap> sparse_map_;
  std::unique_ptr<GridMapProducts> products_;
  std::int64_t maximum_snapshot_skew_ns_{200'000'000LL};
  bool configured_{};
  bool active_{};
  ObservedMapNodeState state_{ObservedMapNodeState::kUnconfigured};
  std::string last_reason_{"UNCONFIGURED"};
  std::optional<std::int64_t> last_map_time_ns_;
  std::optional<std::int64_t> last_generated_observation_time_ns_;
  std::optional<std::int64_t> last_odometry_time_ns_;
  std::optional<nav_msgs::msg::Odometry> last_odometry_;
  std::optional<geometry_msgs::msg::PoseStamped> goal_;
  std::optional<std::int64_t> goal_time_ns_;
  std::optional<grid_map_msgs::msg::GridMap> last_local_map_;
  std::optional<grid_map_msgs::msg::GridMap> last_global_map_;
  std::optional<MapPoint2D> l0_origin_map_;
  std::uint64_t generation_{};
  rclcpp_lifecycle::LifecyclePublisher<
      grid_map_msgs::msg::GridMap>::SharedPtr local_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<
      grid_map_msgs::msg::GridMap>::SharedPtr global_publisher_;
  rclcpp_lifecycle::LifecyclePublisher<
      diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_publisher_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr
      observed_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      goal_subscription_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      bridge_status_subscription_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameter_callback_;
};

ObservedMapNode::ObservedMapNode(const rclcpp::NodeOptions& options)
    : LifecycleNode("lunar_observed_map", options),
      impl_(std::make_unique<Impl>(*this)) {}

ObservedMapNode::~ObservedMapNode() = default;

CallbackReturn ObservedMapNode::on_configure(
    const rclcpp_lifecycle::State&) {
  return impl_->Configure();
}

CallbackReturn ObservedMapNode::on_activate(
    const rclcpp_lifecycle::State&) {
  return impl_->Activate();
}

CallbackReturn ObservedMapNode::on_deactivate(
    const rclcpp_lifecycle::State&) {
  return impl_->Deactivate();
}

CallbackReturn ObservedMapNode::on_cleanup(
    const rclcpp_lifecycle::State&) {
  return impl_->Cleanup();
}

CallbackReturn ObservedMapNode::on_shutdown(
    const rclcpp_lifecycle::State&) {
  return impl_->Cleanup();
}

CallbackReturn ObservedMapNode::on_error(
    const rclcpp_lifecycle::State&) {
  return impl_->Cleanup();
}

void ObservedMapNode::ReceiveObservedMapForTesting(
    const grid_map_msgs::msg::GridMap& message) {
  impl_->ReceiveObservedMap(message);
}

void ObservedMapNode::ReceiveOdometryForTesting(
    const nav_msgs::msg::Odometry& message) {
  impl_->ReceiveOdometry(message);
}

void ObservedMapNode::ReceiveGoalForTesting(
    const geometry_msgs::msg::PoseStamped& message) {
  impl_->ReceiveGoal(message);
}

void ObservedMapNode::ReceiveBridgeStatusForTesting(
    const diagnostic_msgs::msg::DiagnosticArray& message) {
  impl_->ReceiveBridgeStatus(message);
}

FuseResult ObservedMapNode::FusePatchForTesting(
    const ObservedPatch& patch) {
  return impl_->FusePatch(patch);
}

void ObservedMapNode::ResetSessionForTesting(std::string session_id) {
  impl_->ResetSession(std::move(session_id));
}

std::optional<grid_map_msgs::msg::GridMap>
ObservedMapNode::last_local_map_for_testing() const {
  return impl_->last_local();
}

std::optional<grid_map_msgs::msg::GridMap>
ObservedMapNode::last_global_map_for_testing() const {
  return impl_->last_global();
}

std::uint64_t ObservedMapNode::generation_for_testing() const noexcept {
  return impl_->generation();
}

ObservedMapNodeState ObservedMapNode::state_for_testing() const noexcept {
  return impl_->state();
}

std::string ObservedMapNode::last_reason_for_testing() const {
  return impl_->reason();
}

}  // namespace lunar::observed_map
