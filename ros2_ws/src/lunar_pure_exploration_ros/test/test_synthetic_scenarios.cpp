#include "lunar_pure_exploration_ros/exploration_node.hpp"
#include "lunar_pure_exploration_ros/platform_config_loader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_task.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <yaml-cpp/yaml.h>

namespace lunar::pure_exploration_ros {

class ExplorationNodeTestPeer final {
public:
  static std::optional<lunar::pure_exploration::Pose2>
  ActiveTarget(const ExplorationNode &node) {
    return node.SnapshotActiveTargetForTest();
  }
};

namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using ServerGoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;
using Task = lunar_pure_exploration_msgs::msg::PureExplorationTask;
using namespace std::chrono_literals;

constexpr std::chrono::seconds kCaseDeadline{20};
const std::filesystem::path kFixtureDirectory{TASK15_FIXTURE_DIRECTORY};

class RosEnvironment final : public ::testing::Environment {
public:
  void SetUp() override {
    if (!rclcpp::ok()) {
      int argc = 0;
      rclcpp::init(argc, nullptr);
    }
  }
  void TearDown() override {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

const auto *const kRosEnvironment =
    ::testing::AddGlobalTestEnvironment(new RosEnvironment{});

template <typename Predicate>
bool WaitFor(Predicate &&predicate,
             const std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

geometry_msgs::msg::Quaternion YawQuaternion(const double yaw) {
  geometry_msgs::msg::Quaternion quaternion;
  quaternion.z = std::sin(yaw / 2.0);
  quaternion.w = std::cos(yaw / 2.0);
  return quaternion;
}

struct CellKey final {
  std::int32_t x{};
  std::int32_t y{};
  auto operator<=>(const CellKey &) const = default;
};

std::string CellString(const CellKey cell) {
  return std::to_string(cell.x) + "," + std::to_string(cell.y);
}

std::string CandidateKeyString(const Action::Goal &goal) {
  return std::to_string(static_cast<std::int64_t>(
             std::llround(goal.goal.point.x * 1000.0))) +
         "," +
         std::to_string(static_cast<std::int64_t>(
             std::llround(goal.goal.point.y * 1000.0))) +
         "," +
         std::to_string(static_cast<std::int64_t>(
             std::llround(goal.goal.yaw_rad * 1800.0 / std::numbers::pi)));
}

std::string PoseKeyString(const geometry_msgs::msg::Pose &pose) {
  const double yaw = 2.0 * std::atan2(pose.orientation.z, pose.orientation.w);
  return std::to_string(static_cast<std::int64_t>(
             std::llround(pose.position.x * 1000.0))) +
         "," +
         std::to_string(static_cast<std::int64_t>(
             std::llround(pose.position.y * 1000.0))) +
         "," +
         std::to_string(static_cast<std::int64_t>(
             std::llround(yaw * 1800.0 / std::numbers::pi)));
}

CellKey WorldCell(const double x, const double y, const double resolution,
                  const double origin_x = 0.0, const double origin_y = 0.0) {
  return {static_cast<std::int32_t>(std::floor((x - origin_x) / resolution)),
          static_cast<std::int32_t>(std::floor((y - origin_y) / resolution))};
}

enum class ResponseKind {
  kSuccess,
  kNoPath,
  kTimeout,
  kDelayed,
  kInvalidSuccess
};

const char *ResponseName(const ResponseKind kind) {
  switch (kind) {
  case ResponseKind::kSuccess:
    return "SUCCESS";
  case ResponseKind::kNoPath:
    return "NO_PATH";
  case ResponseKind::kTimeout:
    return "TIMEOUT";
  case ResponseKind::kDelayed:
    return "DELAYED";
  case ResponseKind::kInvalidSuccess:
    return "INVALID_SUCCESS";
  }
  return "UNKNOWN";
}

struct ResponseSpec final {
  ResponseKind kind{ResponseKind::kNoPath};
  std::optional<std::pair<double, double>> executable_endpoint{};
  std::optional<std::string> expected_candidate_key{};
  std::optional<std::string> expected_request_id{};
};

struct ScriptedCell final {
  std::string candidate_id;
  std::deque<ResponseSpec> responses;
};

struct RequestRecord final {
  std::string request_id;
  std::string candidate_key;
  CellKey goal_cell;
  std::string response;
};

class GoalCellScripts final {
public:
  void EnableStrict() {
    std::scoped_lock lock{mutex_};
    strict_ = true;
  }
  void Set(CellKey cell, std::string candidate_id,
           std::vector<ResponseSpec> responses) {
    std::scoped_lock lock{mutex_};
    scripts_.insert_or_assign(
        cell, ScriptedCell{std::move(candidate_id),
                           {responses.begin(), responses.end()}});
  }

  void SetCanonical(CellKey cell, std::vector<ResponseSpec> responses) {
    std::scoped_lock lock{mutex_};
    strict_ = true;
    scripts_.insert_or_assign(
        cell, ScriptedCell{"", {responses.begin(), responses.end()}});
  }

  struct Assigned final {
    ResponseSpec response;
    std::size_t record_index;
  };

  std::optional<Assigned> Assign(const CellKey cell,
                                 const std::string &candidate_key,
                                 const std::string &request_id) {
    std::scoped_lock lock{mutex_};
    const auto found = scripts_.find(cell);
    if (found == scripts_.end() || found->second.responses.empty()) {
      if (strict_) {
        error_ = "unknown or exhausted goal cell " + CellString(cell) +
                 " key=" + candidate_key + " for request " + request_id;
        return std::nullopt;
      }
      const ResponseSpec response{ResponseKind::kNoPath, std::nullopt};
      records_.push_back(
          {request_id, candidate_key, cell, ResponseName(response.kind)});
      return Assigned{response, records_.size() - 1U};
    }
    auto response = found->second.responses.front();
    found->second.responses.pop_front();
    if ((response.expected_candidate_key &&
         *response.expected_candidate_key != candidate_key) ||
        (response.expected_request_id &&
         *response.expected_request_id != request_id)) {
      error_ = "canonical response mismatch at cell " + CellString(cell) +
               ": observed key=" + candidate_key + " request=" + request_id;
      return std::nullopt;
    }
    records_.push_back(
        {request_id, candidate_key, cell, ResponseName(response.kind)});
    return Assigned{response, records_.size() - 1U};
  }

  void UpdateRecord(const std::size_t index, const std::string &response) {
    std::scoped_lock lock{mutex_};
    records_.at(index).response = response;
  }

  std::vector<RequestRecord> Records() const {
    std::scoped_lock lock{mutex_};
    return records_;
  }

  std::string Error() const {
    std::scoped_lock lock{mutex_};
    return error_;
  }

  std::size_t Remaining() const {
    std::scoped_lock lock{mutex_};
    std::size_t count = 0U;
    for (const auto &[_, script] : scripts_) {
      count += script.responses.size();
    }
    return count;
  }

  std::string Trace() const {
    std::scoped_lock lock{mutex_};
    std::ostringstream stream;
    stream << "ordered candidate/request trace:";
    for (const auto &record : records_) {
      stream << "\n  key=" << record.candidate_key
             << " cell=" << CellString(record.goal_cell)
             << " request=" << record.request_id
             << " response=" << record.response;
    }
    if (!error_.empty()) {
      stream << "\n  ERROR: " << error_;
    }
    return stream.str();
  }

private:
  mutable std::mutex mutex_;
  std::map<CellKey, ScriptedCell> scripts_;
  std::vector<RequestRecord> records_;
  std::string error_;
  bool strict_{false};
};

class GoalCellPlannerServer final {
public:
  GoalCellPlannerServer(std::shared_ptr<rclcpp::Node> node,
                        const std::string &action_name,
                        std::shared_ptr<GoalCellScripts> scripts,
                        const double resolution, const double origin_x = 0.0,
                        const double origin_y = 0.0)
      : node_(std::move(node)), scripts_(std::move(scripts)),
        resolution_(resolution), origin_x_(origin_x), origin_y_(origin_y) {
    server_ = rclcpp_action::create_server<Action>(
        node_, action_name,
        [this](const rclcpp_action::GoalUUID &,
               const std::shared_ptr<const Action::Goal> goal) {
          const CellKey cell = WorldCell(goal->goal.point.x, goal->goal.point.y,
                                         resolution_, origin_x_, origin_y_);
          auto assigned = scripts_->Assign(cell, CandidateKeyString(*goal),
                                           goal->request_id);
          if (!assigned) {
            return rclcpp_action::GoalResponse::REJECT;
          }
          std::scoped_lock lock{mutex_};
          assignments_.insert_or_assign(goal->request_id, *assigned);
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](const std::shared_ptr<ServerGoalHandle> handle) {
          std::scoped_lock lock{mutex_};
          canceled_request_ids_.push_back(handle->get_goal()->request_id);
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<ServerGoalHandle> handle) {
          GoalCellScripts::Assigned assigned;
          {
            std::scoped_lock lock{mutex_};
            const auto request_id = handle->get_goal()->request_id;
            handles_.insert_or_assign(request_id, handle);
            assigned = assignments_.at(request_id);
          }
          if (assigned.response.kind != ResponseKind::kDelayed) {
            Finish(handle->get_goal()->request_id, assigned.response);
          }
        });
  }

  bool HasHandle(const std::string &request_id) const {
    std::scoped_lock lock{mutex_};
    return handles_.contains(request_id);
  }

  std::size_t CancelCount() const {
    std::scoped_lock lock{mutex_};
    return canceled_request_ids_.size();
  }

  void Finish(const std::string &request_id, const ResponseSpec response) {
    std::shared_ptr<ServerGoalHandle> handle;
    std::size_t record_index{};
    {
      std::scoped_lock lock{mutex_};
      handle = handles_.at(request_id);
      record_index = assignments_.at(request_id).record_index;
    }
    auto result = std::make_shared<Action::Result>();
    if (response.kind == ResponseKind::kSuccess) {
      result->planning_outcome = Action::Result::NEW_REFERENCE_AVAILABLE;
      result->execution_directive = Action::Result::ACTIVATE_NEW_REFERENCE;
      result->reason_code = "PLAN_FOUND";
      result->has_reference = true;
      result->reference.header.frame_id = "map";
      result->reference.header.stamp.sec = 31;
      result->reference.input_time.sec = 29;
      result->reference.plan_id = "scenario-plan:" + request_id;
      result->reference.platform_type = result->reference.WHEELED;
      result->reference.path_preview.header.frame_id = "map";
      geometry_msgs::msg::PoseStamped preview;
      preview.header.frame_id = "map";
      preview.header.stamp.sec = 23;
      preview.pose.position.x = handle->get_goal()->goal.point.x;
      preview.pose.position.y = handle->get_goal()->goal.point.y;
      preview.pose.orientation =
          YawQuaternion(handle->get_goal()->goal.yaw_rad);
      result->reference.path_preview.poses.push_back(std::move(preview));
      auto &point = result->reference.trajectory.points.emplace_back();
      auto &transform = point.transforms.emplace_back();
      transform.translation.x = response.executable_endpoint
                                    ? response.executable_endpoint->first
                                    : handle->get_goal()->goal.point.x;
      transform.translation.y = response.executable_endpoint
                                    ? response.executable_endpoint->second
                                    : handle->get_goal()->goal.point.y;
      transform.rotation.w = 1.0;
      scripts_->UpdateRecord(record_index, "SUCCESS");
      handle->succeed(result);
      return;
    }
    if (response.kind == ResponseKind::kNoPath) {
      result->planning_outcome = Action::Result::GOAL_INFEASIBLE;
      result->execution_directive = Action::Result::NO_SAFE_REFERENCE;
      result->reason_code = "NO_PATH";
      result->has_reference = false;
      scripts_->UpdateRecord(record_index, "NO_PATH");
      handle->abort(result);
      return;
    }
    if (response.kind == ResponseKind::kTimeout) {
      result->planning_outcome = Action::Result::RESOURCE_EXHAUSTED;
      result->execution_directive = Action::Result::NO_SAFE_REFERENCE;
      result->reason_code = "TIMEOUT";
      result->has_reference = false;
      scripts_->UpdateRecord(record_index, "TIMEOUT");
      handle->abort(result);
      return;
    }
    if (response.kind == ResponseKind::kInvalidSuccess) {
      result->planning_outcome = Action::Result::NEW_REFERENCE_AVAILABLE;
      result->execution_directive = Action::Result::ACTIVATE_NEW_REFERENCE;
      result->reason_code = "PLAN_FOUND";
      result->has_reference = true;
      result->reference.header.frame_id = "map";
      result->reference.plan_id = "invalid-plan:" + request_id;
      result->reference.platform_type = result->reference.WHEELED;
      scripts_->UpdateRecord(record_index, "INVALID_SUCCESS");
      handle->succeed(result);
      return;
    }
    throw std::logic_error{"delayed response requires an explicit terminal"};
  }

  void FinishCanceled(const std::string &request_id) {
    std::shared_ptr<ServerGoalHandle> handle;
    std::size_t record_index{};
    {
      std::scoped_lock lock{mutex_};
      handle = handles_.at(request_id);
      record_index = assignments_.at(request_id).record_index;
    }
    auto result = std::make_shared<Action::Result>();
    result->planning_outcome = Action::Result::CANCELED;
    result->execution_directive = Action::Result::NO_SAFE_REFERENCE;
    result->reason_code = "REQUEST_CANCELED";
    result->has_reference = false;
    scripts_->UpdateRecord(record_index, "CANCELED");
    handle->canceled(result);
  }

private:
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<GoalCellScripts> scripts_;
  double resolution_;
  double origin_x_;
  double origin_y_;
  rclcpp_action::Server<Action>::SharedPtr server_;
  mutable std::mutex mutex_;
  std::map<std::string, GoalCellScripts::Assigned> assignments_;
  std::map<std::string, std::shared_ptr<ServerGoalHandle>> handles_;
  std::vector<std::string> canceled_request_ids_;
};

nav_msgs::msg::OccupancyGrid ExplorationMap() {
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.header.stamp.sec = 1;
  map.info.map_load_time.sec = 7;
  map.info.width = 20U;
  map.info.height = 20U;
  map.info.resolution = 0.5F;
  map.info.origin.orientation.w = 1.0;
  map.data.assign(400U, 0);
  for (std::uint32_t y = 0U; y < map.info.height; ++y) {
    for (std::uint32_t x = 10U; x < map.info.width; ++x) {
      map.data[static_cast<std::size_t>(y) * map.info.width + x] = -1;
    }
  }
  return map;
}

nav_msgs::msg::OccupancyGrid NarrowPassageMap() {
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.header.stamp.sec = 2;
  map.info.map_load_time.sec = 11;
  map.info.width = 20U;
  map.info.height = 16U;
  map.info.resolution = 0.5F;
  map.info.origin.orientation.w = 1.0;
  map.data.assign(static_cast<std::size_t>(map.info.width) * map.info.height,
                  0);
  const auto set = [&map](const std::uint32_t x, const std::uint32_t y,
                          const std::int8_t value) {
    map.data[static_cast<std::size_t>(y) * map.info.width + x] = value;
  };
  for (std::uint32_t y = 0U; y < map.info.height; ++y) {
    if (!((y >= 2U && y <= 5U) || y == 12U)) {
      set(7U, y, 100);
    }
  }
  for (std::uint32_t x = 8U; x < map.info.width; ++x) {
    set(x, 8U, 100);
  }
  for (std::uint32_t y = 0U; y < map.info.height; ++y) {
    for (std::uint32_t x = 15U; x < map.info.width; ++x) {
      set(x, y, -1);
    }
  }
  return map;
}

nav_msgs::msg::Odometry Odometry(const double x = 2.0, const double y = 2.0,
                                 const double yaw = 0.0) {
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.position.x = x;
  odometry.pose.pose.position.y = y;
  odometry.pose.pose.orientation = YawQuaternion(yaw);
  return odometry;
}

tf2_msgs::msg::TFMessage MapFromOdom() {
  tf2_msgs::msg::TFMessage message;
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.transform.rotation.w = 1.0;
  message.transforms.push_back(std::move(transform));
  return message;
}

struct CanonicalFixture final {
  std::string fixture_id;
  Task task;
  nav_msgs::msg::OccupancyGrid initial_map;
  nav_msgs::msg::Odometry odometry;
  tf2_msgs::msg::TFMessage transforms;
  std::optional<nav_msgs::msg::OccupancyGrid> growth_map;
  std::vector<nav_msgs::msg::Odometry> execution_odometries;
  YAML::Node response_table;
  YAML::Node expected;
  YAML::Node variants;
};

ResponseKind ParseResponseKind(const std::string &kind) {
  if (kind == "SUCCESS") {
    return ResponseKind::kSuccess;
  }
  if (kind == "NO_PATH") {
    return ResponseKind::kNoPath;
  }
  if (kind == "TIMEOUT") {
    return ResponseKind::kTimeout;
  }
  if (kind == "DELAYED") {
    return ResponseKind::kDelayed;
  }
  if (kind == "INVALID_SUCCESS") {
    return ResponseKind::kInvalidSuccess;
  }
  throw std::runtime_error{"unsupported canonical response kind " + kind};
}

CellKey ParseCell(const std::string &value) {
  const auto separator = value.find(',');
  if (separator == std::string::npos) {
    throw std::runtime_error{"invalid canonical cell " + value};
  }
  return {static_cast<std::int32_t>(std::stoi(value.substr(0, separator))),
          static_cast<std::int32_t>(std::stoi(value.substr(separator + 1U)))};
}

std::shared_ptr<GoalCellScripts>
LoadCanonicalScripts(const CanonicalFixture &fixture) {
  auto scripts = std::make_shared<GoalCellScripts>();
  scripts->EnableStrict();
  for (const auto row : fixture.response_table) {
    const CellKey cell = ParseCell(row.first.as<std::string>());
    std::vector<ResponseSpec> responses;
    for (const auto response : row.second["responses"]) {
      const auto kind = response["kind"].as<std::string>();
      if (kind == "TWO_SEGMENT") {
        for (const auto segment : response["segments"]) {
          const auto request_id = segment["request_id"].as<std::string>();
          responses.push_back({.kind = ResponseKind::kSuccess,
                               .executable_endpoint = std::nullopt,
                               .expected_candidate_key =
                                   segment["candidate_key"].as<std::string>(),
                               .expected_request_id = request_id});
        }
      } else {
        std::optional<std::pair<double, double>> endpoint;
        if (response["executable_endpoint_xy"]) {
          endpoint =
              std::pair{response["executable_endpoint_xy"][0].as<double>(),
                        response["executable_endpoint_xy"][1].as<double>()};
        }
        responses.push_back(
            {.kind = ParseResponseKind(kind),
             .executable_endpoint = endpoint,
             .expected_candidate_key =
                 response["candidate_key"].as<std::string>(),
             .expected_request_id = response["request_id"].as<std::string>()});
      }
    }
    scripts->SetCanonical(cell, std::move(responses));
  }
  return scripts;
}

nav_msgs::msg::OccupancyGrid LoadGrid(const YAML::Node &node) {
  nav_msgs::msg::OccupancyGrid map;
  const auto header = node["header"];
  const auto geometry = node["geometry"];
  map.header.frame_id = header["frame_id"].as<std::string>();
  const auto stamp_ns = header["stamp_ns"].as<std::uint64_t>();
  map.header.stamp.sec = static_cast<std::int32_t>(stamp_ns / 1000000000ULL);
  map.header.stamp.nanosec =
      static_cast<std::uint32_t>(stamp_ns % 1000000000ULL);
  const auto map_version = header["map_version"].as<std::uint64_t>();
  map.info.map_load_time.sec = static_cast<std::int32_t>(map_version);
  map.info.resolution = geometry["resolution_m"].as<float>();
  map.info.width = geometry["width"].as<std::uint32_t>();
  map.info.height = geometry["height"].as<std::uint32_t>();
  map.info.origin.position.x = geometry["origin"]["x"].as<double>();
  map.info.origin.position.y = geometry["origin"]["y"].as<double>();
  map.info.origin.orientation.w = 1.0;
  if (node["data"]) {
    map.data = node["data"].as<std::vector<std::int8_t>>();
  } else {
    for (const auto row_node : node["rows"]) {
      const auto row = row_node.as<std::string>();
      for (const char cell : row) {
        if (cell == '.') {
          map.data.push_back(0);
        } else if (cell == '#') {
          map.data.push_back(100);
        } else if (cell == '?') {
          map.data.push_back(-1);
        } else {
          throw std::runtime_error{"canonical fixture row has invalid cell"};
        }
      }
    }
  }
  if (map.data.size() !=
      static_cast<std::size_t>(map.info.width) * map.info.height) {
    throw std::runtime_error{"canonical fixture grid size mismatch"};
  }
  return map;
}

CanonicalFixture LoadFixture(const std::string &name) {
  const auto root =
      YAML::LoadFile((kFixtureDirectory / (name + ".yaml")).string());
  CanonicalFixture fixture;
  fixture.fixture_id = root["fixture_id"].as<std::string>();
  if (root["schema_version"].as<int>() != 1 || fixture.fixture_id != name ||
      root["authority"].as<std::string>() != "test-only/non-authoritative") {
    throw std::runtime_error{"canonical fixture identity mismatch"};
  }
  fixture.task.header.frame_id = "map";
  fixture.task.task_id = root["task"]["task_id"].as<std::string>();
  fixture.task.command = Task::START;
  for (const auto point : root["task"]["boundary_xy"]) {
    auto &output = fixture.task.boundary.points.emplace_back();
    output.x = point[0].as<float>();
    output.y = point[1].as<float>();
  }
  const auto inputs = root["initial_inputs"];
  fixture.initial_map = LoadGrid(inputs["occupancy_grid"]);
  const auto pose = inputs["odometry"]["pose_xy_yaw"];
  fixture.odometry.header.frame_id =
      inputs["odometry"]["frame_id"].as<std::string>();
  fixture.odometry.child_frame_id =
      inputs["odometry"]["child_frame_id"].as<std::string>();
  fixture.odometry.pose.pose.position.x = pose[0].as<double>();
  fixture.odometry.pose.pose.position.y = pose[1].as<double>();
  fixture.odometry.pose.pose.orientation = YawQuaternion(pose[2].as<double>());
  const auto covariance =
      inputs["odometry"]["covariance"].as<std::vector<double>>();
  std::copy(covariance.begin(), covariance.end(),
            fixture.odometry.pose.covariance.begin());
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  const auto map_to_odom = inputs["transforms"]["map_to_odom"];
  transform.transform.translation.x =
      map_to_odom["translation_xy"][0].as<double>();
  transform.transform.translation.y =
      map_to_odom["translation_xy"][1].as<double>();
  transform.transform.rotation =
      YawQuaternion(map_to_odom["yaw_rad"].as<double>());
  fixture.transforms.transforms.push_back(std::move(transform));
  for (const auto event : root["events"]) {
    const auto kind = event["kind"].as<std::string>();
    if (kind == "publish_map_growth") {
      fixture.growth_map = LoadGrid(event["occupancy_grid"]);
    } else if (kind == "publish_odometry") {
      auto execution_odometry = fixture.odometry;
      const auto event_pose = event["pose_xy_yaw"];
      execution_odometry.pose.pose.position.x = event_pose[0].as<double>();
      execution_odometry.pose.pose.position.y = event_pose[1].as<double>();
      execution_odometry.pose.pose.orientation =
          YawQuaternion(event_pose[2].as<double>());
      fixture.execution_odometries.push_back(std::move(execution_odometry));
    }
  }
  fixture.response_table = root["candidate_goal_cell_response_table"];
  fixture.expected = root["expected"];
  if (root["variants"]) {
    fixture.variants = root["variants"];
  }
  return fixture;
}

Task StartTask(
    const std::string &task_id,
    const std::vector<std::pair<float, float>> &boundary = {
        {0.0F, 0.0F}, {10.0F, 0.0F}, {10.0F, 10.0F}, {0.0F, 10.0F}}) {
  Task task;
  task.header.frame_id = "map";
  task.task_id = task_id;
  task.command = Task::START;
  for (const auto [x, y] : boundary) {
    auto &point = task.boundary.points.emplace_back();
    point.x = x;
    point.y = y;
  }
  return task;
}

std::vector<lunar::pure_exploration::CandidateView> ControlledCandidates(
    const std::span<const lunar::pure_exploration::FrontierCluster> frontiers,
    const std::vector<lunar::pure_exploration::Pose2> &poses) {
  if (frontiers.empty()) {
    throw std::logic_error{"scenario requires a real WFD frontier"};
  }
  std::vector<lunar::pure_exploration::CandidateView> candidates;
  candidates.reserve(poses.size());
  for (std::size_t index = 0U; index < poses.size(); ++index) {
    const auto &pose = poses[index];
    candidates.push_back(
        {.id = static_cast<std::uint64_t>(1000U + index),
         .frontier_id = frontiers.front().id,
         .frontier_index = 0U,
         .key = {.x_mm =
                     static_cast<std::int64_t>(std::llround(pose.x * 1000.0)),
                 .y_mm =
                     static_cast<std::int64_t>(std::llround(pose.y * 1000.0)),
                 .yaw_tenth_deg = static_cast<std::int64_t>(
                     std::llround(pose.yaw * 1800.0 / std::numbers::pi))},
         .pose = pose,
         .frontier_distance_m = 1.0,
         .frontier_canonical_key =
             std::make_shared<const std::vector<std::int64_t>>(
                 frontiers.front().canonical_key)});
  }
  return candidates;
}

ExplorationNodeParameters ScenarioParameters(
    const std::string &prefix,
    const std::shared_ptr<std::chrono::steady_clock::time_point> &now,
    std::shared_ptr<ExplorationPipelineSeams> seams = {}) {
  const auto wheel = LoadPlatformConfig(
      std::filesystem::path{ament_index_cpp::get_package_share_directory(
          "lunar_pure_planner_ros")} /
          "config" / "wheel.yaml",
      "wheel");
  return ExplorationNodeParameters{
      .platform = wheel.geometry,
      .candidate_parameters = {{-std::numbers::pi / 4.0,
                                -std::numbers::pi / 8.0, 0.0,
                                std::numbers::pi / 8.0,
                                std::numbers::pi / 4.0}},
      .candidate_limits = {4096U, 64U, 100000U},
      .task_raster_limits = {1048576U},
      .sensor_model = {10.0, std::numbers::pi / 2.0},
      .information_gain_limits = {100000U},
      .score_weights = {},
      .failure_memory_limits = {8U, 256U, 1024U},
      .global_occupied_threshold = 50,
      .maximum_path_preview_poses = 64U,
      .maximum_executable_path_points = 64U,
      .maximum_replans = 2U,
      .goal_yaw_tolerance_rad = std::numbers::pi / 16.0,
      // Scripted candidates are the test input under evaluation.  The
      // production prefilter has its own node-level regression coverage.
      .filter_global_goal_cell = false,
      .planner_result_timeout = 2s,
      .global_map_topic = prefix + "/global_map",
      .odometry_topic = prefix + "/odometry",
      .tf_topic = prefix + "/tf",
      .task_topic = prefix + "/task",
      .planner_action = prefix + "/plan_motion",
      .planner_diagnostics_topic = prefix + "/planner_diagnostics",
      .motion_reference_topic = prefix + "/motion_reference",
      .execution_cancel_topic = prefix + "/execution_cancel",
      .status_topic = prefix + "/status",
      .current_goal_topic = prefix + "/current_goal",
      .frontiers_topic = prefix + "/frontiers",
      .diagnostics_topic = prefix + "/diagnostics",
      .steady_now = [now] { return *now; },
      .pipeline_seams = std::move(seams)};
}

class ScenarioHarness final {
public:
  explicit ScenarioHarness(std::shared_ptr<GoalCellScripts> scripts,
                           std::shared_ptr<ExplorationPipelineSeams> seams = {},
                           const double resolution = 0.5)
      : started_(std::chrono::steady_clock::now()),
        scripts_(std::move(scripts)),
        now_(std::make_shared<std::chrono::steady_clock::time_point>()) {
    const auto suffix = sequence_.fetch_add(1U);
    prefix_ = "/task15_scenario_" + std::to_string(suffix);
    parameters_ = ScenarioParameters(prefix_, now_, std::move(seams));
    server_node_ = std::make_shared<rclcpp::Node>("task15_scenario_server_" +
                                                  std::to_string(suffix));
    io_node_ = std::make_shared<rclcpp::Node>("task15_scenario_io_" +
                                              std::to_string(suffix));
    server_ = std::make_unique<GoalCellPlannerServer>(
        server_node_, parameters_.planner_action, scripts_, resolution);
    explorer_ = std::make_shared<ExplorationNode>(parameters_);
    task_publisher_ =
        io_node_->create_publisher<Task>(parameters_.task_topic, 10);
    map_publisher_ = io_node_->create_publisher<nav_msgs::msg::OccupancyGrid>(
        parameters_.global_map_topic, 10);
    odometry_publisher_ = io_node_->create_publisher<nav_msgs::msg::Odometry>(
        parameters_.odometry_topic, 10);
    tf_publisher_ = io_node_->create_publisher<tf2_msgs::msg::TFMessage>(
        parameters_.tf_topic, 10);
    reference_subscription_ = io_node_->create_subscription<
        lunar_planning_msgs::msg::MotionReference>(
        parameters_.motion_reference_topic, 10,
        [this](lunar_planning_msgs::msg::MotionReference::SharedPtr value) {
          std::scoped_lock lock{messages_mutex_};
          references_.push_back(*value);
        });
    cancel_subscription_ = io_node_->create_subscription<std_msgs::msg::String>(
        parameters_.execution_cancel_topic, 10,
        [this](std_msgs::msg::String::SharedPtr value) {
          std::scoped_lock lock{messages_mutex_};
          execution_cancels_.push_back(value->data);
        });
    status_subscription_ = io_node_->create_subscription<Status>(
        parameters_.status_topic, rclcpp::QoS{10}.reliable().transient_local(),
        [this](Status::SharedPtr value) {
          std::scoped_lock lock{messages_mutex_};
          statuses_.push_back(*value);
          if (status_transitions_.empty() ||
              status_transitions_.back() != value->state) {
            status_transitions_.push_back(value->state);
          }
        });
    current_goal_subscription_ =
        io_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            parameters_.current_goal_topic,
            rclcpp::QoS{1}.reliable().transient_local(),
            [this](geometry_msgs::msg::PoseStamped::SharedPtr value) {
              std::scoped_lock lock{messages_mutex_};
              current_goals_.push_back(*value);
            });
    executor_ = std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
        rclcpp::ExecutorOptions{}, 4U);
    executor_->add_node(server_node_);
    executor_->add_node(io_node_);
    executor_->add_node(explorer_);
    spin_thread_ = std::jthread([this] { executor_->spin(); });
    auto probe = rclcpp_action::create_client<Action>(
        io_node_, parameters_.planner_action);
    if (!probe->wait_for_action_server(3s) || !WaitFor([this] {
          return task_publisher_->get_subscription_count() == 1U &&
                 map_publisher_->get_subscription_count() == 1U &&
                 odometry_publisher_->get_subscription_count() == 1U &&
                 tf_publisher_->get_subscription_count() == 1U;
        })) {
      throw std::runtime_error{"scenario ROS graph did not become ready"};
    }
  }

  ~ScenarioHarness() {
    executor_->cancel();
    if (spin_thread_.joinable()) {
      spin_thread_.join();
    }
    executor_->remove_node(explorer_);
    executor_->remove_node(io_node_);
    executor_->remove_node(server_node_);
  }

  void PublishInputs(Task task, nav_msgs::msg::OccupancyGrid map,
                     nav_msgs::msg::Odometry odometry = Odometry()) {
    map_publisher_->publish(std::move(map));
    odometry_publisher_->publish(std::move(odometry));
    tf_publisher_->publish(MapFromOdom());
    task_publisher_->publish(std::move(task));
  }
  void PublishMap(nav_msgs::msg::OccupancyGrid map) {
    map_publisher_->publish(std::move(map));
  }
  void PublishOdometry(nav_msgs::msg::Odometry odometry) {
    odometry_publisher_->publish(std::move(odometry));
  }
  void PublishTask(Task task) { task_publisher_->publish(std::move(task)); }
  void PollExecution() { explorer_->PollExecution(); }
  bool HasActiveGoal() const {
    return ExplorationNodeTestPeer::ActiveTarget(*explorer_).has_value();
  }

  std::optional<Status> LatestStatus() const {
    std::scoped_lock lock{messages_mutex_};
    return statuses_.empty() ? std::nullopt
                             : std::optional<Status>{statuses_.back()};
  }
  std::size_t StatusCount() const {
    std::scoped_lock lock{messages_mutex_};
    return statuses_.size();
  }
  std::vector<std::uint8_t> StatusTransitions() const {
    std::scoped_lock lock{messages_mutex_};
    return status_transitions_;
  }
  std::size_t StuckRetryCount() const {
    std::scoped_lock lock{messages_mutex_};
    return static_cast<std::size_t>(
        std::ranges::count_if(statuses_, [](const Status &status) {
          return status.reason_code == "STUCK_RETRY";
        }));
  }
  std::vector<lunar_planning_msgs::msg::MotionReference> References() const {
    std::scoped_lock lock{messages_mutex_};
    return references_;
  }
  std::vector<std::string> ExecutionCancels() const {
    std::scoped_lock lock{messages_mutex_};
    return execution_cancels_;
  }
  std::optional<geometry_msgs::msg::PoseStamped> LatestCurrentGoal() const {
    std::scoped_lock lock{messages_mutex_};
    return current_goals_.empty()
               ? std::nullopt
               : std::optional<geometry_msgs::msg::PoseStamped>{
                     current_goals_.back()};
  }
  GoalCellPlannerServer &Server() { return *server_; }
  const std::shared_ptr<GoalCellScripts> &Scripts() const { return scripts_; }

  std::string Trace() const {
    std::ostringstream stream;
    stream << scripts_->Trace() << "\nstatus transitions:";
    for (const auto state : StatusTransitions()) {
      stream << ' ' << static_cast<int>(state);
    }
    stream << "\nreference plan ids:";
    for (const auto &reference : References()) {
      stream << ' ' << reference.plan_id;
    }
    stream << "\nexecution cancellation plan ids:";
    for (const auto &plan_id : ExecutionCancels()) {
      stream << ' ' << plan_id;
    }
    stream << "\nwall elapsed ms="
           << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - started_)
                  .count();
    return stream.str();
  }
  void ExpectWithinDeadline() const {
    EXPECT_LT(std::chrono::steady_clock::now() - started_, kCaseDeadline)
        << Trace();
  }

private:
  static std::atomic<std::uint64_t> sequence_;
  std::chrono::steady_clock::time_point started_;
  std::shared_ptr<GoalCellScripts> scripts_;
  std::shared_ptr<std::chrono::steady_clock::time_point> now_;
  std::string prefix_;
  ExplorationNodeParameters parameters_;
  std::shared_ptr<rclcpp::Node> server_node_;
  std::shared_ptr<rclcpp::Node> io_node_;
  std::shared_ptr<ExplorationNode> explorer_;
  std::unique_ptr<GoalCellPlannerServer> server_;
  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::jthread spin_thread_;
  rclcpp::Publisher<Task>::SharedPtr task_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_publisher_;
  rclcpp::Subscription<lunar_planning_msgs::msg::MotionReference>::SharedPtr
      reference_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cancel_subscription_;
  rclcpp::Subscription<Status>::SharedPtr status_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      current_goal_subscription_;
  mutable std::mutex messages_mutex_;
  std::vector<Status> statuses_;
  std::vector<std::uint8_t> status_transitions_;
  std::vector<lunar_planning_msgs::msg::MotionReference> references_;
  std::vector<std::string> execution_cancels_;
  std::vector<geometry_msgs::msg::PoseStamped> current_goals_;
};

std::atomic<std::uint64_t> ScenarioHarness::sequence_{0U};

std::string StatusName(const std::uint8_t state) {
  switch (state) {
  case Status::IDLE:
    return "IDLE";
  case Status::WAITING_FOR_INPUT:
    return "WAITING_FOR_INPUT";
  case Status::SELECTING_FRONTIER:
    return "SELECTING_FRONTIER";
  case Status::PLANNING:
    return "PLANNING";
  case Status::EXECUTING:
    return "EXECUTING";
  case Status::REPLANNING:
    return "REPLANNING";
  case Status::PAUSED:
    return "PAUSED";
  case Status::COMPLETED:
    return "COMPLETED";
  case Status::ERROR:
    return "ERROR";
  default:
    return "UNKNOWN";
  }
}

std::string StableHash(const std::string &value) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char byte : value) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  std::ostringstream output;
  output << "fnv1a64:" << std::hex << hash;
  return output.str();
}

std::string NormalizedTrace(const ScenarioHarness &scenario) {
  std::ostringstream trace;
  for (const auto &record : scenario.Scripts()->Records()) {
    trace << "request=" << record.request_id << "|key=" << record.candidate_key
          << "|cell=" << CellString(record.goal_cell)
          << "|result=" << record.response << '\n';
  }
  trace << "states=";
  for (const auto state : scenario.StatusTransitions()) {
    trace << StatusName(state) << ',';
  }
  trace << "\nreferences=";
  for (const auto &reference : scenario.References()) {
    trace << reference.plan_id << ',';
  }
  trace << "\ncancels=";
  for (const auto &plan_id : scenario.ExecutionCancels()) {
    trace << plan_id << ',';
  }
  const auto status = scenario.LatestStatus();
  trace << "\ncoverage=";
  if (status) {
    trace << std::fixed << std::setprecision(9) << status->coverage_ratio
          << "|completed=" << (status->state == Status::COMPLETED)
          << "|replan=" << status->replan_count
          << "|stuck=" << scenario.StuckRetryCount();
    if (status->completed_goal_count > 0U) {
      trace << "\narrival=completed_goals=" << status->completed_goal_count
            << "|active_goal=" << scenario.HasActiveGoal()
            << "|failed_candidates=" << status->failed_candidate_count;
    }
  }
  const auto goal = scenario.LatestCurrentGoal();
  trace << "\ncommitted="
        << (goal ? PoseKeyString(goal->pose) : std::string{"none"});
  return trace.str();
}

void ExpectCanonicalTrace(const ScenarioHarness &scenario,
                          const CanonicalFixture &fixture) {
  const std::string diagnostic =
      scenario.Trace() + "\nnormalized trace:\n" + NormalizedTrace(scenario) +
      "\nhash=" + StableHash(NormalizedTrace(scenario));
  const auto records = scenario.Scripts()->Records();
  const auto requests = fixture.expected["ordered_requests"];
  ASSERT_EQ(records.size(), requests.size()) << diagnostic;
  for (std::size_t index = 0U; index < records.size(); ++index) {
    EXPECT_EQ(records[index].request_id,
              requests[index]["request_id"].as<std::string>())
        << diagnostic;
    EXPECT_EQ(records[index].candidate_key,
              requests[index]["candidate_key"].as<std::string>())
        << diagnostic;
    EXPECT_EQ(CellString(records[index].goal_cell),
              requests[index]["goal_cell"].as<std::string>())
        << diagnostic;
    EXPECT_EQ(records[index].response,
              requests[index]["response_kind"].as<std::string>())
        << diagnostic;
  }
  std::vector<std::string> observed_states;
  for (const auto state : scenario.StatusTransitions()) {
    observed_states.push_back(StatusName(state));
  }
  EXPECT_EQ(observed_states,
            fixture.expected["status_sequence"].as<std::vector<std::string>>())
      << diagnostic;
  std::vector<std::string> reference_ids;
  for (const auto &reference : scenario.References()) {
    reference_ids.push_back(reference.plan_id);
  }
  EXPECT_EQ(
      reference_ids,
      fixture.expected["reference_plan_ids"].as<std::vector<std::string>>())
      << diagnostic;
  EXPECT_EQ(
      scenario.ExecutionCancels(),
      fixture.expected["cancellation_plan_ids"].as<std::vector<std::string>>())
      << diagnostic;
  const auto status = scenario.LatestStatus();
  ASSERT_TRUE(status.has_value()) << diagnostic;
  EXPECT_DOUBLE_EQ(status->coverage_ratio,
                   fixture.expected["coverage_ratio"].as<double>())
      << diagnostic;
  EXPECT_EQ(status->state == Status::COMPLETED,
            fixture.expected["completed"].as<bool>())
      << diagnostic;
  EXPECT_EQ(status->replan_count,
            fixture.expected["replan_count"].as<std::uint32_t>())
      << diagnostic;
  EXPECT_EQ(scenario.StuckRetryCount(),
            fixture.expected["stuck_retries"].as<std::size_t>())
      << diagnostic;
  const auto goal = scenario.LatestCurrentGoal();
  const auto observed_goal =
      goal ? PoseKeyString(goal->pose) : std::string{"none"};
  EXPECT_EQ(observed_goal,
            fixture.expected["committed_goal_key"].as<std::string>())
      << diagnostic;
  EXPECT_EQ(scenario.Scripts()->Remaining(), 0U) << diagnostic;
  EXPECT_TRUE(scenario.Scripts()->Error().empty()) << diagnostic;
  EXPECT_EQ(StableHash(NormalizedTrace(scenario)),
            fixture.expected["ordered_trace_id"].as<std::string>())
      << diagnostic;
  scenario.ExpectWithinDeadline();
}

std::shared_ptr<GoalCellScripts>
LoadVariantScripts(const CanonicalFixture &fixture,
                   const std::string &variant_name) {
  const auto variant = fixture.variants[variant_name];
  const auto count = variant["request_count"].as<std::size_t>();
  const auto success_index = variant["success_request_index"].as<std::size_t>();
  std::map<CellKey, std::vector<ResponseSpec>> by_cell;
  const auto requests = fixture.expected["ordered_requests"];
  for (std::size_t index = 0U; index < count; ++index) {
    const auto request = requests[index];
    by_cell[ParseCell(request["goal_cell"].as<std::string>())].push_back(
        {.kind = index == success_index ? ResponseKind::kSuccess
                                        : ResponseKind::kNoPath,
         .executable_endpoint = std::nullopt,
         .expected_candidate_key = request["candidate_key"].as<std::string>(),
         .expected_request_id = request["request_id"].as<std::string>()});
  }
  auto scripts = std::make_shared<GoalCellScripts>();
  scripts->EnableStrict();
  for (auto &[cell, responses] : by_cell) {
    scripts->SetCanonical(cell, std::move(responses));
  }
  return scripts;
}

void ExpectVariantTrace(const ScenarioHarness &scenario,
                        const CanonicalFixture &fixture,
                        const std::string &variant_name) {
  const auto variant = fixture.variants[variant_name];
  const auto expected = variant["expected"];
  const auto count = variant["request_count"].as<std::size_t>();
  const auto success_index = variant["success_request_index"].as<std::size_t>();
  const auto records = scenario.Scripts()->Records();
  const std::string diagnostic =
      scenario.Trace() + "\nnormalized trace:\n" + NormalizedTrace(scenario) +
      "\nhash=" + StableHash(NormalizedTrace(scenario));
  ASSERT_EQ(records.size(), count) << diagnostic;
  const auto base_requests = fixture.expected["ordered_requests"];
  for (std::size_t index = 0U; index < count; ++index) {
    EXPECT_EQ(records[index].request_id,
              base_requests[index]["request_id"].as<std::string>())
        << diagnostic;
    EXPECT_EQ(records[index].candidate_key,
              base_requests[index]["candidate_key"].as<std::string>())
        << diagnostic;
    EXPECT_EQ(CellString(records[index].goal_cell),
              base_requests[index]["goal_cell"].as<std::string>())
        << diagnostic;
    EXPECT_EQ(records[index].response,
              index == success_index ? "SUCCESS" : "NO_PATH")
        << diagnostic;
  }
  std::vector<std::string> states;
  for (const auto state : scenario.StatusTransitions()) {
    states.push_back(StatusName(state));
  }
  EXPECT_EQ(states, expected["status_sequence"].as<std::vector<std::string>>())
      << diagnostic;
  std::vector<std::string> reference_ids;
  for (const auto &reference : scenario.References()) {
    reference_ids.push_back(reference.plan_id);
  }
  EXPECT_EQ(reference_ids,
            expected["reference_plan_ids"].as<std::vector<std::string>>())
      << diagnostic;
  EXPECT_EQ(scenario.ExecutionCancels(),
            expected["cancellation_plan_ids"].as<std::vector<std::string>>())
      << diagnostic;
  const auto status = scenario.LatestStatus();
  ASSERT_TRUE(status.has_value()) << diagnostic;
  EXPECT_DOUBLE_EQ(status->coverage_ratio,
                   expected["coverage_ratio"].as<double>())
      << diagnostic;
  EXPECT_EQ(status->state == Status::COMPLETED,
            expected["completed"].as<bool>())
      << diagnostic;
  EXPECT_EQ(status->replan_count, expected["replan_count"].as<std::uint32_t>())
      << diagnostic;
  EXPECT_EQ(scenario.StuckRetryCount(),
            expected["stuck_retries"].as<std::size_t>())
      << diagnostic;
  const auto goal = scenario.LatestCurrentGoal();
  EXPECT_EQ(goal ? PoseKeyString(goal->pose) : std::string{"none"},
            expected["committed_goal_key"].as<std::string>())
      << diagnostic;
  EXPECT_EQ(scenario.Scripts()->Remaining(), 0U) << diagnostic;
  EXPECT_TRUE(scenario.Scripts()->Error().empty()) << diagnostic;
  EXPECT_EQ(StableHash(NormalizedTrace(scenario)),
            expected["ordered_trace_id"].as<std::string>())
      << diagnostic;
  scenario.ExpectWithinDeadline();
}

std::shared_ptr<ExplorationPipelineSeams>
ControlledPipeline(std::vector<lunar::pure_exploration::Pose2> poses) {
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [poses = std::move(poses)](
          const lunar::pure_exploration::TaskRaster &,
          const std::span<const lunar::pure_exploration::FrontierCluster>
              frontiers) { return ControlledCandidates(frontiers, poses); };
  seams->evaluate_gain =
      [](const lunar::pure_exploration::TaskRaster &,
         const lunar::pure_exploration::CandidateView &candidate) {
        return 10000.0 - static_cast<double>(candidate.id);
      };
  return seams;
}

TEST(SyntheticScenarioRunnerTest,
     ConcaveReachableFrontierPublishesReferenceAndRemainsExecuting) {
  const auto fixture = LoadFixture("concave_region");
  const auto scripts = LoadCanonicalScripts(fixture);
  ScenarioHarness scenario(scripts, {}, fixture.initial_map.info.resolution);
  scenario.PublishInputs(fixture.task, fixture.initial_map, fixture.odometry);

  ASSERT_TRUE(WaitFor([&scenario] {
    const auto status = scenario.LatestStatus();
    return scenario.References().size() == 1U && status &&
           status->state == Status::EXECUTING;
  })) << scenario.Trace();
  const auto status = scenario.LatestStatus();
  ASSERT_TRUE(status.has_value()) << scenario.Trace();
  EXPECT_EQ(status->state, Status::EXECUTING) << scenario.Trace();
  EXPECT_LT(status->coverage_ratio, 1.0) << scenario.Trace();
  ASSERT_EQ(scripts->Records().size(), 15U) << scenario.Trace();
  EXPECT_FALSE(scripts->Records().front().candidate_key.empty())
      << scenario.Trace();
  EXPECT_TRUE(scripts->Error().empty()) << scenario.Trace();
  ExpectCanonicalTrace(scenario, fixture);
}

TEST(SyntheticScenarioRunnerTest,
     PointReachableNarrowPassageSurvivesRealWfdAndCandidatePrefilter) {
  const auto fixture = LoadFixture("unreachable_frontier");
  const auto scripts = LoadVariantScripts(fixture, "narrow_passage_continue");
  ScenarioHarness scenario(scripts, {}, fixture.initial_map.info.resolution);
  scenario.PublishInputs(fixture.task, fixture.initial_map, fixture.odometry);

  ASSERT_TRUE(WaitFor([&scenario] {
    const auto status = scenario.LatestStatus();
    return scenario.References().size() == 1U &&
           scenario.LatestCurrentGoal().has_value() && status &&
           status->state == Status::EXECUTING;
  })) << scenario.Trace();
  const auto records = scripts->Records();
  ASSERT_EQ(records.size(), 16U) << scenario.Trace();
  EXPECT_EQ(records[0].response, "NO_PATH") << scenario.Trace();
  EXPECT_EQ(records[1].response, "SUCCESS") << scenario.Trace();
  ExpectVariantTrace(scenario, fixture, "narrow_passage_continue");
}

TEST(SyntheticScenarioRunnerTest,
     MapContentGrowthReevaluatesFailedCellAndTwoSegmentsKeepOneCommit) {
  const auto fixture = LoadFixture("map_growth");
  const auto scripts = LoadCanonicalScripts(fixture);
  ScenarioHarness scenario(scripts, {}, fixture.initial_map.info.resolution);
  scenario.PublishInputs(fixture.task, fixture.initial_map, fixture.odometry);

  ASSERT_TRUE(WaitFor([scripts] { return scripts->Records().size() == 1U; }))
      << scenario.Trace();
  const std::string delayed_request = scripts->Records().front().request_id;
  ASSERT_TRUE(WaitFor([&scenario, &delayed_request] {
    return scenario.Server().HasHandle(delayed_request);
  })) << scenario.Trace();
  ASSERT_TRUE(fixture.growth_map.has_value()) << scenario.Trace();
  scenario.PublishMap(*fixture.growth_map);
  scenario.Server().Finish(delayed_request,
                           {ResponseKind::kNoPath, std::nullopt});
  ASSERT_TRUE(WaitFor([&scenario] {
    return scenario.References().size() == 1U &&
           scenario.LatestCurrentGoal().has_value();
  })) << scenario.Trace();
  const auto committed = scenario.LatestCurrentGoal();
  ASSERT_TRUE(committed.has_value()) << scenario.Trace();
  const auto first_reference = scenario.References().front();
  ASSERT_EQ(fixture.execution_odometries.size(), 2U) << scenario.Trace();
  scenario.PublishOdometry(fixture.execution_odometries.front());
  ASSERT_TRUE(WaitFor([&scenario] {
    scenario.PollExecution();
    const auto status = scenario.LatestStatus();
    return scenario.References().size() == 2U && status &&
           status->state == Status::EXECUTING;
  })) << scenario.Trace();
  const auto retained = scenario.LatestCurrentGoal();
  ASSERT_TRUE(retained.has_value()) << scenario.Trace();
  EXPECT_EQ(PoseKeyString(retained->pose), PoseKeyString(committed->pose))
      << scenario.Trace();
  ASSERT_EQ(scenario.ExecutionCancels().size(), 1U) << scenario.Trace();
  EXPECT_EQ(scenario.ExecutionCancels().front(), first_reference.plan_id)
      << scenario.Trace();
  const auto before_arrival = scenario.LatestStatus();
  ASSERT_TRUE(before_arrival.has_value()) << scenario.Trace();
  EXPECT_TRUE(scenario.HasActiveGoal()) << scenario.Trace();
  EXPECT_EQ(before_arrival->completed_goal_count, 0U) << scenario.Trace();

  const auto two_segment = fixture.expected["two_segment"];
  const auto endpoint_cells =
      two_segment["endpoint_cells"].as<std::vector<std::string>>();
  const auto references = scenario.References();
  ASSERT_EQ(references.size(), endpoint_cells.size()) << scenario.Trace();
  for (std::size_t index = 0U; index < references.size(); ++index) {
    ASSERT_FALSE(references[index].trajectory.points.empty())
        << scenario.Trace();
    const auto &transforms =
        references[index].trajectory.points.back().transforms;
    ASSERT_FALSE(transforms.empty()) << scenario.Trace();
    const auto &translation = transforms.back().translation;
    EXPECT_EQ(CellString(WorldCell(translation.x, translation.y,
                                   fixture.initial_map.info.resolution)),
              endpoint_cells[index])
        << scenario.Trace();
  }
  const auto &final_odometry = fixture.execution_odometries.back();
  EXPECT_EQ(PoseKeyString(final_odometry.pose.pose),
            two_segment["committed_candidate_key"].as<std::string>())
      << scenario.Trace();
  EXPECT_EQ(CellString(WorldCell(final_odometry.pose.pose.position.x,
                                 final_odometry.pose.pose.position.y,
                                 fixture.initial_map.info.resolution)),
            two_segment["final_goal_cell"].as<std::string>())
      << scenario.Trace();
  scenario.PublishOdometry(final_odometry);
  ASSERT_TRUE(WaitFor([&scenario] {
    scenario.PollExecution();
    const auto status = scenario.LatestStatus();
    return status && status->completed_goal_count == 1U &&
           status->state == Status::SELECTING_FRONTIER &&
           !scenario.HasActiveGoal();
  })) << scenario.Trace();
  const auto arrived_status = scenario.LatestStatus();
  ASSERT_TRUE(arrived_status.has_value()) << scenario.Trace();
  EXPECT_EQ(scenario.HasActiveGoal(),
            two_segment["active_goal_after_arrival"].as<bool>())
      << scenario.Trace();
  EXPECT_EQ(arrived_status->completed_goal_count,
            two_segment["completed_goal_count"].as<std::uint32_t>())
      << scenario.Trace();
  EXPECT_EQ(arrived_status->failed_candidate_count,
            before_arrival->failed_candidate_count +
                two_segment["failed_candidate_growth"].as<std::uint32_t>())
      << scenario.Trace();
  EXPECT_EQ(arrived_status->replan_count, 0U) << scenario.Trace();
  EXPECT_EQ(scenario.StuckRetryCount(), 0U) << scenario.Trace();
  ExpectCanonicalTrace(scenario, fixture);
}

TEST(SyntheticScenarioRunnerTest,
     FullyKnownTaskCompletesNormallyWithCoverageOneWithoutPlannerCalls) {
  const auto fixture = LoadFixture("fully_known_completion");
  const auto scripts = LoadCanonicalScripts(fixture);
  ScenarioHarness scenario(scripts, {}, fixture.initial_map.info.resolution);
  scenario.PublishInputs(fixture.task, fixture.initial_map, fixture.odometry);

  ASSERT_TRUE(WaitFor([&scenario] {
    const auto status = scenario.LatestStatus();
    return status && status->state == Status::COMPLETED;
  })) << scenario.Trace();
  const auto status = scenario.LatestStatus();
  ASSERT_TRUE(status.has_value()) << scenario.Trace();
  EXPECT_DOUBLE_EQ(status->coverage_ratio, 1.0) << scenario.Trace();
  EXPECT_EQ(status->reason_code, "COMPLETED_NO_REACHABLE_FRONTIER")
      << scenario.Trace();
  EXPECT_TRUE(scripts->Records().empty()) << scenario.Trace();
  EXPECT_TRUE(scripts->Error().empty()) << scenario.Trace();
  ExpectCanonicalTrace(scenario, fixture);
}

TEST(SyntheticScenarioRunnerTest,
     PartlyUnknownTaskCompletesBelowOneOnlyAfterEveryCandidateNoPath) {
  const auto fixture = LoadFixture("unreachable_frontier");
  const auto scripts = LoadCanonicalScripts(fixture);
  ScenarioHarness scenario(scripts, {}, fixture.initial_map.info.resolution);
  scenario.PublishInputs(fixture.task, fixture.initial_map, fixture.odometry);

  ASSERT_TRUE(WaitFor([&scenario] {
    const auto status = scenario.LatestStatus();
    return status && status->state == Status::COMPLETED;
  })) << scenario.Trace();
  const auto status = scenario.LatestStatus();
  ASSERT_TRUE(status.has_value()) << scenario.Trace();
  EXPECT_LT(status->coverage_ratio, 1.0) << scenario.Trace();
  EXPECT_EQ(status->reason_code, "COMPLETED_NO_REACHABLE_FRONTIER")
      << scenario.Trace();
  const auto records = scripts->Records();
  ASSERT_EQ(records.size(), 18U) << scenario.Trace();
  EXPECT_TRUE(std::ranges::all_of(records, [](const auto &record) {
    return record.response == "NO_PATH";
  })) << scenario.Trace();
  EXPECT_TRUE(scripts->Error().empty()) << scenario.Trace();
  ExpectCanonicalTrace(scenario, fixture);
}

enum class MetadataMutation { kTimestamp, kMapVersion, kCovariance };

class MetadataInvariantScenarioTest
    : public ::testing::TestWithParam<MetadataMutation> {};

TEST_P(MetadataInvariantScenarioTest,
       MetadataOnlyUpdatePreservesExactActiveActionBehavior) {
  const auto fixture = LoadFixture("concave_region");
  const auto scripts = LoadCanonicalScripts(fixture);
  ScenarioHarness scenario(scripts, {}, fixture.initial_map.info.resolution);
  auto map = fixture.initial_map;
  scenario.PublishInputs(fixture.task, map, fixture.odometry);
  ASSERT_TRUE(WaitFor([&scenario] {
    const auto status = scenario.LatestStatus();
    return scenario.References().size() == 1U &&
           scenario.LatestCurrentGoal().has_value() && status &&
           status->state == Status::EXECUTING;
  })) << scenario.Trace();
  const auto status_before = scenario.LatestStatus();
  const auto goal_before = scenario.LatestCurrentGoal();
  const auto status_count = scenario.StatusCount();
  ASSERT_TRUE(status_before.has_value()) << scenario.Trace();
  ASSERT_TRUE(goal_before.has_value()) << scenario.Trace();

  if (GetParam() == MetadataMutation::kTimestamp) {
    map.header.stamp.sec += 100;
    scenario.PublishMap(map);
  } else if (GetParam() == MetadataMutation::kMapVersion) {
    map.info.map_load_time.sec += 100;
    scenario.PublishMap(map);
  } else {
    auto covariance = fixture.odometry;
    covariance.pose.covariance[0] = 1000000.0;
    covariance.pose.covariance[7] = 1000000.0;
    covariance.pose.covariance[35] = 1000000.0;
    scenario.PublishOdometry(covariance);
  }

  ASSERT_TRUE(WaitFor([&scenario, status_count] {
    return scenario.StatusCount() > status_count;
  })) << scenario.Trace();
  const auto status_after = scenario.LatestStatus();
  const auto goal_after = scenario.LatestCurrentGoal();
  ASSERT_TRUE(status_after.has_value()) << scenario.Trace();
  ASSERT_TRUE(goal_after.has_value()) << scenario.Trace();
  EXPECT_EQ(status_after->state, Status::EXECUTING) << scenario.Trace();
  EXPECT_EQ(status_after->current_plan_id, status_before->current_plan_id)
      << scenario.Trace();
  EXPECT_EQ(scripts->Records().size(), 15U) << scenario.Trace();
  EXPECT_EQ(scenario.References().size(), 1U) << scenario.Trace();
  EXPECT_TRUE(scenario.ExecutionCancels().empty()) << scenario.Trace();
  EXPECT_DOUBLE_EQ(goal_after->pose.position.x, goal_before->pose.position.x)
      << scenario.Trace();
  EXPECT_DOUBLE_EQ(goal_after->pose.position.y, goal_before->pose.position.y)
      << scenario.Trace();
  EXPECT_TRUE(scripts->Error().empty()) << scenario.Trace();
  ExpectCanonicalTrace(scenario, fixture);
}

INSTANTIATE_TEST_SUITE_P(StampVersionCovariance, MetadataInvariantScenarioTest,
                         ::testing::Values(MetadataMutation::kTimestamp,
                                           MetadataMutation::kMapVersion,
                                           MetadataMutation::kCovariance));

TEST(SyntheticScenarioRunnerTest,
     ExecutionTimeoutCannotCompleteAndRetriesTheSameCommittedGoal) {
  const auto scripts = std::make_shared<GoalCellScripts>();
  scripts->Set({9, 4}, "timeout-retry",
               {{ResponseKind::kSuccess, std::pair{3.5, 2.0}},
                {ResponseKind::kTimeout, std::nullopt},
                {ResponseKind::kDelayed, std::nullopt}});
  ScenarioHarness scenario(scripts, ControlledPipeline({{4.5, 2.0, 0.0}}));
  scenario.PublishInputs(StartTask("task15-timeout"), ExplorationMap());
  ASSERT_TRUE(WaitFor([&scenario] {
    return scenario.References().size() == 1U &&
           scenario.LatestCurrentGoal().has_value();
  })) << scenario.Trace();
  const auto committed = scenario.LatestCurrentGoal();
  ASSERT_TRUE(committed.has_value()) << scenario.Trace();

  scenario.PublishOdometry(Odometry(3.5, 2.0, 0.0));
  ASSERT_TRUE(WaitFor([&scenario, scripts] {
    scenario.PollExecution();
    return scripts->Records().size() >= 3U;
  })) << scenario.Trace();
  const auto records = scripts->Records();
  ASSERT_EQ(records.size(), 3U) << scenario.Trace();
  EXPECT_EQ(records[1].response, "TIMEOUT") << scenario.Trace();
  EXPECT_EQ(records[2].response, "DELAYED") << scenario.Trace();
  EXPECT_EQ(records[0].goal_cell, records[1].goal_cell) << scenario.Trace();
  EXPECT_EQ(records[1].goal_cell, records[2].goal_cell) << scenario.Trace();
  EXPECT_NE(records[0].request_id, records[1].request_id) << scenario.Trace();
  EXPECT_NE(records[1].request_id, records[2].request_id) << scenario.Trace();
  const auto before_success = scenario.LatestStatus();
  ASSERT_TRUE(before_success.has_value()) << scenario.Trace();
  EXPECT_EQ(before_success->state, Status::REPLANNING) << scenario.Trace();
  EXPECT_EQ(before_success->completed_goal_count, 0U) << scenario.Trace();
  EXPECT_EQ(scenario.References().size(), 1U) << scenario.Trace();
  ASSERT_TRUE(WaitFor([&scenario, &records] {
    return scenario.Server().HasHandle(records[2].request_id);
  })) << scenario.Trace();
  scenario.Server().Finish(records[2].request_id,
                           {ResponseKind::kSuccess, std::pair{4.5, 2.0}});

  ASSERT_TRUE(WaitFor([&scenario] {
    return scenario.References().size() == 2U;
  })) << scenario.Trace();
  const auto after_success = scenario.LatestStatus();
  ASSERT_TRUE(after_success.has_value()) << scenario.Trace();
  EXPECT_EQ(after_success->state, Status::EXECUTING) << scenario.Trace();
  EXPECT_EQ(after_success->replan_count, 0U) << scenario.Trace();
  EXPECT_EQ(after_success->failed_candidate_count, 0U) << scenario.Trace();
  const auto retained = scenario.LatestCurrentGoal();
  ASSERT_TRUE(retained.has_value()) << scenario.Trace();
  EXPECT_DOUBLE_EQ(retained->pose.position.x, committed->pose.position.x)
      << scenario.Trace();
  EXPECT_DOUBLE_EQ(retained->pose.position.y, committed->pose.position.y)
      << scenario.Trace();
  EXPECT_EQ(scenario.ExecutionCancels().size(), 1U) << scenario.Trace();
  EXPECT_TRUE(scripts->Error().empty()) << scenario.Trace();
  scenario.ExpectWithinDeadline();
}

TEST(SyntheticScenarioRunnerTest, InvalidSuccessPayloadFailsClosed) {
  const auto scripts = std::make_shared<GoalCellScripts>();
  scripts->Set({9, 4}, "invalid-success",
               {{ResponseKind::kInvalidSuccess, std::nullopt}});
  ScenarioHarness scenario(scripts, ControlledPipeline({{4.5, 2.0, 0.0}}));
  scenario.PublishInputs(StartTask("task15-invalid-success"), ExplorationMap());

  ASSERT_TRUE(WaitFor([&scenario] {
    const auto status = scenario.LatestStatus();
    return status && status->state == Status::ERROR;
  })) << scenario.Trace();
  const auto status = scenario.LatestStatus();
  ASSERT_TRUE(status.has_value()) << scenario.Trace();
  EXPECT_EQ(status->reason_code, "PLANNER_CONTRACT_PATH_PREVIEW_INVALID")
      << scenario.Trace();
  EXPECT_TRUE(scenario.References().empty()) << scenario.Trace();
  scenario.ExpectWithinDeadline();
}

TEST(SyntheticScenarioRunnerTest,
     AcceptedActionCancelReachesPausedOnlyAfterCanceledTerminal) {
  const auto scripts = std::make_shared<GoalCellScripts>();
  scripts->Set({9, 4}, "cancel-pending",
               {{ResponseKind::kDelayed, std::nullopt}});
  ScenarioHarness scenario(scripts, ControlledPipeline({{4.5, 2.0, 0.0}}));
  scenario.PublishInputs(StartTask("task15-cancel"), ExplorationMap());
  ASSERT_TRUE(WaitFor([scripts] { return scripts->Records().size() == 1U; }))
      << scenario.Trace();
  const auto request_id = scripts->Records().front().request_id;
  ASSERT_TRUE(WaitFor([&scenario, &request_id] {
    return scenario.Server().HasHandle(request_id);
  })) << scenario.Trace();

  Task pause;
  pause.command = Task::PAUSE;
  scenario.PublishTask(pause);
  ASSERT_TRUE(WaitFor([&scenario] {
    return scenario.Server().CancelCount() == 1U;
  })) << scenario.Trace();
  const auto before_terminal = scenario.LatestStatus();
  ASSERT_TRUE(before_terminal.has_value()) << scenario.Trace();
  EXPECT_NE(before_terminal->state, Status::PAUSED) << scenario.Trace();
  scenario.Server().FinishCanceled(request_id);
  ASSERT_TRUE(WaitFor([&scenario] {
    const auto status = scenario.LatestStatus();
    return status && status->state == Status::PAUSED;
  })) << scenario.Trace();
  EXPECT_EQ(scripts->Records().front().response, "CANCELED")
      << scenario.Trace();
  scenario.ExpectWithinDeadline();
}

} // namespace
} // namespace lunar::pure_exploration_ros
