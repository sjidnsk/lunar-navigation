#include "lunar_pure_exploration_ros/platform_config_loader.hpp"

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace lunar::pure_exploration_ros {
namespace {

[[noreturn]] void InvalidConfig() {
  throw std::invalid_argument{"invalid wheel platform config"};
}

void RequireExactKeys(
    const YAML::Node& node,
    const std::initializer_list<std::string_view> expected_keys) {
  if (!node.IsMap() || node.size() != expected_keys.size()) {
    InvalidConfig();
  }

  std::unordered_set<std::string> seen;
  for (const auto& entry : node) {
    if (!entry.first.IsScalar()) {
      InvalidConfig();
    }
    const std::string key = entry.first.as<std::string>();
    bool expected = false;
    for (const std::string_view candidate : expected_keys) {
      if (key == candidate) {
        expected = true;
        break;
      }
    }
    if (!expected || !seen.insert(key).second) {
      InvalidConfig();
    }
  }
}

YAML::Node Require(const YAML::Node& node, const std::string_view key) {
  const YAML::Node value = node[std::string{key}];
  if (!value) {
    InvalidConfig();
  }
  return value;
}

std::string RequireNonemptyString(const YAML::Node& node,
                                  const std::string_view key) {
  const YAML::Node value = Require(node, key);
  if (!value.IsScalar()) {
    InvalidConfig();
  }
  const std::string result = value.as<std::string>();
  if (result.empty()) {
    InvalidConfig();
  }
  return result;
}

double RequireFiniteDouble(const YAML::Node& node,
                           const std::string_view key) {
  const YAML::Node value = Require(node, key);
  if (!value.IsScalar()) {
    InvalidConfig();
  }
  const double result = value.as<double>();
  if (!std::isfinite(result)) {
    InvalidConfig();
  }
  return result;
}

lunar::pure_exploration::PlatformGeometry ParseWheel(
    const YAML::Node& root) {
  RequireExactKeys(root, {"platform", "platform_id", "platform_type",
                          "capability_version", "base_frame_id",
                          "capability"});

  if (RequireNonemptyString(root, "platform") != "wheel" ||
      RequireNonemptyString(root, "platform_type") != "WHEELED" ||
      RequireNonemptyString(root, "base_frame_id") != "base_footprint") {
    InvalidConfig();
  }
  const std::string platform_id =
      RequireNonemptyString(root, "platform_id");
  static_cast<void>(RequireNonemptyString(root, "capability_version"));

  const YAML::Node capability = Require(root, "capability");
  RequireExactKeys(
      capability,
      {"footprint_xy_m", "body_extent_m", "wheel_diameter_m",
       "wheel_width_m", "wheelbase_m", "track_width_m",
       "minimum_underbody_clearance_m",
       "maximum_local_obstacle_relief_m", "allow_unsupported_gap",
       "maximum_forward_speed_mps", "maximum_reverse_speed_mps",
       "maximum_spin_rate_radps", "maximum_acceleration_mps2",
       "maximum_braking_deceleration_mps2",
       "maximum_yaw_acceleration_radps2",
       "maximum_lateral_acceleration_mps2", "maximum_curvature_per_m",
       "maximum_slope_rad", "minimum_clearance_m", "motion_primitives"});

  const YAML::Node footprint = Require(capability, "footprint_xy_m");
  if (!footprint.IsSequence() || footprint.size() < 3U) {
    InvalidConfig();
  }
  std::vector<lunar::pure_exploration::Vec2> vertices;
  vertices.reserve(footprint.size());
  for (const YAML::Node& point : footprint) {
    if (!point.IsSequence() || point.size() != 2U ||
        !point[0U].IsScalar() || !point[1U].IsScalar()) {
      InvalidConfig();
    }
    const double x = point[0U].as<double>();
    const double y = point[1U].as<double>();
    if (!std::isfinite(x) || !std::isfinite(y)) {
      InvalidConfig();
    }
    vertices.push_back({x, y});
  }

  const double minimum_clearance_m =
      RequireFiniteDouble(capability, "minimum_clearance_m");
  if (minimum_clearance_m < 0.0) {
    InvalidConfig();
  }

  return lunar::pure_exploration::PlatformGeometry{
      .platform_id = platform_id,
      .platform_type = "WHEELED",
      .base_frame_id = "base_footprint",
      .footprint_vertices = std::move(vertices),
      .minimum_clearance_m = minimum_clearance_m,
  };
}

}  // namespace

LoadedPlatformConfig LoadPlatformConfig(
    const std::filesystem::path& yaml_path,
    const std::string_view platform_selector) {
  if (platform_selector != "wheel") {
    throw std::invalid_argument{"unsupported exploration platform selector"};
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(yaml_path.string());
  } catch (const YAML::BadFile& error) {
    throw std::runtime_error{"failed to read platform config: " +
                             std::string{error.what()}};
  } catch (const YAML::Exception& error) {
    throw std::invalid_argument{"malformed platform config: " +
                                std::string{error.what()}};
  }

  try {
    return LoadedPlatformConfig{.geometry = ParseWheel(root)};
  } catch (const YAML::Exception& error) {
    throw std::invalid_argument{"malformed platform config: " +
                                std::string{error.what()}};
  }
}

}  // namespace lunar::pure_exploration_ros
