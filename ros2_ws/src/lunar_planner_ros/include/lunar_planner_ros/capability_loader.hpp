#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "lunar_planner_core/types/platform_capability.hpp"

namespace lunar::planning::ros {

struct ObservationCapability final {
  double sensor_range_m{};
  double sensor_fov_rad{};
};

enum class GeometrySourceKind : std::uint8_t {
  kUrdfMesh,
  kParametricEnvelope,
};

struct ParametricWheeledGeometry final {
  std::size_t wheel_count{};
  std::vector<lunar::planning::Vec2> wheel_center_xy_m;
};

struct LoadedCapabilities final {
  std::string platform_id;
  std::string capability_version;
  std::string base_frame_id;
  std::string reference_point;
  std::string actuator_profile_id;
  std::optional<double> maximum_obstacle_height_m;
  std::vector<std::string> source_motion_primitive_ids;
  std::map<std::string, std::string, std::less<>> field_source_types;
  GeometrySourceKind geometry_source_kind{GeometrySourceKind::kUrdfMesh};
  std::optional<ParametricWheeledGeometry> parametric_wheeled_geometry;
  std::filesystem::path urdf_path;
  std::vector<std::filesystem::path> mesh_paths;
  ObservationCapability observation;
  lunar::planning::PlatformCapability platform;
};

enum class CapabilityLoadErrorCode : std::uint8_t {
  kPackageNotFound,
  kUnsafePath,
  kFileMissing,
  kParseError,
  kSchemaInvalid,
  kValueInvalid,
  kUrdfInvalid,
  kMeshMissing,
};

struct CapabilityLoadError final {
  CapabilityLoadErrorCode code{CapabilityLoadErrorCode::kSchemaInvalid};
  std::string reason_code;
  std::string detail;
};

struct CapabilityLoadResult final {
  std::optional<LoadedCapabilities> capabilities;
  std::optional<CapabilityLoadError> error;

  [[nodiscard]] bool ok() const noexcept {
    return capabilities.has_value() && !error.has_value();
  }
};

class CapabilityLoader final {
 public:
  [[nodiscard]] CapabilityLoadResult LoadFromPackageShare(
      const std::string& package_name,
      const std::filesystem::path& platform_capability_file,
      const std::filesystem::path& observation_capability_file) const;

  [[nodiscard]] CapabilityLoadResult LoadFromShareDirectory(
      const std::filesystem::path& package_share_directory,
      const std::filesystem::path& platform_capability_file,
      const std::filesystem::path& observation_capability_file) const;
};

}  // namespace lunar::planning::ros
