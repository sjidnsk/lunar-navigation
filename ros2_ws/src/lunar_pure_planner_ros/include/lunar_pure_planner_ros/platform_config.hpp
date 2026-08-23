#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "lunar_pure_planner_core/types/platform_capability.hpp"

namespace lunar::pure_planner_ros {

struct PlatformConfigResult final {
  std::optional<lunar::pure_planning::PlatformCapability> capability;
  std::string reason_code;
};

[[nodiscard]] PlatformConfigResult LoadPlatformConfig(
    const std::filesystem::path& path, std::string_view platform);

}  // namespace lunar::pure_planner_ros
