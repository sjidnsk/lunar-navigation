#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation_ros {

struct PlatformConfigResult final {
  std::optional<lunar::incremental_navigation::PlatformCapability> capability;
  std::optional<double> start_blind_zone_margin_m;
  std::string reason_code;
};

[[nodiscard]] PlatformConfigResult LoadPlatformConfig(
    const std::filesystem::path& path, std::string_view platform);

}  // namespace lunar::incremental_navigation_ros
