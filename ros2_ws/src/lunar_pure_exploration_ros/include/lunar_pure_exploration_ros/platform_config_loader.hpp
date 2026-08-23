#pragma once

#include <filesystem>
#include <string_view>

#include "lunar_pure_exploration_core/candidate_generator.hpp"

namespace lunar::pure_exploration_ros {

struct LoadedPlatformConfig {
  lunar::pure_exploration::PlatformGeometry geometry;
};

[[nodiscard]] LoadedPlatformConfig LoadPlatformConfig(
    const std::filesystem::path& yaml_path,
    std::string_view platform_selector);

}  // namespace lunar::pure_exploration_ros
