#include "lunar_incremental_navigation_ros/platform_config.hpp"

#include <filesystem>
#include <fstream>
#include <variant>
#include <yaml-cpp/yaml.h>
#include <gtest/gtest.h>

TEST(PlatformConfig, ModifiedProfileUsesConfiguredValuesAndRejectsInvalidSpeed) {
  auto config = YAML::LoadFile(std::string{PLATFORM_CONFIG_DIR} + "/wheel.yaml");
  config["capability"]["maximum_forward_speed_mps"] = 0.1;
  config["capability"]["maximum_acceleration_mps2"] = 0.25;
  config["capability_version"] = "custom-profile";
  auto primitives = config["capability"]["motion_primitives"];
  primitives.remove(primitives.size() - 1U);
  const auto path = std::filesystem::temp_directory_path() / "incremental_flexible_config_test.yaml";
  const auto write = [&] { std::ofstream stream(path); stream << config; };
  write();
  const auto result = lunar::incremental_navigation_ros::LoadPlatformConfig(path, "wheel");
  ASSERT_TRUE(result.capability) << result.error_detail;
  const auto& wheel = std::get<lunar::incremental_navigation::WheeledCapability>(*result.capability);
  EXPECT_DOUBLE_EQ(wheel.maximum_forward_speed_mps, 0.1);
  EXPECT_DOUBLE_EQ(wheel.maximum_acceleration_mps2, 0.25);
  EXPECT_EQ(wheel.motion_primitives.size(), 8U);
  config["capability"]["maximum_forward_speed_mps"] = -0.1;
  write();
  const auto invalid = lunar::incremental_navigation_ros::LoadPlatformConfig(path, "wheel");
  EXPECT_FALSE(invalid.capability);
  EXPECT_NE(invalid.error_detail.find("maximum_forward_speed_mps"), std::string::npos);
  std::filesystem::remove(path);
}
