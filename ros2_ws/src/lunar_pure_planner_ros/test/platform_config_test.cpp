#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/platform_config.hpp"

namespace {

std::filesystem::path ConfigPath(const std::string& filename) {
  return std::filesystem::path{LUNAR_PURE_PLANNER_CONFIG_DIR} / filename;
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path);
  EXPECT_TRUE(input.is_open());
  return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

std::string ReplaceExactlyOnce(std::string text, const std::string_view from,
                               const std::string_view to) {
  const std::size_t position = text.find(from);
  EXPECT_NE(position, std::string::npos);
  EXPECT_EQ(text.find(from, position + from.size()), std::string::npos);
  if (position != std::string::npos) {
    text.replace(position, from.size(), to);
  }
  return text;
}

std::filesystem::path WriteTemporaryConfig(const std::string_view name,
                                           const std::string& contents) {
  const auto path = std::filesystem::temp_directory_path() /
      ("lunar_pure_planner_" + std::string{name} + ".yaml");
  std::ofstream output(path);
  EXPECT_TRUE(output.is_open());
  output << contents;
  return path;
}

void ExpectRejected(const std::filesystem::path& path,
                    const std::string_view platform) {
  const auto result = lunar::pure_planner_ros::LoadPlatformConfig(path, platform);
  EXPECT_FALSE(result.capability.has_value());
  EXPECT_EQ(result.reason_code, "PLANNER_ERROR");
}

TEST(PlatformConfig, LoadsExactlyThreeFixedPlatforms) {
  EXPECT_TRUE(lunar::pure_planner_ros::LoadPlatformConfig(
                  ConfigPath("wheel.yaml"), "wheel")
                  .capability.has_value());
  EXPECT_TRUE(lunar::pure_planner_ros::LoadPlatformConfig(
                  ConfigPath("legged.yaml"), "legged")
                  .capability.has_value());
  EXPECT_TRUE(lunar::pure_planner_ros::LoadPlatformConfig(
                  ConfigPath("hopper.yaml"), "hopper")
                  .capability.has_value());
  EXPECT_EQ(lunar::pure_planner_ros::LoadPlatformConfig(
                ConfigPath("wheel.yaml"), "unknown")
                .reason_code,
            "PLANNER_ERROR");
}

TEST(PlatformConfig, UsesApprovedPrimitiveIdsAndOrder) {
  const auto wheel_result = lunar::pure_planner_ros::LoadPlatformConfig(
      ConfigPath("wheel.yaml"), "wheel");
  ASSERT_TRUE(wheel_result.capability.has_value());
  const auto& wheel = std::get<lunar::pure_planning::WheeledCapability>(
      *wheel_result.capability);
  const std::vector<std::string> wheel_ids{
      "forward", "reverse", "forward-arc-left", "forward-arc-right",
      "reverse-arc-left", "reverse-arc-right", "spin-left", "spin-right",
      "stop-switch"};
  ASSERT_EQ(wheel.motion_primitives.size(), wheel_ids.size());
  for (std::size_t index = 0U; index < wheel_ids.size(); ++index) {
    EXPECT_EQ(wheel.motion_primitives[index].primitive_id, wheel_ids[index]);
  }

  const auto legged_result = lunar::pure_planner_ros::LoadPlatformConfig(
      ConfigPath("legged.yaml"), "legged");
  ASSERT_TRUE(legged_result.capability.has_value());
  const auto& legged = std::get<lunar::pure_planning::LeggedCapability>(
      *legged_result.capability);
  const std::vector<std::string> legged_ids{
      "forward", "backward", "lateral-left", "lateral-right", "spin-left",
      "spin-right"};
  ASSERT_EQ(legged.motion_primitives.size(), legged_ids.size());
  for (std::size_t index = 0U; index < legged_ids.size(); ++index) {
    EXPECT_EQ(legged.motion_primitives[index].primitive_id, legged_ids[index]);
  }
}

TEST(PlatformConfig, RejectsAConfigWhoseDeclaredPlatformDoesNotMatch) {
  const auto result = lunar::pure_planner_ros::LoadPlatformConfig(
      ConfigPath("wheel.yaml"), "legged");

  EXPECT_FALSE(result.capability.has_value());
  EXPECT_EQ(result.reason_code, "PLANNER_ERROR");
}

TEST(PlatformConfig, RejectsMalformedYamlWithoutReturningCapability) {
  const auto path = std::filesystem::temp_directory_path() /
      "lunar_pure_planner_platform_config_malformed.yaml";
  {
    std::ofstream output(path);
    ASSERT_TRUE(output.is_open());
    output << "platform: wheel\\ncapability: [not-a-map\\n";
  }

  const auto result = lunar::pure_planner_ros::LoadPlatformConfig(path, "wheel");
  std::filesystem::remove(path);

  EXPECT_FALSE(result.capability.has_value());
  EXPECT_EQ(result.reason_code, "PLANNER_ERROR");
}

TEST(PlatformConfig, RejectsNonFiniteValueInAnOtherwiseValidConfig) {
  const auto path = WriteTemporaryConfig(
      "platform_config_nonfinite",
      ReplaceExactlyOnce(ReadText(ConfigPath("hopper.yaml")),
                         "specific_impulse_s: 301.0",
                         "specific_impulse_s: .nan"));
  ExpectRejected(path, "hopper");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsAChangedFixedCapabilityValue) {
  const auto path = WriteTemporaryConfig(
      "platform_config_value_drift",
      ReplaceExactlyOnce(ReadText(ConfigPath("wheel.yaml")),
                         "wheel_diameter_m: 0.319", "wheel_diameter_m: 0.320"));
  ExpectRejected(path, "wheel");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsCapabilityVersionDrift) {
  const auto path = WriteTemporaryConfig(
      "platform_config_version_drift",
      ReplaceExactlyOnce(ReadText(ConfigPath("legged.yaml")),
                         "capability_version: quad48-approved-baseline-v1",
                         "capability_version: quad48-approved-baseline-v2"));
  ExpectRejected(path, "legged");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsDuplicatePrimitiveId) {
  const auto path = WriteTemporaryConfig(
      "platform_config_duplicate_id",
      ReplaceExactlyOnce(ReadText(ConfigPath("wheel.yaml")),
                         "primitive_id: reverse,", "primitive_id: forward,"));
  ExpectRejected(path, "wheel");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsPrimitiveOrderDrift) {
  std::string contents = ReadText(ConfigPath("wheel.yaml"));
  contents = ReplaceExactlyOnce(contents, "primitive_id: forward,",
                                "primitive_id: placeholder,");
  contents = ReplaceExactlyOnce(contents, "primitive_id: reverse,",
                                "primitive_id: forward,");
  contents = ReplaceExactlyOnce(contents, "primitive_id: placeholder,",
                                "primitive_id: reverse,");
  const auto path = WriteTemporaryConfig("platform_config_order_drift", contents);
  ExpectRejected(path, "wheel");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsPrimitiveKindDrift) {
  const auto path = WriteTemporaryConfig(
      "platform_config_kind_drift",
      ReplaceExactlyOnce(ReadText(ConfigPath("wheel.yaml")),
                         "primitive_id: forward, kind: FORWARD",
                         "primitive_id: forward, kind: REVERSE"));
  ExpectRejected(path, "wheel");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsFixedBooleanDrift) {
  const auto path = WriteTemporaryConfig(
      "platform_config_bool_drift",
      ReplaceExactlyOnce(ReadText(ConfigPath("wheel.yaml")),
                         "allow_unsupported_gap: false",
                         "allow_unsupported_gap: true"));
  ExpectRejected(path, "wheel");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsMissingCapabilityKey) {
  const auto path = WriteTemporaryConfig(
      "platform_config_missing_key",
      ReplaceExactlyOnce(ReadText(ConfigPath("wheel.yaml")),
                         "  wheel_width_m: 0.148\n", ""));
  ExpectRejected(path, "wheel");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsNegativeBodyDimension) {
  const auto path = WriteTemporaryConfig(
      "platform_config_negative_dimension",
      ReplaceExactlyOnce(ReadText(ConfigPath("wheel.yaml")),
                         "body_extent_m: [1.182, 0.818, 1.29996]",
                         "body_extent_m: [-1.182, 0.818, 1.29996]"));
  ExpectRejected(path, "wheel");
  std::filesystem::remove(path);
}

TEST(PlatformConfig, RejectsExactPrimitiveProjectionDrift) {
  const auto path = WriteTemporaryConfig(
      "platform_config_projection_drift",
      ReplaceExactlyOnce(ReadText(ConfigPath("wheel.yaml")),
                         "position_m: [0.19509032201612825, 0.01921471959676957, 0.0]",
                         "position_m: [0.19509032201612825, 0.01921471959676958, 0.0]"));
  ExpectRejected(path, "wheel");
  std::filesystem::remove(path);
}

}  // namespace
