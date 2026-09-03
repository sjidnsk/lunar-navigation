#include "lunar_pure_exploration_ros/platform_config_loader.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace lunar::pure_exploration_ros {
namespace {

std::filesystem::path PlannerConfigPath(std::string_view file_name) {
  return std::filesystem::path{
             ament_index_cpp::get_package_share_directory(
                 "lunar_pure_planner_ros")} /
         "config" / file_name;
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream stream{path};
  if (!stream) {
    throw std::runtime_error{"failed to read test fixture"};
  }
  std::ostringstream text;
  text << stream.rdbuf();
  return text.str();
}

std::string ReplaceOnce(std::string text, std::string_view old_value,
                        std::string_view new_value) {
  const std::size_t position = text.find(old_value);
  if (position == std::string::npos ||
      text.find(old_value, position + old_value.size()) != std::string::npos) {
    throw std::logic_error{"test replacement must match exactly once"};
  }
  text.replace(position, old_value.size(), new_value);
  return text;
}

class TemporaryYaml final {
 public:
  explicit TemporaryYaml(std::string contents) {
    static std::atomic<unsigned long> sequence{0UL};
    path_ = std::filesystem::temp_directory_path() /
            ("lunar-pure-exploration-platform-" +
             std::to_string(sequence.fetch_add(1UL)) + ".yaml");
    std::ofstream stream{path_};
    if (!stream) {
      throw std::runtime_error{"failed to create test fixture"};
    }
    stream << contents;
    if (!stream) {
      throw std::runtime_error{"failed to write test fixture"};
    }
  }

  TemporaryYaml(const TemporaryYaml&) = delete;
  TemporaryYaml& operator=(const TemporaryYaml&) = delete;

  ~TemporaryYaml() { std::filesystem::remove(path_); }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::string WheelYaml() { return ReadText(PlannerConfigPath("wheel.yaml")); }

std::string LeggedYaml() { return ReadText(PlannerConfigPath("legged.yaml")); }

TEST(PlatformConfigLoader, LoadsPlannerInstalledWheelGeometry) {
  const LoadedPlatformConfig config =
      LoadPlatformConfig(PlannerConfigPath("wheel.yaml"), "wheel");

  EXPECT_EQ(config.geometry.platform_id, "wheeled-lunar-explorer");
  EXPECT_EQ(config.geometry.platform_type, "WHEELED");
  EXPECT_EQ(config.geometry.base_frame_id, "base_footprint");
  ASSERT_EQ(config.geometry.footprint_vertices.size(), 4U);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[0].x, 0.591);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[0].y, 0.409);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[1].x, 0.591);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[1].y, -0.409);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[2].x, -0.591);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[2].y, -0.409);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[3].x, -0.591);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[3].y, 0.409);
  EXPECT_DOUBLE_EQ(config.geometry.minimum_clearance_m, 0.2);
}

TEST(PlatformConfigLoader, LoadsPlannerInstalledLeggedGeometry) {
  const LoadedPlatformConfig config =
      LoadIncrementalPlatformConfig(PlannerConfigPath("legged.yaml"),
                                    "legged");

  EXPECT_EQ(config.geometry.platform_id, "yobotics-quad48");
  EXPECT_EQ(config.geometry.platform_type, "LEGGED");
  EXPECT_EQ(config.geometry.base_frame_id, "base_link");
  ASSERT_EQ(config.geometry.footprint_vertices.size(), 4U);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[0].x, 0.34);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[0].y, 0.165);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[1].x, 0.34);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[1].y, -0.165);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[2].x, -0.34);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[2].y, -0.165);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[3].x, -0.34);
  EXPECT_DOUBLE_EQ(config.geometry.footprint_vertices[3].y, 0.165);
  EXPECT_DOUBLE_EQ(config.geometry.minimum_clearance_m, 0.3);
}

TEST(PlatformConfigLoader, RejectsMalformedLeggedGeometry) {
  for (const auto& replacement : {
           std::pair{"body_extent_m: [0.68, 0.33, 0.35]",
                     "body_extent_m: [0.68, 0.33]"},
           std::pair{"body_extent_m: [0.68, 0.33, 0.35]",
                     "body_extent_m: [0.68, .nan, 0.35]"},
           std::pair{"body_extent_m: [0.68, 0.33, 0.35]",
                     "body_extent_m: [-0.68, 0.33, 0.35]"},
           std::pair{"minimum_body_clearance_m: 0.3",
                     "minimum_body_clearance_m: -0.01"},
           std::pair{"minimum_body_clearance_m: 0.3",
                     "minimum_body_clearance_m: .inf"},
           std::pair{"capability:\n",
                     "capability:\n  unexpected_capability: true\n"}}) {
    const TemporaryYaml yaml{
        ReplaceOnce(LeggedYaml(), replacement.first, replacement.second)};
    EXPECT_THROW(LoadIncrementalPlatformConfig(yaml.path(), "legged"),
                 std::invalid_argument);
  }
}

TEST(PlatformConfigLoader, PreservesLegacyWheelOnlySelectorContract) {
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("wheel.yaml"), ""),
               std::invalid_argument);
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("wheel.yaml"), "Wheel"),
               std::invalid_argument);
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("wheel.yaml"), "WHEELED"),
               std::invalid_argument);
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("wheel.yaml"), "legged"),
               std::invalid_argument);
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("legged.yaml"), "wheel"),
               std::invalid_argument);
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("legged.yaml"), "legged"),
               std::invalid_argument);
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("wheel.yaml"), "hopper"),
               std::invalid_argument);
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("wheel.yaml"), "unknown"),
               std::invalid_argument);
  EXPECT_THROW(LoadPlatformConfig(PlannerConfigPath("hopper.yaml"), "hopper"),
               std::invalid_argument);
}

TEST(PlatformConfigLoader,
     IncrementalApiLoadsWheelAndRejectsUnsupportedSelectorsAndMismatches) {
  EXPECT_EQ(LoadIncrementalPlatformConfig(PlannerConfigPath("wheel.yaml"),
                                           "wheel")
                .geometry.platform_type,
            "WHEELED");
  EXPECT_THROW(LoadIncrementalPlatformConfig(PlannerConfigPath("wheel.yaml"),
                                              "legged"),
               std::invalid_argument);
  EXPECT_THROW(LoadIncrementalPlatformConfig(PlannerConfigPath("legged.yaml"),
                                              "wheel"),
               std::invalid_argument);
  EXPECT_THROW(LoadIncrementalPlatformConfig(PlannerConfigPath("hopper.yaml"),
                                              "hopper"),
               std::invalid_argument);
  EXPECT_THROW(LoadIncrementalPlatformConfig(PlannerConfigPath("wheel.yaml"),
                                              "unknown"),
               std::invalid_argument);
}

TEST(PlatformConfigLoader, RejectsSelectorPayloadAndFrozenIdentityMismatch) {
  for (const auto& replacement : {
           std::pair{"platform: wheel", "platform: legged"},
           std::pair{"platform_type: WHEELED", "platform_type: wheel"},
           std::pair{"base_frame_id: base_footprint",
                     "base_frame_id: base_link"}}) {
    const TemporaryYaml yaml{
        ReplaceOnce(WheelYaml(), replacement.first, replacement.second)};
    EXPECT_THROW(LoadPlatformConfig(yaml.path(), "wheel"),
                 std::invalid_argument);
  }
}

TEST(PlatformConfigLoader, RejectsEmptyOrDuplicatePlatformId) {
  const TemporaryYaml empty{
      ReplaceOnce(WheelYaml(), "platform_id: wheeled-lunar-explorer",
                  "platform_id: ''")};
  EXPECT_THROW(LoadPlatformConfig(empty.path(), "wheel"),
               std::invalid_argument);

  const TemporaryYaml duplicate{ReplaceOnce(
      WheelYaml(), "platform_id: wheeled-lunar-explorer",
      "platform_id: wheeled-lunar-explorer\nplatform_id: second-id")};
  EXPECT_THROW(LoadPlatformConfig(duplicate.path(), "wheel"),
               std::invalid_argument);
}

TEST(PlatformConfigLoader, AcceptsAnyFinitePolygonWithAtLeastThreeVertices) {
  std::string contents = ReplaceOnce(
      WheelYaml(), "platform_id: wheeled-lunar-explorer",
      "platform_id: alternate-wheel-id");
  contents = ReplaceOnce(
      std::move(contents),
      "footprint_xy_m: [[0.591, 0.409], [0.591, -0.409], [-0.591, -0.409], [-0.591, 0.409]]",
      "footprint_xy_m: [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]");
  contents = ReplaceOnce(std::move(contents), "minimum_clearance_m: 0.2",
                         "minimum_clearance_m: 0.0");
  const TemporaryYaml yaml{std::move(contents)};

  const LoadedPlatformConfig config = LoadPlatformConfig(yaml.path(), "wheel");
  EXPECT_EQ(config.geometry.platform_id, "alternate-wheel-id");
  ASSERT_EQ(config.geometry.footprint_vertices.size(), 3U);
  EXPECT_DOUBLE_EQ(config.geometry.minimum_clearance_m, 0.0);
}

TEST(PlatformConfigLoader, RejectsMalformedFootprintAndClearance) {
  for (const auto& replacement : {
           std::pair{"footprint_xy_m: [[0.591, 0.409], [0.591, -0.409], [-0.591, -0.409], [-0.591, 0.409]]",
                     "footprint_xy_m: [[0.0, 0.0], [1.0, 0.0]]"},
           std::pair{"footprint_xy_m: [[0.591, 0.409], [0.591, -0.409], [-0.591, -0.409], [-0.591, 0.409]]",
                     "footprint_xy_m: [[.nan, 0.0], [1.0, 0.0], [0.0, 1.0]]"},
           std::pair{"footprint_xy_m: [[0.591, 0.409], [0.591, -0.409], [-0.591, -0.409], [-0.591, 0.409]]",
                     "footprint_xy_m: [[.inf, 0.0], [1.0, 0.0], [0.0, 1.0]]"},
           std::pair{"footprint_xy_m: [[0.591, 0.409], [0.591, -0.409], [-0.591, -0.409], [-0.591, 0.409]]",
                     "footprint_xy_m: [[-.inf, 0.0], [1.0, 0.0], [0.0, 1.0]]"},
           std::pair{"minimum_clearance_m: 0.2",
                     "minimum_clearance_m: -0.01"},
           std::pair{"minimum_clearance_m: 0.2",
                     "minimum_clearance_m: .nan"},
           std::pair{"minimum_clearance_m: 0.2",
                     "minimum_clearance_m: .inf"},
           std::pair{"minimum_clearance_m: 0.2",
                     "minimum_clearance_m: -.inf"}}) {
    const TemporaryYaml yaml{
        ReplaceOnce(WheelYaml(), replacement.first, replacement.second)};
    EXPECT_THROW(LoadPlatformConfig(yaml.path(), "wheel"),
                 std::invalid_argument);
  }
}

TEST(PlatformConfigLoader, RejectsUnknownRootOrCapabilityKeys) {
  const TemporaryYaml root_unknown{
      ReplaceOnce(WheelYaml(), "platform: wheel",
                  "platform: wheel\nunexpected_root: true")};
  EXPECT_THROW(LoadPlatformConfig(root_unknown.path(), "wheel"),
               std::invalid_argument);

  const TemporaryYaml capability_unknown{
      ReplaceOnce(WheelYaml(), "capability:\n",
                  "capability:\n  unexpected_capability: true\n")};
  EXPECT_THROW(LoadPlatformConfig(capability_unknown.path(), "wheel"),
               std::invalid_argument);
}

TEST(PlatformConfigLoader, ClassifiesMalformedYamlAndIoFailures) {
  const TemporaryYaml malformed{"platform: [unterminated"};
  EXPECT_THROW(LoadPlatformConfig(malformed.path(), "wheel"),
               std::invalid_argument);

  EXPECT_THROW(
      LoadPlatformConfig(
          std::filesystem::temp_directory_path() /
              "lunar-pure-exploration-definitely-missing.yaml",
          "wheel"),
      std::runtime_error);
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
