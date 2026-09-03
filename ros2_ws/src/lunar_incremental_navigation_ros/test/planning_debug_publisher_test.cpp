#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "lunar_incremental_navigation_ros/planning_debug_publisher.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

namespace core = lunar::incremental_navigation;

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

const auto* const kRosEnvironment =
    ::testing::AddGlobalTestEnvironment(new RosEnvironment{});

[[nodiscard]] PlanningDebugPublisherConfig EnabledConfig() {
  return {.enabled = true,
          .topic_prefix = "/planning_demo/debug",
          .fine_window_m = 40.0,
          .cost_display_max = 2.0};
}

TEST(PlanningDebugPublisher, DisabledCreatesNoDebugPublishers) {
  rclcpp::Node node("debug_publisher_disabled");
  PlanningDebugPublisher publisher(node, {.enabled = false});

  EXPECT_FALSE(publisher.enabled());
  EXPECT_TRUE(
      node.get_publishers_info_by_topic("/planning_demo/debug/fine_state")
          .empty());
  EXPECT_TRUE(
      node.get_publishers_info_by_topic("/planning_demo/debug/fine_cost")
          .empty());
  EXPECT_TRUE(
      node.get_publishers_info_by_topic("/planning_demo/debug/execution_risk")
          .empty());
  EXPECT_TRUE(
      node.get_publishers_info_by_topic("/planning_demo/debug/traversability")
          .empty());
  EXPECT_TRUE(node.get_publishers_info_by_topic(
                        "/planning_demo/debug/guidance_state")
                  .empty());
  EXPECT_TRUE(node.get_publishers_info_by_topic(
                        "/planning_demo/debug/start_patch_cells")
                  .empty());
}

TEST(PlanningDebugPublisher, EnabledUsesReliableTransientLocalQos) {
  rclcpp::Node node("debug_publisher_enabled");
  PlanningDebugPublisher publisher(node, EnabledConfig());

  ASSERT_TRUE(publisher.enabled());
  const auto profile = DebugVisualizationQos().get_rmw_qos_profile();
  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.depth, 1U);
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  EXPECT_FALSE(node.get_publishers_info_by_topic(
                   "/planning_demo/debug/execution_risk")
                   .empty());
  EXPECT_FALSE(node.get_publishers_info_by_topic(
                   "/planning_demo/debug/traversability")
                   .empty());
}

TEST(PlanningDebugPublisher, DeduplicatesFineByRevisionAndCenter) {
  const core::Point2 center{.x = 1.0, .y = 2.0};

  EXPECT_TRUE(ShouldPublishDebugFineSnapshot(std::nullopt, std::nullopt, 7U,
                                              center));
  EXPECT_FALSE(ShouldPublishDebugFineSnapshot(7U, center, 7U, center));
  EXPECT_TRUE(ShouldPublishDebugFineSnapshot(7U, center, 7U,
                                              {.x = 2.0, .y = 2.0}));
  EXPECT_TRUE(ShouldPublishDebugFineSnapshot(7U, center, 8U, center));
}

TEST(PlanningDebugPublisher, FirstZeroRevisionIsNotDeduplicated) {
  const core::Point2 center{.x = 1.0, .y = 2.0};

  EXPECT_TRUE(ShouldPublishDebugFineSnapshot(std::nullopt, std::nullopt, 0U,
                                              center));
  EXPECT_FALSE(ShouldPublishDebugFineSnapshot(0U, center, 0U, center));
  EXPECT_TRUE(ShouldPublishDebugRevision(std::nullopt, 0U));
  EXPECT_FALSE(ShouldPublishDebugRevision(0U, 0U));
}

TEST(PlanningDebugPublisher, NormalizesRelativePlanningDemoPrefix) {
  EXPECT_EQ(NormalizeDebugTopicPrefix("planning_demo/debug"),
            "/planning_demo/debug");
  EXPECT_EQ(NormalizeDebugTopicPrefix("/planning_demo/debug/"),
            "/planning_demo/debug");
}

TEST(PlanningDebugPublisher, RejectsInvalidDebugTopicPrefixes) {
  for (const std::string& prefix : {"", "/", "/planning_demo", "/other"}) {
    EXPECT_THROW(static_cast<void>(NormalizeDebugTopicPrefix(prefix)),
                 std::invalid_argument)
        << prefix;
  }
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
