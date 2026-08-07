#include "lunar_planner_ros/route_marker_publisher.hpp"

#include <chrono>
#include <ranges>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <visualization_msgs/msg/marker.hpp>

namespace lunar::planning::ros {
namespace {

using namespace std::chrono_literals;
using Marker = visualization_msgs::msg::Marker;

const Marker* Find(
    const visualization_msgs::msg::MarkerArray& markers,
    const std::string& marker_namespace, const std::int32_t id = 0,
    const std::int32_t action = Marker::ADD) {
  const auto found = std::ranges::find_if(
      markers.markers, [&](const Marker& marker) {
        return marker.ns == marker_namespace && marker.id == id &&
            marker.action == action;
      });
  return found == markers.markers.end() ? nullptr : &*found;
}

RouteMarkerContext Context() {
  return RouteMarkerContext{
      .stamp = builtin_interfaces::msg::Time{}.set__sec(10),
      .route_id = "route/request-1",
  };
}

PlannerInput BaseInput(
    PlatformCapability capability, PlatformState state,
    const Vec3 goal = {10.0, 2.0, 0.0}) {
  return PlannerInput{
      .request_id = "request-1",
      .mission_id = "mission-1",
      .mission_revision = 7U,
      .platform_id = "platform-1",
      .capability_version = "capability-v1",
      .global_map_generation = 31U,
      .local_map_generation = 37U,
      .map_from_odom_generation = 41U,
      .state_time = {10'000'000'000LL},
      .current_state = std::move(state),
      .goal_map = GoalRegion{
          .goal_id = "goal-1",
          .target = PointGoal{.position_m = goal, .tolerance_m = 0.0},
      },
      .world = WorldSnapshot{
          .map_from_odom = RigidTransform{
              .parent_frame = "map",
              .child_frame = "odom",
              .stamp = {10'000'000'000LL},
              .rotation = {},
          },
      },
      .capability = std::move(capability),
  };
}

PlannerOutput GroundOutput(
    const PlatformType platform, const TrajectorySemantics semantics) {
  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "REFERENCE_AVAILABLE",
      .reference = MotionReference{
          .plan_id = "ground/request-1",
          .platform_type = platform,
          .input_time = {10'000'000'000LL},
          .preview = GlobalRoutePreview{
              .poses_map = {
                  Pose3{.position_m = {0.0, 0.0, 0.0}},
                  Pose3{.position_m = {5.0, 0.0, 0.0}},
                  Pose3{.position_m = {10.0, 2.0, 0.0}},
              },
          },
          .data = TrajectoryReference{
              .semantics = semantics,
              .points = {
                  TrajectoryPoint{
                      .time_from_start = 0s,
                      .pose = Pose3{.position_m = {0.0, 0.0, 0.0}},
                  },
                  TrajectoryPoint{
                      .time_from_start = 2s,
                      .pose = Pose3{.position_m = {2.0, 0.5, 0.0}},
                      .velocity = Twist3{.linear_mps = {1.0, 0.0, 0.0}},
                  },
              },
          },
      },
  };
}

WheeledCapability WheelCapability() {
  return WheeledCapability{
      .footprint_xy_m = {
          {-0.591, -0.409}, {0.591, -0.409},
          {0.591, 0.409}, {-0.591, 0.409},
      },
      .body_extent_m = {1.182, 0.818, 1.29996},
      .wheel_diameter_m = 0.319,
      .wheel_width_m = 0.148,
      .wheelbase_m = 0.8175,
      .track_width_m = 0.670,
      .minimum_clearance_m = 0.20,
  };
}

LeggedCapability LegCapability() {
  return LeggedCapability{
      .body_extent_m = {0.68, 0.33, 0.35},
      .minimum_body_clearance_m = 0.30,
  };
}

PlannerOutput HopperOutput() {
  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "HOPPER_SINGLE_HOP_AVAILABLE",
      .reference = MotionReference{
          .plan_id = "hopper/request-1",
          .platform_type = PlatformType::kHopper,
          .input_time = {10'000'000'000LL},
          .preview = GlobalRoutePreview{
              .poses_map = {
                  Pose3{.position_m = {0.0, 0.0, 0.0}},
                  Pose3{.position_m = {4.0, 0.0, 0.76}},
              },
          },
          .data = HopReference{
              .segments = {HopSegment{
                  .segment_id = "hop-1",
                  .launch_pose = {},
                  .landing_region_boundary_m = {
                      {3.8, -0.2, 0.76}, {4.2, -0.2, 0.76},
                      {4.2, 0.2, 0.76}, {3.8, 0.2, 0.76},
                  },
                  .flight_time = 2s,
                  .launch_velocity_mps = {2.0, 0.0, 2.0},
                  .flight_tube_radius_m = 0.75,
                  .nominal_landing_point_m = {4.0, 0.0, 0.76},
                  .ideal_fuel_required_kg = 0.08,
                  .certified_fuel_required_kg = 0.10,
                  .expected_remaining_usable_fuel_kg = 0.10,
                  .required_delta_v_mps = 28.0,
                  .available_delta_v_mps = 29.0,
                  .capability_version = "capability-v1",
                  .global_map_generation = 31U,
                  .local_map_generation = 37U,
              }},
          },
      },
      .certified_hops = {CertifiedHopPreview{
          .segment_id = "hop-1",
          .launch_pose_map = Pose3{.position_m = {0.0, 0.0, 0.0}},
          .landing_pose_map = Pose3{.position_m = {4.0, 0.0, 0.76}},
          .launch_velocity_mps = {2.0, 0.0, 2.0},
          .flight_time = 2s,
          .flight_tube_radius_m = 0.75,
          .landing_region_map = {
              {3.8, -0.2, 0.76}, {4.2, -0.2, 0.76},
              {4.2, 0.2, 0.76}, {3.8, 0.2, 0.76},
          },
      }},
  };
}

TEST(RouteMarkerPublisher, RendersWheeledGeometryRouteMarginAndDirection) {
  RouteMarkerPublisher publisher;
  const PlannerInput input = BaseInput(
      WheelCapability(), WheeledState{.pose = {}});
  const auto markers = publisher.Replace(
      GroundOutput(
          PlatformType::kWheeled, TrajectorySemantics::kWheeledBase),
      input, Context());

  const Marker* body = Find(markers, "wheeled_platform_body");
  ASSERT_NE(body, nullptr);
  EXPECT_DOUBLE_EQ(body->scale.x, 1.182);
  EXPECT_DOUBLE_EQ(body->scale.y, 0.818);
  EXPECT_DOUBLE_EQ(body->scale.z, 1.29996);
  for (std::int32_t id = 0; id < 4; ++id) {
    const Marker* wheel = Find(markers, "wheeled_platform_wheel", id);
    ASSERT_NE(wheel, nullptr);
    EXPECT_DOUBLE_EQ(wheel->scale.x, 0.319);
    EXPECT_DOUBLE_EQ(wheel->scale.y, 0.319);
    EXPECT_DOUBLE_EQ(wheel->scale.z, 0.148);
  }
  ASSERT_NE(Find(markers, "wheeled_raw_footprint"), nullptr);
  const Marker* margin = Find(markers, "wheeled_clearance_margin");
  ASSERT_NE(margin, nullptr);
  ASSERT_GE(margin->points.size(), 5U);
  EXPECT_NEAR(margin->points.front().x, -0.791, 1.0e-12);
  EXPECT_NEAR(margin->points.front().y, -0.609, 1.0e-12);
  EXPECT_NE(Find(markers, "planning_start"), nullptr);
  EXPECT_NE(Find(markers, "planning_goal"), nullptr);
  EXPECT_EQ(Find(markers, "planning_global_route"), nullptr);
  EXPECT_NE(Find(markers, "certified_local_execution"), nullptr);
  EXPECT_NE(Find(markers, "planning_direction"), nullptr);
  const Marker* roughness = Find(markers, "terrain_semantics", 0);
  const Marker* obstacle = Find(markers, "terrain_semantics", 1);
  const Marker* unknown = Find(markers, "terrain_semantics", 2);
  ASSERT_NE(roughness, nullptr);
  ASSERT_NE(obstacle, nullptr);
  ASSERT_NE(unknown, nullptr);
  EXPECT_NE(roughness->color, obstacle->color);
  EXPECT_NE(roughness->color, unknown->color);
  EXPECT_NE(obstacle->color, unknown->color);
}

TEST(RouteMarkerPublisher, RendersLeggedBodyAndGroundRouteEvidence) {
  RouteMarkerPublisher publisher;
  const PlannerInput input = BaseInput(
      LegCapability(), LeggedState{.body_pose = {}});
  const auto markers = publisher.Replace(
      GroundOutput(
          PlatformType::kLegged,
          TrajectorySemantics::kLeggedBodyReference),
      input, Context());

  const Marker* body = Find(markers, "legged_platform_body");
  ASSERT_NE(body, nullptr);
  EXPECT_DOUBLE_EQ(body->scale.x, 0.68);
  EXPECT_DOUBLE_EQ(body->scale.y, 0.33);
  EXPECT_DOUBLE_EQ(body->scale.z, 0.35);
  EXPECT_NE(Find(markers, "planning_start"), nullptr);
  EXPECT_NE(Find(markers, "planning_goal"), nullptr);
  EXPECT_EQ(Find(markers, "planning_global_route"), nullptr);
  EXPECT_NE(Find(markers, "certified_local_execution"), nullptr);
  EXPECT_EQ(Find(markers, "wheeled_platform_body"), nullptr);
}

TEST(RouteMarkerPublisher, RendersHopperTargetRegionTubeAndAuditText) {
  RouteMarkerPublisher publisher;
  HopperCapability capability{
      .specific_impulse_s = 301.0,
      .landing_support_radius_m = 0.45,
      .flight_collision_radius_m = 0.55,
      .landing_lateral_margin_m = 0.20,
  };
  const PlannerInput input = BaseInput(
      capability, HopperState{.pose = {}}, {4.0, 0.0, 0.76});
  const auto markers = publisher.Replace(HopperOutput(), input, Context());

  const Marker* platform = Find(markers, "hopper_platform");
  ASSERT_NE(platform, nullptr);
  EXPECT_DOUBLE_EQ(platform->scale.x, 1.10);
  const Marker* target = Find(markers, "hopper_exact_target");
  ASSERT_NE(target, nullptr);
  EXPECT_DOUBLE_EQ(target->pose.position.x, 4.0);
  const Marker* disk = Find(markers, "hopper_landing_support_disk");
  ASSERT_NE(disk, nullptr);
  EXPECT_DOUBLE_EQ(disk->scale.x, 1.30);
  EXPECT_DOUBLE_EQ(disk->scale.y, 1.30);
  const Marker* filled = Find(markers, "hopper_landing_region_filled");
  ASSERT_NE(filled, nullptr);
  EXPECT_EQ(filled->type, Marker::TRIANGLE_LIST);
  EXPECT_GE(filled->points.size(), 6U);
  const Marker* nominal = Find(markers, "hopper_nominal_landing_point");
  ASSERT_NE(nominal, nullptr);
  EXPECT_DOUBLE_EQ(nominal->pose.position.x, 4.0);
  EXPECT_NE(Find(markers, "hopper_nominal_arc"), nullptr);
  const Marker* tube = Find(markers, "hopper_flight_tube");
  ASSERT_NE(tube, nullptr);
  EXPECT_DOUBLE_EQ(tube->scale.x, 1.50);
  const Marker* evidence = Find(markers, "hopper_certification_evidence");
  ASSERT_NE(evidence, nullptr);
  EXPECT_NE(evidence->text.find("fuel=0.100000 kg"), std::string::npos);
  EXPECT_NE(evidence->text.find("dv=28.000000/29.000000 m/s"),
            std::string::npos);
  EXPECT_NE(evidence->text.find("version=capability-v1"),
            std::string::npos);
  EXPECT_EQ(Find(markers, "certified_hop_promotion_region"), nullptr);
}

TEST(RouteMarkerPublisher, ReplacesPlatformMarkersAndDeletesOwnedIds) {
  RouteMarkerPublisher publisher;
  const PlannerInput wheel_input = BaseInput(
      WheelCapability(), WheeledState{.pose = {}});
  const auto wheel = publisher.Replace(
      GroundOutput(
          PlatformType::kWheeled, TrajectorySemantics::kWheeledBase),
      wheel_input, Context());
  const std::size_t wheel_owned = std::ranges::count_if(
      wheel.markers, [](const Marker& marker) {
        return marker.action == Marker::ADD;
      });

  const PlannerInput legged_input = BaseInput(
      LegCapability(), LeggedState{.body_pose = {}});
  const auto switched = publisher.Replace(
      GroundOutput(
          PlatformType::kLegged,
          TrajectorySemantics::kLeggedBodyReference),
      legged_input, Context());
  EXPECT_EQ(
      std::ranges::count_if(switched.markers, [](const Marker& marker) {
        return marker.action == Marker::DELETE;
      }),
      wheel_owned);
  EXPECT_NE(Find(switched, "wheeled_platform_body", 0, Marker::DELETE),
            nullptr);
  EXPECT_NE(Find(switched, "legged_platform_body"), nullptr);

  const auto deleted = publisher.DeleteOwned(
      builtin_interfaces::msg::Time{}.set__sec(11));
  EXPECT_TRUE(std::ranges::all_of(deleted.markers, [](const Marker& marker) {
    return marker.action == Marker::DELETE &&
        marker.action != Marker::DELETEALL;
  }));
  EXPECT_TRUE(publisher.DeleteOwned(
      builtin_interfaces::msg::Time{}.set__sec(12)).markers.empty());
}

}  // namespace
}  // namespace lunar::planning::ros
