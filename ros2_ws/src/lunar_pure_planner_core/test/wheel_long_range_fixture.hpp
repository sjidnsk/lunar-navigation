#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "hierarchical/global_route_planner.hpp"
#include "lunar_pure_planner_core/planner.hpp"
#include "shared/local_terrain_projection.hpp"
#include "shared/map_snapshot.hpp"
#include "wheel/anytime_wheel_planner.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::pure_planning::wheel {

class WheelLongRangeFixture final {
 public:
  struct RunRecord final {
    bool success{};
    std::int64_t failed_segment{-1};
    double global_elapsed_ms{};
    double local_elapsed_ms{};
    double elapsed_ms{};
    std::uint64_t expanded_states{};
    std::size_t generated_states{};
    std::size_t edge_evaluations{};
    std::size_t state_labels{};
    std::size_t sweep_cell_checks{};
    std::size_t rolling_segments{};
    Pose3 final_pose{};
  };

  [[nodiscard]] RunRecord Run750MeterRollingScenario() const {
    const auto end_to_end_started = SteadyClock::now();
    RunRecord record;
    const auto finish = [&](RunRecord completed) {
      completed.elapsed_ms = std::chrono::duration<double, std::milli>(
                              SteadyClock::now() - end_to_end_started)
                              .count();
      std::cout << "[planner-metrics] {\"scenario\":\"750m\","
                << "\"success\":" << (completed.success ? "true" : "false") << ','
                << "\"failed_segment\":" << completed.failed_segment << ','
                << "\"elapsed_ms\":" << completed.elapsed_ms << ','
                << "\"expanded_states\":" << completed.expanded_states << ','
                << "\"edge_evaluations\":" << completed.edge_evaluations << ','
                << "\"state_labels\":" << completed.state_labels << ','
                << "\"sweep_cell_checks\":" << completed.sweep_cell_checks << "}\n";
      return completed;
    };

    constexpr std::size_t kGlobalWidth = 1000U;
    constexpr std::size_t kGlobalHeight = 1000U;
    constexpr double kGlobalResolutionM = 1.0;
    constexpr std::size_t kLocalWidth = 320U;
    constexpr std::size_t kLocalHeight = 320U;
    constexpr double kLocalResolutionM = 0.2;
    std::vector<std::int8_t> obstacle_world(kGlobalWidth * kGlobalHeight, 0);
    std::minstd_rand generator{0x5EED1234U};
    std::uniform_int_distribution<std::size_t> x_distribution{80U, 770U};
    std::uniform_int_distribution<std::size_t> y_distribution{40U, 960U};
    for (std::size_t index = 0U; index < 1800U; ++index) {
      const std::size_t x = x_distribution(generator);
      const std::size_t y = y_distribution(generator);
      if (std::hypot(static_cast<double>(x) - 50.0,
                     static_cast<double>(y) - 500.0) > 8.0 &&
          std::hypot(static_cast<double>(x) - 800.0,
                     static_cast<double>(y) - 500.0) > 8.0) {
        obstacle_world[y * kGlobalWidth + x] = 100;
      }
    }
    for (const auto [x, gap_y] : {std::pair{180U, 350U},
                                  std::pair{360U, 650U},
                                  std::pair{540U, 350U},
                                  std::pair{700U, 650U}}) {
      for (std::size_t y = 80U; y < 920U; ++y) {
        if (y < gap_y || y > gap_y + 80U) {
          obstacle_world[y * kGlobalWidth + x] = 100;
        }
      }
    }
    const GridMap global_map{
        .frame_id = "map",
        .width = kGlobalWidth,
        .height = kGlobalHeight,
        .resolution_m = kGlobalResolutionM,
        .layers = {
            {"occupancy", GridLayer{.values = obstacle_world}},
        },
    };
    const WheeledCapability capability = MakeProjectWheelCapability();
    Pose3 current = MakePose(50.0, 500.0);
    const GoalRegion final_goal{
        .goal_id = "750m-goal",
        .target = PointGoal{.position_m = {.x = 800.0, .y = 500.0, .z = 0.0},
                            .tolerance_m = 0.2},
        .yaw_rad = 0.0,
        .yaw_tolerance_rad = 0.1,
    };

    const auto make_local_map = [&](const Pose3& center) {
      std::vector<float> occupancy(kLocalWidth * kLocalHeight, 0.0F);
      for (std::size_t y = 0U; y < kLocalHeight; ++y) {
        for (std::size_t x = 0U; x < kLocalWidth; ++x) {
          const double world_x = center.position_m.x - 32.0 +
                                 (static_cast<double>(x) + 0.5) *
                                     kLocalResolutionM;
          const double world_y = center.position_m.y - 32.0 +
                                 (static_cast<double>(y) + 0.5) *
                                     kLocalResolutionM;
          const auto global_x = static_cast<std::int64_t>(std::floor(world_x));
          const auto global_y = static_cast<std::int64_t>(std::floor(world_y));
          if (global_x >= 0 && global_y >= 0 &&
              global_x < static_cast<std::int64_t>(kGlobalWidth) &&
              global_y < static_cast<std::int64_t>(kGlobalHeight) &&
              obstacle_world[static_cast<std::size_t>(global_y) * kGlobalWidth +
                             static_cast<std::size_t>(global_x)] >= 50) {
            occupancy[y * kLocalWidth + x] = 1.0F;
          }
        }
      }
      return GridMap{
          .frame_id = "odom",
          .width = kLocalWidth,
          .height = kLocalHeight,
          .resolution_m = kLocalResolutionM,
          .origin_m = {.x = center.position_m.x - 32.0,
                       .y = center.position_m.y - 32.0},
          .layers = {
              {"occupancy", GridLayer{.values = std::move(occupancy)}},
              {"elevation", GridLayer{.values = std::vector<float>(
                                kLocalWidth * kLocalHeight, 0.0F)}},
          },
      };
    };

    PlanningRequest global_request{
        .request_id = "750m-global",
        .environment_mode = EnvironmentMode::kLunarSurface,
        .current_state = WheeledState{.pose = current},
        .goal_map = final_goal,
        .world = {.global_map = global_map,
                  .local_map = make_local_map(current),
                  .map_from_odom = {.parent_frame = "map", .child_frame = "odom"}},
        .capability = capability,
        .config = {.global_budget = std::chrono::seconds{20},
                   .search = {.stop_after_first_solution = true}},
        .control = {.deadline = SteadyClock::now() + std::chrono::seconds{20}},
    };
    const auto global_started = SteadyClock::now();
    const GlobalStageResult global = hierarchical::PlanSurfaceGlobal(
        global_request, {.deadline = SteadyClock::now() + std::chrono::seconds{20}});
    record.global_elapsed_ms =
        std::chrono::duration<double, std::milli>(SteadyClock::now() - global_started)
            .count();
    if (!global.route.has_value()) {
      return finish(record);
    }
    record.expanded_states += global.route->expanded_states;

    const auto route_point_at_horizon = [&](const Pose3& pose)
        -> std::optional<Vec3> {
      const auto& route = global.route->poses_map;
      if (route.size() < 2U) {
        return std::nullopt;
      }
      std::size_t nearest_segment{};
      double nearest_ratio{};
      double nearest_distance = std::numeric_limits<double>::infinity();
      for (std::size_t index = 1U; index < route.size(); ++index) {
        const Vec3 from = route[index - 1U].position_m;
        const Vec3 to = route[index].position_m;
        const double dx = to.x - from.x;
        const double dy = to.y - from.y;
        const double squared_length = dx * dx + dy * dy;
        const double ratio = squared_length > 0.0
                                 ? std::clamp(
                                       ((pose.position_m.x - from.x) * dx +
                                        (pose.position_m.y - from.y) * dy) /
                                           squared_length,
                                       0.0, 1.0)
                                 : 0.0;
        const double projected_x = from.x + ratio * dx;
        const double projected_y = from.y + ratio * dy;
        const double distance = std::hypot(pose.position_m.x - projected_x,
                                           pose.position_m.y - projected_y);
        if (distance < nearest_distance) {
          nearest_distance = distance;
          nearest_segment = index - 1U;
          nearest_ratio = ratio;
        }
      }
      constexpr double kRollingHorizonM = 8.0;
      double remaining = kRollingHorizonM;
      Vec3 from = route[nearest_segment].position_m;
      Vec3 to = route[nearest_segment + 1U].position_m;
      from.x += nearest_ratio * (to.x - from.x);
      from.y += nearest_ratio * (to.y - from.y);
      for (std::size_t index = nearest_segment + 1U;; ++index) {
        const double distance = std::hypot(to.x - from.x, to.y - from.y);
        if (distance >= remaining) {
          return Vec3{.x = from.x + remaining * (to.x - from.x) / distance,
                      .y = from.y + remaining * (to.y - from.y) / distance};
        }
        remaining -= distance;
        if (index + 1U == route.size()) {
          return to;
        }
        from = to;
        to = route[index + 1U].position_m;
      }
    };

    while (std::hypot(current.position_m.x - 800.0,
                      current.position_m.y - 500.0) > 0.2) {
      if (record.rolling_segments >= 200U) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      const auto local_started = SteadyClock::now();
      PlanningRequest local_request = global_request;
      local_request.request_id = "750m-local-" +
                                 std::to_string(record.rolling_segments);
      local_request.current_state = WheeledState{.pose = current};
      local_request.world.local_map = make_local_map(current);
      const auto target = route_point_at_horizon(current);
      if (!target.has_value()) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      const GoalRegion bounded_local_goal{
          .goal_id = "750m-local-goal",
          .target = PointGoal{.position_m = *target, .tolerance_m = 0.2},
      };
      const auto snapshot = shared::MapSnapshot::Create(
          local_request.world.local_map, shared::MapContract::kLocalElevation);
      if (!snapshot.ok()) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      const auto terrain = shared::BuildLocalTerrainProjection(
          snapshot.snapshot,
          static_cast<float>(local_request.config.local_occupancy_threshold));
      if (!terrain.ok()) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      const WheelPlanResult local = PlanWheel({
          .start = WheeledState{.pose = current},
          .goal_odom = bounded_local_goal,
          .terrain = &*terrain.value,
          .capability = &capability,
          .control = {.deadline = SteadyClock::now() + std::chrono::seconds{1}},
      });
      record.local_elapsed_ms +=
          std::chrono::duration<double, std::milli>(SteadyClock::now() - local_started)
              .count();
      record.expanded_states += local.metrics.expanded_states;
      record.generated_states += local.quantized_state_count;
      record.edge_evaluations += local.metrics.edge_validation_evaluations;
      record.state_labels += local.quantized_state_count;
      record.sweep_cell_checks += local.sweep_cell_checks;
      if (!local.ok()) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      current = local.trajectory.back().pose;
      ++record.rolling_segments;
    }
    record.success = true;
    record.final_pose = current;
    return finish(record);
  }

 private:
  [[nodiscard]] static Pose3 MakePose(const double x, const double y) {
    return Pose3{
        .position_m = Vec3{.x = x, .y = y, .z = 0.0},
        .orientation = QuaternionFromYaw(0.0),
    };
  }

  [[nodiscard]] static WheelMotionPrimitive MakePrimitive(
      const char* id, const WheelPrimitiveKind kind, const double x,
      const double y = 0.0, const double yaw = 0.0) {
    return WheelMotionPrimitive{
        .primitive_id = id,
        .kind = kind,
        .relative_end_pose = Pose3{
            .position_m = Vec3{.x = x, .y = y, .z = 0.0},
            .orientation = QuaternionFromYaw(yaw),
        },
    };
  }

  [[nodiscard]] static WheeledCapability MakeProjectWheelCapability() {
    return WheeledCapability{
        .footprint_xy_m = {
            Vec2{.x = 0.591, .y = 0.409},
            Vec2{.x = 0.591, .y = -0.409},
            Vec2{.x = -0.591, .y = -0.409},
            Vec2{.x = -0.591, .y = 0.409},
        },
        .body_extent_m = Vec3{.x = 1.182, .y = 0.818, .z = 1.29996},
        .wheel_diameter_m = 0.319,
        .wheel_width_m = 0.148,
        .wheelbase_m = 0.8175,
        .track_width_m = 0.67,
        .minimum_underbody_clearance_m = 0.21,
        .maximum_local_obstacle_relief_m = 0.2,
        .allow_unsupported_gap = false,
        .minimum_body_z_m = 0.0,
        .maximum_body_z_m = 1.29996,
        .maximum_forward_speed_mps = 1.5,
        .maximum_reverse_speed_mps = 1.5,
        .maximum_spin_rate_radps = 1.0,
        .maximum_acceleration_mps2 = 0.5,
        .maximum_braking_deceleration_mps2 = 0.5,
        .maximum_yaw_acceleration_radps2 = 0.5,
        .maximum_lateral_acceleration_mps2 = 0.5,
        .maximum_curvature_per_m = 1.0,
        .maximum_slope_rad = 0.3490658503988659,
        .minimum_clearance_m = 0.2,
        .motion_primitives = {
            MakePrimitive("forward", WheelPrimitiveKind::kForward, 0.2),
            MakePrimitive("forward-arc-left", WheelPrimitiveKind::kForwardArc,
                          0.19509032201612825, 0.01921471959676957,
                          std::numbers::pi / 16.0),
            MakePrimitive("forward-arc-right", WheelPrimitiveKind::kForwardArc,
                          0.19509032201612825, -0.01921471959676957,
                          -std::numbers::pi / 16.0),
            MakePrimitive("reverse", WheelPrimitiveKind::kReverse, -0.2),
            MakePrimitive("reverse-arc-left", WheelPrimitiveKind::kReverseArc,
                          -0.19509032201612825, 0.01921471959676957,
                          -std::numbers::pi / 16.0),
            MakePrimitive("reverse-arc-right", WheelPrimitiveKind::kReverseArc,
                          -0.19509032201612825, -0.01921471959676957,
                          std::numbers::pi / 16.0),
            MakePrimitive("spin-left", WheelPrimitiveKind::kSpinCounterclockwise,
                          0.0, 0.0, std::numbers::pi / 16.0),
            MakePrimitive("spin-right", WheelPrimitiveKind::kSpinClockwise, 0.0,
                          0.0, -std::numbers::pi / 16.0),
            MakePrimitive("stop-switch", WheelPrimitiveKind::kStopAndSwitch, 0.0),
        },
    };
  }
};

}  // namespace lunar::pure_planning::wheel
