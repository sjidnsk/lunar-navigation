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
#include "hierarchical/surface_portal_set.hpp"
#include "hierarchical/surface_rolling_session.hpp"
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
    double max_cycle_elapsed_ms{};
    double elapsed_ms{};
    std::uint64_t expanded_states{};
    std::size_t generated_states{};
    std::size_t edge_evaluations{};
    std::size_t state_labels{};
    std::size_t sweep_cell_checks{};
    std::size_t rolling_segments{};
    std::size_t cycles_under_one_second{};
    std::size_t cycles_one_to_two_seconds{};
    std::size_t cycles_two_to_three_seconds{};
    std::size_t cycles_at_least_three_seconds{};
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
                << "\"max_cycle_elapsed_ms\":"
                << completed.max_cycle_elapsed_ms << ','
                << "\"cycles_under_one_second\":"
                << completed.cycles_under_one_second << ','
                << "\"cycles_one_to_two_seconds\":"
                << completed.cycles_one_to_two_seconds << ','
                << "\"cycles_two_to_three_seconds\":"
                << completed.cycles_two_to_three_seconds << ','
                << "\"cycles_at_least_three_seconds\":"
                << completed.cycles_at_least_three_seconds << ','
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
      const double origin_x =
          std::floor((center.position_m.x - 32.0) / kLocalResolutionM) *
          kLocalResolutionM;
      const double origin_y =
          std::floor((center.position_m.y - 32.0) / kLocalResolutionM) *
          kLocalResolutionM;
      std::vector<float> occupancy(kLocalWidth * kLocalHeight, 0.0F);
      for (std::size_t y = 0U; y < kLocalHeight; ++y) {
        for (std::size_t x = 0U; x < kLocalWidth; ++x) {
          const double world_x = origin_x +
                                 (static_cast<double>(x) + 0.5) *
                                     kLocalResolutionM;
          const double world_y = origin_y +
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
          .origin_m = {.x = origin_x, .y = origin_y},
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
        .config = {.search = {.stop_after_first_solution = true}},
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
    const hierarchical::SurfaceRollingSession rolling_session(
        *global.route, final_goal,
        hierarchical::SurfaceRollingConfig{.horizon_m = 8.0,
                                            .max_deviation_m = 2.0});
    double confirmed_route_progress_m{};

    while (std::hypot(current.position_m.x - 800.0,
                      current.position_m.y - 500.0) > 0.2) {
      if (record.rolling_segments >= 320U) {
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
      const auto cycle_deadline = local_started + std::chrono::seconds{3};
      const SearchControl cycle_control{.deadline = cycle_deadline};
      hierarchical::SurfaceRollingDecision decision =
          rolling_session.Decide(current, confirmed_route_progress_m);
      const hierarchical::SurfacePortalSetResult portals =
          hierarchical::BuildSurfacePortalSet(
              local_request, *global.route, decision, 32U, cycle_control);
      if (!portals.ok()) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      const hierarchical::LocalGoalSetResult converted =
          hierarchical::ConvertSurfacePortalsToLocalGoals(
              portals, decision, current, *local_request.world.global_map,
              local_request.world.local_map);
      if (!converted.ok()) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      LocalGoalSet local_goals = *converted.goals;
      const auto snapshot = shared::MapSnapshot::Create(
          local_request.world.local_map, shared::MapContract::kLocalElevation,
          cycle_control);
      if (!snapshot.ok()) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      const auto terrain = shared::BuildLocalTerrainProjection(
          snapshot.snapshot,
          static_cast<float>(local_request.config.local_occupancy_threshold),
          cycle_control);
      if (!terrain.ok()) {
        record.failed_segment = static_cast<std::int64_t>(record.rolling_segments);
        record.final_pose = current;
        return finish(record);
      }
      const WheelPlanResult local = PlanWheel({
          .start = WheeledState{.pose = current},
          .goals_odom = local_goals,
          .terrain = &*terrain.value,
          .capability = &capability,
          .control = cycle_control,
          .search = {.stop_after_first_solution = true},
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
      const double cycle_elapsed_ms =
          std::chrono::duration<double, std::milli>(SteadyClock::now() -
                                                    local_started)
              .count();
      record.max_cycle_elapsed_ms =
          std::max(record.max_cycle_elapsed_ms, cycle_elapsed_ms);
      if (cycle_elapsed_ms < 1000.0) {
        ++record.cycles_under_one_second;
      } else if (cycle_elapsed_ms < 2000.0) {
        ++record.cycles_one_to_two_seconds;
      } else if (cycle_elapsed_ms < 3000.0) {
        ++record.cycles_two_to_three_seconds;
      } else {
        ++record.cycles_at_least_three_seconds;
      }
      if (!decision.targets_final_goal &&
          local.selected_goal_index.has_value() &&
          *local.selected_goal_index < local_goals.goals_odom.size()) {
        const auto& selected = std::get<PointGoal>(
            local_goals.goals_odom[*local.selected_goal_index].target);
        for (const hierarchical::SurfacePortalCandidate& portal :
             portals.candidates) {
          const auto& candidate = std::get<PointGoal>(portal.goal_odom.target);
          if (candidate.position_m.x == selected.position_m.x &&
              candidate.position_m.y == selected.position_m.y) {
            confirmed_route_progress_m = std::max(
                confirmed_route_progress_m, portal.route_progress_m);
            break;
          }
        }
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
