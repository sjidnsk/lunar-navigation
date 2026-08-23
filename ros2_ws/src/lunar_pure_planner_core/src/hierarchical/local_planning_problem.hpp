#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_pure_planner_core/types/execution_context.hpp"
#include "lunar_pure_planner_core/types/goal.hpp"
#include "lunar_pure_planner_core/types/planner_config.hpp"
#include "lunar_pure_planner_core/types/planner_io.hpp"
#include "lunar_pure_planner_core/types/platform_capability.hpp"
#include "lunar_pure_planner_core/types/world_snapshot.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::hierarchical {

class LocalSearchDomain final {
 public:
  LocalSearchDomain(std::size_t width, std::size_t height,
                    std::vector<std::uint8_t> allowed);

  [[nodiscard]] bool Contains(shared::GridCell cell) const noexcept;
  [[nodiscard]] std::size_t width() const noexcept;
  [[nodiscard]] std::size_t height() const noexcept;
  [[nodiscard]] std::span<const std::uint8_t> allowed() const noexcept;
  [[nodiscard]] std::size_t allowed_cell_count() const noexcept;
  [[nodiscard]] const std::string &sha256() const noexcept;

 private:
  std::size_t width_{};
  std::size_t height_{};
  std::vector<std::uint8_t> allowed_;
  std::size_t allowed_cell_count_{};
  std::string sha256_;
};

struct LocalPlanningProblem final {
  std::string request_id;
  std::string platform_id;
  std::string capability_version;
  std::uint64_t local_map_generation{};
  TimePoint state_time;
  PlatformState current_state;
  GoalRegion goal_odom;
  GridMap local_map;
  LocalSearchDomain search_domain;
  std::vector<Vec3> route_prefix_odom;
  std::size_t frontier_attempt_index{};
  double frontier_distance_m{};
  PlatformCapability capability;
  PlannerConfig config;
  std::optional<ExecutionContext> previous_execution;
  std::stop_token stop_token;
};

} // namespace lunar::pure_planning::hierarchical
