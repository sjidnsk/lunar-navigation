#include "lunar_pure_exploration_core/boundary_guidance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "detail/visibility_traversal.hpp"

namespace lunar::pure_exploration {
namespace {

using Wide = long double;

constexpr std::array<std::pair<std::int32_t, std::int32_t>, 4>
    kCardinalSteps{{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}};
constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

struct SearchCost {
  std::uint32_t unknown_cell_count;
  std::uint64_t step_count;
};

struct SearchLabel {
  bool reached{false};
  bool settled{false};
  SearchCost cost{0U, 0U};
  std::optional<GridIndex> predecessor;
};

struct QueueEntry {
  SearchCost cost;
  GridIndex cell;
};

struct ActualCandidateKey {
  CandidateKey candidate_key;
  ApproachCandidateKind candidate_kind;
  auto operator<=>(const ActualCandidateKey&) const = default;
};

struct VisibilityCounts {
  std::uint32_t guidance_unknown_cell_count{0U};
  std::uint32_t task_unknown_cell_count{0U};
};

bool Finite(Vec2 point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool Finite(Pose2 pose) {
  return std::isfinite(pose.x) && std::isfinite(pose.y) &&
         std::isfinite(pose.yaw);
}

bool SameGeometry(const GridGeometry& left, const GridGeometry& right) {
  return left.width == right.width && left.height == right.height &&
         left.resolution == right.resolution &&
         left.origin_x == right.origin_x &&
         left.origin_y == right.origin_y &&
         left.origin_yaw == right.origin_yaw;
}

CellState ClassifyGuidanceCell(const OccupancyGridView& map,
                               GridIndex cell) {
  const std::optional<std::int8_t> raw = map.RawValue(cell);
  if (!raw.has_value()) {
    return CellState::kOutsideMap;
  }
  if (*raw == -1) {
    return CellState::kUnknown;
  }
  if (*raw < 0 || *raw > 100) {
    return CellState::kOccupied;
  }
  return map.Classify(cell);
}

void ValidateSharedGeometry(const OccupancyGridView& map,
                            const TaskRaster& raster) {
  if (!SameGeometry(map.geometry(), raster.geometry())) {
    throw std::invalid_argument(
        "boundary guidance map and task raster geometry must match");
  }
}

std::size_t ValidateConstructorInputs(SensorModel sensor_model,
                                      double goal_yaw_tolerance_rad,
                                      BoundaryGuidance::Limits limits) {
  const double full_turn = 2.0 * std::numbers::pi;
  if (!std::isfinite(sensor_model.range_m) || sensor_model.range_m <= 0.0) {
    throw std::invalid_argument(
        "boundary guidance sensor range must be finite and positive");
  }
  if (!std::isfinite(sensor_model.field_of_view_rad) ||
      sensor_model.field_of_view_rad <= 0.0 ||
      sensor_model.field_of_view_rad > full_turn) {
    throw std::invalid_argument(
        "boundary guidance sensor FOV must be in (0, 2*pi]");
  }
  if (!std::isfinite(goal_yaw_tolerance_rad) ||
      goal_yaw_tolerance_rad < 0.0) {
    throw std::invalid_argument(
        "boundary guidance yaw tolerance must be finite and nonnegative");
  }
  if (limits.maximum_guidance_grid_cells == 0U ||
      limits.maximum_guidance_work_units == 0U ||
      limits.maximum_approach_candidates == 0U) {
    throw std::invalid_argument(
        "boundary guidance resource limits must be positive");
  }
  return limits.maximum_guidance_work_units;
}

void ConsumeWork(std::size_t amount, std::size_t limit, std::size_t& used) {
  if (used > limit) {
    throw std::overflow_error("boundary guidance work counter is inconsistent");
  }
  if (amount > limit - used) {
    throw std::length_error("boundary guidance work limit exhausted");
  }
  used += amount;
}

void ConsumeWork(std::size_t limit, std::size_t& used) {
  ConsumeWork(1U, limit, used);
}

std::size_t CheckedMapCellCount(const GridGeometry& geometry,
                                std::size_t maximum_grid_cells) {
  const std::size_t width = static_cast<std::size_t>(geometry.width);
  const std::size_t height = static_cast<std::size_t>(geometry.height);
  if (height != 0U && width > std::numeric_limits<std::size_t>::max() / height) {
    throw std::overflow_error("boundary guidance map cell count overflows size_t");
  }
  const std::size_t count = width * height;
  if (count > maximum_grid_cells) {
    throw std::length_error("boundary guidance map exceeds grid cell limit");
  }
  if (count > std::vector<SearchLabel>{}.max_size()) {
    throw std::length_error("boundary guidance map exceeds search storage");
  }
  return count;
}

std::size_t CellOffset(const GridGeometry& geometry, GridIndex cell) {
  if (cell.x < 0 || cell.y < 0 ||
      static_cast<std::uint32_t>(cell.x) >= geometry.width ||
      static_cast<std::uint32_t>(cell.y) >= geometry.height) {
    throw std::out_of_range("boundary guidance cell is outside the map");
  }
  return static_cast<std::size_t>(cell.y) * geometry.width +
         static_cast<std::size_t>(cell.x);
}

std::optional<GridIndex> OffsetCell(GridIndex cell, std::int32_t dx,
                                    std::int32_t dy) {
  const std::int64_t x = static_cast<std::int64_t>(cell.x) + dx;
  const std::int64_t y = static_cast<std::int64_t>(cell.y) + dy;
  if (x < std::numeric_limits<std::int32_t>::min() ||
      x > std::numeric_limits<std::int32_t>::max() ||
      y < std::numeric_limits<std::int32_t>::min() ||
      y > std::numeric_limits<std::int32_t>::max()) {
    return std::nullopt;
  }
  return GridIndex{static_cast<std::int32_t>(x),
                   static_cast<std::int32_t>(y)};
}

int CompareSearchCost(SearchCost left, SearchCost right) {
  if (left.unknown_cell_count != right.unknown_cell_count) {
    return left.unknown_cell_count < right.unknown_cell_count ? -1 : 1;
  }
  if (left.step_count != right.step_count) {
    return left.step_count < right.step_count ? -1 : 1;
  }
  return 0;
}

bool GuidanceCostLess(const GuidanceCost& left, const GuidanceCost& right) {
  if (left.unknown_cell_count != right.unknown_cell_count) {
    return left.unknown_cell_count < right.unknown_cell_count;
  }
  return left.path_length_m < right.path_length_m;
}

bool WorldCellLess(const OccupancyGridView& map, GridIndex left,
                   GridIndex right) {
  const Vec2 left_center = map.CellCenter(left);
  const Vec2 right_center = map.CellCenter(right);
  if (left_center.x != right_center.x) {
    return left_center.x < right_center.x;
  }
  if (left_center.y != right_center.y) {
    return left_center.y < right_center.y;
  }
  return left < right;
}

struct QueueEntryLess {
  const OccupancyGridView* map;

  bool operator()(const QueueEntry& left, const QueueEntry& right) const {
    const int cost_order = CompareSearchCost(left.cost, right.cost);
    if (cost_order != 0) {
      return cost_order < 0;
    }
    return WorldCellLess(*map, left.cell, right.cell);
  }
};

SearchCost ExtendCost(SearchCost current, CellState entered_state) {
  if (current.step_count == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("boundary guidance path step count overflow");
  }
  ++current.step_count;
  if (entered_state == CellState::kUnknown) {
    if (current.unknown_cell_count ==
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("boundary guidance unknown count overflow");
    }
    ++current.unknown_cell_count;
  }
  return current;
}

double PathLength(std::uint64_t step_count, double resolution) {
  const Wide length = static_cast<Wide>(step_count) * resolution;
  if (!std::isfinite(length) ||
      length > static_cast<Wide>(std::numeric_limits<double>::max())) {
    throw std::overflow_error("boundary guidance path length overflow");
  }
  const double converted = static_cast<double>(length);
  if (!std::isfinite(converted) || (step_count != 0U && converted == 0.0)) {
    throw std::overflow_error(
        "boundary guidance path length is not representable");
  }
  return converted == 0.0 ? 0.0 : converted;
}

double NormalizeYaw(double yaw) {
  if (!std::isfinite(yaw)) {
    throw std::invalid_argument("boundary guidance yaw must be finite");
  }
  if (yaw >= -std::numbers::pi && yaw < std::numbers::pi) {
    return yaw == 0.0 ? 0.0 : yaw;
  }
  if (yaw == std::numbers::pi) {
    return -std::numbers::pi;
  }
  const double full_turn = 2.0 * std::numbers::pi;
  double normalized = std::fmod(yaw + std::numbers::pi, full_turn);
  if (normalized < 0.0) {
    normalized += full_turn;
  }
  normalized -= std::numbers::pi;
  return normalized == 0.0 ? 0.0 : normalized;
}

double HeadingChange(double target_yaw, double robot_yaw) {
  if (!std::isfinite(target_yaw) || !std::isfinite(robot_yaw)) {
    throw std::invalid_argument("boundary guidance heading must be finite");
  }
  const double change = std::abs(
      std::remainder(NormalizeYaw(target_yaw) - NormalizeYaw(robot_yaw),
                     2.0 * std::numbers::pi));
  if (!std::isfinite(change) || change > std::numbers::pi) {
    throw std::overflow_error("boundary guidance heading change overflow");
  }
  return change == 0.0 ? 0.0 : change;
}

double EuclideanDistance(Pose2 candidate, Pose2 robot) {
  const double distance =
      std::hypot(candidate.x - robot.x, candidate.y - robot.y);
  if (!std::isfinite(distance)) {
    throw std::overflow_error(
        "boundary guidance Euclidean distance is not finite");
  }
  return distance == 0.0 ? 0.0 : distance;
}

bool IsBoundaryCell(const TaskRaster& raster, GridIndex cell) {
  if (raster.Classify(cell) == CellState::kOutsideTask) {
    return false;
  }
  for (const auto& [dx, dy] : kCardinalSteps) {
    const std::optional<GridIndex> neighbor = OffsetCell(cell, dx, dy);
    if (!neighbor.has_value() ||
        raster.Classify(*neighbor) == CellState::kOutsideTask) {
      return true;
    }
  }
  return false;
}

bool IsLegalIntent(const OccupancyGridView& map, const TaskRaster& raster,
                   GridIndex intent) {
  if (!map.Contains(intent) || !raster.IsMapBacked(intent) ||
      !IsBoundaryCell(raster, intent)) {
    return false;
  }
  const CellState state = ClassifyGuidanceCell(map, intent);
  return state == CellState::kFree || state == CellState::kUnknown;
}

std::vector<SearchLabel> SearchGlobalMap(
    const OccupancyGridView& map, GridIndex start, std::size_t map_cell_count,
    std::size_t work_limit, std::size_t& consumed_work) {
  std::vector<SearchLabel> labels(map_cell_count);
  std::set<QueueEntry, QueueEntryLess> queue(QueueEntryLess{&map});
  SearchLabel& start_label = labels[CellOffset(map.geometry(), start)];
  start_label.reached = true;
  start_label.cost = SearchCost{0U, 0U};
  queue.insert(QueueEntry{start_label.cost, start});

  while (!queue.empty()) {
    ConsumeWork(work_limit, consumed_work);
    const QueueEntry current = *queue.begin();
    queue.erase(queue.begin());
    SearchLabel& current_label =
        labels[CellOffset(map.geometry(), current.cell)];
    if (current_label.settled ||
        CompareSearchCost(current.cost, current_label.cost) != 0) {
      throw std::logic_error("boundary guidance queue label mismatch");
    }
    current_label.settled = true;

    for (const auto& [dx, dy] : kCardinalSteps) {
      ConsumeWork(work_limit, consumed_work);
      const std::optional<GridIndex> neighbor =
          OffsetCell(current.cell, dx, dy);
      if (!neighbor.has_value() || !map.Contains(*neighbor)) {
        continue;
      }
      const CellState state = ClassifyGuidanceCell(map, *neighbor);
      if (state != CellState::kFree && state != CellState::kUnknown) {
        continue;
      }
      const SearchCost proposed = ExtendCost(current.cost, state);
      SearchLabel& neighbor_label =
          labels[CellOffset(map.geometry(), *neighbor)];
      const int cost_order = neighbor_label.reached
                                 ? CompareSearchCost(proposed,
                                                     neighbor_label.cost)
                                 : -1;
      const bool better_predecessor =
          neighbor_label.reached && cost_order == 0 &&
          (!neighbor_label.predecessor.has_value() ||
           WorldCellLess(map, current.cell, *neighbor_label.predecessor));
      if (cost_order > 0 || (cost_order == 0 && !better_predecessor)) {
        continue;
      }
      if (neighbor_label.settled && cost_order < 0) {
        throw std::logic_error(
            "boundary guidance improved an already settled label");
      }
      if (cost_order < 0 && neighbor_label.reached &&
          !neighbor_label.settled) {
        const std::size_t erased =
            queue.erase(QueueEntry{neighbor_label.cost, *neighbor});
        if (erased != 1U) {
          throw std::logic_error(
              "boundary guidance could not replace queued label");
        }
      }
      neighbor_label.reached = true;
      neighbor_label.cost = proposed;
      neighbor_label.predecessor = current.cell;
      if (cost_order < 0 && !neighbor_label.settled) {
        const bool inserted =
            queue.insert(QueueEntry{proposed, *neighbor}).second;
        if (!inserted) {
          throw std::logic_error(
              "boundary guidance queue rejected improved label");
        }
      }
    }
  }
  return labels;
}

std::vector<GridIndex> ReconstructRoute(
    GridIndex start, GridIndex target, const GridGeometry& geometry,
    const std::vector<SearchLabel>& labels, std::size_t work_limit,
    std::size_t& consumed_work) {
  const SearchLabel& target_label = labels[CellOffset(geometry, target)];
  if (!target_label.reached) {
    return {};
  }
  if (target_label.cost.step_count >=
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("boundary guidance route length overflows size_t");
  }
  const std::size_t expected_size =
      static_cast<std::size_t>(target_label.cost.step_count) + 1U;
  if (expected_size > std::vector<GridIndex>{}.max_size()) {
    throw std::length_error("boundary guidance route exceeds storage capacity");
  }
  if (consumed_work > work_limit ||
      expected_size > work_limit - consumed_work) {
    throw std::length_error("boundary guidance route exceeds remaining work");
  }

  std::vector<GridIndex> reverse_route;
  reverse_route.reserve(expected_size);
  GridIndex current = target;
  for (std::size_t count = 0U; count < expected_size; ++count) {
    ConsumeWork(work_limit, consumed_work);
    reverse_route.push_back(current);
    if (current == start) {
      break;
    }
    const SearchLabel& label = labels[CellOffset(geometry, current)];
    if (!label.predecessor.has_value()) {
      throw std::logic_error("boundary guidance route has no predecessor");
    }
    current = *label.predecessor;
  }
  if (reverse_route.size() != expected_size || reverse_route.back() != start) {
    throw std::logic_error("boundary guidance route reconstruction mismatch");
  }
  std::reverse(reverse_route.begin(), reverse_route.end());
  return reverse_route;
}

Pose2 RoutePose(const OccupancyGridView& map,
                const std::vector<GridIndex>& route, std::size_t index,
                Pose2 robot_pose) {
  if (route.empty() || index >= route.size()) {
    throw std::out_of_range("boundary guidance route pose index is invalid");
  }
  const Vec2 position =
      index == 0U ? Vec2{robot_pose.x, robot_pose.y}
                  : map.CellCenter(route[index]);
  if (route.size() == 1U) {
    return Pose2{position.x, position.y, NormalizeYaw(robot_pose.yaw)};
  }

  Vec2 from = position;
  Vec2 to{};
  if (index + 1U < route.size()) {
    to = map.CellCenter(route[index + 1U]);
  } else {
    from = map.CellCenter(route[index - 1U]);
    to = position;
  }
  const double dx = to.x - from.x;
  const double dy = to.y - from.y;
  if (!std::isfinite(dx) || !std::isfinite(dy) ||
      (dx == 0.0 && dy == 0.0)) {
    throw std::overflow_error("boundary guidance route tangent is invalid");
  }
  return Pose2{position.x, position.y,
               NormalizeYaw(std::atan2(dy, dx))};
}

std::uint32_t RemainingUnknownCount(
    const OccupancyGridView& map, const std::vector<GridIndex>& route,
    std::size_t candidate_index, std::size_t work_limit,
    std::size_t& consumed_work) {
  std::uint32_t count = 0U;
  for (std::size_t index = candidate_index + 1U; index < route.size();
       ++index) {
    ConsumeWork(work_limit, consumed_work);
    if (ClassifyGuidanceCell(map, route[index]) != CellState::kUnknown) {
      continue;
    }
    if (count == std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error(
          "boundary guidance remaining unknown count overflow");
    }
    ++count;
  }
  return count;
}

bool TraceCellRepresentable(const detail::TraceCell& cell) {
  return cell.x >= std::numeric_limits<std::int32_t>::min() &&
         cell.x <= std::numeric_limits<std::int32_t>::max() &&
         cell.y >= std::numeric_limits<std::int32_t>::min() &&
         cell.y <= std::numeric_limits<std::int32_t>::max();
}

GridIndex ToGridIndex(const detail::TraceCell& cell) {
  if (!TraceCellRepresentable(cell)) {
    throw std::overflow_error(
        "boundary guidance visibility cell exceeds GridIndex");
  }
  return GridIndex{static_cast<std::int32_t>(cell.x),
                   static_cast<std::int32_t>(cell.y)};
}

std::int32_t CheckedVisibilityBound(Wide value) {
  if (!std::isfinite(value) ||
      value < static_cast<Wide>(std::numeric_limits<std::int32_t>::min()) ||
      value > static_cast<Wide>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error(
        "boundary guidance visibility bound exceeds int32");
  }
  return static_cast<std::int32_t>(value);
}

VisibilityCounts EvaluateVisibility(
    const OccupancyGridView& map, const TaskRaster& raster,
    SensorModel sensor_model, Pose2 pose,
    const std::set<GridIndex>& guidance_route_cells,
    std::size_t work_limit, std::size_t& consumed_work) {
  if (!Finite(pose)) {
    throw std::invalid_argument(
        "boundary guidance visibility pose must be finite");
  }
  const auto start = map.WorldToGrid(Vec2{pose.x, pose.y});
  if (!start.has_value()) {
    throw std::overflow_error(
        "boundary guidance visibility start is not representable");
  }
  if (map.geometry().width == 0U || map.geometry().height == 0U) {
    return {};
  }

  const double range_cells_double =
      sensor_model.range_m / map.geometry().resolution;
  if (!std::isfinite(range_cells_double)) {
    throw std::overflow_error(
        "boundary guidance sensor range in cells overflowed");
  }
  const Wide range_cells = range_cells_double;
  const Wide lower_x = std::ceil(static_cast<Wide>(start->x) - range_cells -
                                 0.5L);
  const Wide upper_x = std::floor(static_cast<Wide>(start->x) + range_cells -
                                  0.5L);
  const Wide lower_y = std::ceil(static_cast<Wide>(start->y) - range_cells -
                                 0.5L);
  const Wide upper_y = std::floor(static_cast<Wide>(start->y) + range_cells -
                                  0.5L);
  const Wide map_max_x = static_cast<Wide>(map.geometry().width - 1U);
  const Wide map_max_y = static_cast<Wide>(map.geometry().height - 1U);
  const Wide clipped_lower_x = std::max(lower_x, 0.0L);
  const Wide clipped_upper_x = std::min(upper_x, map_max_x);
  const Wide clipped_lower_y = std::max(lower_y, 0.0L);
  const Wide clipped_upper_y = std::min(upper_y, map_max_y);
  if (clipped_lower_x > clipped_upper_x ||
      clipped_lower_y > clipped_upper_y) {
    return {};
  }
  const std::int32_t minimum_x = CheckedVisibilityBound(clipped_lower_x);
  const std::int32_t maximum_x = CheckedVisibilityBound(clipped_upper_x);
  const std::int32_t minimum_y = CheckedVisibilityBound(clipped_lower_y);
  const std::int32_t maximum_y = CheckedVisibilityBound(clipped_upper_y);
  const std::size_t width = static_cast<std::size_t>(
      static_cast<std::int64_t>(maximum_x) - minimum_x + 1);
  const std::size_t height = static_cast<std::size_t>(
      static_cast<std::int64_t>(maximum_y) - minimum_y + 1);
  if (height != 0U && width > std::numeric_limits<std::size_t>::max() / height) {
    throw std::overflow_error(
        "boundary guidance visibility AABB cell count overflow");
  }
  const std::size_t aabb_cells = width * height;
  if (consumed_work > work_limit ||
      aabb_cells > work_limit - consumed_work) {
    throw std::length_error(
        "boundary guidance visibility AABB exceeds remaining work");
  }

  const double yaw_delta = pose.yaw - map.geometry().origin_yaw;
  if (!std::isfinite(yaw_delta)) {
    throw std::overflow_error("boundary guidance grid yaw overflowed");
  }
  const double full_turn = 2.0 * std::numbers::pi;
  const double yaw_grid = std::remainder(yaw_delta, full_turn);
  const double half_fov = sensor_model.field_of_view_rad / 2.0;
  const bool omnidirectional = sensor_model.field_of_view_rad == full_turn;
  const Wide range_squared = range_cells * range_cells;
  VisibilityCounts counts;

  for (std::int64_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::int64_t x = minimum_x; x <= maximum_x; ++x) {
      ConsumeWork(work_limit, consumed_work);
      const GridIndex target{static_cast<std::int32_t>(x),
                             static_cast<std::int32_t>(y)};
      const Wide delta_x =
          static_cast<Wide>(x) + 0.5L - static_cast<Wide>(start->x);
      const Wide delta_y =
          static_cast<Wide>(y) + 0.5L - static_cast<Wide>(start->y);
      const Wide distance_squared = delta_x * delta_x + delta_y * delta_y;
      if (distance_squared > range_squared) {
        continue;
      }
      if (!omnidirectional && distance_squared != 0.0L) {
        const double bearing =
            std::atan2(static_cast<double>(delta_y),
                       static_cast<double>(delta_x));
        const double difference =
            std::abs(std::remainder(bearing - yaw_grid, full_turn));
        if (difference > half_fov) {
          continue;
        }
      }
      if (ClassifyGuidanceCell(map, target) != CellState::kUnknown) {
        continue;
      }

      bool blocked = false;
      detail::VisibilityWorkBudget trace_budget{work_limit, consumed_work};
      detail::TraceClosedSegment(
          *start,
          Vec2{static_cast<double>(x) + 0.5,
               static_cast<double>(y) + 0.5},
          trace_budget,
          [&](long double, std::span<const detail::TraceCell> group) {
            for (const detail::TraceCell& traced : group) {
              if (!TraceCellRepresentable(traced)) {
                blocked = true;
                break;
              }
              const GridIndex visited = ToGridIndex(traced);
              if (visited == target) {
                continue;
              }
              if (!map.Contains(visited)) {
                blocked = true;
                break;
              }
              const CellState state = ClassifyGuidanceCell(map, visited);
              if (state == CellState::kOccupied ||
                  state == CellState::kOutsideMap) {
                blocked = true;
                break;
              }
            }
            return blocked ? detail::TraceControl::kStop
                           : detail::TraceControl::kContinue;
          });
      consumed_work = trace_budget.used;
      if (blocked) {
        continue;
      }
      if (guidance_route_cells.contains(target)) {
        detail::CheckedVisibleIncrement(
            counts.guidance_unknown_cell_count);
      }
      if (raster.Classify(target) == CellState::kUnknown) {
        detail::CheckedVisibleIncrement(counts.task_unknown_cell_count);
      }
    }
  }
  return counts;
}

double TaskUnknownArea(std::uint32_t count, double resolution) {
  const Wide area = static_cast<Wide>(count) * resolution * resolution;
  if (!std::isfinite(area) ||
      area > static_cast<Wide>(std::numeric_limits<double>::max())) {
    throw std::overflow_error("boundary guidance task unknown area overflow");
  }
  const double converted = static_cast<double>(area);
  if (!std::isfinite(converted) || (count != 0U && converted == 0.0)) {
    throw std::overflow_error(
        "boundary guidance task unknown area is not representable");
  }
  return converted == 0.0 ? 0.0 : converted;
}

std::vector<Vec2> GuidanceRouteWorld(
    const OccupancyGridView& map, const std::vector<GridIndex>& route,
    std::size_t first_index, Pose2 first_pose, std::size_t work_limit,
    std::size_t& consumed_work) {
  if (first_index >= route.size()) {
    throw std::out_of_range("boundary guidance route suffix is empty");
  }
  const std::size_t count = route.size() - first_index;
  if (count > std::vector<Vec2>{}.max_size()) {
    throw std::length_error(
        "boundary guidance world route exceeds storage capacity");
  }
  if (consumed_work > work_limit || count > work_limit - consumed_work) {
    throw std::length_error(
        "boundary guidance world route exceeds remaining work");
  }
  std::vector<Vec2> world_route;
  world_route.reserve(count);
  for (std::size_t index = first_index; index < route.size(); ++index) {
    ConsumeWork(work_limit, consumed_work);
    world_route.push_back(
        index == first_index
            ? Vec2{first_pose.x, first_pose.y}
            : map.CellCenter(route[index]));
  }
  return world_route;
}

std::set<GridIndex> RouteCellSet(const std::vector<GridIndex>& route,
                                 std::size_t first_index) {
  if (first_index >= route.size()) {
    throw std::out_of_range("boundary guidance route suffix is empty");
  }
  return std::set<GridIndex>(route.begin() +
                                 static_cast<std::ptrdiff_t>(first_index),
                             route.end());
}

void HashByte(std::uint64_t& hash, std::uint8_t value) {
  hash ^= value;
  hash *= kFnvPrime;
}

void HashInt64(std::uint64_t& hash, std::int64_t value) {
  const std::uint64_t bits = static_cast<std::uint64_t>(value);
  for (unsigned shift = 0U; shift < 64U; shift += 8U) {
    HashByte(hash,
             static_cast<std::uint8_t>((bits >> shift) & UINT64_C(0xff)));
  }
}

std::uint64_t CandidateDisplayId(
    const BoundaryApproachGoalIdentity& identity) {
  std::uint64_t hash = kFnvOffset;
  HashInt64(hash, identity.intent_cell.x);
  HashInt64(hash, identity.intent_cell.y);
  HashInt64(hash, identity.candidate_key.x_mm);
  HashInt64(hash, identity.candidate_key.y_mm);
  HashInt64(hash, identity.candidate_key.yaw_tenth_deg);
  HashByte(hash, static_cast<std::uint8_t>(identity.candidate_kind));
  return hash;
}

bool BetterDuplicate(const OccupancyGridView& map,
                     const ApproachCandidate& candidate,
                     const ApproachCandidate& incumbent) {
  if (GuidanceCostLess(candidate.remaining_cost, incumbent.remaining_cost)) {
    return true;
  }
  if (GuidanceCostLess(incumbent.remaining_cost, candidate.remaining_cost)) {
    return false;
  }
  if (candidate.task_unknown_area_m2 != incumbent.task_unknown_area_m2) {
    return candidate.task_unknown_area_m2 >
           incumbent.task_unknown_area_m2;
  }
  return WorldCellLess(map, candidate.identity.intent_cell,
                       incumbent.identity.intent_cell);
}

void ValidateCandidateForRanking(const ApproachCandidate& candidate) {
  if (!Finite(candidate.pose) ||
      !std::isfinite(candidate.remaining_cost.path_length_m) ||
      candidate.remaining_cost.path_length_m < 0.0 ||
      !std::isfinite(candidate.task_unknown_area_m2) ||
      candidate.task_unknown_area_m2 < 0.0) {
    throw std::invalid_argument(
        "boundary guidance candidate metrics must be finite and nonnegative");
  }
  for (const Vec2 point : candidate.guidance_route) {
    if (!Finite(point)) {
      throw std::invalid_argument(
          "boundary guidance route points must be finite");
    }
  }
  if (candidate.identity.candidate_key != MakeCandidateKey(candidate.pose)) {
    throw std::invalid_argument(
        "boundary guidance candidate identity does not match pose");
  }
}

struct RankingMetrics {
  std::vector<double> euclidean_distance;
  std::vector<double> heading_change;
};

RankingMetrics ValidateRankingAuthority(const BoundaryGuidanceResult& result,
                                        Pose2 robot_pose) {
  if (!Finite(robot_pose)) {
    throw std::invalid_argument(
        "boundary guidance ranking robot pose must be finite");
  }
  RankingMetrics metrics;
  metrics.euclidean_distance.reserve(result.candidates.size());
  metrics.heading_change.reserve(result.candidates.size());
  std::set<ActualCandidateKey> identities;
  for (const ApproachCandidate& candidate : result.candidates) {
    ValidateCandidateForRanking(candidate);
    const ActualCandidateKey actual{candidate.identity.candidate_key,
                                    candidate.identity.candidate_kind};
    if (!identities.insert(actual).second) {
      throw std::invalid_argument(
          "boundary guidance candidates contain a duplicate actual pose");
    }
    metrics.euclidean_distance.push_back(
        EuclideanDistance(candidate.pose, robot_pose));
    metrics.heading_change.push_back(
        HeadingChange(candidate.pose.yaw, robot_pose.yaw));
  }
  return metrics;
}

bool HigherLevelCandidateLess(const ApproachCandidate& left,
                              const ApproachCandidate& right) {
  if (left.fully_inside_task != right.fully_inside_task) {
    return left.fully_inside_task;
  }
  if (left.remaining_cost.unknown_cell_count !=
      right.remaining_cost.unknown_cell_count) {
    return left.remaining_cost.unknown_cell_count <
           right.remaining_cost.unknown_cell_count;
  }
  if (left.remaining_cost.path_length_m != right.remaining_cost.path_length_m) {
    return left.remaining_cost.path_length_m < right.remaining_cost.path_length_m;
  }
  if (left.task_unknown_area_m2 != right.task_unknown_area_m2) {
    return left.task_unknown_area_m2 > right.task_unknown_area_m2;
  }
  return false;
}

bool HigherLevelCandidateEquivalent(const ApproachCandidate& left,
                                    const ApproachCandidate& right) {
  return left.fully_inside_task == right.fully_inside_task &&
         left.remaining_cost.unknown_cell_count ==
             right.remaining_cost.unknown_cell_count &&
         left.remaining_cost.path_length_m == right.remaining_cost.path_length_m &&
         left.task_unknown_area_m2 == right.task_unknown_area_m2;
}

}  // namespace

BoundaryGuidance::BoundaryGuidance(PlatformGeometry platform,
                                   SensorModel sensor_model,
                                   double goal_yaw_tolerance_rad,
                                   Limits limits)
    : validator_(std::move(platform),
                 ValidateConstructorInputs(sensor_model,
                                           goal_yaw_tolerance_rad, limits)),
      sensor_model_(sensor_model),
      goal_yaw_tolerance_rad_(goal_yaw_tolerance_rad),
      limits_(limits) {}

BoundaryGuidanceResult BoundaryGuidance::Build(
    const OccupancyGridView& map, const TaskRaster& raster,
    Pose2 robot_pose) const {
  ValidateSharedGeometry(map, raster);
  if (!Finite(robot_pose)) {
    throw std::invalid_argument(
        "boundary guidance robot pose must be finite");
  }
  const std::size_t map_cell_count =
      CheckedMapCellCount(map.geometry(), limits_.maximum_guidance_grid_cells);
  if (raster.task_cells().size() > limits_.maximum_guidance_grid_cells) {
    throw std::length_error(
        "boundary guidance task raster exceeds grid cell limit");
  }

  BoundaryGuidanceResult result{
      .phase = NavigationPhase::kApproachTask,
      .wait_reason = ApproachWaitReason::kNone,
      .fully_inside_task = false,
      .intents = {},
      .candidates = {},
      .consumed_work_units = 0U,
  };
  std::size_t& work = result.consumed_work_units;
  if (!validator_.IsMapFree(map, robot_pose, work)) {
    result.wait_reason = ApproachWaitReason::kWaitingForSafeStart;
    return result;
  }
  if (validator_.IsTaskFree(raster, robot_pose, work)) {
    result.phase = NavigationPhase::kExploreTask;
    result.fully_inside_task = true;
    return result;
  }

  std::vector<GridIndex> legal_boundary_cells;
  if (raster.task_cells().size() >
      std::vector<GridIndex>{}.max_size()) {
    throw std::length_error(
        "boundary guidance intent set exceeds storage capacity");
  }
  legal_boundary_cells.reserve(raster.task_cells().size());
  bool uncovered_boundary = false;
  for (const GridIndex cell : raster.task_cells()) {
    ConsumeWork(limits_.maximum_guidance_work_units, work);
    if (!IsBoundaryCell(raster, cell)) {
      continue;
    }
    if (!map.Contains(cell) ||
        raster.Classify(cell) == CellState::kOutsideMap) {
      uncovered_boundary = true;
      continue;
    }
    const CellState state = ClassifyGuidanceCell(map, cell);
    if (state == CellState::kFree || state == CellState::kUnknown) {
      legal_boundary_cells.push_back(cell);
    }
  }
  if (uncovered_boundary) {
    result.wait_reason = ApproachWaitReason::kWaitingForTaskMapCoverage;
    return result;
  }
  if (legal_boundary_cells.empty()) {
    result.wait_reason = ApproachWaitReason::kNoGuidanceRoute;
    return result;
  }

  const auto start_cell = map.WorldToCell(Vec2{robot_pose.x, robot_pose.y});
  if (!start_cell.has_value() || !map.Contains(*start_cell) ||
      map.Classify(*start_cell) != CellState::kFree) {
    result.wait_reason = ApproachWaitReason::kWaitingForSafeStart;
    return result;
  }
  const std::vector<SearchLabel> labels = SearchGlobalMap(
      map, *start_cell, map_cell_count,
      limits_.maximum_guidance_work_units, work);

  result.intents.reserve(legal_boundary_cells.size());
  for (const GridIndex intent_cell : legal_boundary_cells) {
    const SearchLabel& label =
        labels[CellOffset(map.geometry(), intent_cell)];
    if (!label.reached) {
      continue;
    }
    result.intents.push_back(ApproachIntent{
        .cell = intent_cell,
        .total_cost = GuidanceCost{
            label.cost.unknown_cell_count,
            PathLength(label.cost.step_count, map.geometry().resolution)},
        .route = ReconstructRoute(
            *start_cell, intent_cell, map.geometry(), labels,
            limits_.maximum_guidance_work_units, work),
    });
  }
  std::sort(result.intents.begin(), result.intents.end(),
            [&map](const ApproachIntent& left,
                   const ApproachIntent& right) {
              if (GuidanceCostLess(left.total_cost, right.total_cost)) {
                return true;
              }
              if (GuidanceCostLess(right.total_cost, left.total_cost)) {
                return false;
              }
              return WorldCellLess(map, left.cell, right.cell);
            });
  if (result.intents.empty()) {
    result.wait_reason = ApproachWaitReason::kNoGuidanceRoute;
    return result;
  }

  std::map<ActualCandidateKey, ApproachCandidate> deduplicated;
  for (const ApproachIntent& intent : result.intents) {
    std::optional<std::size_t> last_safe_index;
    Pose2 last_safe_pose{};
    bool last_safe_fully_inside = false;
    std::optional<std::size_t> last_fully_inside_index;
    Pose2 last_fully_inside_pose{};
    for (std::size_t index = 0U; index < intent.route.size(); ++index) {
      const Pose2 pose = RoutePose(map, intent.route, index, robot_pose);
      if (!validator_.IsMapFree(map, pose, work)) {
        break;
      }
      last_safe_index = index;
      last_safe_pose = pose;
      last_safe_fully_inside = validator_.IsTaskFree(raster, pose, work);
      if (last_safe_fully_inside) {
        last_fully_inside_index = index;
        last_fully_inside_pose = pose;
      }
    }
    if (!last_safe_index.has_value()) {
      continue;
    }

    std::size_t candidate_index = *last_safe_index;
    Pose2 candidate_pose = last_safe_pose;
    bool fully_inside = last_safe_fully_inside;
    ApproachCandidateKind kind =
        candidate_index == 0U ? ApproachCandidateKind::kRotation
                              : ApproachCandidateKind::kTranslation;
    if (kind == ApproachCandidateKind::kRotation &&
        !(HeadingChange(candidate_pose.yaw, robot_pose.yaw) >
          goal_yaw_tolerance_rad_)) {
      continue;
    }

    auto remaining_cost_at = [&](std::size_t index) {
      const std::uint64_t remaining_steps =
          static_cast<std::uint64_t>(intent.route.size() - 1U - index);
      return GuidanceCost{
          RemainingUnknownCount(map, intent.route, index,
                                limits_.maximum_guidance_work_units, work),
          PathLength(remaining_steps, map.geometry().resolution),
      };
    };
    GuidanceCost remaining_cost = remaining_cost_at(candidate_index);

    // A known route can cross the task and end at another boundary cell.  If
    // that final safe pose neither enters the task nor exposes route UNKNOWN,
    // retain the greatest-index fully-inside pose from the same safe prefix.
    if (kind == ApproachCandidateKind::kTranslation && !fully_inside &&
        remaining_cost.unknown_cell_count == 0U &&
        last_fully_inside_index.has_value()) {
      candidate_index = *last_fully_inside_index;
      candidate_pose = last_fully_inside_pose;
      fully_inside = true;
      kind = ApproachCandidateKind::kTranslation;
      remaining_cost = remaining_cost_at(candidate_index);
    }
    if (kind == ApproachCandidateKind::kTranslation &&
        !GuidanceCostLess(remaining_cost, intent.total_cost)) {
      continue;
    }

    const std::set<GridIndex> route_cells =
        RouteCellSet(intent.route, candidate_index);
    VisibilityCounts visibility = EvaluateVisibility(
        map, raster, sensor_model_, candidate_pose, route_cells,
        limits_.maximum_guidance_work_units, work);

    // Occlusion can make a suffix containing UNKNOWN provide no observation
    // value.  A fully-inside pose already proven in the safe prefix is still
    // a valid entry target, and the greatest route index is deterministic.
    if (kind == ApproachCandidateKind::kTranslation && !fully_inside &&
        visibility.guidance_unknown_cell_count == 0U &&
        last_fully_inside_index.has_value() &&
        *last_fully_inside_index != candidate_index) {
      candidate_index = *last_fully_inside_index;
      candidate_pose = last_fully_inside_pose;
      fully_inside = true;
      remaining_cost = remaining_cost_at(candidate_index);
      const std::set<GridIndex> fallback_route_cells =
          RouteCellSet(intent.route, candidate_index);
      visibility = EvaluateVisibility(
          map, raster, sensor_model_, candidate_pose, fallback_route_cells,
          limits_.maximum_guidance_work_units, work);
    }
    if (kind == ApproachCandidateKind::kRotation &&
        visibility.guidance_unknown_cell_count == 0U) {
      continue;
    }
    if (kind == ApproachCandidateKind::kTranslation && !fully_inside &&
        visibility.guidance_unknown_cell_count == 0U) {
      continue;
    }

    const CandidateKey key = MakeCandidateKey(candidate_pose);
    const BoundaryApproachGoalIdentity identity{
        .intent_cell = intent.cell,
        .candidate_key = key,
        .candidate_kind = kind,
    };
    ApproachCandidate candidate{
        .id = CandidateDisplayId(identity),
        .identity = identity,
        .pose = candidate_pose,
        .remaining_cost = remaining_cost,
        .task_unknown_area_m2 =
            TaskUnknownArea(visibility.task_unknown_cell_count,
                            map.geometry().resolution),
        .guidance_unknown_cell_count =
            visibility.guidance_unknown_cell_count,
        .fully_inside_task = fully_inside,
        .guidance_route = GuidanceRouteWorld(
            map, intent.route, candidate_index, candidate_pose,
            limits_.maximum_guidance_work_units, work),
    };
    const ActualCandidateKey actual{key, kind};
    const auto incumbent = deduplicated.find(actual);
    if (incumbent == deduplicated.end()) {
      if (deduplicated.size() == limits_.maximum_approach_candidates) {
        throw std::length_error(
            "boundary guidance approach candidate limit exceeded");
      }
      deduplicated.emplace(actual, std::move(candidate));
    } else if (BetterDuplicate(map, candidate, incumbent->second)) {
      incumbent->second = std::move(candidate);
    }
  }

  result.candidates.reserve(deduplicated.size());
  for (auto& [unused, candidate] : deduplicated) {
    (void)unused;
    result.candidates.push_back(std::move(candidate));
  }
  std::sort(result.candidates.begin(), result.candidates.end(),
            [](const ApproachCandidate& left,
               const ApproachCandidate& right) {
              return left.identity < right.identity;
            });
  result.wait_reason = result.candidates.empty()
                           ? ApproachWaitReason::kNoSafeCandidate
                           : ApproachWaitReason::kNone;
  return result;
}

std::vector<std::size_t> BoundaryGuidance::CoarseOrder(
    const BoundaryGuidanceResult& result, Pose2 robot_pose) const {
  const RankingMetrics metrics =
      ValidateRankingAuthority(result, robot_pose);
  std::vector<std::size_t> order(result.candidates.size());
  for (std::size_t index = 0U; index < order.size(); ++index) {
    order[index] = index;
  }
  std::sort(order.begin(), order.end(),
            [&result, &metrics](std::size_t left_index,
                               std::size_t right_index) {
              const ApproachCandidate& left = result.candidates[left_index];
              const ApproachCandidate& right = result.candidates[right_index];
              if (!HigherLevelCandidateEquivalent(left, right)) {
                return HigherLevelCandidateLess(left, right);
              }
              if (metrics.euclidean_distance[left_index] !=
                  metrics.euclidean_distance[right_index]) {
                return metrics.euclidean_distance[left_index] <
                       metrics.euclidean_distance[right_index];
              }
              return left.identity < right.identity;
            });
  return order;
}

std::vector<std::size_t> BoundaryGuidance::FinalOrder(
    const BoundaryGuidanceResult& result,
    std::span<const PlannedCandidate> planned, Pose2 robot_pose) const {
  const RankingMetrics metrics =
      ValidateRankingAuthority(result, robot_pose);
  std::set<std::size_t> seen;
  std::vector<std::size_t> order;
  std::vector<double> path_lengths(result.candidates.size(), 0.0);
  order.reserve(planned.size());
  for (const PlannedCandidate& row : planned) {
    if (row.candidate_index >= result.candidates.size()) {
      throw std::invalid_argument(
          "boundary guidance planned candidate index is out of range");
    }
    if (!std::isfinite(row.path_length_m) || row.path_length_m < 0.0) {
      throw std::invalid_argument(
          "boundary guidance planned path length must be finite and nonnegative");
    }
    if (!seen.insert(row.candidate_index).second) {
      throw std::invalid_argument(
          "boundary guidance planned candidates contain duplicate indices");
    }
    path_lengths[row.candidate_index] = row.path_length_m;
    order.push_back(row.candidate_index);
  }
  std::sort(order.begin(), order.end(),
            [&result, &metrics, &path_lengths](std::size_t left_index,
                                              std::size_t right_index) {
              const ApproachCandidate& left = result.candidates[left_index];
              const ApproachCandidate& right = result.candidates[right_index];
              if (!HigherLevelCandidateEquivalent(left, right)) {
                return HigherLevelCandidateLess(left, right);
              }
              if (path_lengths[left_index] != path_lengths[right_index]) {
                return path_lengths[left_index] < path_lengths[right_index];
              }
              if (metrics.heading_change[left_index] !=
                  metrics.heading_change[right_index]) {
                return metrics.heading_change[left_index] <
                       metrics.heading_change[right_index];
              }
              return left.identity < right.identity;
            });
  return order;
}

bool BoundaryGuidance::IsCandidateStillValid(
    const OccupancyGridView& latest_map, const TaskRaster& latest_raster,
    const ApproachCandidate& frozen_candidate) const {
  ValidateSharedGeometry(latest_map, latest_raster);
  ValidateCandidateForRanking(frozen_candidate);
  (void)CheckedMapCellCount(latest_map.geometry(),
                            limits_.maximum_guidance_grid_cells);
  if (latest_raster.task_cells().size() >
      limits_.maximum_guidance_grid_cells) {
    throw std::length_error(
        "boundary guidance task raster exceeds grid cell limit");
  }
  if (!IsLegalIntent(latest_map, latest_raster,
                     frozen_candidate.identity.intent_cell)) {
    return false;
  }

  std::size_t work = 0U;
  if (!validator_.IsMapFree(latest_map, frozen_candidate.pose, work)) {
    return false;
  }
  if (validator_.IsTaskFree(latest_raster, frozen_candidate.pose, work)) {
    return true;
  }

  std::set<GridIndex> route_cells;
  for (const Vec2 point : frozen_candidate.guidance_route) {
    ConsumeWork(limits_.maximum_guidance_work_units, work);
    const auto cell = latest_map.WorldToCell(point);
    if (!cell.has_value()) {
      return false;
    }
    if (latest_map.Contains(*cell)) {
      route_cells.insert(*cell);
    }
  }
  if (route_cells.empty()) {
    return false;
  }
  const VisibilityCounts visibility = EvaluateVisibility(
      latest_map, latest_raster, sensor_model_, frozen_candidate.pose,
      route_cells, limits_.maximum_guidance_work_units, work);
  return visibility.guidance_unknown_cell_count > 0U;
}

}  // namespace lunar::pure_exploration
