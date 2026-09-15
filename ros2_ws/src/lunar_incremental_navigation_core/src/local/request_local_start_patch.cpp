#include "lunar_incremental_navigation_core/request_local_start_patch.hpp"

#include "lunar_incremental_navigation_core/local_planning_window.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "map/grid_bounds.hpp"

namespace lunar::incremental_navigation {
namespace {

struct Plane final {
  long double x_slope{};
  long double y_slope{};
  long double offset{};
};

[[nodiscard]] Point2 CellCenter(const SparseGridGeometry& geometry,
                                const GridIndex index) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  return Point2{
      .x = std::fma(static_cast<double>(index.x) + 0.5, resolution_m,
                    origin.x),
      .y = std::fma(static_cast<double>(index.y) + 0.5, resolution_m,
                    origin.y),
  };
}

[[nodiscard]] std::optional<GridIndex> WorldToCell(
    const SparseGridGeometry& geometry, const Point2 point) noexcept {
  if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
    return std::nullopt;
  }
  const Vec3 origin = geometry.origin_m();
  const long double x_cells =
      (static_cast<long double>(point.x) - origin.x) /
      geometry.resolution_m();
  const long double y_cells =
      (static_cast<long double>(point.y) - origin.y) /
      geometry.resolution_m();
  const long double floored_x = std::floor(x_cells);
  const long double floored_y = std::floor(y_cells);
  if (!std::isfinite(x_cells) || !std::isfinite(y_cells) ||
      floored_x < static_cast<long double>(geometry.min_inclusive().x) ||
      floored_y < static_cast<long double>(geometry.min_inclusive().y) ||
      floored_x >= static_cast<long double>(geometry.max_exclusive().x) ||
      floored_y >= static_cast<long double>(geometry.max_exclusive().y)) {
    return std::nullopt;
  }
  const GridIndex result{.x = static_cast<std::int64_t>(floored_x),
                         .y = static_cast<std::int64_t>(floored_y)};
  return geometry.Contains(result) ? std::optional<GridIndex>(result)
                                   : std::nullopt;
}

[[nodiscard]] GridIndex ClampedLowerBound(
    const SparseGridGeometry& geometry, const Point2 center,
    const double radius_m) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  const auto lower = [](const double value, const std::int64_t minimum,
                        const std::int64_t maximum) {
    if (value <= static_cast<double>(minimum)) {
      return minimum;
    }
    if (value >= static_cast<double>(maximum)) {
      return maximum;
    }
    return static_cast<std::int64_t>(std::floor(value));
  };
  return {
      .x = lower((center.x - radius_m - origin.x) / resolution_m - 1.0,
                 geometry.min_inclusive().x,
                 geometry.max_exclusive().x),
      .y = lower((center.y - radius_m - origin.y) / resolution_m - 1.0,
                 geometry.min_inclusive().y,
                 geometry.max_exclusive().y),
  };
}

[[nodiscard]] GridIndex ClampedUpperBound(
    const SparseGridGeometry& geometry, const Point2 center,
    const double radius_m) noexcept {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  const auto upper = [](const double value, const std::int64_t minimum,
                        const std::int64_t maximum) {
    if (value <= static_cast<double>(minimum)) {
      return minimum;
    }
    if (value >= static_cast<double>(maximum)) {
      return maximum;
    }
    return static_cast<std::int64_t>(std::ceil(value));
  };
  return {
      .x = upper((center.x + radius_m - origin.x) / resolution_m + 1.0,
                 geometry.min_inclusive().x,
                 geometry.max_exclusive().x),
      .y = upper((center.y + radius_m - origin.y) / resolution_m + 1.0,
                 geometry.min_inclusive().y,
                 geometry.max_exclusive().y),
  };
}

[[nodiscard]] std::set<GridIndex> PatchCells(
    const SparseGridGeometry& geometry, const Point2 anchor,
    const double radius_m) {
  std::set<GridIndex> result;
  const GridIndex minimum = ClampedLowerBound(geometry, anchor, radius_m);
  const GridIndex maximum = ClampedUpperBound(geometry, anchor, radius_m);
  for (std::int64_t y = minimum.y; y < maximum.y; ++y) {
    for (std::int64_t x = minimum.x; x < maximum.x; ++x) {
      const GridIndex cell{.x = x, .y = y};
      if (CircleIntersectsCellArea(anchor, radius_m, geometry, cell)) {
        result.insert(cell);
      }
    }
  }
  return result;
}

[[nodiscard]] std::set<GridIndex> WithOneCellRing(
    const SparseGridGeometry& geometry,
    const std::set<GridIndex>& domain) {
  std::set<GridIndex> result = domain;
  for (const GridIndex cell : domain) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        if (const std::optional<GridIndex> neighbor =
                detail::OffsetWithinGeometry(geometry, cell, dx, dy)) {
          result.insert(*neighbor);
        }
      }
    }
  }
  return result;
}

[[nodiscard]] std::optional<Plane> FitSupportPlane(
    const ElevationSnapshot& raw,
    const PlatformElevationEvaluator& evaluator,
    const std::set<GridIndex>& sample_domain, const Point2 anchor) {
  std::array<std::array<long double, 4>, 3> matrix{};
  std::size_t sample_count = 0U;
  std::optional<GridIndex> first_sample;
  std::optional<GridIndex> second_sample;
  bool has_noncollinear_samples = false;
  for (const GridIndex cell : sample_domain) {
    const std::optional<float> elevation = raw.ElevationAt(cell);
    if (!elevation || evaluator.Evaluate(raw, cell).state ==
                          IntrinsicCellState::kBlocked) {
      continue;
    }
    const Point2 center = CellCenter(raw.geometry(), cell);
    const long double x = static_cast<long double>(center.x - anchor.x);
    const long double y = static_cast<long double>(center.y - anchor.y);
    const long double z = static_cast<long double>(*elevation);
    matrix[0][0] += x * x;
    matrix[0][1] += x * y;
    matrix[0][2] += x;
    matrix[0][3] += x * z;
    matrix[1][0] += x * y;
    matrix[1][1] += y * y;
    matrix[1][2] += y;
    matrix[1][3] += y * z;
    matrix[2][0] += x;
    matrix[2][1] += y;
    matrix[2][2] += 1.0L;
    matrix[2][3] += z;
    if (!first_sample) {
      first_sample = cell;
    } else if (!second_sample) {
      second_sample = cell;
    } else {
      const long double first_dx =
          static_cast<long double>(second_sample->x) - first_sample->x;
      const long double first_dy =
          static_cast<long double>(second_sample->y) - first_sample->y;
      const long double next_dx =
          static_cast<long double>(cell.x) - first_sample->x;
      const long double next_dy =
          static_cast<long double>(cell.y) - first_sample->y;
      has_noncollinear_samples =
          has_noncollinear_samples ||
          first_dx * next_dy - first_dy * next_dx != 0.0L;
    }
    ++sample_count;
  }
  if (sample_count < 3U || !has_noncollinear_samples) {
    return std::nullopt;
  }

  for (std::size_t column = 0U; column < 3U; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1U; row < 3U; ++row) {
      if (std::abs(matrix[row][column]) >
          std::abs(matrix[pivot][column])) {
        pivot = row;
      }
    }
    if (matrix[pivot][column] == 0.0L) {
      return std::nullopt;
    }
    if (pivot != column) {
      std::swap(matrix[pivot], matrix[column]);
    }
    const long double divisor = matrix[column][column];
    for (std::size_t entry = column; entry < 4U; ++entry) {
      matrix[column][entry] /= divisor;
    }
    for (std::size_t row = 0U; row < 3U; ++row) {
      if (row == column) {
        continue;
      }
      const long double factor = matrix[row][column];
      for (std::size_t entry = column; entry < 4U; ++entry) {
        matrix[row][entry] -= factor * matrix[column][entry];
      }
    }
  }
  const Plane plane{.x_slope = matrix[0][3],
                    .y_slope = matrix[1][3],
                    .offset = matrix[2][3]};
  if (!std::isfinite(plane.x_slope) || !std::isfinite(plane.y_slope) ||
      !std::isfinite(plane.offset)) {
    return std::nullopt;
  }
  return plane;
}

class PatchedElevationView final : public ElevationRangeView {
 public:
  PatchedElevationView(std::shared_ptr<const ElevationSnapshot> base,
                       const std::set<GridIndex>& patch_cells,
                       const Plane plane, const Point2 anchor) noexcept
      : base_(std::move(base)),
        patch_cells_(patch_cells),
        plane_(plane),
        anchor_(anchor) {}

  [[nodiscard]] const SparseGridGeometry& geometry() const noexcept override {
    return base_->geometry();
  }

  [[nodiscard]] std::optional<ElevationRange> ElevationRangeAt(
      const GridIndex index) const noexcept override {
    if (const std::optional<ElevationRange> measured =
            base_->ElevationRangeAt(index)) {
      return measured;
    }
    if (!patch_cells_.contains(index)) {
      return std::nullopt;
    }
    const Point2 center = CellCenter(base_->geometry(), index);
    const long double fitted =
        plane_.x_slope * static_cast<long double>(center.x - anchor_.x) +
        plane_.y_slope * static_cast<long double>(center.y - anchor_.y) +
        plane_.offset;
    if (!std::isfinite(fitted) ||
        std::abs(fitted) >
            static_cast<long double>(std::numeric_limits<float>::max())) {
      return std::nullopt;
    }
    const float elevation_m = static_cast<float>(fitted);
    return ElevationRange{.min_m = elevation_m, .max_m = elevation_m};
  }

  [[nodiscard]] std::optional<LocalTerrainMeasurements> TerrainMeasurementsAt(
      const GridIndex index) const noexcept override {
    return base_->TerrainMeasurementsAt(index);
  }

 private:
  std::shared_ptr<const ElevationSnapshot> base_;
  const std::set<GridIndex>& patch_cells_;
  Plane plane_;
  Point2 anchor_;
};

[[nodiscard]] bool HasEvidenceExit(
    const RequestLocalPlanningView& view, const GridIndex start) {
  constexpr std::array<GridIndex, 8> kNeighbors{{
      {.x = -1, .y = -1}, {.x = 0, .y = -1}, {.x = 1, .y = -1},
      {.x = -1, .y = 0},                         {.x = 1, .y = 0},
      {.x = -1, .y = 1},  {.x = 0, .y = 1},  {.x = 1, .y = 1},
  }};
  std::queue<GridIndex> frontier;
  std::set<GridIndex> visited;
  frontier.push(start);
  visited.insert(start);
  while (!frontier.empty()) {
    const GridIndex current = frontier.front();
    frontier.pop();
    for (const GridIndex offset : kNeighbors) {
      const std::optional<GridIndex> next = detail::OffsetWithinGeometry(
          view.geometry(), current, offset.x, offset.y);
      if (!next) {
        continue;
      }
      if (view.CanBeEndpoint(*next)) {
        return true;
      }
      if (view.Source(*next) != LocalCellSource::kStartAssumedFree ||
          !visited.insert(*next).second) {
        continue;
      }
      frontier.push(*next);
    }
  }
  return false;
}

}  // namespace

StartPatchResult RequestLocalStartPatchBuilder::Build(
    std::shared_ptr<const FineTraversabilitySnapshot> fine, const Pose2& p0,
    const PlatformCapability& capability,
    const TraversabilityProfile& profile) const {
  if (!fine) {
    throw std::invalid_argument("start patch requires a fine snapshot");
  }
  return Build(fine, fine->geometry(), p0, capability, profile);
}

StartPatchResult RequestLocalStartPatchBuilder::Build(
    std::shared_ptr<const FineTraversabilitySnapshot> fine,
    const SparseGridGeometry& local_window, const Pose2& p0,
    const PlatformCapability& capability,
    const TraversabilityProfile& profile) const {
  if (!fine) {
    throw std::invalid_argument("start patch requires a fine snapshot");
  }
  if (!IsLocalPlanningWindowFor(local_window, fine->geometry())) {
    throw std::invalid_argument(
        "start patch requires a local window on the base lattice");
  }
  if (!std::isfinite(p0.position_m.x) || !std::isfinite(p0.position_m.y) ||
      !std::isfinite(p0.yaw_rad) ||
      !std::isfinite(profile.start_blind_zone_margin_m) ||
      profile.start_blind_zone_margin_m < 0.0) {
    throw std::invalid_argument(
        "start patch pose and margin must be finite and nonnegative");
  }
  const double patch_radius_m = fine->hard_inflation_radius_m() +
                                profile.start_blind_zone_margin_m;
  if (!std::isfinite(patch_radius_m)) {
    throw std::invalid_argument("start patch radius must be finite");
  }
  const std::optional<GridIndex> start =
      WorldToCell(local_window, p0.position_m);
  if (!start) {
    return {.status = StartPatchResult::Status::kUnresolved};
  }

  if (fine->State(*start) == FineCellState::kBlocked) {
    return {.status = StartPatchResult::Status::kStartBlocked};
  }
  if (fine->State(*start) == FineCellState::kFree) {
    return {
        .status = StartPatchResult::Status::kNotNeeded,
        .view = std::make_shared<const RequestLocalPlanningView>(
            fine, local_window, p0, patch_radius_m,
            std::vector<LocalCellOverride>{}),
    };
  }

  const std::unique_ptr<const PlatformElevationEvaluator> evaluator =
      MakePlatformElevationEvaluator(capability);
  const std::set<GridIndex> patch_cells =
      PatchCells(local_window, p0.position_m, patch_radius_m);
  const std::optional<Plane> plane = FitSupportPlane(
      *fine->elevation(), *evaluator,
      WithOneCellRing(local_window, patch_cells), p0.position_m);
  if (!plane) {
    return {.status = StartPatchResult::Status::kUnresolved};
  }
  const PatchedElevationView patched(fine->elevation(), patch_cells, *plane,
                                     p0.position_m);
  FineCellEvaluator patched_evaluator(patched, capability, profile);

  std::vector<LocalCellOverride> overrides;
  overrides.reserve(patch_cells.size());
  FineCellState patched_start_state = FineCellState::kUnknown;
  for (const GridIndex cell : patch_cells) {
    const FineCellEvaluation evaluation = patched_evaluator.Evaluate(cell);
    if (cell == *start) {
      patched_start_state = evaluation.state;
    }
    if (fine->State(cell) == FineCellState::kUnknown &&
        evaluation.state == FineCellState::kFree) {
      overrides.push_back(LocalCellOverride{
          .index = cell,
          .source = LocalCellSource::kStartAssumedFree,
          .traversal_cost = evaluation.traversal_cost,
      });
    }
  }
  if (patched_start_state == FineCellState::kBlocked) {
    return {.status = StartPatchResult::Status::kStartBlocked};
  }
  if (patched_start_state != FineCellState::kFree) {
    return {.status = StartPatchResult::Status::kUnresolved};
  }

  auto view = std::make_shared<const RequestLocalPlanningView>(
      fine, local_window, p0, patch_radius_m, std::move(overrides));
  const std::size_t assumed_cells = view->overrides().size();
  if (!HasEvidenceExit(*view, *start)) {
    return {.status = StartPatchResult::Status::kUnresolved,
            .assumed_cells = assumed_cells};
  }
  return {.status = StartPatchResult::Status::kReady,
          .view = std::move(view),
          .assumed_cells = assumed_cells};
}

StartConnectionsResult RequestLocalStartPatchBuilder::BuildStartConnections(
    std::shared_ptr<const FineTraversabilitySnapshot> fine,
    const SparseGridGeometry& local_window, const Pose2& p0,
    const PlatformCapability& capability,
    const TraversabilityProfile& profile) const {
  const StartPatchResult patch = Build(fine, local_window, p0, capability, profile);
  StartConnectionsResult result{.status = patch.status};
  if (!patch.view || (patch.status != StartPatchResult::Status::kReady &&
                      patch.status != StartPatchResult::Status::kNotNeeded)) {
    return result;
  }
  const std::optional<GridIndex> start = WorldToCell(local_window, p0.position_m);
  if (!start) return result;
  const StartPhase initial = patch.view->Source(*start) == LocalCellSource::kEvidenceFree
                                 ? StartPhase::kNormal : StartPhase::kStartPrefix;
  if (initial == StartPhase::kNormal) {
    result.connections.push_back({.index = *start, .phase = initial});
    return result;
  }
  constexpr std::array<GridIndex, 8> kNeighbors{{
      {.x = -1, .y = -1}, {.x = 0, .y = -1}, {.x = 1, .y = -1},
      {.x = -1, .y = 0},                         {.x = 1, .y = 0},
      {.x = -1, .y = 1},  {.x = 0, .y = 1},  {.x = 1, .y = 1},
  }};
  std::queue<std::pair<GridIndex, StartPhase>> frontier;
  std::set<GridIndex> visited;
  frontier.push({*start, initial});
  visited.insert(*start);
  while (!frontier.empty()) {
    const auto [current, phase] = frontier.front();
    frontier.pop();
    for (const GridIndex offset : kNeighbors) {
      const auto next = detail::OffsetWithinGeometry(local_window, current,
                                                     offset.x, offset.y);
      if (!next || visited.contains(*next)) continue;
      const auto next_phase = patch.view->AdvanceGridStep(phase, current, *next);
      if (!next_phase) continue;
      visited.insert(*next);
      if (*next_phase == StartPhase::kNormal) {
        result.connections.push_back({.index = *next, .phase = *next_phase});
      } else {
        frontier.push({*next, *next_phase});
      }
    }
  }
  if (result.connections.empty()) result.status = StartPatchResult::Status::kUnresolved;
  return result;
}

}  // namespace lunar::incremental_navigation
