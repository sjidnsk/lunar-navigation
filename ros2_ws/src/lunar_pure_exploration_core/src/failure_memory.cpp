#include "lunar_pure_exploration_core/failure_memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lunar::pure_exploration {
namespace {

bool IsFinitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

void ValidateGeometry(const GridGeometry& geometry) {
  if (!IsFinitePositive(geometry.resolution) ||
      !std::isfinite(geometry.origin_x) ||
      !std::isfinite(geometry.origin_y) ||
      !std::isfinite(geometry.origin_yaw)) {
    throw std::invalid_argument("failure patch geometry must be finite");
  }
}

std::int32_t CheckedGridBound(long double value) {
  if (!std::isfinite(value) ||
      value < static_cast<long double>(
                  std::numeric_limits<std::int32_t>::min()) ||
      value > static_cast<long double>(
                  std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("failure patch bound exceeds GridIndex range");
  }
  return static_cast<std::int32_t>(value);
}

}  // namespace

FailureMemory::FailureMemory(double platform_length_m,
                             FailureMemoryLimits limits)
    : platform_length_m_(platform_length_m), limits_(limits) {
  if (!IsFinitePositive(platform_length_m_)) {
    throw std::invalid_argument("platform length must be finite and positive");
  }
  if (limits_.maximum_entries == 0U ||
      limits_.maximum_patch_cells_per_entry == 0U ||
      limits_.maximum_total_patch_cells == 0U) {
    throw std::invalid_argument("failure memory limits must be positive");
  }
}

void FailureMemory::BeginTask(std::string task_id) {
  if (task_id.empty()) {
    throw std::invalid_argument("failure memory task id must not be empty");
  }
  task_id_.swap(task_id);
  entries_.clear();
  total_patch_cells_ = 0U;
}

std::vector<FailureMemory::PatchCell> FailureMemory::BuildPatch(
    Vec2 world_center, double failure_radius_m, const TaskRaster& raster,
    std::size_t maximum_work_units) {
  if (!std::isfinite(world_center.x) || !std::isfinite(world_center.y)) {
    throw std::invalid_argument("failure center must be finite");
  }
  const GridGeometry& geometry = raster.geometry();
  ValidateGeometry(geometry);
  if (!IsFinitePositive(failure_radius_m)) {
    throw std::overflow_error("failure radius must be finite and positive");
  }

  const double radius_cells = failure_radius_m / geometry.resolution;
  if (!IsFinitePositive(radius_cells)) {
    throw std::overflow_error(
        "failure radius in cells must be finite and positive");
  }
  const auto grid_center = raster.WorldToGrid(world_center);
  if (!grid_center.has_value() || !std::isfinite(grid_center->x) ||
      !std::isfinite(grid_center->y)) {
    throw std::overflow_error("failure center grid transform overflow");
  }

  const long double center_x = static_cast<long double>(grid_center->x);
  const long double center_y = static_cast<long double>(grid_center->y);
  const long double promoted_radius =
      static_cast<long double>(radius_cells);
  const long double minimum_x =
      std::ceil(center_x - promoted_radius - 0.5L);
  const long double maximum_x =
      std::floor(center_x + promoted_radius - 0.5L);
  const long double minimum_y =
      std::ceil(center_y - promoted_radius - 0.5L);
  const long double maximum_y =
      std::floor(center_y + promoted_radius - 0.5L);
  const long double width = maximum_x - minimum_x + 1.0L;
  const long double height = maximum_y - minimum_y + 1.0L;
  const long double limit =
      static_cast<long double>(maximum_work_units);
  if (!std::isfinite(width) || !std::isfinite(height) || width <= 0.0L ||
      height <= 0.0L || width > limit || height > limit) {
    throw std::length_error("failure patch theoretical work exceeds limit");
  }
  const long double theoretical_work = width * height;
  if (!std::isfinite(theoretical_work) || theoretical_work > limit) {
    throw std::length_error("failure patch theoretical work exceeds limit");
  }

  const std::int32_t lower_x = CheckedGridBound(minimum_x);
  const std::int32_t upper_x = CheckedGridBound(maximum_x);
  const std::int32_t lower_y = CheckedGridBound(minimum_y);
  const std::int32_t upper_y = CheckedGridBound(maximum_y);
  const std::uint64_t exact_width = static_cast<std::uint64_t>(
      static_cast<std::int64_t>(upper_x) - lower_x + 1);
  const std::uint64_t exact_height = static_cast<std::uint64_t>(
      static_cast<std::int64_t>(upper_y) - lower_y + 1);
  if (exact_height != 0U &&
      exact_width >
          std::numeric_limits<std::size_t>::max() / exact_height) {
    throw std::length_error("failure patch theoretical work exceeds limit");
  }
  const std::size_t work_count =
      static_cast<std::size_t>(exact_width) *
      static_cast<std::size_t>(exact_height);
  if (work_count > maximum_work_units) {
    throw std::length_error("failure patch theoretical work exceeds limit");
  }

  std::vector<PatchCell> patch;
  if (work_count > patch.max_size()) {
    throw std::overflow_error("failure patch exceeds storage capacity");
  }
  patch.reserve(work_count);
  std::size_t work_used = 0U;
  const long double radius_squared = promoted_radius * promoted_radius;
  for (std::int64_t y = lower_y; y <= upper_y; ++y) {
    for (std::int64_t x = lower_x; x <= upper_x; ++x) {
      if (work_used == maximum_work_units) {
        throw std::length_error("failure patch work budget exhausted");
      }
      ++work_used;
      const long double dx =
          static_cast<long double>(x) + 0.5L - center_x;
      const long double dy =
          static_cast<long double>(y) + 0.5L - center_y;
      if (dx * dx + dy * dy > radius_squared) {
        continue;
      }
      if (patch.size() == maximum_work_units) {
        throw std::length_error("failure patch cell limit exhausted");
      }
      const GridIndex index{static_cast<std::int32_t>(x),
                            static_cast<std::int32_t>(y)};
      patch.push_back(PatchCell{index, raster.Classify(index)});
    }
  }
  if (work_used != work_count) {
    throw std::logic_error("failure patch work accounting mismatch");
  }
  return patch;
}

bool FailureMemory::SameGeometry(const GridGeometry& left,
                                 const GridGeometry& right) {
  return left.width == right.width && left.height == right.height &&
         left.resolution == right.resolution &&
         left.origin_x == right.origin_x && left.origin_y == right.origin_y &&
         left.origin_yaw == right.origin_yaw;
}

void FailureMemory::RecordPersistentFailure(
    const CandidateView& candidate, PersistentFailureReason reason,
    const TaskRaster& raster) {
  switch (reason) {
    case PersistentFailureReason::kExecutionReplansExhausted:
    case PersistentFailureReason::kNavigationNoPath:
    case PersistentFailureReason::kNavigationTimeout:
      break;
    default:
      throw std::invalid_argument("unsupported persistent failure reason");
  }
  if (!std::isfinite(candidate.pose.x) || !std::isfinite(candidate.pose.y)) {
    throw std::invalid_argument("failure candidate world center must be finite");
  }
  const GridGeometry& geometry = raster.geometry();
  ValidateGeometry(geometry);
  const auto existing =
      std::find_if(entries_.begin(), entries_.end(),
                   [&candidate](const Entry& entry) {
                     return entry.key == candidate.key;
                   });
  const bool is_replacement = existing != entries_.end();
  Vec2 world_center{candidate.pose.x, candidate.pose.y};
  double failure_radius = 0.0;
  if (is_replacement) {
    world_center = existing->world_center;
    failure_radius = existing->failure_radius_m;
    if (!IsFinitePositive(failure_radius)) {
      throw std::logic_error("saved failure radius invariant violated");
    }
  } else {
    const double twice_resolution = 2.0 * geometry.resolution;
    if (!IsFinitePositive(twice_resolution)) {
      throw std::overflow_error(
          "twice failure-map resolution must be finite and positive");
    }
    failure_radius = std::max(platform_length_m_, twice_resolution);
    if (!IsFinitePositive(failure_radius)) {
      throw std::overflow_error("failure radius must be finite and positive");
    }
  }

  Entry replacement{
      candidate.key, world_center, failure_radius, geometry,
      BuildPatch(world_center, failure_radius, raster,
                 limits_.maximum_patch_cells_per_entry)};

  if (!is_replacement && entries_.size() >= limits_.maximum_entries) {
    throw std::length_error("failure entry limit exhausted");
  }
  const std::size_t old_patch_cells =
      is_replacement ? existing->patch.size() : 0U;
  if (old_patch_cells > total_patch_cells_) {
    throw std::logic_error("failure patch total invariant violated");
  }
  const std::size_t retained_patch_cells =
      total_patch_cells_ - old_patch_cells;
  if (retained_patch_cells > limits_.maximum_total_patch_cells ||
      replacement.patch.size() >
          limits_.maximum_total_patch_cells - retained_patch_cells) {
    throw std::length_error("failure total patch cell limit exhausted");
  }
  const std::size_t new_total =
      retained_patch_cells + replacement.patch.size();

  static_assert(std::is_nothrow_move_assignable_v<Entry>);
  static_assert(std::is_nothrow_move_constructible_v<Entry>);
  if (is_replacement) {
    *existing = std::move(replacement);
  } else {
    entries_.push_back(std::move(replacement));
  }
  total_patch_cells_ = new_total;
}

void FailureMemory::EraseEntry(std::size_t index) {
  if (index >= entries_.size()) {
    throw std::logic_error("failure erase index invariant violated");
  }
  const std::size_t removed_cells = entries_[index].patch.size();
  if (removed_cells > total_patch_cells_) {
    throw std::logic_error("failure patch total invariant violated");
  }
  total_patch_cells_ -= removed_cells;
  entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
}

bool FailureMemory::IsSuppressed(const CandidateView& candidate,
                                 const TaskRaster& raster) {
  const auto existing =
      std::find_if(entries_.begin(), entries_.end(),
                   [&candidate](const Entry& entry) {
                     return entry.key == candidate.key;
                   });
  if (existing == entries_.end()) {
    return false;
  }
  const std::size_t index =
      static_cast<std::size_t>(existing - entries_.begin());
  if (!SameGeometry(existing->geometry, raster.geometry())) {
    EraseEntry(index);
    return false;
  }

  const auto current_patch =
      BuildPatch(existing->world_center, existing->failure_radius_m, raster,
                 limits_.maximum_patch_cells_per_entry);
  if (current_patch != existing->patch) {
    EraseEntry(index);
    return false;
  }
  return true;
}

std::size_t FailureMemory::size() const { return entries_.size(); }

}  // namespace lunar::pure_exploration
