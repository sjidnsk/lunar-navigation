#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lunar_pure_exploration_core/candidate_generator.hpp"

namespace lunar::pure_exploration {

enum class PersistentFailureReason : std::uint8_t {
  kExecutionReplansExhausted,
  kNavigationNoPath,
  kNavigationTimeout,
};

struct FailureMemoryLimits {
  std::size_t maximum_entries;
  std::size_t maximum_patch_cells_per_entry;
  std::size_t maximum_total_patch_cells;
};

class FailureMemory {
 public:
  FailureMemory(double platform_length_m, FailureMemoryLimits limits);

  void BeginTask(std::string task_id);
  void RecordPersistentFailure(const CandidateView& candidate,
                               PersistentFailureReason reason,
                               const TaskRaster& raster);
  bool IsSuppressed(const CandidateView& candidate, const TaskRaster& raster);
  std::size_t size() const;

 private:
  struct PatchCell {
    GridIndex index;
    CellState state;
    bool operator==(const PatchCell&) const = default;
  };

  struct Entry {
    CandidateKey key;
    Vec2 world_center;
    double failure_radius_m;
    GridGeometry geometry;
    std::vector<PatchCell> patch;
  };

  static std::vector<PatchCell> BuildPatch(
      Vec2 world_center, double failure_radius_m, const TaskRaster& raster,
      std::size_t maximum_work_units);
  static bool SameGeometry(const GridGeometry& left,
                           const GridGeometry& right);
  void EraseEntry(std::size_t index);

  double platform_length_m_;
  FailureMemoryLimits limits_;
  std::string task_id_;
  std::vector<Entry> entries_;
  std::size_t total_patch_cells_{0U};
};

}  // namespace lunar::pure_exploration
