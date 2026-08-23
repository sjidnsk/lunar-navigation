#pragma once

#include <cstdint>
#include <vector>

#include "lunar_pure_exploration_core/task_raster.hpp"

namespace lunar::pure_exploration {

struct FrontierCluster {
  std::uint64_t id;
  std::vector<GridIndex> cells;

  struct InterfaceEdge {
    GridIndex free_cell;
    GridIndex unknown_cell;
    std::uint8_t direction;
    Vec2 midpoint;
  };

  std::vector<InterfaceEdge> interface_edges;
  std::vector<std::int64_t> canonical_key;
  Vec2 centroid;
  double length_m;
};

enum class FrontierDetectionReason : std::uint8_t {
  kOk,
  kNoReachableFreeStart,
};

struct FrontierDetection {
  std::vector<FrontierCluster> clusters;
  std::uint32_t reachable_free_cell_count;
  bool has_reachable_free_start;
  FrontierDetectionReason reason;
};

struct FrontierParameters {
  double minimum_cluster_length_m;
};

class FrontierDetector {
 public:
  explicit FrontierDetector(FrontierParameters parameters);
  FrontierDetection Detect(const TaskRaster& raster,
                           GridIndex robot_cell) const;

 private:
  FrontierParameters parameters_;
};

}  // namespace lunar::pure_exploration
