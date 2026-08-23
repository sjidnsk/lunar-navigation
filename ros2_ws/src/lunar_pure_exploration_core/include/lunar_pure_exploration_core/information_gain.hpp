#pragma once

#include <cstddef>
#include <cstdint>

#include "lunar_pure_exploration_core/candidate_generator.hpp"

namespace lunar::pure_exploration {

struct SensorModel {
  double range_m;
  double field_of_view_rad;
};

struct GainEvaluation {
  std::uint32_t visible_unknown_cells;
  double visible_unknown_area_m2;
};

class InformationGainEvaluator {
 public:
  struct Limits {
    std::size_t maximum_visibility_work_units;
  };

  InformationGainEvaluator(SensorModel model, Limits limits);
  GainEvaluation Evaluate(const TaskRaster& raster,
                          const CandidateView& candidate) const;

 private:
  SensorModel model_;
  Limits limits_;
};

}  // namespace lunar::pure_exploration
