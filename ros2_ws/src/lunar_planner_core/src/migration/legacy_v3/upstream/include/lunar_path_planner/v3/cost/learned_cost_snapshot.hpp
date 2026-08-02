#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/contracts/base_types.hpp"
#include "lunar_path_planner/v3/contracts/profiles.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lunar::planning::v3 {

struct LearnedCostSnapshotInput final {
  ContentRef snapshot_ref;
  ContentRef model_ref;
  ContentRef source_map_snapshot_ref;
  std::uint32_t source_map_revision{};
  std::string frame_id;
  std::string feature_contract_id;
  std::string output_contract_id;
  ClockStamp generated_at;
  Interval certified_energy_correction_bounds;
  Interval certified_nonfatal_risk_correction_bounds;
  std::vector<float> energy_correction;
  std::vector<float> nonfatal_risk_correction;
};

class LearnedCostSnapshot final {
 public:
  [[nodiscard]] static Result<
      std::shared_ptr<const LearnedCostSnapshot>>
  Create(const LearnedCostSnapshotInput& input,
         std::size_t expected_cell_count);

  [[nodiscard]] const ContentRef& snapshot_ref() const noexcept;
  [[nodiscard]] const ContentRef& model_ref() const noexcept;
  [[nodiscard]] const ContentRef& source_map_snapshot_ref() const noexcept;
  [[nodiscard]] std::uint32_t source_map_revision() const noexcept;
  [[nodiscard]] const std::string& frame_id() const noexcept;
  [[nodiscard]] const Interval&
  certified_energy_correction_bounds() const noexcept;
  [[nodiscard]] const Interval&
  certified_nonfatal_risk_correction_bounds() const noexcept;
  [[nodiscard]] std::span<const float> EnergyCorrection() const noexcept;
  [[nodiscard]] std::span<const float>
  NonfatalRiskCorrection() const noexcept;

 private:
  struct Storage;

  explicit LearnedCostSnapshot(std::shared_ptr<const Storage> storage);

  std::shared_ptr<const Storage> storage_;
};

enum class SoftCostSource {
  kAnalyticOnly,
  kAnalyticPlusPinnedLearned,
};

struct ResolvedSoftCost final {
  std::vector<float> energy;
  std::vector<float> nonfatal_risk;
  SoftCostSource source{SoftCostSource::kAnalyticOnly};
  std::optional<std::string> fallback_reason_code;
};

[[nodiscard]] Result<ResolvedSoftCost> ComposeBoundedSoftCost(
    std::span<const float> analytic_energy,
    std::span<const float> analytic_nonfatal_risk,
    const std::shared_ptr<const LearnedCostSnapshot>& learned,
    const ImmutableMapSnapshot& map,
    const PlannerAlgorithmConfig& config);

}  // namespace lunar::planning::v3
