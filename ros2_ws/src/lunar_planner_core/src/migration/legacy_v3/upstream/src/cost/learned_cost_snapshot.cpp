#include "lunar_path_planner/v3/cost/learned_cost_snapshot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr std::uint32_t kMaximumRevision = 2'147'483'647U;

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool IsIdentifierCharacter(char value) noexcept {
  const auto byte = static_cast<unsigned char>(value);
  return (byte >= static_cast<unsigned char>('A') &&
          byte <= static_cast<unsigned char>('Z')) ||
         (byte >= static_cast<unsigned char>('a') &&
          byte <= static_cast<unsigned char>('z')) ||
         (byte >= static_cast<unsigned char>('0') &&
          byte <= static_cast<unsigned char>('9')) ||
         value == '.' || value == '_' || value == ':' || value == '/' ||
         value == '-';
}

[[nodiscard]] bool IsAsciiAlphanumeric(char value) noexcept {
  const auto byte = static_cast<unsigned char>(value);
  return (byte >= static_cast<unsigned char>('A') &&
          byte <= static_cast<unsigned char>('Z')) ||
         (byte >= static_cast<unsigned char>('a') &&
          byte <= static_cast<unsigned char>('z')) ||
         (byte >= static_cast<unsigned char>('0') &&
          byte <= static_cast<unsigned char>('9'));
}

[[nodiscard]] bool IsIdentifier(std::string_view value) noexcept {
  return !value.empty() && value.size() <= 128U &&
         IsAsciiAlphanumeric(value.front()) &&
         std::ranges::all_of(value, IsIdentifierCharacter);
}

[[nodiscard]] bool IsLowerSha256(std::string_view value) noexcept {
  return value.size() == 64U &&
         std::ranges::all_of(value, [](char digit) {
           return (digit >= '0' && digit <= '9') ||
                  (digit >= 'a' && digit <= 'f');
         });
}

[[nodiscard]] bool IsContentRef(const ContentRef& value) noexcept {
  return IsIdentifier(value.id) && value.revision >= 1U &&
         value.revision <= kMaximumRevision &&
         IsLowerSha256(value.content_hash);
}

[[nodiscard]] bool IsFiniteOrdered(const Interval& value) noexcept {
  return std::isfinite(value.lower) && std::isfinite(value.upper) &&
         value.lower <= value.upper;
}

[[nodiscard]] bool CorrectionsFit(
    std::span<const float> values,
    const Interval& certified_bounds) noexcept {
  return std::ranges::all_of(values, [&](float value) {
    const double widened = static_cast<double>(value);
    return std::isfinite(value) && widened >= certified_bounds.lower &&
           widened <= certified_bounds.upper;
  });
}

[[nodiscard]] bool AllFinite(std::span<const float> values) noexcept {
  return std::ranges::all_of(
      values, [](float value) { return std::isfinite(value); });
}

[[nodiscard]] ResolvedSoftCost AnalyticOnly(
    std::span<const float> analytic_energy,
    std::span<const float> analytic_nonfatal_risk,
    std::string reason_code) {
  return {
      .energy =
          std::vector<float>(analytic_energy.begin(), analytic_energy.end()),
      .nonfatal_risk = std::vector<float>(
          analytic_nonfatal_risk.begin(), analytic_nonfatal_risk.end()),
      .source = SoftCostSource::kAnalyticOnly,
      .fallback_reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool IsValidPolicyBound(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] double TightenCorrection(
    float correction,
    const Interval& certified_bounds,
    double maximum_absolute_correction) noexcept {
  const double certified = std::clamp(
      static_cast<double>(correction), certified_bounds.lower,
      certified_bounds.upper);
  return std::clamp(certified, -maximum_absolute_correction,
                    maximum_absolute_correction);
}

[[nodiscard]] bool CanStoreAsFiniteFloat(double value) noexcept {
  constexpr double kMaximumFloat =
      static_cast<double>(std::numeric_limits<float>::max());
  return std::isfinite(value) && value >= -kMaximumFloat &&
         value <= kMaximumFloat;
}

}  // namespace

struct LearnedCostSnapshot::Storage final {
  explicit Storage(const LearnedCostSnapshotInput& input)
      : snapshot_ref(input.snapshot_ref),
        model_ref(input.model_ref),
        source_map_snapshot_ref(input.source_map_snapshot_ref),
        source_map_revision(input.source_map_revision),
        frame_id(input.frame_id),
        feature_contract_id(input.feature_contract_id),
        output_contract_id(input.output_contract_id),
        generated_at(input.generated_at),
        certified_energy_correction_bounds(
            input.certified_energy_correction_bounds),
        certified_nonfatal_risk_correction_bounds(
            input.certified_nonfatal_risk_correction_bounds),
        energy_correction(input.energy_correction),
        nonfatal_risk_correction(input.nonfatal_risk_correction) {}

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

Result<std::shared_ptr<const LearnedCostSnapshot>>
LearnedCostSnapshot::Create(const LearnedCostSnapshotInput& input,
                            std::size_t expected_cell_count) {
  if (!IsContentRef(input.snapshot_ref)) {
    return Invalid("learned_cost_snapshot.snapshot_ref",
                   "snapshot_ref must be a valid ContentRef");
  }
  if (!IsContentRef(input.model_ref)) {
    return Invalid("learned_cost_snapshot.model_ref",
                   "model_ref must be a valid ContentRef");
  }
  if (!IsContentRef(input.source_map_snapshot_ref)) {
    return Invalid("learned_cost_snapshot.source_map_snapshot_ref",
                   "source_map_snapshot_ref must be a valid ContentRef");
  }
  if (input.source_map_revision < 1U ||
      input.source_map_revision > kMaximumRevision) {
    return Invalid("learned_cost_snapshot.source_map_revision",
                   "source_map_revision must be within the schema range");
  }
  if (!IsIdentifier(input.frame_id)) {
    return Invalid("learned_cost_snapshot.frame_id",
                   "frame_id must be a valid identifier");
  }
  if (!IsIdentifier(input.feature_contract_id)) {
    return Invalid("learned_cost_snapshot.feature_contract_id",
                   "feature_contract_id must be a valid identifier");
  }
  if (!IsIdentifier(input.output_contract_id)) {
    return Invalid("learned_cost_snapshot.output_contract_id",
                   "output_contract_id must be a valid identifier");
  }
  if (!IsIdentifier(input.generated_at.clock_id)) {
    return Invalid("learned_cost_snapshot.generated_at.clock_id",
                   "generated_at clock_id must be a valid identifier");
  }
  if (expected_cell_count == 0U) {
    return Invalid("learned_cost_snapshot.expected_cell_count",
                   "expected cell count must be positive");
  }
  if (input.energy_correction.size() != expected_cell_count) {
    return Invalid("learned_cost_snapshot.energy_correction",
                   "energy correction shape must match the expected grid");
  }
  if (input.nonfatal_risk_correction.size() != expected_cell_count) {
    return Invalid(
        "learned_cost_snapshot.nonfatal_risk_correction",
        "nonfatal-risk correction shape must match the expected grid");
  }
  if (!IsFiniteOrdered(input.certified_energy_correction_bounds)) {
    return Invalid(
        "learned_cost_snapshot.certified_energy_correction_bounds",
        "certified energy bounds must be finite and ordered");
  }
  if (!IsFiniteOrdered(
          input.certified_nonfatal_risk_correction_bounds)) {
    return Invalid(
        "learned_cost_snapshot.certified_nonfatal_risk_correction_bounds",
        "certified nonfatal-risk bounds must be finite and ordered");
  }
  if (!CorrectionsFit(
          input.energy_correction,
          input.certified_energy_correction_bounds)) {
    return Invalid(
        "learned_cost_snapshot.energy_correction",
        "each energy correction must be finite and certified");
  }
  if (!CorrectionsFit(
          input.nonfatal_risk_correction,
          input.certified_nonfatal_risk_correction_bounds)) {
    return Invalid(
        "learned_cost_snapshot.nonfatal_risk_correction",
        "each nonfatal-risk correction must be finite and certified");
  }

  auto storage = std::make_shared<const Storage>(input);
  return std::shared_ptr<const LearnedCostSnapshot>(
      new LearnedCostSnapshot(std::move(storage)));
}

LearnedCostSnapshot::LearnedCostSnapshot(
    std::shared_ptr<const Storage> storage)
    : storage_(std::move(storage)) {}

const ContentRef& LearnedCostSnapshot::snapshot_ref() const noexcept {
  return storage_->snapshot_ref;
}

const ContentRef& LearnedCostSnapshot::model_ref() const noexcept {
  return storage_->model_ref;
}

const ContentRef&
LearnedCostSnapshot::source_map_snapshot_ref() const noexcept {
  return storage_->source_map_snapshot_ref;
}

std::uint32_t LearnedCostSnapshot::source_map_revision() const noexcept {
  return storage_->source_map_revision;
}

const std::string& LearnedCostSnapshot::frame_id() const noexcept {
  return storage_->frame_id;
}

const Interval&
LearnedCostSnapshot::certified_energy_correction_bounds()
    const noexcept {
  return storage_->certified_energy_correction_bounds;
}

const Interval&
LearnedCostSnapshot::certified_nonfatal_risk_correction_bounds()
    const noexcept {
  return storage_->certified_nonfatal_risk_correction_bounds;
}

std::span<const float>
LearnedCostSnapshot::EnergyCorrection() const noexcept {
  return storage_->energy_correction;
}

std::span<const float>
LearnedCostSnapshot::NonfatalRiskCorrection() const noexcept {
  return storage_->nonfatal_risk_correction;
}

Result<ResolvedSoftCost> ComposeBoundedSoftCost(
    std::span<const float> analytic_energy,
    std::span<const float> analytic_nonfatal_risk,
    const std::shared_ptr<const LearnedCostSnapshot>& learned,
    const ImmutableMapSnapshot& map,
    const PlannerAlgorithmConfig& config) {
  const std::size_t expected_cell_count = map.geometry().CellCount();
  if (expected_cell_count == 0U ||
      analytic_energy.size() != expected_cell_count ||
      analytic_nonfatal_risk.size() != expected_cell_count ||
      !AllFinite(analytic_energy) ||
      !AllFinite(analytic_nonfatal_risk)) {
    return Invalid("analytic_cost", "analytic_cost_invalid");
  }

  if (!learned) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_snapshot_missing");
  }
  if (config.learned_cost_policy.mode !=
      LearnedCostPolicy::Mode::kOptionalBoundedSoftCost) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_cost_disabled");
  }
  if (learned->source_map_snapshot_ref() != map.snapshot_ref()) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_map_snapshot_mismatch");
  }
  if (learned->source_map_revision() != map.map_revision()) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_map_revision_mismatch");
  }
  if (learned->frame_id() != map.frame_id()) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_frame_mismatch");
  }
  if (!config.learned_cost_model_ref.has_value()) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_model_unconfigured");
  }
  if (*config.learned_cost_model_ref != learned->model_ref()) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_model_mismatch");
  }
  if (learned->EnergyCorrection().size() != expected_cell_count ||
      learned->NonfatalRiskCorrection().size() !=
          expected_cell_count) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_shape_mismatch");
  }

  const double maximum_energy =
      config.learned_cost_policy.maximum_absolute_energy_correction;
  const double maximum_risk =
      config.learned_cost_policy
          .maximum_absolute_nonfatal_risk_correction;
  if (!IsValidPolicyBound(maximum_energy) ||
      !IsValidPolicyBound(maximum_risk)) {
    return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                        "learned_policy_bounds_invalid");
  }

  std::vector<float> resolved_energy;
  std::vector<float> resolved_risk;
  resolved_energy.reserve(expected_cell_count);
  resolved_risk.reserve(expected_cell_count);
  for (std::size_t index = 0U; index < expected_cell_count; ++index) {
    const double energy_correction = TightenCorrection(
        learned->EnergyCorrection()[index],
        learned->certified_energy_correction_bounds(), maximum_energy);
    const double risk_correction = TightenCorrection(
        learned->NonfatalRiskCorrection()[index],
        learned->certified_nonfatal_risk_correction_bounds(),
        maximum_risk);
    const double energy =
        static_cast<double>(analytic_energy[index]) + energy_correction;
    const double risk =
        static_cast<double>(analytic_nonfatal_risk[index]) +
        risk_correction;
    if (!CanStoreAsFiniteFloat(energy) ||
        !CanStoreAsFiniteFloat(risk)) {
      return AnalyticOnly(analytic_energy, analytic_nonfatal_risk,
                          "learned_composition_non_finite");
    }
    resolved_energy.push_back(static_cast<float>(energy));
    resolved_risk.push_back(static_cast<float>(risk));
  }

  return ResolvedSoftCost{
      .energy = std::move(resolved_energy),
      .nonfatal_risk = std::move(resolved_risk),
      .source = SoftCostSource::kAnalyticPlusPinnedLearned,
      .fallback_reason_code = std::nullopt,
  };
}

}  // namespace lunar::planning::v3
