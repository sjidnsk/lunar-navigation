#include "lunar_path_planner/v3/legged/legged_lattice.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <tuple>
#include <utility>

namespace lunar::planning::v3 {
namespace {

constexpr std::uint64_t kInvalidPoseKey =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kPoseIndexLimit = UINT64_C(1) << 24U;
constexpr std::uint64_t kYawIndexLimit = UINT64_C(1) << 16U;
constexpr double kIntervalTolerance = 1.0e-12;

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] double MaximumAbsolute(
    const Interval& interval) noexcept {
  return std::max(std::abs(interval.lower),
                  std::abs(interval.upper));
}

[[nodiscard]] std::int32_t WrapYawBin(
    std::int64_t value, std::size_t bin_count) noexcept {
  const auto count = static_cast<std::int64_t>(bin_count);
  const std::int64_t wrapped = ((value % count) + count) % count;
  return static_cast<std::int32_t>(wrapped);
}

[[nodiscard]] double YawForBin(
    std::int32_t bin, std::size_t bin_count) noexcept {
  return static_cast<double>(bin) *
         (2.0 * std::numbers::pi /
          static_cast<double>(bin_count));
}

[[nodiscard]] double CanonicalYaw(double yaw) noexcept {
  const double period = 2.0 * std::numbers::pi;
  double result =
      std::fmod(yaw + std::numbers::pi, period);
  if (result < 0.0) {
    result += period;
  }
  return result - std::numbers::pi;
}

[[nodiscard]] bool YawInside(
    double yaw, const CircularYawInterval& interval) noexcept {
  if (!std::isfinite(yaw) || !std::isfinite(interval.start_rad) ||
      !std::isfinite(interval.span_rad) ||
      interval.span_rad < 0.0 ||
      interval.span_rad > 2.0 * std::numbers::pi ||
      !interval.closed) {
    return false;
  }
  if (interval.span_rad >=
      2.0 * std::numbers::pi - kIntervalTolerance) {
    return true;
  }
  const double period = 2.0 * std::numbers::pi;
  double offset =
      std::fmod(CanonicalYaw(yaw) -
                    CanonicalYaw(interval.start_rad),
                period);
  if (offset < 0.0) {
    offset += period;
  }
  return offset <= interval.span_rad + kIntervalTolerance;
}

[[nodiscard]] std::string HexBits(double value) {
  constexpr char kDigits[] = "0123456789abcdef";
  std::uint64_t bits =
      std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value);
  std::string result(16U, '0');
  for (std::size_t index = result.size(); index > 0U; --index) {
    result[index - 1U] = kDigits[bits & 0xFU];
    bits >>= 4U;
  }
  return result;
}

[[nodiscard]] std::string StableEdgeId(
    const BodyMotionPrimitive& primitive,
    const LeggedLatticeState& target) {
  return primitive.id + ":" + std::to_string(target.ix) + ":" +
         std::to_string(target.iy) + ":" +
         std::to_string(target.iyaw) + ":" +
         HexBits(target.reachable_z.min_m) + ":" +
         HexBits(target.reachable_z.max_m);
}

[[nodiscard]] bool ValidGrid(const GridConfig& grid) noexcept {
  return std::isfinite(grid.xy_resolution_m) &&
         grid.xy_resolution_m > 0.0 && grid.yaw_bin_count > 0U &&
         grid.yaw_bin_count < kYawIndexLimit;
}

}  // namespace

bool HeightLabelDominates(
    DurationNanoseconds lhs_time,
    const HeightInterval& lhs_interval,
    DurationNanoseconds rhs_time,
    const HeightInterval& rhs_interval) noexcept {
  return lhs_time.value <= rhs_time.value &&
         IsValidHeightInterval(lhs_interval) &&
         IsValidHeightInterval(rhs_interval) &&
         lhs_interval.min_m <=
             rhs_interval.min_m + kIntervalTolerance &&
         lhs_interval.max_m >=
             rhs_interval.max_m - kIntervalTolerance;
}

std::vector<HeightLabelRecord> FilterNonDominatedHeightLabels(
    std::span<const HeightLabelRecord> labels) {
  std::vector<HeightLabelRecord> ordered(labels.begin(), labels.end());
  std::ranges::sort(
      ordered, [](const HeightLabelRecord& lhs,
                  const HeightLabelRecord& rhs) {
        return std::tuple{lhs.pose_key, lhs.arrival_time.value,
                          lhs.reachable_z.min_m,
                          lhs.reachable_z.max_m,
                          lhs.stable_label_id} <
               std::tuple{rhs.pose_key, rhs.arrival_time.value,
                          rhs.reachable_z.min_m,
                          rhs.reachable_z.max_m,
                          rhs.stable_label_id};
      });
  std::vector<HeightLabelRecord> result;
  result.reserve(ordered.size());
  for (std::size_t candidate = 0U; candidate < ordered.size();
       ++candidate) {
    if (!IsValidHeightInterval(ordered[candidate].reachable_z) ||
        ordered[candidate].arrival_time.value.count() < 0) {
      continue;
    }
    bool dominated = false;
    for (std::size_t other = 0U; other < ordered.size(); ++other) {
      if (candidate == other ||
          ordered[candidate].pose_key != ordered[other].pose_key ||
          !HeightLabelDominates(
              ordered[other].arrival_time,
              ordered[other].reachable_z,
              ordered[candidate].arrival_time,
              ordered[candidate].reachable_z)) {
        continue;
      }
      const bool equivalent =
          HeightLabelDominates(
              ordered[candidate].arrival_time,
              ordered[candidate].reachable_z,
              ordered[other].arrival_time,
              ordered[other].reachable_z);
      if (!equivalent ||
          ordered[other].stable_label_id <
              ordered[candidate].stable_label_id) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      result.push_back(ordered[candidate]);
    }
  }
  return result;
}

std::optional<LeggedLatticeState>
ApplyLeggedPrimitiveKinematics(
    const LeggedLatticeState& source,
    const BodyMotionPrimitive& primitive,
    const GridConfig& grid) noexcept {
  if (!ValidGrid(grid) ||
      !IsValidHeightInterval(source.reachable_z) ||
      source.iyaw < 0 ||
      static_cast<std::size_t>(source.iyaw) >=
          grid.yaw_bin_count ||
      !IsFinite(primitive.relative_end.position_m) ||
      !std::isfinite(primitive.relative_end.yaw_rad)) {
    return std::nullopt;
  }
  const double yaw = YawForBin(source.iyaw, grid.yaw_bin_count);
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  const double world_dx =
      cosine * primitive.relative_end.position_m.x -
      sine * primitive.relative_end.position_m.y;
  const double world_dy =
      sine * primitive.relative_end.position_m.x +
      cosine * primitive.relative_end.position_m.y;
  const long long delta_x =
      std::llround(world_dx / grid.xy_resolution_m);
  const long long delta_y =
      std::llround(world_dy / grid.xy_resolution_m);
  const double yaw_step =
      2.0 * std::numbers::pi /
      static_cast<double>(grid.yaw_bin_count);
  const long long delta_yaw =
      std::llround(primitive.relative_end.yaw_rad / yaw_step);
  const std::int64_t target_x =
      static_cast<std::int64_t>(source.ix) + delta_x;
  const std::int64_t target_y =
      static_cast<std::int64_t>(source.iy) + delta_y;
  if (target_x < std::numeric_limits<std::int32_t>::min() ||
      target_x > std::numeric_limits<std::int32_t>::max() ||
      target_y < std::numeric_limits<std::int32_t>::min() ||
      target_y > std::numeric_limits<std::int32_t>::max()) {
    return std::nullopt;
  }
  return LeggedLatticeState{
      .ix = static_cast<std::int32_t>(target_x),
      .iy = static_cast<std::int32_t>(target_y),
      .iyaw = WrapYawBin(
          static_cast<std::int64_t>(source.iyaw) + delta_yaw,
          grid.yaw_bin_count),
      .reachable_z = source.reachable_z,
  };
}

LeggedLatticeAdapter::LeggedLatticeAdapter(
    const SafeProjection& projection,
    std::span<const TerminalCandidate> terminals,
    GridConfig grid,
    const BodyMotionPrimitiveCatalog& catalog) noexcept
    : projection_(&projection),
      terminals_(terminals),
      grid_(grid),
      catalog_(&catalog) {}

StateKey LeggedLatticeAdapter::Key(
    const LeggedLatticeState& state) const noexcept {
  if (state.ix < 0 || state.iy < 0 || state.iyaw < 0 ||
      static_cast<std::uint64_t>(state.ix) >= kPoseIndexLimit ||
      static_cast<std::uint64_t>(state.iy) >= kPoseIndexLimit ||
      static_cast<std::uint64_t>(state.iyaw) >= kYawIndexLimit) {
    return kInvalidPoseKey;
  }
  return (static_cast<std::uint64_t>(state.ix) << 40U) |
         (static_cast<std::uint64_t>(state.iy) << 16U) |
         static_cast<std::uint64_t>(state.iyaw);
}

std::vector<SearchTransition<LeggedLatticeState, LeggedLatticeEdge>>
LeggedLatticeAdapter::Expand(
    const LeggedLatticeState& state) const {
  std::vector<
      SearchTransition<LeggedLatticeState, LeggedLatticeEdge>>
      transitions;
  if (projection_ == nullptr || catalog_ == nullptr ||
      !ValidGrid(grid_) ||
      Key(state) == kInvalidPoseKey ||
      !IsValidHeightInterval(state.reachable_z)) {
    return transitions;
  }

  const double source_yaw =
      YawForBin(state.iyaw, grid_.yaw_bin_count);
  const Vec2 source_position{
      .x = projection_->geometry.origin_m.x +
           (static_cast<double>(state.ix) + 0.5) *
               grid_.xy_resolution_m,
      .y = projection_->geometry.origin_m.y +
           (static_cast<double>(state.iy) + 0.5) *
               grid_.xy_resolution_m,
  };
  LeggedTerrainEvaluator evaluator(*projection_,
                                   catalog_->capability());

  for (const auto& primitive : catalog_->ordered_primitives()) {
    auto target =
        ApplyLeggedPrimitiveKinematics(state, primitive, grid_);
    if (!target.has_value() ||
        !projection_->InBounds({target->ix, target->iy})) {
      continue;
    }
    std::vector<LeggedTerrainEvaluation> samples;
    samples.reserve(primitive.normalized_samples.size());
    bool valid_samples = !primitive.normalized_samples.empty();
    double previous_sample = -1.0;
    for (const double normalized : primitive.normalized_samples) {
      if (!std::isfinite(normalized) || normalized < 0.0 ||
          normalized > 1.0 || normalized <= previous_sample) {
        valid_samples = false;
        break;
      }
      previous_sample = normalized;
      const double body_x =
          primitive.relative_end.position_m.x * normalized;
      const double body_y =
          primitive.relative_end.position_m.y * normalized;
      const PoseXyzYaw pose{
          .position_m =
              {
                  source_position.x +
                      std::cos(source_yaw) * body_x -
                      std::sin(source_yaw) * body_y,
                  source_position.y +
                      std::sin(source_yaw) * body_x +
                      std::cos(source_yaw) * body_y,
                  0.0,
              },
          .yaw_rad =
              source_yaw +
              primitive.relative_end.yaw_rad * normalized,
      };
      samples.push_back(evaluator.EvaluatePose(
          pose, catalog_->capability().collision_envelope));
    }
    if (!valid_samples ||
        primitive.normalized_samples.front() != 0.0 ||
        primitive.normalized_samples.back() != 1.0) {
      continue;
    }
    const auto reachable = evaluator.PropagateEdgeInterval(
        state.reachable_z, primitive, samples);
    if (!reachable.has_value()) {
      continue;
    }
    target->reachable_z = *reachable;

    DurationNanoseconds transition_time =
        primitive.nominal_duration;
    SecondaryCostVector costs{};
    for (const auto& sample : samples) {
      transition_time.value =
          std::max(transition_time.value,
                   sample.terrain_scaled_duration.value);
      costs += sample.secondary_costs;
    }
    const double sample_count =
        static_cast<double>(samples.size());
    costs.energy /= sample_count;
    costs.risk /= sample_count;
    costs.smoothness =
        std::abs(primitive.relative_end.yaw_rad);
    LeggedLatticeEdge edge{
        .source = state,
        .target = *target,
        .primitive_id = primitive.id,
        .motion_kind = primitive.kind,
        .sampled_body_sweep_ref =
            primitive.sampled_body_sweep_ref,
        .transition_time = transition_time,
        .secondary_costs = costs,
    };
    transitions.push_back(
        SearchTransition<LeggedLatticeState, LeggedLatticeEdge>{
            .stable_edge_id = StableEdgeId(primitive, *target),
            .successor = *target,
            .edge = std::move(edge),
        });
  }
  return transitions;
}

bool LeggedLatticeAdapter::HardFeasible(
    const SearchTransition<LeggedLatticeState,
                           LeggedLatticeEdge>& transition) const {
  if (projection_ == nullptr ||
      !projection_->InBounds(
          {transition.successor.ix, transition.successor.iy}) ||
      !projection_->HardFeasible(
          {transition.successor.ix, transition.successor.iy}) ||
      !IsValidHeightInterval(transition.successor.reachable_z) ||
      transition.edge.transition_time.value.count() < 0) {
    return false;
  }
  const auto& costs = transition.edge.secondary_costs;
  return std::isfinite(costs.energy) && costs.energy >= 0.0 &&
         std::isfinite(costs.risk) && costs.risk >= 0.0 &&
         std::isfinite(costs.smoothness) && costs.smoothness >= 0.0;
}

DurationNanoseconds LeggedLatticeAdapter::TransitionTime(
    const SearchTransition<LeggedLatticeState,
                           LeggedLatticeEdge>& transition) const noexcept {
  return transition.edge.transition_time;
}

DurationNanoseconds
LeggedLatticeAdapter::AdmissibleTimeHeuristic(
    const LeggedLatticeState& state,
    const SearchProblem<LeggedLatticeState>& problem) const noexcept {
  static_cast<void>(problem);
  if (catalog_ == nullptr || terminals_.empty() || !ValidGrid(grid_)) {
    return {};
  }
  const auto& envelope = catalog_->capability().velocity_envelope;
  const double maximum_planar_speed =
      std::hypot(MaximumAbsolute(envelope.forward_mps),
                 MaximumAbsolute(envelope.lateral_mps));
  if (!std::isfinite(maximum_planar_speed) ||
      maximum_planar_speed <= 0.0) {
    return {};
  }
  double minimum_seconds = std::numeric_limits<double>::infinity();
  for (const auto& terminal : terminals_) {
    const double dx =
        static_cast<double>(terminal.cell.x - state.ix) *
        grid_.xy_resolution_m;
    const double dy =
        static_cast<double>(terminal.cell.y - state.iy) *
        grid_.xy_resolution_m;
    minimum_seconds =
        std::min(minimum_seconds,
                 std::hypot(dx, dy) / maximum_planar_speed);
  }
  if (!std::isfinite(minimum_seconds) || minimum_seconds <= 0.0) {
    return {};
  }
  const long double nanoseconds =
      static_cast<long double>(minimum_seconds) * 1.0e9L;
  const auto bounded = static_cast<std::int64_t>(
      std::min(
          nanoseconds,
          static_cast<long double>(
              std::numeric_limits<std::int64_t>::max())));
  return DurationNanoseconds{std::chrono::nanoseconds{bounded}};
}

SecondaryCostVector LeggedLatticeAdapter::SecondaryCosts(
    const SearchTransition<LeggedLatticeState,
                           LeggedLatticeEdge>& transition) const noexcept {
  return transition.edge.secondary_costs;
}

bool LeggedLatticeAdapter::IsTerminal(
    const LeggedLatticeState& state,
    const SearchProblem<LeggedLatticeState>& problem) const noexcept {
  static_cast<void>(problem);
  if (!IsValidHeightInterval(state.reachable_z) ||
      !ValidGrid(grid_)) {
    return false;
  }
  const double yaw = YawForBin(state.iyaw, grid_.yaw_bin_count);
  return std::ranges::any_of(
      terminals_, [&](const TerminalCandidate& terminal) {
        return terminal.cell.x == state.ix &&
               terminal.cell.y == state.iy &&
               (!terminal.yaw_interval.has_value() ||
                YawInside(yaw, *terminal.yaw_interval));
      });
}

}  // namespace lunar::planning::v3
