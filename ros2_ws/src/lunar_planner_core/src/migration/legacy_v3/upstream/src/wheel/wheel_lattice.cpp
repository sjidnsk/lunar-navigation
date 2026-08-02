#include "lunar_path_planner/v3/wheel/wheel_lattice.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>

namespace lunar::planning::v3 {
namespace {

constexpr std::uint64_t kFnvOffsetBasis =
    UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);
constexpr std::size_t kMaximumSweepSamples = 64U;
constexpr double kGeometryTolerance = 1.0e-9;

void HashByte(std::uint64_t& hash, std::uint8_t byte) noexcept {
  hash ^= static_cast<std::uint64_t>(byte);
  hash *= kFnvPrime;
}

void HashU32(std::uint64_t& hash, std::uint32_t value) noexcept {
  for (unsigned shift = 0U; shift < 32U; shift += 8U) {
    HashByte(hash, static_cast<std::uint8_t>(value >> shift));
  }
}

[[nodiscard]] int ModeOrder(WheelMotionMode mode) noexcept {
  switch (mode) {
    case WheelMotionMode::kForward:
      return 0;
    case WheelMotionMode::kReverse:
      return 1;
    case WheelMotionMode::kSpinClockwise:
      return 2;
    case WheelMotionMode::kSpinCounterClockwise:
      return 3;
    case WheelMotionMode::kStart:
      return 4;
  }
  return 5;
}

[[nodiscard]] std::int32_t NormalizeYawBin(
    std::int64_t bin, std::size_t bin_count) noexcept {
  if (bin_count == 0U ||
      bin_count >
          static_cast<std::size_t>(
              std::numeric_limits<std::int32_t>::max())) {
    return 0;
  }
  const auto count = static_cast<std::int64_t>(bin_count);
  std::int64_t normalized = bin % count;
  if (normalized < 0) {
    normalized += count;
  }
  return static_cast<std::int32_t>(normalized);
}

[[nodiscard]] double WrapPositive(double angle) noexcept {
  const double two_pi = 2.0 * std::numbers::pi;
  double wrapped = std::fmod(angle, two_pi);
  if (wrapped < 0.0) {
    wrapped += two_pi;
  }
  return wrapped;
}

[[nodiscard]] double ShortestAngularDistance(
    double lhs, double rhs) noexcept {
  return std::abs(std::remainder(lhs - rhs,
                                 2.0 * std::numbers::pi));
}

[[nodiscard]] bool YawInInterval(
    double yaw, const CircularYawInterval& interval) noexcept {
  if (!std::isfinite(yaw) || !std::isfinite(interval.start_rad) ||
      !std::isfinite(interval.span_rad) ||
      interval.span_rad < 0.0 ||
      interval.span_rad > 2.0 * std::numbers::pi +
                              kGeometryTolerance) {
    return false;
  }
  if (interval.span_rad >=
      2.0 * std::numbers::pi - kGeometryTolerance) {
    return true;
  }
  const double offset =
      WrapPositive(yaw - interval.start_rad);
  return offset <= interval.span_rad + kGeometryTolerance;
}

[[nodiscard]] double YawDistanceToInterval(
    double yaw, const CircularYawInterval& interval) noexcept {
  if (YawInInterval(yaw, interval)) {
    return 0.0;
  }
  return std::min(
      ShortestAngularDistance(yaw, interval.start_rad),
      ShortestAngularDistance(
          yaw, interval.start_rad + interval.span_rad));
}

[[nodiscard]] std::size_t CellIndex(
    const GridGeometry& geometry, Cell cell) noexcept {
  return static_cast<std::size_t>(cell.y) * geometry.width +
         static_cast<std::size_t>(cell.x);
}

[[nodiscard]] bool FiniteNonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] double SafeLayerValue(
    std::span<const float> layer,
    std::size_t index) noexcept {
  if (index >= layer.size()) {
    return 0.0;
  }
  const double value = static_cast<double>(layer[index]);
  return FiniteNonnegative(value) ? value : 0.0;
}

[[nodiscard]] std::chrono::nanoseconds SecondsToNanoseconds(
    double seconds) noexcept {
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    return std::chrono::nanoseconds::zero();
  }
  constexpr long double kNanosecondsPerSecond = 1.0e9L;
  const long double nanoseconds =
      static_cast<long double>(seconds) * kNanosecondsPerSecond;
  const long double maximum = static_cast<long double>(
      std::numeric_limits<std::int64_t>::max());
  if (nanoseconds >= maximum) {
    return std::chrono::nanoseconds{
        std::numeric_limits<std::int64_t>::max()};
  }
  return std::chrono::nanoseconds{
      static_cast<std::int64_t>(std::ceil(nanoseconds))};
}

}  // namespace

StateKey EncodeWheelStateKey(
    std::int32_t ix,
    std::int32_t iy,
    std::int32_t iyaw,
    WheelMotionMode mode) noexcept {
  std::uint64_t hash = kFnvOffsetBasis;
  HashU32(hash, std::bit_cast<std::uint32_t>(ix));
  HashU32(hash, std::bit_cast<std::uint32_t>(iy));
  HashU32(hash, std::bit_cast<std::uint32_t>(iyaw));
  HashByte(hash, static_cast<std::uint8_t>(mode));
  return hash;
}

WheelLatticeAdapter::WheelLatticeAdapter(
    const WheelPrimitiveCatalog& catalog,
    const SafeProjection& projection,
    WheelCapabilityView capability,
    std::span<const TerminalCandidate> terminals,
    GridConfig grid_config)
    : catalog_(&catalog),
      projection_(&projection),
      capability_(std::move(capability)),
      terminals_(terminals.begin(), terminals.end()),
      grid_config_(grid_config) {}

StateKey WheelLatticeAdapter::Key(
    const WheelLatticeState& state) const noexcept {
  return EncodeWheelStateKey(state.ix, state.iy, state.iyaw,
                             state.motion_mode);
}

PoseXyzYaw WheelLatticeAdapter::Pose(
    const WheelLatticeState& state) const noexcept {
  const auto& geometry = projection_->geometry;
  const double resolution =
      grid_config_.xy_resolution_m > 0.0
          ? grid_config_.xy_resolution_m
          : geometry.resolution_m;
  const double yaw_step =
      grid_config_.yaw_bin_count > 0U
          ? 2.0 * std::numbers::pi /
                static_cast<double>(grid_config_.yaw_bin_count)
          : 0.0;
  double z_m = 0.0;
  const Cell cell{state.ix, state.iy};
  const auto elevation = projection_->ElevationMeters();
  if (projection_->InBounds(cell)) {
    const std::size_t index = CellIndex(geometry, cell);
    if (index < elevation.size() &&
        std::isfinite(static_cast<double>(elevation[index]))) {
      z_m = static_cast<double>(elevation[index]);
    }
  }
  return {
      .position_m =
          {
              geometry.origin_m.x +
                  (static_cast<double>(state.ix) + 0.5) *
                      resolution,
              geometry.origin_m.y +
                  (static_cast<double>(state.iy) + 0.5) *
                      resolution,
              z_m,
          },
      .yaw_rad = static_cast<double>(state.iyaw) * yaw_step,
  };
}

WheelLatticeState WheelLatticeAdapter::Quantize(
    const PoseXyzYaw& pose,
    WheelMotionMode mode) const noexcept {
  const auto& geometry = projection_->geometry;
  const double resolution =
      grid_config_.xy_resolution_m > 0.0
          ? grid_config_.xy_resolution_m
          : geometry.resolution_m;
  const double yaw_step =
      grid_config_.yaw_bin_count > 0U
          ? 2.0 * std::numbers::pi /
                static_cast<double>(grid_config_.yaw_bin_count)
          : 1.0;
  return {
      .ix = static_cast<std::int32_t>(std::llround(
          (pose.position_m.x - geometry.origin_m.x) /
              resolution -
          0.5)),
      .iy = static_cast<std::int32_t>(std::llround(
          (pose.position_m.y - geometry.origin_m.y) /
              resolution -
          0.5)),
      .iyaw = NormalizeYawBin(
          static_cast<std::int64_t>(
              std::llround(pose.yaw_rad / yaw_step)),
          grid_config_.yaw_bin_count),
      .motion_mode = mode,
  };
}

std::vector<SearchTransition<WheelLatticeState, WheelLatticeEdge>>
WheelLatticeAdapter::Expand(
    const WheelLatticeState& state) const {
  std::vector<
      SearchTransition<WheelLatticeState, WheelLatticeEdge>>
      transitions;
  const auto outgoing = catalog_->Outgoing(state.motion_mode);
  transitions.reserve(outgoing.size());
  const PoseXyzYaw source_pose = Pose(state);
  const double cosine = std::cos(source_pose.yaw_rad);
  const double sine = std::sin(source_pose.yaw_rad);

  for (const WheelPrimitive& primitive : outgoing) {
    PoseXyzYaw target_pose{
        .position_m =
            {
                source_pose.position_m.x +
                    cosine * primitive.relative_end.position_m.x -
                    sine * primitive.relative_end.position_m.y,
                source_pose.position_m.y +
                    sine * primitive.relative_end.position_m.x +
                    cosine * primitive.relative_end.position_m.y,
                source_pose.position_m.z +
                    primitive.relative_end.position_m.z,
            },
        .yaw_rad =
            source_pose.yaw_rad + primitive.relative_end.yaw_rad,
    };
    WheelLatticeState target =
        Quantize(target_pose, primitive.target_mode);
    target_pose = Pose(target);

    SecondaryCostVector costs = primitive.secondary_costs;
    const Cell target_cell{target.ix, target.iy};
    if (projection_->InBounds(target_cell)) {
      const std::size_t index =
          CellIndex(projection_->geometry, target_cell);
      const double distance_m =
          std::hypot(target_pose.position_m.x -
                         source_pose.position_m.x,
                     target_pose.position_m.y -
                         source_pose.position_m.y);
      const double weight =
          primitive.kind == WheelPrimitiveKind::kModeSwitch
              ? 0.0
              : std::max(
                    distance_m,
                    std::abs(target_pose.yaw_rad -
                             source_pose.yaw_rad));
      costs.energy +=
          SafeLayerValue(projection_->resolved_energy, index) *
          weight;
      costs.risk += SafeLayerValue(
                        projection_->resolved_nonfatal_risk, index) *
                    weight;
      costs.smoothness +=
          std::abs(target_pose.yaw_rad - source_pose.yaw_rad);
    }

    WheelLatticeEdge edge{
        .source = state,
        .target = target,
        .primitive_id = primitive.id,
        .capability_primitive_id =
            primitive.capability_primitive_id,
        .primitive_kind = primitive.kind,
        .source_pose = source_pose,
        .target_pose = target_pose,
        .transition_time = primitive.nominal_duration,
        .secondary_costs = costs,
        .validation_ref = primitive.sweep.validation_ref,
    };
    transitions.push_back(
        {
            .stable_edge_id =
                primitive.id + ":" + std::to_string(Key(state)),
            .successor = target,
            .edge = std::move(edge),
        });
  }

  std::ranges::sort(
      transitions, [](const auto& lhs, const auto& rhs) {
        return std::tuple{
                   ModeOrder(lhs.successor.motion_mode),
                   lhs.edge.capability_primitive_id,
                   lhs.stable_edge_id} <
               std::tuple{
                   ModeOrder(rhs.successor.motion_mode),
                   rhs.edge.capability_primitive_id,
                   rhs.stable_edge_id};
      });
  return transitions;
}

bool WheelLatticeAdapter::HardFeasible(
    const SearchTransition<WheelLatticeState, WheelLatticeEdge>&
        transition) const {
  const Cell source{transition.edge.source.ix,
                    transition.edge.source.iy};
  const Cell target{transition.edge.target.ix,
                    transition.edge.target.iy};
  if (!projection_->InBounds(source) ||
      !projection_->InBounds(target)) {
    return false;
  }
  const std::int64_t delta_x =
      static_cast<std::int64_t>(target.x) - source.x;
  const std::int64_t delta_y =
      static_cast<std::int64_t>(target.y) - source.y;
  const std::uint64_t maximum_delta = static_cast<std::uint64_t>(
      std::max(std::llabs(delta_x), std::llabs(delta_y)));
  if (maximum_delta > kMaximumSweepSamples) {
    return false;
  }
  const std::size_t samples =
      std::max<std::size_t>(1U,
                            static_cast<std::size_t>(maximum_delta));
  const double required_clearance =
      capability_.footprint_support_radius_m() +
      capability_.certified_horizontal_position_error_m() +
      capability_.hard_limits().minimum_clearance_m;
  for (std::size_t index = 0U; index <= samples; ++index) {
    const double fraction =
        static_cast<double>(index) / static_cast<double>(samples);
    const Cell cell{
        static_cast<std::int32_t>(std::llround(
            static_cast<double>(source.x) +
            fraction * static_cast<double>(delta_x))),
        static_cast<std::int32_t>(std::llround(
            static_cast<double>(source.y) +
            fraction * static_cast<double>(delta_y))),
    };
    if (!projection_->Known(cell) ||
        !projection_->HardFeasible(cell) ||
        static_cast<double>(projection_->ClearanceMeters(cell)) +
                kGeometryTolerance <
            required_clearance) {
      return false;
    }
  }
  return true;
}

DurationNanoseconds WheelLatticeAdapter::TransitionTime(
    const SearchTransition<WheelLatticeState, WheelLatticeEdge>&
        transition) const noexcept {
  return transition.edge.transition_time;
}

DurationNanoseconds
WheelLatticeAdapter::AdmissibleTimeHeuristic(
    const WheelLatticeState& state,
    const SearchProblem<WheelLatticeState>& problem) const noexcept {
  static_cast<void>(problem);
  const PoseXyzYaw pose = Pose(state);
  const auto& limits = capability_.hard_limits();
  const double maximum_drive_speed =
      std::max(limits.maximum_forward_speed_mps,
               limits.maximum_reverse_speed_mps);
  double best_seconds = std::numeric_limits<double>::infinity();
  for (const TerminalCandidate& terminal : terminals_) {
    const double distance_m =
        std::hypot(terminal.position_m.x - pose.position_m.x,
                   terminal.position_m.y - pose.position_m.y);
    const double translation_seconds =
        maximum_drive_speed > 0.0
            ? distance_m / maximum_drive_speed
            : std::numeric_limits<double>::infinity();
    double rotation_seconds = 0.0;
    if (terminal.yaw_interval.has_value() &&
        limits.maximum_spin_rate_radps > 0.0) {
      rotation_seconds =
          YawDistanceToInterval(pose.yaw_rad,
                                *terminal.yaw_interval) /
          limits.maximum_spin_rate_radps;
    }
    best_seconds =
        std::min(best_seconds,
                 std::max(translation_seconds, rotation_seconds));
  }
  return DurationNanoseconds{
      SecondsToNanoseconds(best_seconds)};
}

SecondaryCostVector WheelLatticeAdapter::SecondaryCosts(
    const SearchTransition<WheelLatticeState, WheelLatticeEdge>&
        transition) const noexcept {
  return transition.edge.secondary_costs;
}

bool WheelLatticeAdapter::IsTerminal(
    const WheelLatticeState& state,
    const SearchProblem<WheelLatticeState>& problem) const noexcept {
  static_cast<void>(problem);
  const PoseXyzYaw pose = Pose(state);
  return std::ranges::any_of(
      terminals_, [&](const TerminalCandidate& terminal) {
        if (terminal.cell.x != state.ix ||
            terminal.cell.y != state.iy) {
          return false;
        }
        if (terminal.requires_zero_speed &&
            state.motion_mode != WheelMotionMode::kStart) {
          return false;
        }
        return !terminal.yaw_interval.has_value() ||
               YawInInterval(pose.yaw_rad,
                             *terminal.yaw_interval);
      });
}

}  // namespace lunar::planning::v3
