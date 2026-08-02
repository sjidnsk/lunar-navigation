#include "fixtures/system/planner_v3_system_fixture.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/crypto/sha256.hpp"
#include "lunar_path_planner/v3/hopper/hopper_config.hpp"

namespace lunar::planning::v3::system_test {
namespace {

using namespace std::chrono_literals;

constexpr std::size_t kGroundWidth = 12U;
constexpr std::size_t kGroundHeight = 8U;

struct MultiscaleDimensions final {
  std::size_t width;
  std::size_t height;
  double resolution_m;
  double ground_distance_m;
  double hopper_distance_m;
};

struct DetourLayout final {
  std::size_t barrier_x;
  std::size_t gap_begin_y;
  std::size_t gap_end_y;
};

struct G1ReferenceSpec final {
  std::string_view scenario_id;
  std::string_view scenario_hash;
  std::string_view proxy_seed_hex;
  double hard_obstacle_fraction;
};

struct G1HopperEndpoints final {
  std::size_t start_x;
  std::size_t start_y;
  std::size_t goal_x;
  std::size_t goal_y;
};

[[nodiscard]] std::vector<std::uint8_t>
GenerateG1ProxyObstacleMask(
    MapScale scale, G1ReferenceScenario reference,
    bool reserve_hopper_endpoints);

[[nodiscard]] G1HopperEndpoints
SelectG1HopperEndpoints(
    MapScale scale, G1ReferenceScenario reference);

[[nodiscard]] constexpr G1ReferenceSpec G1ReferenceFor(
    const G1ReferenceScenario reference) {
  switch (reference) {
    case G1ReferenceScenario::kLowKnown:
      return {
          "validation/scenario-0027/standard-proxy/v1",
          "564835143c269d1ef4fb8d01cb7517cf4efb674bb1554c36467a1d59eb39f16c",
          "0647281ae562cc9fac85a745a0c183c2",
          0.006683349609375,
      };
    case G1ReferenceScenario::kMediumKnown:
      return {
          "validation/scenario-0007/standard-proxy/v1",
          "6f754169bb3c4837978e3e852258493f4b1c971e8a59148a9a8269fb306a95bc",
          "6b6d97f83a6183e4119cbd01f8e0e25e",
          0.01348876953125,
      };
    case G1ReferenceScenario::kHighFrontier:
      return {
          "validation/scenario-0116/standard-proxy/v1",
          "49e4254dc8e5602be67cc2e9eb68af65093ec049b463e3581edfc9b5f352b554",
          "9ded9466c337bbf7037cd239265150d5",
          0.03814697265625,
      };
  }
  throw std::invalid_argument{"unsupported G1 reference scenario"};
}

[[nodiscard]] MultiscaleDimensions DimensionsFor(
    const MapScale map_scale) {
  switch (map_scale) {
    case MapScale::kTenMeter:
      return {24U, 24U, 0.5, 8.0, 4.0};
    case MapScale::kHundredMeter:
      return {60U, 40U, 2.0, 80.0, 8.0};
    case MapScale::kThousandMeter:
      return {240U, 40U, 5.0, 500.0, 10.0};
  }
  throw std::invalid_argument{"unsupported multiscale map scale"};
}

[[nodiscard]] double GroundPrimitiveDistanceFor(
    const MapScale map_scale) {
  switch (map_scale) {
    case MapScale::kTenMeter:
      return 1.0;
    case MapScale::kHundredMeter:
      return 4.0;
    case MapScale::kThousandMeter:
      return 10.0;
  }
  throw std::invalid_argument{"unsupported multiscale map scale"};
}

[[nodiscard]] std::size_t G1UnknownBeginX(
    const MapScale map_scale) {
  const MultiscaleDimensions dimensions = DimensionsFor(map_scale);
  const double primitive_m = GroundPrimitiveDistanceFor(map_scale);
  const double known_distance_m =
      std::floor(
          0.6 * dimensions.ground_distance_m / primitive_m) *
      primitive_m;
  const std::size_t ground_frontier_x =
      2U + static_cast<std::size_t>(
                   known_distance_m / dimensions.resolution_m);
  if (map_scale == MapScale::kTenMeter) {
    return ground_frontier_x;
  }
  const double physical_height_m =
      static_cast<double>(dimensions.height) *
      dimensions.resolution_m;
  const double hopper_start_x =
      map_scale == MapScale::kTenMeter
          ? 4.5 * dimensions.resolution_m
          : std::floor(
                (physical_height_m / 2.0 -
                 dimensions.hopper_distance_m / 2.0) /
                dimensions.resolution_m) *
                    dimensions.resolution_m +
                dimensions.resolution_m / 2.0;
  const std::size_t hopper_goal_x =
      static_cast<std::size_t>(
          (hopper_start_x + dimensions.hopper_distance_m) /
          dimensions.resolution_m);
  return std::max(ground_frontier_x, hopper_goal_x + 1U);
}

[[nodiscard]] ScenarioRegion RectangleRegion(
    const ScenarioRegion::Kind kind,
    const double minimum_x, const double minimum_y,
    const double maximum_x, const double maximum_y) {
  return {
      .kind = kind,
      .polygon_xy_m =
          {
              {minimum_x, minimum_y},
              {maximum_x, minimum_y},
              {maximum_x, maximum_y},
              {minimum_x, maximum_y},
          },
  };
}

[[nodiscard]] DetourLayout DetourFor(
    const PlatformType platform_type,
    const ScenarioDescription& description) {
  const MultiscaleDimensions dimensions =
      DimensionsFor(description.scale);
  const std::size_t height = dimensions.height;
  const std::size_t center_y =
      static_cast<std::size_t>(
          description.start_xy_m.y / dimensions.resolution_m);
  const std::size_t gap_begin_y =
      std::min(
          center_y > 0U ? center_y - 1U : 0U,
          height - 3U);
  return {
      .barrier_x =
          platform_type == PlatformType::kHopper
              ? 0U
              : static_cast<std::size_t>(
                    (description.start_xy_m.x +
                     description.goal_xy_m.x) /
                    (2.0 * dimensions.resolution_m)),
      .gap_begin_y = gap_begin_y,
      .gap_end_y = gap_begin_y + 3U,
  };
}

[[nodiscard]] std::size_t UnknownBeginX(
    const PlatformType platform_type,
    const ScenarioDescription& description) {
  const MultiscaleDimensions dimensions =
      DimensionsFor(description.scale);
  if (platform_type == PlatformType::kHopper) {
    return static_cast<std::size_t>(
               description.goal_xy_m.x / dimensions.resolution_m) +
           2U;
  }
  const double primitive_m =
      GroundPrimitiveDistanceFor(description.scale);
  const double known_distance_m =
      std::floor(
          0.6 * description.nominal_plan_distance_m /
          primitive_m) *
      primitive_m;
  return 2U + static_cast<std::size_t>(
                  known_distance_m / dimensions.resolution_m);
}

[[nodiscard]] ScenarioDescription DescribeMultiscaleScenario(
    const PlatformType platform_type,
    const MapScale map_scale,
    const MapScenario map_scenario) {
  const MultiscaleDimensions dimensions = DimensionsFor(map_scale);
  const bool hopper = platform_type == PlatformType::kHopper;
  const double distance =
      hopper ? dimensions.hopper_distance_m
             : dimensions.ground_distance_m;
  const double physical_height_m =
      static_cast<double>(dimensions.height) *
      dimensions.resolution_m;
  double start_x = 1.5 * dimensions.resolution_m;
  double start_y =
      physical_height_m / 2.0 - dimensions.resolution_m / 2.0;
  if (hopper) {
    if (map_scenario ==
        MapScenario::kUnknownGoalWithSafeFrontier) {
      start_x =
          distance + 2.5 * dimensions.resolution_m;
      start_y = start_x;
    } else {
      start_x =
          std::floor(
              (physical_height_m / 2.0 - distance / 2.0) /
              dimensions.resolution_m) *
              dimensions.resolution_m +
          dimensions.resolution_m / 2.0;
    }
  }
  ScenarioDescription description{
      .scale = map_scale,
      .scene = map_scenario,
      .width_m =
          static_cast<double>(dimensions.width) *
          dimensions.resolution_m,
      .height_m = physical_height_m,
      .start_xy_m = {start_x, start_y},
      .goal_xy_m = {start_x + distance, start_y},
      .nominal_plan_distance_m = distance,
      .expected_outcome =
          map_scenario ==
                      MapScenario::kUnknownGoalWithSafeFrontier &&
                  !hopper
              ? ExpectedExperimentOutcome::
                    kSafeFrontierReferenceReady
              : ExpectedExperimentOutcome::kNewReferenceReady,
  };

  if (map_scenario == MapScenario::kDetour) {
    const DetourLayout layout =
        DetourFor(platform_type, description);
    const double barrier_x =
        static_cast<double>(layout.barrier_x) *
        dimensions.resolution_m;
    description.regions.push_back(RectangleRegion(
        ScenarioRegion::Kind::kHardObstacle,
        barrier_x, 0.0,
        barrier_x + dimensions.resolution_m,
        static_cast<double>(layout.gap_begin_y) *
            dimensions.resolution_m));
    description.regions.push_back(RectangleRegion(
        ScenarioRegion::Kind::kHardObstacle,
        barrier_x,
        static_cast<double>(layout.gap_end_y) *
            dimensions.resolution_m,
        barrier_x + dimensions.resolution_m,
        description.height_m));
  } else if (
      map_scenario ==
      MapScenario::kUnknownGoalWithSafeFrontier) {
    const double unknown_begin_x = static_cast<double>(
        UnknownBeginX(platform_type, description)) *
        dimensions.resolution_m;
    description.regions.push_back(RectangleRegion(
        ScenarioRegion::Kind::kUnknown,
        unknown_begin_x, 0.0, description.width_m,
        description.height_m));
  }
  return description;
}

[[nodiscard]] ScenarioDescription DescribeG1MultiscaleScenario(
    const PlatformType platform_type,
    const MapScale map_scale,
    const G1ReferenceScenario reference) {
  const MultiscaleDimensions dimensions = DimensionsFor(map_scale);
  const G1ReferenceSpec source = G1ReferenceFor(reference);
  const bool hopper = platform_type == PlatformType::kHopper;
  const double distance =
      hopper ? dimensions.hopper_distance_m
             : dimensions.ground_distance_m;
  const double resolution_m = dimensions.resolution_m;
  const double physical_height_m =
      static_cast<double>(dimensions.height) * resolution_m;
  const bool vertical_hopper_frontier =
      hopper && map_scale == MapScale::kTenMeter &&
      reference == G1ReferenceScenario::kHighFrontier;
  const std::optional<G1HopperEndpoints> derived_hopper =
      hopper && map_scale != MapScale::kTenMeter
          ? std::optional<G1HopperEndpoints>{
                SelectG1HopperEndpoints(map_scale, reference)}
          : std::nullopt;
  const double start_x =
      derived_hopper.has_value()
          ? (static_cast<double>(derived_hopper->start_x) + 0.5) *
                resolution_m
          : (hopper ? 4.5 * resolution_m
                    : 1.5 * resolution_m);
  const double start_y =
      derived_hopper.has_value()
          ? (static_cast<double>(derived_hopper->start_y) + 0.5) *
                resolution_m
          : (vertical_hopper_frontier
                 ? 7.5 * resolution_m
                 : physical_height_m / 2.0 -
                       resolution_m / 2.0);
  const double goal_x =
      derived_hopper.has_value()
          ? (static_cast<double>(derived_hopper->goal_x) + 0.5) *
                resolution_m
          : (vertical_hopper_frontier
                 ? start_x
                 : start_x + distance);
  const double goal_y =
      derived_hopper.has_value()
          ? (static_cast<double>(derived_hopper->goal_y) + 0.5) *
                resolution_m
          : (vertical_hopper_frontier
                 ? start_y + distance
                 : start_y);
  ScenarioDescription description{
      .scale = map_scale,
      .scene =
          reference == G1ReferenceScenario::kHighFrontier
              ? MapScenario::kUnknownGoalWithSafeFrontier
              : MapScenario::kOpenKnown,
      .g1_reference =
          G1ReferenceProvenance{
              .reference = reference,
              .source_scenario_id = std::string{source.scenario_id},
              .source_scenario_hash = std::string{source.scenario_hash},
              .proxy_seed_hex = std::string{source.proxy_seed_hex},
              .source_hard_obstacle_fraction =
                  source.hard_obstacle_fraction,
              .source_kind =
                  "synthetic_terrain_obstacle_proxy/v1",
              .physical_obstacle_cells_written = false,
              .derivation =
                  "g1_validation_reference_scaled/v1",
          },
      .width_m =
          static_cast<double>(dimensions.width) * resolution_m,
      .height_m = physical_height_m,
      .start_xy_m = {start_x, start_y},
      .goal_xy_m = {goal_x, goal_y},
      .nominal_plan_distance_m = distance,
      .expected_outcome =
          reference == G1ReferenceScenario::kHighFrontier &&
                  !hopper
              ? ExpectedExperimentOutcome::
                    kSafeFrontierReferenceReady
              : ExpectedExperimentOutcome::kNewReferenceReady,
  };
  if (reference == G1ReferenceScenario::kHighFrontier) {
    const double unknown_begin_x =
        static_cast<double>(G1UnknownBeginX(map_scale)) *
        resolution_m;
    description.regions.push_back(RectangleRegion(
        ScenarioRegion::Kind::kUnknown,
        unknown_begin_x, 0.0, description.width_m,
        description.height_m));
  }
  return description;
}

[[nodiscard]] ContentRef Ref(
    std::string id, const char digit) {
  return {
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, digit),
  };
}

[[nodiscard]] AxisAlignedBox3 ZeroBox() {
  return {
      .center = {},
      .half_extent = {},
  };
}

[[nodiscard]] WheeledOrLeggedErrorBounds ZeroGroundError() {
  return {
      .position_bound_m = ZeroBox(),
      .yaw_bound_rad = {},
      .linear_velocity_bound_mps = ZeroBox(),
      .yaw_rate_bound_radps = {},
  };
}

[[nodiscard]] HopperErrorBounds ZeroHopperError() {
  return {
      .position_bound_m = ZeroBox(),
      .orientation_bound = {},
      .linear_velocity_bound_mps = ZeroBox(),
      .angular_velocity_bound_radps = ZeroBox(),
  };
}

[[nodiscard]] ConvexPolytope3 BoxPolytope(
    const double half_x, const double half_y,
    const double minimum_z, const double maximum_z) {
  return {
      .halfspaces =
          {
              {{1.0, 0.0, 0.0}, half_x},
              {{-1.0, 0.0, 0.0}, half_x},
              {{0.0, 1.0, 0.0}, half_y},
              {{0.0, -1.0, 0.0}, half_y},
              {{0.0, 0.0, 1.0}, maximum_z},
              {{0.0, 0.0, -1.0}, -minimum_z},
          },
  };
}

template <class T>
[[nodiscard]] std::shared_ptr<const T> OpaqueResolvedObject() {
  auto owner = std::make_shared<int>(1);
  const auto* opaque =
      reinterpret_cast<const T*>(owner.get());
  return std::shared_ptr<const T>(
      std::move(owner), opaque);
}

[[nodiscard]] MapSnapshotInput BaseMapInput(
    const PlatformType platform_type,
    const MapScenario map_scenario) {
  const bool hopper = platform_type == PlatformType::kHopper;
  const std::size_t width = hopper ? 16U : kGroundWidth;
  const std::size_t height = hopper ? 12U : kGroundHeight;
  const double resolution_m = hopper ? 0.5 : 1.0;
  const std::size_t count = width * height;
  const char platform_digit =
      platform_type == PlatformType::kWheeled
          ? '1'
          : (platform_type == PlatformType::kLegged ? '2' : '3');

  MapSnapshotInput input{
      .snapshot_ref =
          Ref("system-map-" +
                  std::to_string(
                      static_cast<int>(platform_type)) +
                  "-" +
                  std::to_string(
                      static_cast<int>(map_scenario)),
              platform_digit),
      .map_revision = 1U,
      .immutable_data_handle =
          "system-map-handle-" +
          std::to_string(static_cast<int>(platform_type)) +
          "-" +
          std::to_string(static_cast<int>(map_scenario)),
      .source_time =
          ClockStamp{"mission", std::chrono::nanoseconds{100}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -2.0},
              .maximum_m =
                  {
                      static_cast<double>(width) * resolution_m,
                      static_cast<double>(height) * resolution_m,
                      8.0,
                  },
          },
      .geometry =
          {
              .width = width,
              .height = height,
              .resolution_m = resolution_m,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {LayerKind::kKnownMask, Ref("system-known", '4')},
              {LayerKind::kElevation, Ref("system-elevation", '5')},
              {LayerKind::kTerrainNormal, Ref("system-normal", '6')},
              {LayerKind::kRoughness, Ref("system-roughness", '7')},
              {LayerKind::kHardObstacle, Ref("system-obstacle", '8')},
              {LayerKind::kConfidence, Ref("system-confidence", '9')},
          },
      .known_mask = std::vector<std::uint8_t>(count, 1U),
      .elevation_m = std::vector<float>(count, 0.0F),
      .normal_x = std::vector<float>(count, 0.0F),
      .normal_y = std::vector<float>(count, 0.0F),
      .normal_z = std::vector<float>(count, 1.0F),
      .roughness_m = std::vector<float>(count, 0.0F),
      .hard_obstacle_mask =
          std::vector<std::uint8_t>(count, 0U),
      .confidence = std::vector<float>(count, 1.0F),
  };

  if (!hopper &&
      map_scenario == MapScenario::kKnownObstacleAtGoal) {
    input.hard_obstacle_mask[3U * width + 4U] = 1U;
  }
  if (!hopper &&
      map_scenario ==
          MapScenario::kUnknownGoalWithSafeFrontier) {
    for (std::size_t y = 0U; y < height; ++y) {
      for (std::size_t x = 6U; x < width; ++x) {
        const std::size_t index = y * width + x;
        input.known_mask[index] = 0U;
        input.confidence[index] = 0.0F;
      }
    }
  }
  return input;
}

[[nodiscard]] MapSnapshotInput MultiscaleMapInput(
    const PlatformType platform_type,
    const ScenarioDescription& description) {
  const MultiscaleDimensions dimensions =
      DimensionsFor(description.scale);
  const std::size_t width = dimensions.width;
  const std::size_t height = dimensions.height;
  const std::size_t count = width * height;
  const char platform_digit =
      platform_type == PlatformType::kWheeled
          ? '1'
          : (platform_type == PlatformType::kLegged ? '2' : '3');
  MapSnapshotInput input{
      .snapshot_ref =
          Ref("multiscale-map-" +
                  std::to_string(
                      static_cast<int>(platform_type)) +
                  "-" +
                  std::to_string(
                      static_cast<int>(description.scale)) +
                  "-" +
                  std::to_string(
                      static_cast<int>(description.scene)),
              platform_digit),
      .map_revision = 1U,
      .immutable_data_handle =
          "multiscale-map-handle-" +
          std::to_string(static_cast<int>(platform_type)) +
          "-" +
          std::to_string(static_cast<int>(description.scale)) +
          "-" +
          std::to_string(static_cast<int>(description.scene)),
      .source_time =
          ClockStamp{"mission", std::chrono::nanoseconds{100}},
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -2.0},
              .maximum_m =
                  {
                      description.width_m,
                      description.height_m,
                      8.0,
                  },
          },
      .geometry =
          {
              .width = width,
              .height = height,
              .resolution_m = dimensions.resolution_m,
              .origin_m = {0.0, 0.0},
              .frame_id = "map",
          },
      .layer_manifest =
          {
              {LayerKind::kKnownMask, Ref("multiscale-known", '4')},
              {LayerKind::kElevation, Ref("multiscale-elevation", '5')},
              {LayerKind::kTerrainNormal, Ref("multiscale-normal", '6')},
              {LayerKind::kRoughness, Ref("multiscale-roughness", '7')},
              {LayerKind::kHardObstacle, Ref("multiscale-obstacle", '8')},
              {LayerKind::kConfidence, Ref("multiscale-confidence", '9')},
          },
      .known_mask = std::vector<std::uint8_t>(count, 1U),
      .elevation_m = std::vector<float>(count, 0.0F),
      .normal_x = std::vector<float>(count, 0.0F),
      .normal_y = std::vector<float>(count, 0.0F),
      .normal_z = std::vector<float>(count, 1.0F),
      .roughness_m = std::vector<float>(count, 0.0F),
      .hard_obstacle_mask =
          std::vector<std::uint8_t>(count, 0U),
      .confidence = std::vector<float>(count, 1.0F),
  };

  if (description.scene == MapScenario::kDetour) {
    const DetourLayout layout =
        DetourFor(platform_type, description);
    for (std::size_t y = 0U; y < height; ++y) {
      if (y < layout.gap_begin_y || y >= layout.gap_end_y) {
        input.hard_obstacle_mask[y * width + layout.barrier_x] = 1U;
      }
    }
  } else if (
      description.scene ==
      MapScenario::kUnknownGoalWithSafeFrontier) {
    const std::size_t unknown_begin_x =
        UnknownBeginX(platform_type, description);
    for (std::size_t y = 0U; y < height; ++y) {
      for (std::size_t x = unknown_begin_x; x < width; ++x) {
        const std::size_t index = y * width + x;
        input.known_mask[index] = 0U;
        input.confidence[index] = 0.0F;
      }
    }
  }
  return input;
}

class ProxyCellGenerator final {
 public:
  ProxyCellGenerator(
      const std::string_view seed_hex,
      const MapScale scale) {
    state_ = 1469598103934665603ULL;
    for (const char value : seed_hex) {
      state_ ^= static_cast<std::uint64_t>(
          static_cast<unsigned char>(value));
      state_ *= 1099511628211ULL;
    }
    state_ ^= static_cast<std::uint64_t>(
        static_cast<unsigned int>(scale) + 1U) *
        0x9e3779b97f4a7c15ULL;
    if (state_ == 0U) {
      state_ = 0x6a09e667f3bcc909ULL;
    }
  }

  [[nodiscard]] std::uint64_t Next() {
    state_ ^= state_ >> 12U;
    state_ ^= state_ << 25U;
    state_ ^= state_ >> 27U;
    return state_ * 2685821657736338717ULL;
  }

 private:
  std::uint64_t state_{};
};

[[nodiscard]] bool G1ProtectedCell(
    const std::size_t x, const std::size_t y,
    const MultiscaleDimensions& dimensions,
    const MapScale scale,
    const G1ReferenceScenario reference) {
  const std::size_t ground_start_x = 1U;
  const std::size_t ground_y = dimensions.height / 2U - 1U;
  const std::size_t ground_goal_x =
      ground_start_x +
      static_cast<std::size_t>(
          dimensions.ground_distance_m /
          dimensions.resolution_m);
  const std::size_t hopper_start_x = 4U;
  const std::size_t hopper_goal_x =
      scale == MapScale::kTenMeter &&
              reference == G1ReferenceScenario::kHighFrontier
          ? hopper_start_x
          : hopper_start_x +
                static_cast<std::size_t>(
                    dimensions.hopper_distance_m /
                    dimensions.resolution_m);
  const std::size_t hopper_start_y =
      scale == MapScale::kTenMeter &&
              reference == G1ReferenceScenario::kHighFrontier
          ? 7U
          : ground_y;
  const std::size_t hopper_goal_y =
      scale == MapScale::kTenMeter &&
              reference == G1ReferenceScenario::kHighFrontier
          ? hopper_start_y +
                static_cast<std::size_t>(
                    dimensions.hopper_distance_m /
                    dimensions.resolution_m)
          : ground_y;
  const auto near = [x, y](
                        const std::size_t protected_x,
                        const std::size_t protected_y,
                        const std::size_t radius) {
    const std::size_t dx =
        x > protected_x ? x - protected_x : protected_x - x;
    const std::size_t dy =
        y > protected_y ? y - protected_y : protected_y - y;
    return dx <= radius && dy <= radius;
  };
  const std::size_t protected_ground_goal_x =
      reference == G1ReferenceScenario::kHighFrontier
          ? std::min(
                ground_goal_x,
                G1UnknownBeginX(scale) - 1U)
          : ground_goal_x;
  const bool ground_corridor =
      x >= ground_start_x && x <= protected_ground_goal_x &&
      (y > ground_y ? y - ground_y : ground_y - y) <= 1U;
  const bool ten_meter_hopper_corridor =
      scale == MapScale::kTenMeter &&
      reference == G1ReferenceScenario::kHighFrontier &&
      x >= hopper_start_x - 1U &&
      x <= hopper_start_x + 1U &&
      y >= hopper_start_y && y <= hopper_goal_y;
  return ground_corridor || ten_meter_hopper_corridor ||
         near(ground_start_x, ground_y, 1U) ||
         near(ground_goal_x, ground_y, 1U) ||
         (scale == MapScale::kTenMeter &&
          (near(hopper_start_x, hopper_start_y, 3U) ||
           near(hopper_goal_x, hopper_goal_y, 1U)));
}

[[nodiscard]] std::vector<std::uint8_t>
GenerateG1ProxyObstacleMask(
    const MapScale scale,
    const G1ReferenceScenario reference,
    const bool reserve_hopper_endpoints) {
  constexpr std::array<std::array<int, 2U>, 9U> offsets{{
      {0, 0},
      {1, 0},
      {0, 1},
      {-1, 0},
      {0, -1},
      {1, 1},
      {-1, 1},
      {-1, -1},
      {1, -1},
  }};
  const MultiscaleDimensions dimensions = DimensionsFor(scale);
  const G1ReferenceSpec source = G1ReferenceFor(reference);
  const std::size_t cell_count =
      dimensions.width * dimensions.height;
  const std::size_t target_count =
      static_cast<std::size_t>(std::llround(
          source.hard_obstacle_fraction *
          static_cast<double>(cell_count)));
  const std::size_t eligible_width =
      reference == G1ReferenceScenario::kHighFrontier
          ? G1UnknownBeginX(scale)
          : dimensions.width;
  std::vector<std::uint8_t> mask(cell_count, 0U);
  ProxyCellGenerator generator(source.proxy_seed_hex, scale);
  std::size_t placed = 0U;
  const std::size_t maximum_attempts =
      cell_count * 16U + target_count * 64U;
  for (std::size_t attempt = 0U;
       attempt < maximum_attempts && placed < target_count;
       ++attempt) {
    const std::size_t anchor_x =
        static_cast<std::size_t>(
            generator.Next() % eligible_width);
    const std::size_t anchor_y =
        static_cast<std::size_t>(
            generator.Next() % dimensions.height);
    const std::size_t cluster_size =
        5U + static_cast<std::size_t>(generator.Next() % 5U);
    const std::size_t rotation =
        static_cast<std::size_t>(
            generator.Next() % (offsets.size() - 1U));
    for (std::size_t member = 0U;
         member < cluster_size && placed < target_count;
         ++member) {
      const auto& offset =
          member == 0U
              ? offsets.front()
              : offsets[
                    1U +
                    (rotation + member - 1U) %
                        (offsets.size() - 1U)];
      const auto candidate_x =
          static_cast<std::ptrdiff_t>(anchor_x) + offset[0];
      const auto candidate_y =
          static_cast<std::ptrdiff_t>(anchor_y) + offset[1];
      if (candidate_x < 0 || candidate_y < 0 ||
          candidate_x >= static_cast<std::ptrdiff_t>(eligible_width) ||
          candidate_y >=
              static_cast<std::ptrdiff_t>(dimensions.height)) {
        continue;
      }
      const std::size_t x =
          static_cast<std::size_t>(candidate_x);
      const std::size_t y =
          static_cast<std::size_t>(candidate_y);
      const std::size_t index = y * dimensions.width + x;
      if (mask[index] != 0U ||
          G1ProtectedCell(
              x, y, dimensions, scale, reference)) {
        continue;
      }
      mask[index] = 1U;
      ++placed;
    }
  }
  std::optional<G1HopperEndpoints> reserved_hopper;
  if (reserve_hopper_endpoints &&
      scale != MapScale::kTenMeter) {
    reserved_hopper =
        SelectG1HopperEndpoints(scale, reference);
    for (std::size_t y = 0U; y < dimensions.height; ++y) {
      for (std::size_t x = 0U; x < eligible_width; ++x) {
        const std::size_t minimum_x =
            reserved_hopper->start_x > 1U
                ? reserved_hopper->start_x - 1U
                : 0U;
        const std::size_t maximum_x =
            std::min(
                reserved_hopper->goal_x + 1U,
                eligible_width - 1U);
        const std::size_t minimum_y =
            reserved_hopper->start_y > 1U
                ? reserved_hopper->start_y - 1U
                : 0U;
        const std::size_t maximum_y =
            std::min(
                reserved_hopper->start_y + 1U,
                dimensions.height - 1U);
        const std::size_t index = y * dimensions.width + x;
        if (x >= minimum_x && x <= maximum_x &&
            y >= minimum_y && y <= maximum_y &&
            mask[index] != 0U) {
          mask[index] = 0U;
          --placed;
        }
      }
    }
  }
  const std::size_t fallback_offset =
      static_cast<std::size_t>(generator.Next() % cell_count);
  for (std::size_t step = 0U;
       step < cell_count && placed < target_count;
       ++step) {
    const std::size_t index =
        (fallback_offset + step) % cell_count;
    const std::size_t x = index % dimensions.width;
    const std::size_t y = index / dimensions.width;
    const bool hopper_reserved =
        reserved_hopper.has_value() &&
        x + 1U >= reserved_hopper->start_x &&
        x <= reserved_hopper->goal_x + 1U &&
        y + 1U >= reserved_hopper->start_y &&
        y <= reserved_hopper->start_y + 1U;
    if (x >= eligible_width || mask[index] != 0U ||
        hopper_reserved ||
        G1ProtectedCell(
            x, y, dimensions, scale, reference)) {
      continue;
    }
    mask[index] = 1U;
    ++placed;
  }
  if (placed != target_count) {
    throw std::runtime_error{
        "G1 proxy obstacle placement exhausted"};
  }
  return mask;
}

[[nodiscard]] G1HopperEndpoints
SelectG1HopperEndpoints(
    const MapScale scale,
    const G1ReferenceScenario reference) {
  const MultiscaleDimensions dimensions = DimensionsFor(scale);
  const std::vector<std::uint8_t> hard_obstacles =
      GenerateG1ProxyObstacleMask(scale, reference, false);
  const std::size_t known_width =
      reference == G1ReferenceScenario::kHighFrontier
          ? G1UnknownBeginX(scale)
          : dimensions.width;
  const std::size_t hop_cells =
      static_cast<std::size_t>(
          dimensions.hopper_distance_m /
          dimensions.resolution_m);
  std::vector<std::uint8_t> safe(
      dimensions.width * dimensions.height, 0U);
  for (std::size_t y = 1U; y + 1U < dimensions.height; ++y) {
    for (std::size_t x = 1U; x + 1U < known_width; ++x) {
      bool footprint_safe = true;
      for (std::size_t covered_y = y - 1U;
           covered_y <= y + 1U && footprint_safe;
           ++covered_y) {
        for (std::size_t covered_x = x - 1U;
             covered_x <= x + 1U; ++covered_x) {
          if (hard_obstacles[
                  covered_y * dimensions.width + covered_x] != 0U) {
            footprint_safe = false;
            break;
          }
        }
      }
      if (footprint_safe) {
        safe[y * dimensions.width + x] = 1U;
      }
    }
  }
  std::vector<std::array<std::size_t, 2U>> unsafe_cells;
  unsafe_cells.reserve(safe.size());
  for (std::size_t y = 0U; y < dimensions.height; ++y) {
    for (std::size_t x = 0U; x < known_width; ++x) {
      if (safe[y * dimensions.width + x] == 0U) {
        unsafe_cells.push_back({x, y});
      }
    }
  }
  std::optional<G1HopperEndpoints> selected;
  double selected_clearance_squared = -1.0;
  for (std::size_t y = 1U; y + 1U < dimensions.height; ++y) {
    for (std::size_t x = 1U;
         x + hop_cells + 1U < known_width; ++x) {
      const std::size_t goal_x = x + hop_cells;
      if (safe[y * dimensions.width + x] == 0U ||
          safe[y * dimensions.width + goal_x] == 0U) {
        continue;
      }
      double clearance_squared =
          std::numeric_limits<double>::infinity();
      for (const auto& unsafe : unsafe_cells) {
        for (const std::size_t endpoint_x : {x, goal_x}) {
          const double dx =
              static_cast<double>(endpoint_x) -
              static_cast<double>(unsafe[0]);
          const double dy =
              static_cast<double>(y) -
              static_cast<double>(unsafe[1]);
          clearance_squared = std::min(
              clearance_squared, dx * dx + dy * dy);
        }
      }
      if (clearance_squared > selected_clearance_squared) {
        selected_clearance_squared = clearance_squared;
        selected = G1HopperEndpoints{x, y, goal_x, y};
      }
    }
  }
  if (!selected.has_value()) {
    throw std::runtime_error{
        "G1 proxy mask has no Hopper landing pair"};
  }
  return *selected;
}

[[nodiscard]] MapSnapshotInput G1MultiscaleMapInput(
    const PlatformType platform_type,
    const ScenarioDescription& description) {
  if (!description.g1_reference.has_value()) {
    throw std::invalid_argument{
        "G1 multiscale map requires provenance"};
  }
  ScenarioDescription all_known = description;
  all_known.scene = MapScenario::kOpenKnown;
  all_known.regions.clear();
  MapSnapshotInput input =
      MultiscaleMapInput(platform_type, all_known);
  const char platform_digit =
      platform_type == PlatformType::kWheeled
          ? '1'
          : (platform_type == PlatformType::kLegged ? '2' : '3');
  const auto reference =
      description.g1_reference->reference;
  input.snapshot_ref =
      Ref(
          "multiscale-g1-map-" +
              std::to_string(static_cast<int>(platform_type)) +
              "-" +
              std::to_string(static_cast<int>(description.scale)) +
              "-" +
              std::to_string(static_cast<int>(reference)),
          platform_digit);
  input.immutable_data_handle =
      "multiscale-g1-map-handle-" +
      std::to_string(static_cast<int>(platform_type)) +
      "-" +
      std::to_string(static_cast<int>(description.scale)) +
      "-" +
      std::to_string(static_cast<int>(reference));
  input.hard_obstacle_mask =
      GenerateG1ProxyObstacleMask(
          description.scale, reference, true);
  if (reference == G1ReferenceScenario::kHighFrontier) {
    const MultiscaleDimensions dimensions =
        DimensionsFor(description.scale);
    const std::size_t unknown_begin_x =
        G1UnknownBeginX(description.scale);
    for (std::size_t y = 0U; y < dimensions.height; ++y) {
      for (std::size_t x = unknown_begin_x;
           x < dimensions.width; ++x) {
        const std::size_t index = y * dimensions.width + x;
        input.known_mask[index] = 0U;
        input.confidence[index] = 0.0F;
        input.hard_obstacle_mask[index] = 0U;
      }
    }
  }
  return input;
}

void AddG1ProxyRegions(
    ScenarioDescription& description,
    const MapSnapshotInput& input) {
  const GridGeometry& geometry = input.geometry;
  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      const std::size_t index = y * geometry.width + x;
      if (input.hard_obstacle_mask[index] == 0U) {
        continue;
      }
      const double minimum_x =
          geometry.origin_m.x +
          static_cast<double>(x) * geometry.resolution_m;
      const double minimum_y =
          geometry.origin_m.y +
          static_cast<double>(y) * geometry.resolution_m;
      description.regions.push_back(RectangleRegion(
          ScenarioRegion::Kind::
              kSyntheticTerrainObstacleProxy,
          minimum_x, minimum_y,
          minimum_x + geometry.resolution_m,
          minimum_y + geometry.resolution_m));
    }
  }
}

[[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot>
MakeMap(
    const PlatformType platform_type,
    const MapScenario map_scenario) {
  auto created = ImmutableMapSnapshot::Create(
      BaseMapInput(platform_type, map_scenario));
  if (!IsOk(created)) {
    const Error& error = std::get<Error>(created);
    throw std::runtime_error{
        error.field_path + ": " + error.message};
  }
  return std::get<
      std::shared_ptr<const ImmutableMapSnapshot>>(
      std::move(created));
}

[[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot>
MakeG1MultiscaleMap(
    const PlatformType platform_type,
    ScenarioDescription& description) {
  MapSnapshotInput input =
      G1MultiscaleMapInput(platform_type, description);
  AddG1ProxyRegions(description, input);
  auto created =
      ImmutableMapSnapshot::Create(std::move(input));
  if (!IsOk(created)) {
    const Error& error = std::get<Error>(created);
    throw std::runtime_error{
        error.field_path + ": " + error.message};
  }
  return std::get<
      std::shared_ptr<const ImmutableMapSnapshot>>(
      std::move(created));
}

[[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot>
MakeMultiscaleMap(
    const PlatformType platform_type,
    const ScenarioDescription& description) {
  auto created = ImmutableMapSnapshot::Create(
      MultiscaleMapInput(platform_type, description));
  if (!IsOk(created)) {
    const Error& error = std::get<Error>(created);
    throw std::runtime_error{
        error.field_path + ": " + error.message};
  }
  return std::get<
      std::shared_ptr<const ImmutableMapSnapshot>>(
      std::move(created));
}

[[nodiscard]] WheelMotionPrimitive WheelPrimitive(
    std::string id, const WheelMotionPrimitive::Kind kind,
    const PoseXyzYaw relative_end_pose,
    const std::chrono::nanoseconds duration,
    const char digest_digit) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .relative_end_pose = relative_end_pose,
      .nominal_duration = DurationNanoseconds{duration},
      .swept_geometry_ref =
          Ref(
              "system-wheel-sweep-" +
                  std::string(1U, digest_digit),
              digest_digit),
  };
}

[[nodiscard]] SafetyCapabilityProfile WheelCapability() {
  using Kind = WheelMotionPrimitive::Kind;
  return {
      .content_ref = Ref("system-wheel-capability", 'a'),
      .content =
          WheeledCapability{
              .frame_id = "map",
              .collision_envelope =
                  {
                      .vertices_xy_m =
                          {
                              {-0.2, -0.2},
                              {0.2, -0.2},
                              {0.2, 0.2},
                              {-0.2, 0.2},
                          },
                      .minimum_z_m = -0.1,
                      .maximum_z_m = 0.5,
                  },
              .motion_model_ref =
                  Ref("system-wheel-motion", 'b'),
              .analytic_cost_model_ref =
                  Ref("system-wheel-cost", 'c'),
              .hard_limits =
                  {
                      .maximum_forward_speed_mps = 1.0,
                      .maximum_reverse_speed_mps = 0.8,
                      .maximum_spin_rate_radps = 1.0,
                      .maximum_forward_acceleration_mps2 = 1.0,
                      .maximum_braking_deceleration_mps2 = 1.0,
                      .maximum_yaw_acceleration_radps2 = 1.0,
                      .maximum_lateral_acceleration_mps2 = 1.0,
                      .maximum_drive_curvature_per_m = 1.0,
                      .maximum_slope_rad = 0.5,
                      .minimum_clearance_m = 0.0,
                  },
              .certified_state_error_bounds =
                  ZeroGroundError(),
              .motion_primitives =
                  {
                      WheelPrimitive(
                          "forward", Kind::kDriveForwardLine,
                          {{1.0, 0.0, 0.0}, 0.0}, 1s, '0'),
                      WheelPrimitive(
                          "forward-arc", Kind::kDriveForwardArc,
                          {{1.0, 0.0, 0.0},
                           std::numbers::pi / 2.0},
                          1s, '1'),
                      WheelPrimitive(
                          "reverse", Kind::kDriveReverseLine,
                          {{-1.0, 0.0, 0.0}, 0.0}, 1s, '2'),
                      WheelPrimitive(
                          "reverse-arc", Kind::kDriveReverseArc,
                          {{-1.0, 0.0, 0.0},
                           -std::numbers::pi / 2.0},
                          1s, '3'),
                      WheelPrimitive(
                          "spin-cw", Kind::kSpinCw,
                          {{0.0, 0.0, 0.0},
                           -std::numbers::pi / 2.0},
                          500ms, '4'),
                      WheelPrimitive(
                          "spin-ccw", Kind::kSpinCcw,
                          {{0.0, 0.0, 0.0},
                           std::numbers::pi / 2.0},
                          500ms, '5'),
                      WheelPrimitive(
                          "stop-switch", Kind::kStopAndSwitch,
                          {{0.0, 0.0, 0.0}, 0.0},
                          100ms, '6'),
                  },
          },
  };
}

[[nodiscard]] LeggedBodyPrimitive LeggedPrimitive(
    std::string id, const LeggedBodyPrimitive::Kind kind,
    const Vec3 displacement, const double yaw_change_rad,
    const char digest_digit) {
  return {
      .primitive_id = std::move(id),
      .kind = kind,
      .body_frame_displacement_m = displacement,
      .yaw_change_rad = yaw_change_rad,
      .nominal_duration = DurationNanoseconds{2s},
      .sampled_body_sweep_ref =
          Ref(
              "system-legged-sweep-" +
                  std::string(1U, digest_digit),
              digest_digit),
  };
}

[[nodiscard]] SafetyCapabilityProfile LeggedCapabilityProfile() {
  using Kind = LeggedBodyPrimitive::Kind;
  return {
      .content_ref = Ref("system-legged-capability", 'd'),
      .content =
          LeggedCapability{
              .frame_id = "map",
              .reference_point_id = "base_link",
              .reference_point_definition =
                  LeggedCapability::ReferencePointDefinition::
                      kFixedNominalCom,
              .collision_envelope =
                  {
                      .body_frame_halfspaces =
                          BoxPolytope(
                              0.2, 0.2, -0.3, 0.3),
                  },
              .motion_model_ref =
                  Ref("system-legged-motion", 'e'),
              .analytic_cost_model_ref =
                  Ref("system-legged-cost", 'f'),
              .terrain_thresholds =
                  {
                      .maximum_slope_rad = 0.5,
                      .maximum_roughness_m = 0.2,
                      .maximum_step_height_m = 0.3,
                      .maximum_gap_width_m = 0.4,
                      .minimum_confidence = 0.8,
                      .minimum_body_clearance_m = 0.0,
                      .minimum_body_height_m = 0.4,
                      .maximum_body_height_m = 0.6,
                  },
              .body_velocity_limits =
                  {
                      .forward_mps = {-0.5, 0.5},
                      .lateral_mps = {-0.5, 0.5},
                      .vertical_mps = {-0.1, 0.1},
                      .yaw_rate_radps = {-1.0, 1.0},
                      .linear_acceleration_mps2 = 0.5,
                      .yaw_acceleration_radps2 = 1.0,
                  },
              .certified_state_error_bounds =
                  ZeroGroundError(),
              .motion_primitives =
                  {
                      LeggedPrimitive(
                          "forward", Kind::kForward,
                          {1.0, 0.0, 0.0}, 0.0, '1'),
                      LeggedPrimitive(
                          "backward", Kind::kBackward,
                          {-1.0, 0.0, 0.0}, 0.0, '2'),
                      LeggedPrimitive(
                          "left", Kind::kLateral,
                          {0.0, 1.0, 0.0}, 0.0, '3'),
                      LeggedPrimitive(
                          "right", Kind::kLateral,
                          {0.0, -1.0, 0.0}, 0.0, '4'),
                      LeggedPrimitive(
                          "spin", Kind::kSpin,
                          {0.0, 0.0, 0.0},
                          std::numbers::pi / 2.0, '5'),
                  },
              .feasibility_scope =
                  LeggedCapability::FeasibilityScope::
                      kBodyGeometryAndTerrainThresholdsOnly,
              .footstep_feasibility_guaranteed = false,
          },
  };
}

[[nodiscard]] SafetyCapabilityProfile HopperCapabilityProfile() {
  return {
      .content_ref = Ref("system-hopper-capability", '1'),
      .content =
          HopperCapability{
              .frame_id = "map",
              .collision_envelope =
                  {
                      .body_frame_halfspaces =
                          BoxPolytope(
                              0.35, 0.25, -0.5, 0.5),
                  },
              .motion_model_ref =
                  Ref("system-hopper-motion", '2'),
              .analytic_cost_model_ref =
                  Ref("system-hopper-cost", '3'),
              .gravity_model_ref =
                  Ref("system-lunar-gravity", '4'),
              .landing_terrain_thresholds =
                  {
                      .maximum_slope_rad = 0.55,
                      .maximum_roughness_m = 0.1,
                      .maximum_plane_residual_m = 0.05,
                      .minimum_overhead_clearance_m = 0.0,
                      .minimum_lateral_clearance_m = 0.0,
                      .minimum_landing_region_area_m2 = 0.2,
                  },
              .launch_limits =
                  {
                      .maximum_launch_speed_mps = 8.0,
                      .maximum_launch_impulse_newton_seconds = 100.0,
                      .minimum_flight_time =
                          DurationNanoseconds{500ms},
                      .maximum_flight_time =
                          DurationNanoseconds{10s},
                      .maximum_landing_speed_mps = 8.0,
                      .minimum_downward_impact_speed_mps = 0.1,
                      .minimum_landing_clearance_m = 0.0,
                  },
              .attitude_envelope =
                  {
                      .maximum_angular_speed_radps = 2.0,
                      .maximum_angular_acceleration_radps2 = 4.0,
                      .maximum_initial_angular_speed_radps = 0.2,
                      .minimum_settle_guard =
                          DurationNanoseconds{100ms},
                  },
              .certified_state_error_bounds =
                  ZeroHopperError(),
          },
  };
}

[[nodiscard]] std::shared_ptr<const SafetyCapabilityProfile>
MakeCapability(const PlatformType platform_type) {
  SafetyCapabilityProfile capability =
      platform_type == PlatformType::kWheeled
          ? WheelCapability()
          : (platform_type == PlatformType::kLegged
                 ? LeggedCapabilityProfile()
                 : HopperCapabilityProfile());
  return std::make_shared<const SafetyCapabilityProfile>(
      std::move(capability));
}

[[nodiscard]] std::shared_ptr<const SafetyCapabilityProfile>
MakeMultiscaleCapability(
    const PlatformType platform_type,
    const MapScale map_scale) {
  if (platform_type == PlatformType::kHopper) {
    return MakeCapability(platform_type);
  }
  const double primitive_m =
      GroundPrimitiveDistanceFor(map_scale);
  SafetyCapabilityProfile capability =
      platform_type == PlatformType::kWheeled
          ? WheelCapability()
          : LeggedCapabilityProfile();
  if (platform_type == PlatformType::kWheeled) {
    auto& wheel = std::get<WheeledCapability>(capability.content);
    for (WheelMotionPrimitive& primitive :
         wheel.motion_primitives) {
      using Kind = WheelMotionPrimitive::Kind;
      if (primitive.kind == Kind::kDriveForwardLine ||
          primitive.kind == Kind::kDriveForwardArc ||
          primitive.kind == Kind::kDriveReverseLine ||
          primitive.kind == Kind::kDriveReverseArc) {
        primitive.relative_end_pose.position_m.x *= primitive_m;
        primitive.relative_end_pose.position_m.y *= primitive_m;
        primitive.relative_end_pose.position_m.z *= primitive_m;
        primitive.nominal_duration = DurationNanoseconds{
            std::chrono::milliseconds{
                static_cast<std::int64_t>(
                    primitive_m * 1250.0)}};
      }
    }
  } else {
    auto& legged = std::get<LeggedCapability>(capability.content);
    for (LeggedBodyPrimitive& primitive :
         legged.motion_primitives) {
      if (primitive.kind !=
          LeggedBodyPrimitive::Kind::kSpin) {
        primitive.body_frame_displacement_m.x *= primitive_m;
        primitive.body_frame_displacement_m.y *= primitive_m;
        primitive.body_frame_displacement_m.z *= primitive_m;
        primitive.nominal_duration = DurationNanoseconds{
            std::chrono::seconds{
                static_cast<std::int64_t>(
                    primitive_m * 2.0)}};
      }
    }
  }
  return std::make_shared<const SafetyCapabilityProfile>(
      std::move(capability));
}

[[nodiscard]] CorridorConfig ValidCorridor() {
  return {
      .maximum_regions = 16U,
      .maximum_inflation_iterations = 64U,
      .maximum_halfplanes_per_region = 16U,
      .maximum_split_depth = 4U,
      .minimum_overlap_m = 0.05,
      .sampling_spacing_m = 0.25,
  };
}

[[nodiscard]] SmoothingConfig ValidSmoothing() {
  return {
      .maximum_scp_iterations = 2U,
      .maximum_trust_region_reductions = 2U,
      .initial_trust_region_m = 0.2,
      .minimum_trust_region_m = 0.01,
      .constraint_tolerance = 1.0e-6,
      .maximum_time_increase = DurationNanoseconds{2s},
  };
}

[[nodiscard]] TimeScalingConfig ValidTimeScaling() {
  return {
      .maximum_adaptive_samples = 128U,
      .minimum_parameter_step = 0.01,
      .maximum_forward_passes = 2U,
      .maximum_backward_passes = 2U,
      .enable_jerk_smoothing = false,
      .maximum_jerk_smoothing_iterations = 0U,
  };
}

[[nodiscard]] std::shared_ptr<const PlannerAlgorithmConfig>
MakeAlgorithmConfig(const PlatformType) {
  PlannerAlgorithmConfig config{
      .content_ref =
          Ref("system-algorithm-config", 'a'),
      .time_equivalence_tolerance =
          DurationNanoseconds{100ms},
      .max_input_skew = DurationNanoseconds{1s},
      .error_bound_model_id = "system-error-bound",
      .projection_cache_capacity = 8U,
      .ara_star =
          {
              .initial_epsilon = 2.0,
              .epsilon_decrement = 0.5,
              .target_epsilon = 1.0,
              .resource_caps =
                  {
                      .maximum_expanded_states = 2048U,
                      .maximum_reopened_states = 2048U,
                      .maximum_generated_candidates = 16U,
                      .maximum_open_states = 2048U,
                      .maximum_memory_bytes = 1U << 20U,
                  },
          },
      .wheeled =
          {
              .state_lattice =
                  {
                      .xy_resolution_m = 1.0,
                      .yaw_bin_count = 4U,
                      .maximum_terminal_candidates = 16U,
                  },
              .corridor = ValidCorridor(),
              .smoothing = ValidSmoothing(),
              .time_scaling = ValidTimeScaling(),
              .continuous_validation_maximum_subdivisions = 16U,
          },
      .legged =
          {
              .pose_lattice =
                  {
                      .xy_resolution_m = 1.0,
                      .yaw_bin_count = 4U,
                      .maximum_terminal_candidates = 16U,
                  },
              .maximum_height_interval_splits = 8U,
              .corridor = ValidCorridor(),
              .smoothing = ValidSmoothing(),
              .time_scaling = ValidTimeScaling(),
              .continuous_validation_maximum_subdivisions = 16U,
          },
      .hopper =
          {
              .maximum_landing_regions = 16U,
              .maximum_graph_nodes = 16U,
              .maximum_graph_out_degree = 4U,
              .yaw_partition_count = 4U,
              .support_direction_count = 16U,
              .maximum_nominal_aim_points_per_region = 8U,
              .maximum_full_certification_attempts = 16U,
              .maximum_interval_subdivision_depth = 8U,
              .maximum_collision_subdivision_depth = 8U,
              .maximum_root_iterations = 64U,
              .maximum_flight_tube_sections = 64U,
              .landing_region_inflation_iterations = 16U,
              .landing_region_maximum_split_depth = 4U,
              .landing_region_maximum_vertices = 16U,
          },
      .learned_cost_policy =
          {
              .mode = LearnedCostPolicy::Mode::kDisabled,
              .maximum_inference_evaluations = 0U,
              .maximum_absolute_energy_correction = 0.0,
              .maximum_absolute_nonfatal_risk_correction = 0.0,
          },
      .deterministic_execution =
          {
              .fixed_thread_count = 1U,
              .stable_candidate_order = true,
              .preallocated_memory_pools = true,
          },
  };
  return std::make_shared<const PlannerAlgorithmConfig>(
      std::move(config));
}

[[nodiscard]] std::shared_ptr<const PlannerAlgorithmConfig>
MakeMultiscaleAlgorithmConfig(
    const PlatformType platform_type,
    const MapScale map_scale) {
  PlannerAlgorithmConfig config =
      *MakeAlgorithmConfig(platform_type);
  const double resolution_m =
      DimensionsFor(map_scale).resolution_m;
  config.wheeled.state_lattice.xy_resolution_m = resolution_m;
  config.legged.pose_lattice.xy_resolution_m = resolution_m;
  config.hopper.maximum_landing_regions = 64U;
  config.hopper.maximum_graph_nodes = 64U;
  return std::make_shared<const PlannerAlgorithmConfig>(
      std::move(config));
}

[[nodiscard]] ResolvedCapabilityBindings GroundBindings(
    const SafetyCapabilityProfile& capability) {
  const auto refs = std::visit(
      [](const auto& content) {
        return std::pair{
            content.motion_model_ref,
            content.analytic_cost_model_ref};
      },
      capability.content);
  return {
      .motion_model =
          {
              .content_ref = refs.first,
              .object = OpaqueResolvedObject<MotionModel>(),
          },
      .analytic_cost_model =
          {
              .content_ref = refs.second,
              .object =
                  OpaqueResolvedObject<AnalyticCostModel>(),
          },
  };
}

[[nodiscard]] ResolvedCapabilityBindings HopperBindings(
    const SafetyCapabilityProfile& capability) {
  const auto& hopper =
      std::get<HopperCapability>(capability.content);
  auto gravity = std::make_shared<const GravityModel>(
      GravityModel{
          .content_ref = hopper.gravity_model_ref,
          .frame_id = "map",
          .nominal_acceleration_mps2 = {0.0, 0.0, -1.62},
          .acceleration_error_mps2 =
              {
                  .center = {},
                  .half_extent = {0.0, 0.0, 0.01},
              },
          .spatial_validity_m =
              {
                  .center = {},
                  .half_extent =
                      {2000.0, 2000.0, 2000.0},
              },
          .valid_from =
              ClockStamp{
                  "mission", std::chrono::nanoseconds{0}},
          .valid_until =
              ClockStamp{"mission", std::chrono::hours{24}},
      });
  auto error =
      std::make_shared<const DeterministicErrorModel>(
          DeterministicErrorModel{
              .content_ref =
                  Ref("system-hopper-error", '5'),
              .initial_position_error_m =
                  {
                      .center = {},
                      .half_extent = {0.01, 0.01, 0.01},
                  },
              .initial_velocity_error_mps =
                  {
                      .center = {},
                      .half_extent = {0.01, 0.01, 0.01},
                  },
              .launch_execution_velocity_error_mps =
                  {
                      .center = {},
                      .half_extent = {0.01, 0.01, 0.01},
                  },
              .gravity_error_mps2 =
                  {
                      .center = {},
                      .half_extent = {0.0, 0.0, 0.01},
                  },
              .landing_plane_origin_error_m =
                  {
                      .center = {},
                      .half_extent = {0.01, 0.01, 0.01},
                  },
              .landing_plane_normal_error =
                  RotationVectorBall{0.01},
              .landing_plane_residual_error_m =
                  SymmetricScalarInterval{0.0, 0.01},
          });
  auto actuator =
      std::make_shared<const ActuatorOrImpulseProfile>(
          ActuatorOrImpulseProfile{
              .content_ref =
                  Ref("system-hopper-actuator", '6'),
              .platform_mass_kg = 10.0,
              .launch_preparation_time =
                  DurationNanoseconds{100ms},
              .landing_settle_time =
                  DurationNanoseconds{200ms},
              .nominal_landing_center_normal_offset_m = 0.5,
          });
  auto rotation =
      std::make_shared<const BodyRotationEnvelope>(
          BodyRotationEnvelope{
              .content_ref =
                  Ref("system-hopper-rotation", '7'),
              .arbitrary_attitude_body_envelope =
                  BoxPolytope(0.45, 0.45, -0.45, 0.45),
          });

  return {
      .motion_model =
          {
              .content_ref = hopper.motion_model_ref,
              .object = OpaqueResolvedObject<MotionModel>(),
          },
      .analytic_cost_model =
          {
              .content_ref =
                  hopper.analytic_cost_model_ref,
              .object =
                  OpaqueResolvedObject<AnalyticCostModel>(),
          },
      .gravity_model =
          ResolvedBinding<GravityModel>{
              gravity->content_ref, gravity},
      .error_model =
          ResolvedBinding<DeterministicErrorModel>{
              error->content_ref, error},
      .actuator_or_impulse_profile =
          ResolvedBinding<ActuatorOrImpulseProfile>{
              actuator->content_ref, actuator},
      .body_rotation_envelope =
          ResolvedBinding<BodyRotationEnvelope>{
              rotation->content_ref, rotation},
  };
}

[[nodiscard]] ResolvedCapabilityBindings MakeBindings(
    const SafetyCapabilityProfile& capability,
    const PlatformType platform_type) {
  return platform_type == PlatformType::kHopper
             ? HopperBindings(capability)
             : GroundBindings(capability);
}

class TestContractObject final
    : public ImmutableContractObject {
 public:
  TestContractObject(
      ContentRef ref, const ContractObjectKind kind)
      : ref_(std::move(ref)), kind_(kind) {}

  [[nodiscard]] ContentRef content_ref() const override {
    return ref_;
  }

  [[nodiscard]] ContractObjectKind kind() const override {
    return kind_;
  }

 private:
  ContentRef ref_;
  ContractObjectKind kind_;
};

class TestCertificate final : public CertificationObject {
 public:
  TestCertificate(
      ContentRef ref, CertificationProvenance provenance)
      : ref_(std::move(ref)),
        provenance_(std::move(provenance)) {}

  [[nodiscard]] ContentRef content_ref() const override {
    return ref_;
  }

  [[nodiscard]] ContractObjectKind kind() const override {
    return ContractObjectKind::kCertification;
  }

  [[nodiscard]] const CertificationProvenance& provenance()
      const override {
    return provenance_;
  }

 private:
  ContentRef ref_;
  CertificationProvenance provenance_;
};

[[nodiscard]] bool SameRef(
    const ContentRef& left, const ContentRef& right) {
  return left == right;
}

class FixedSystemRegistry final
    : public ContractObjectRegistry {
 public:
  FixedSystemRegistry(
      std::shared_ptr<const ImmutableMapSnapshot> map,
      std::shared_ptr<const SafetyCapabilityProfile> capability,
      std::shared_ptr<const PlannerAlgorithmConfig> config,
      ResolvedCapabilityBindings bindings,
      const RegistryFault fault)
      : map_(std::move(map)),
        capability_(std::move(capability)),
        config_(std::move(config)),
        bindings_(std::move(bindings)),
        fault_(fault) {}

  [[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot>
  FindMapSnapshot(
      const ContentRef& ref,
      const std::string_view handle) const override {
    return map_ && SameRef(map_->snapshot_ref(), ref) &&
                   map_->immutable_data_handle() == handle
               ? map_
               : nullptr;
  }

  [[nodiscard]]
  std::shared_ptr<const SafetyCapabilityProfile>
  FindSafetyCapability(
      const ContentRef& ref) const override {
    if (fault_ == RegistryFault::kMissingCapability) {
      return nullptr;
    }
    return capability_ &&
                   SameRef(capability_->content_ref, ref)
               ? capability_
               : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const PlannerAlgorithmConfig>
  FindAlgorithmConfig(
      const ContentRef& ref) const override {
    return config_ && SameRef(config_->content_ref, ref)
               ? config_
               : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const LearnedCostSnapshot>
  FindLearnedCost(
      const ContentRef&, std::string_view) const override {
    return nullptr;
  }

  [[nodiscard]] Result<ResolvedCapabilityBindings>
  ResolveCapabilityBindings(
      const SafetyCapabilityProfile& profile) const override {
    if (!capability_ ||
        profile.content_ref != capability_->content_ref) {
      return Error{
          ErrorCode::kMissingRegistryObject,
          "safety_capability_ref",
          "system fixture capability is absent",
      };
    }
    ResolvedCapabilityBindings result = bindings_;
    if (fault_ ==
        RegistryFault::kCapabilityBindingsMismatch) {
      result.motion_model.content_ref =
          Ref("mismatched-motion-model", '0');
    }
    return result;
  }

  [[nodiscard]] Result<
      std::shared_ptr<const ImmutableContractObject>>
  Resolve(
      const ContentRef& ref,
      const ContractObjectKind expected_kind) const override {
    if (expected_kind ==
        ContractObjectKind::kCertification) {
      std::vector<ContentRef> inputs{
          bindings_.motion_model.content_ref,
          bindings_.analytic_cost_model.content_ref,
      };
      const auto append =
          [&inputs](const auto& optional_binding) {
            if (optional_binding.has_value()) {
              inputs.push_back(
                  optional_binding->content_ref);
            }
          };
      append(bindings_.gravity_model);
      append(bindings_.error_model);
      append(bindings_.actuator_or_impulse_profile);
      append(bindings_.body_rotation_envelope);
      append(bindings_.attitude_tightening_table);
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestCertificate>(
              ref,
              CertificationProvenance{
                  .source_map_snapshot_ref =
                      map_->snapshot_ref(),
                  .source_safety_capability_ref =
                      capability_->content_ref,
                  .source_algorithm_config_ref =
                      config_->content_ref,
                  .input_refs = std::move(inputs),
                  .certification_purpose =
                      "system-integration-fixture",
              }));
    }

    if (bindings_.gravity_model.has_value() &&
        bindings_.gravity_model->content_ref == ref &&
        expected_kind == ContractObjectKind::kGravityModel) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    if (bindings_.error_model.has_value() &&
        bindings_.error_model->content_ref == ref &&
        expected_kind ==
            ContractObjectKind::kDeterministicErrorModel) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    if (bindings_.actuator_or_impulse_profile.has_value() &&
        bindings_.actuator_or_impulse_profile->content_ref == ref &&
        expected_kind ==
            ContractObjectKind::kActuatorOrImpulseProfile) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    if (bindings_.body_rotation_envelope.has_value() &&
        bindings_.body_rotation_envelope->content_ref == ref &&
        expected_kind ==
            ContractObjectKind::kBodyRotationEnvelope) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    if (bindings_.attitude_tightening_table.has_value() &&
        bindings_.attitude_tightening_table->content_ref == ref &&
        expected_kind ==
            ContractObjectKind::kAttitudeTighteningTable) {
      return std::shared_ptr<const ImmutableContractObject>(
          std::make_shared<const TestContractObject>(
              ref, expected_kind));
    }
    return Error{
        ErrorCode::kMissingRegistryObject,
        "content_ref",
        "system fixture contract object is absent",
    };
  }

 private:
  std::shared_ptr<const ImmutableMapSnapshot> map_;
  std::shared_ptr<const SafetyCapabilityProfile> capability_;
  std::shared_ptr<const PlannerAlgorithmConfig> config_;
  ResolvedCapabilityBindings bindings_;
  RegistryFault fault_;
};

[[nodiscard]] GoalRegion GoalFor(
    const PlatformType platform_type,
    const MapScenario map_scenario) {
  if (platform_type == PlatformType::kHopper) {
    return {
        .goal_id = "system-hopper-goal",
        .target =
            PointGoal{
                .position_m = {4.0, 3.0, 0.0},
                .position_tolerance_m = 0.5,
            },
        .optional_yaw_interval =
            CircularYawInterval{
                .start_rad = 0.0,
                .span_rad = 0.0,
            },
    };
  }
  const double goal_x =
      map_scenario ==
              MapScenario::kUnknownGoalWithSafeFrontier
          ? 8.5
          : 4.5;
  return {
      .goal_id =
          platform_type == PlatformType::kWheeled
              ? "system-wheel-goal"
              : "system-legged-goal",
      .target =
          PointGoal{
              .position_m = {goal_x, 3.5, 0.0},
              .position_tolerance_m = 0.2,
          },
  };
}

[[nodiscard]] GoalRegion MultiscaleGoalFor(
    const PlatformType platform_type,
    const ScenarioDescription& description) {
  GoalRegion goal{
      .goal_id =
          platform_type == PlatformType::kWheeled
              ? "multiscale-wheel-goal"
              : (platform_type == PlatformType::kLegged
                     ? "multiscale-legged-goal"
                     : "multiscale-hopper-goal"),
      .target =
          PointGoal{
              .position_m =
                  {
                      description.goal_xy_m.x,
                      description.goal_xy_m.y,
                      0.0,
                  },
              .position_tolerance_m =
                  platform_type == PlatformType::kHopper
                      ? 0.5
                      : 0.2,
          },
  };
  if (platform_type == PlatformType::kHopper) {
    goal.optional_yaw_interval =
        CircularYawInterval{
            .start_rad = 0.0,
            .span_rad = 0.0,
        };
  }
  return goal;
}

[[nodiscard]] PlatformState StateFor(
    const PlatformType platform_type) {
  if (platform_type == PlatformType::kHopper) {
    return HopperState{
        .position_m = {3.0, 3.0, 0.5},
        .orientation_body_to_frame =
            {1.0, 0.0, 0.0, 0.0},
        .linear_velocity_mps = {},
        .angular_velocity_radps = {},
        .error_bounds = ZeroHopperError(),
    };
  }
  return WheeledOrLeggedState{
      .position_m =
          {
              2.5,
              3.5,
              platform_type == PlatformType::kLegged
                  ? 0.5
                  : 0.0,
          },
      .yaw_rad = 0.0,
      .linear_velocity_mps = {},
      .yaw_rate_radps = 0.0,
      .error_bounds = ZeroGroundError(),
  };
}

[[nodiscard]] PlatformState MultiscaleStateFor(
    const PlatformType platform_type,
    const ScenarioDescription& description) {
  if (platform_type == PlatformType::kHopper) {
    return HopperState{
        .position_m =
            {
                description.start_xy_m.x,
                description.start_xy_m.y,
                0.5,
            },
        .orientation_body_to_frame =
            {1.0, 0.0, 0.0, 0.0},
        .linear_velocity_mps = {},
        .angular_velocity_radps = {},
        .error_bounds = ZeroHopperError(),
    };
  }
  return WheeledOrLeggedState{
      .position_m =
          {
              description.start_xy_m.x,
              description.start_xy_m.y,
              platform_type == PlatformType::kLegged
                  ? 0.5
                  : 0.0,
          },
      .yaw_rad = 0.0,
      .linear_velocity_mps = {},
      .yaw_rate_radps = 0.0,
      .error_bounds = ZeroGroundError(),
  };
}

}  // namespace

SystemScenario MakeSystemScenario(
    const PlatformType platform_type,
    const MapScenario map_scenario,
    const RegistryFault registry_fault) {
  auto map = MakeMap(platform_type, map_scenario);
  auto capability = MakeCapability(platform_type);
  auto config = MakeAlgorithmConfig(platform_type);
  ResolvedCapabilityBindings bindings =
      MakeBindings(*capability, platform_type);
  auto registry =
      std::make_shared<const FixedSystemRegistry>(
          map, capability, config, bindings, registry_fault);
  PlanningRequest request{
      .request_id =
          "system-request-" +
          std::to_string(static_cast<int>(platform_type)) +
          "-" +
          std::to_string(static_cast<int>(map_scenario)),
      .request_time =
          ClockStamp{
              "mission", std::chrono::nanoseconds{100}},
      .state_time =
          ClockStamp{
              "mission", std::chrono::nanoseconds{100}},
      .frame_id = "map",
      .platform_type = platform_type,
      .current_state = StateFor(platform_type),
      .goal = GoalFor(platform_type, map_scenario),
      .map_snapshot = std::move(map),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(config),
      .capability_bindings = std::move(bindings),
  };
  const std::size_t cache_capacity =
      request.algorithm_config->projection_cache_capacity;
  return {
      .request = std::move(request),
      .registry = std::move(registry),
      .projection_cache =
          std::make_unique<SafeProjectionCache>(
              cache_capacity),
      .description = {},
  };
}

SystemScenario MakeMultiscaleSystemScenario(
    const PlatformType platform_type,
    const MapScale map_scale,
    const MapScenario map_scenario) {
  ScenarioDescription description = DescribeMultiscaleScenario(
      platform_type, map_scale, map_scenario);
  auto map = MakeMultiscaleMap(platform_type, description);
  auto capability =
      MakeMultiscaleCapability(platform_type, map_scale);
  auto config =
      MakeMultiscaleAlgorithmConfig(platform_type, map_scale);
  ResolvedCapabilityBindings bindings =
      MakeBindings(*capability, platform_type);
  auto registry =
      std::make_shared<const FixedSystemRegistry>(
          map, capability, config, bindings, RegistryFault::kNone);
  PlanningRequest request{
      .request_id =
          "multiscale-request-" +
          std::to_string(static_cast<int>(platform_type)) +
          "-" +
          std::to_string(static_cast<int>(map_scale)) +
          "-" +
          std::to_string(static_cast<int>(map_scenario)),
      .request_time =
          ClockStamp{
              "mission", std::chrono::nanoseconds{100}},
      .state_time =
          ClockStamp{
              "mission", std::chrono::nanoseconds{100}},
      .frame_id = "map",
      .platform_type = platform_type,
      .current_state =
          MultiscaleStateFor(platform_type, description),
      .goal = MultiscaleGoalFor(platform_type, description),
      .map_snapshot = std::move(map),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(config),
      .capability_bindings = std::move(bindings),
  };
  const std::size_t cache_capacity =
      request.algorithm_config->projection_cache_capacity;
  return {
      .request = std::move(request),
      .registry = std::move(registry),
      .projection_cache =
          std::make_unique<SafeProjectionCache>(
              cache_capacity),
      .description = std::move(description),
  };
}

SystemScenario MakeMultiscaleSystemScenario(
    const PlatformType platform_type,
    const MapScale map_scale,
    const G1ReferenceScenario reference_scenario) {
  ScenarioDescription description =
      DescribeG1MultiscaleScenario(
          platform_type, map_scale, reference_scenario);
  auto map =
      MakeG1MultiscaleMap(platform_type, description);
  auto capability =
      MakeMultiscaleCapability(platform_type, map_scale);
  auto config =
      MakeMultiscaleAlgorithmConfig(platform_type, map_scale);
  ResolvedCapabilityBindings bindings =
      MakeBindings(*capability, platform_type);
  auto registry =
      std::make_shared<const FixedSystemRegistry>(
          map, capability, config, bindings, RegistryFault::kNone);
  PlanningRequest request{
      .request_id =
          "multiscale-g1-request-" +
          std::to_string(static_cast<int>(platform_type)) +
          "-" +
          std::to_string(static_cast<int>(map_scale)) +
          "-" +
          std::to_string(static_cast<int>(reference_scenario)),
      .request_time =
          ClockStamp{
              "mission", std::chrono::nanoseconds{100}},
      .state_time =
          ClockStamp{
              "mission", std::chrono::nanoseconds{100}},
      .frame_id = "map",
      .platform_type = platform_type,
      .current_state =
          MultiscaleStateFor(platform_type, description),
      .goal = MultiscaleGoalFor(platform_type, description),
      .map_snapshot = std::move(map),
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(config),
      .capability_bindings = std::move(bindings),
  };
  const std::size_t cache_capacity =
      request.algorithm_config->projection_cache_capacity;
  return {
      .request = std::move(request),
      .registry = std::move(registry),
      .projection_cache =
          std::make_unique<SafeProjectionCache>(
              cache_capacity),
      .description = std::move(description),
  };
}

std::unique_ptr<PlannerV3> MakePlanner(
    SystemScenario& scenario) {
  static const SemanticValidator validator;
  return MakeDefaultPlannerV3(
      validator, *scenario.registry,
      *scenario.projection_cache);
}

PreviousExecutionContext MakeGroundExecutionContext(
    const SystemScenario& scenario,
    const bool stale_source_map) {
  ContentRef source_map =
      scenario.request.map_snapshot->snapshot_ref();
  if (stale_source_map) {
    source_map = Ref("stale-system-map", 'f');
  }
  return {
      .active_bundle_ref =
          Ref("active-system-bundle", 'e'),
      .active_bundle_handle = "active-system-bundle-handle",
      .commit_boundary =
          TimeCommitBoundary{
              DurationNanoseconds{500ms}},
      .execution_cursor =
          TimeExecutionCursor{
              .offset = DurationNanoseconds{100ms},
          },
      .controller_status = ControllerStatus::kExecuting,
      .source_map_snapshot_ref = std::move(source_map),
      .source_capability_ref =
          scenario.request.safety_capability->content_ref,
  };
}

Result<Sha256Digest> ResponseHash(
    const PlanningResponse& response) {
  return Sha256Hex(
      JsonCodec::EncodePlanningResponse(response));
}

std::string DescribeIssues(
    const ValidationReport& report) {
  std::ostringstream description;
  for (const ValidationIssue& issue : report.issues) {
    description << issue.field_path << " ["
                << issue.reason_code << "]: "
                << issue.message << '\n';
  }
  return description.str();
}

}  // namespace lunar::planning::v3::system_test
