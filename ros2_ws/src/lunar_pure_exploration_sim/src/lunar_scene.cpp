#include "lunar_pure_exploration_sim/lunar_scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace lunar::pure_exploration_sim {
namespace {

constexpr double kMinCoordinateM = -150.0;
constexpr double kMaxCoordinateM = 150.0;
constexpr double kGlobalResolutionM = 1.0;
constexpr std::size_t kGlobalWidth = 300U;
constexpr std::size_t kGlobalHeight = 300U;
constexpr double kStartClearRadiusM = 4.0;
constexpr double kBackboneHalfWidthM = 2.0;
constexpr double kLoopHalfExtentM = 80.0;
constexpr double kRoughnessOffsetM = 0.2;
constexpr double kSpatialBucketSizeM = 5.0;
constexpr std::size_t kSpatialBucketWidth = 60U;
constexpr std::size_t kSpatialBucketCount =
    kSpatialBucketWidth * kSpatialBucketWidth;
constexpr double kPi = 3.14159265358979323846;

double UnitRandom(std::mt19937& generator) {
  return std::generate_canonical<double, 53>(generator);
}

double UniformRandom(std::mt19937& generator, double minimum,
                     double maximum) {
  return minimum + (maximum - minimum) * UnitRandom(generator);
}

double Distance(double first_x_m, double first_y_m, double second_x_m,
                double second_y_m) {
  return std::hypot(first_x_m - second_x_m, first_y_m - second_y_m);
}

double BaseElevation(double world_x_m, double world_y_m) {
  return 1.4 * std::sin(0.015 * world_x_m) +
         0.9 * std::cos(0.019 * world_y_m) +
         0.45 * std::sin(0.011 * (world_x_m + world_y_m));
}

}  // namespace

LunarScene BuildLunarScene(std::uint32_t seed) { return LunarScene(seed); }

LunarScene::LunarScene(std::uint32_t seed) {
  std::mt19937 generator(seed);

  constexpr std::size_t kStandaloneRockCount = 75U;
  rocks_.reserve(kStandaloneRockCount + 10U * 8U);
  for (std::size_t index = 0; index < kStandaloneRockCount; ++index) {
    rocks_.push_back(Rock{
        .x_m = UniformRandom(generator, -145.0, 145.0),
        .y_m = UniformRandom(generator, -145.0, 145.0),
        .radius_m = UniformRandom(generator, 0.7, 2.4),
        .height_m = UniformRandom(generator, 0.4, 2.2),
    });
  }

  constexpr std::size_t kClusterCount = 10U;
  constexpr std::size_t kRocksPerCluster = 8U;
  for (std::size_t cluster = 0; cluster < kClusterCount; ++cluster) {
    const double cluster_x_m = UniformRandom(generator, -135.0, 135.0);
    const double cluster_y_m = UniformRandom(generator, -135.0, 135.0);
    for (std::size_t index = 0; index < kRocksPerCluster; ++index) {
      const double angle_rad = UniformRandom(generator, 0.0, 2.0 * kPi);
      const double distance_m = 7.0 * std::sqrt(UnitRandom(generator));
      rocks_.push_back(Rock{
          .x_m = cluster_x_m + distance_m * std::cos(angle_rad),
          .y_m = cluster_y_m + distance_m * std::sin(angle_rad),
          .radius_m = UniformRandom(generator, 0.6, 1.8),
          .height_m = UniformRandom(generator, 0.35, 1.7),
      });
    }
  }

  constexpr std::size_t kCraterCount = 16U;
  craters_.reserve(kCraterCount);
  for (std::size_t index = 0; index < kCraterCount; ++index) {
    craters_.push_back(Crater{
        .x_m = UniformRandom(generator, -132.0, 132.0),
        .y_m = UniformRandom(generator, -132.0, 132.0),
        .radius_m = UniformRandom(generator, 5.0, 13.0),
        .rim_half_width_m = UniformRandom(generator, 0.65, 1.45),
        .depth_m = UniformRandom(generator, 0.8, 3.2),
        .rim_height_m = UniformRandom(generator, 0.35, 1.3),
    });
  }

  BuildSpatialIndex();

  global_occupancy_.reserve(kGlobalWidth * kGlobalHeight);
  for (std::size_t row = 0; row < kGlobalHeight; ++row) {
    const double world_y_m =
        kMinCoordinateM + (static_cast<double>(row) + 0.5) *
                              kGlobalResolutionM;
    for (std::size_t column = 0; column < kGlobalWidth; ++column) {
      const double world_x_m =
          kMinCoordinateM + (static_cast<double>(column) + 0.5) *
                                kGlobalResolutionM;
      global_occupancy_.push_back(
          EvaluateSurface(world_x_m, world_y_m).occupied
              ? std::int8_t{100}
              : std::int8_t{0});
    }
  }
}

double LunarScene::min_x_m() const noexcept { return kMinCoordinateM; }

double LunarScene::max_x_m() const noexcept { return kMaxCoordinateM; }

std::size_t LunarScene::global_width() const noexcept { return kGlobalWidth; }

std::size_t LunarScene::global_height() const noexcept {
  return kGlobalHeight;
}

double LunarScene::global_resolution_m() const noexcept {
  return kGlobalResolutionM;
}

const std::vector<std::int8_t>& LunarScene::GlobalOccupancy() const noexcept {
  return global_occupancy_;
}

bool LunarScene::IsOccupied(double world_x_m, double world_y_m) const {
  if (IsReservedFree(world_x_m, world_y_m)) {
    return false;
  }
  const auto bucket = SpatialBucketIndex(world_x_m, world_y_m);
  if (!bucket.has_value()) {
    return false;
  }
  for (const std::uint16_t index : crater_buckets_[*bucket]) {
    const auto& crater = craters_[index];
    const double distance_m =
        Distance(world_x_m, world_y_m, crater.x_m, crater.y_m);
    if (std::abs(distance_m - crater.radius_m) <= crater.rim_half_width_m) {
      return true;
    }
  }
  for (const std::uint16_t index : rock_buckets_[*bucket]) {
    const auto& rock = rocks_[index];
    const double delta_x_m = world_x_m - rock.x_m;
    const double delta_y_m = world_y_m - rock.y_m;
    if (delta_x_m * delta_x_m + delta_y_m * delta_y_m <
        rock.radius_m * rock.radius_m) {
      return true;
    }
  }
  return false;
}

TruthSample LunarScene::Sample(double world_x_m, double world_y_m) const {
  const SurfaceSample center = EvaluateSurface(world_x_m, world_y_m);
  const std::array<SurfaceSample, 4> neighbors{
      EvaluateSurface(world_x_m - kRoughnessOffsetM, world_y_m),
      EvaluateSurface(world_x_m + kRoughnessOffsetM, world_y_m),
      EvaluateSurface(world_x_m, world_y_m - kRoughnessOffsetM),
      EvaluateSurface(world_x_m, world_y_m + kRoughnessOffsetM),
  };

  double roughness = 0.0;
  for (const auto& neighbor : neighbors) {
    roughness = std::max(
        roughness, std::abs(center.elevation_m - neighbor.elevation_m));
  }
  return TruthSample{
      .occupied = center.occupied,
      .semantic_id = center.semantic_id,
      .elevation_m = center.elevation_m,
      .roughness = roughness,
  };
}

LunarScene::SurfaceSample LunarScene::EvaluateSurface(double world_x_m,
                                                      double world_y_m) const {
  double elevation_m = BaseElevation(world_x_m, world_y_m);
  if (IsReservedFree(world_x_m, world_y_m)) {
    return SurfaceSample{
        .occupied = false,
        .semantic_id = 0U,
        .elevation_m = elevation_m,
    };
  }

  bool crater_rim_occupied = false;
  const auto bucket = SpatialBucketIndex(world_x_m, world_y_m);
  if (!bucket.has_value()) {
    return SurfaceSample{
        .occupied = false,
        .semantic_id = 0U,
        .elevation_m = elevation_m,
    };
  }
  for (const std::uint16_t index : crater_buckets_[*bucket]) {
    const auto& crater = craters_[index];
    const double distance_m =
        Distance(world_x_m, world_y_m, crater.x_m, crater.y_m);
    const double rim_distance_m = std::abs(distance_m - crater.radius_m);
    if (rim_distance_m <= crater.rim_half_width_m) {
      crater_rim_occupied = true;
      elevation_m += crater.rim_height_m *
                     (1.0 - rim_distance_m / crater.rim_half_width_m);
    }

    const double bowl_radius_m = crater.radius_m - crater.rim_half_width_m;
    if (distance_m < bowl_radius_m) {
      const double normalized_distance = distance_m / bowl_radius_m;
      elevation_m -= crater.depth_m *
                     (1.0 - normalized_distance * normalized_distance);
    }
  }

  bool rock_occupied = false;
  for (const std::uint16_t index : rock_buckets_[*bucket]) {
    const auto& rock = rocks_[index];
    const double distance_m =
        Distance(world_x_m, world_y_m, rock.x_m, rock.y_m);
    if (distance_m < rock.radius_m) {
      rock_occupied = true;
      const double normalized_distance = distance_m / rock.radius_m;
      const double profile = 1.0 - normalized_distance * normalized_distance;
      elevation_m += rock.height_m * profile * profile;
    }
  }

  return SurfaceSample{
      .occupied = crater_rim_occupied || rock_occupied,
      .semantic_id = static_cast<std::uint8_t>(
          crater_rim_occupied ? 2U : (rock_occupied ? 1U : 0U)),
      .elevation_m = elevation_m,
  };
}

void LunarScene::BuildSpatialIndex() {
  rock_buckets_.assign(kSpatialBucketCount, {});
  crater_buckets_.assign(kSpatialBucketCount, {});
  const auto bucket_coordinate = [](double coordinate_m) {
    return std::clamp(
        static_cast<long>(std::floor(
            (coordinate_m - kMinCoordinateM) / kSpatialBucketSizeM)),
        0L, static_cast<long>(kSpatialBucketWidth) - 1L);
  };
  const auto add_bounds = [&](double center_x_m, double center_y_m,
                              double radius_m, std::uint16_t primitive_index,
                              auto& buckets) {
    const long minimum_x = bucket_coordinate(center_x_m - radius_m);
    const long maximum_x = bucket_coordinate(center_x_m + radius_m);
    const long minimum_y = bucket_coordinate(center_y_m - radius_m);
    const long maximum_y = bucket_coordinate(center_y_m + radius_m);
    for (long y = minimum_y; y <= maximum_y; ++y) {
      for (long x = minimum_x; x <= maximum_x; ++x) {
        buckets[static_cast<std::size_t>(y) * kSpatialBucketWidth +
                static_cast<std::size_t>(x)]
            .push_back(primitive_index);
      }
    }
  };
  for (std::size_t index = 0U; index < rocks_.size(); ++index) {
    const auto& rock = rocks_[index];
    add_bounds(rock.x_m, rock.y_m, rock.radius_m,
               static_cast<std::uint16_t>(index), rock_buckets_);
  }
  for (std::size_t index = 0U; index < craters_.size(); ++index) {
    const auto& crater = craters_[index];
    add_bounds(crater.x_m, crater.y_m,
               crater.radius_m + crater.rim_half_width_m,
               static_cast<std::uint16_t>(index), crater_buckets_);
  }
}

std::optional<std::size_t> LunarScene::SpatialBucketIndex(
    double world_x_m, double world_y_m) const noexcept {
  if (!std::isfinite(world_x_m) || !std::isfinite(world_y_m) ||
      world_x_m < kMinCoordinateM || world_x_m >= kMaxCoordinateM ||
      world_y_m < kMinCoordinateM || world_y_m >= kMaxCoordinateM) {
    return std::nullopt;
  }
  const auto x = static_cast<std::size_t>(
      std::floor((world_x_m - kMinCoordinateM) / kSpatialBucketSizeM));
  const auto y = static_cast<std::size_t>(
      std::floor((world_y_m - kMinCoordinateM) / kSpatialBucketSizeM));
  return y * kSpatialBucketWidth + x;
}

bool LunarScene::IsReservedFree(double world_x_m,
                                double world_y_m) noexcept {
  if (std::hypot(world_x_m, world_y_m) <= kStartClearRadiusM) {
    return true;
  }

  if (std::abs(world_x_m) <= kBackboneHalfWidthM ||
      std::abs(world_y_m) <= kBackboneHalfWidthM) {
    return true;
  }

  const double square_radius_m =
      std::max(std::abs(world_x_m), std::abs(world_y_m));
  return std::abs(square_radius_m - kLoopHalfExtentM) <=
         kBackboneHalfWidthM;
}

}  // namespace lunar::pure_exploration_sim
