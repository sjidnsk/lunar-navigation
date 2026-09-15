// Independent transport integration oracle; deliberately uses the persistent
// producer and public fine builder instead of the extension's dense GridView.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
using namespace lunar::incremental_navigation;
template <class T>
void read(T& value) {
  std::cin.read(reinterpret_cast<char*>(&value), sizeof(T));
}
int main() {
  int width, height, bx, by;
  double resolution, ox, oy;
  read(width);
  read(height);
  read(bx);
  read(by);
  read(resolution);
  read(ox);
  read(oy);
  WheeledCapability wheel;
  read(wheel.maximum_slope_rad);
  read(wheel.maximum_local_obstacle_relief_m);
  read(wheel.minimum_underbody_clearance_m);
  read(wheel.minimum_clearance_m);
  for (int i = 0; i < 4; ++i) {
    double x, y;
    read(x);
    read(y);
    wheel.footprint_xy_m.push_back({x, y});
  }
  std::vector<float> heights(width * height), wire(width * height * 4);
  std::cin.read(reinterpret_cast<char*>(heights.data()),
                heights.size() * sizeof(float));
  std::cin.read(reinterpret_cast<char*>(wire.data()),
                wire.size() * sizeof(float));
  if (!std::cin) return 2;
  std::vector<LocalTerrainMeasurements> stats(heights.size());
  for (std::size_t i = 0; i < heights.size(); ++i)
    if (std::isfinite(heights[i]))
      stats[i] = {true, wire[4 * i + 3] == 1.f, wire[4 * i], wire[4 * i + 1],
                  wire[4 * i + 2]};
  PersistentElevationMap producer;
  GridGeometry geometry{.frame_id = "map",
                        .width = static_cast<std::size_t>(width),
                        .height = static_cast<std::size_t>(height),
                        .resolution_m = resolution,
                        .origin_m = {ox, oy, 0}};
  const RigidTransform identity{.parent_frame = "map", .child_frame = "map"};
  GridGeometry bootstrap = geometry;
  bootstrap.width = 1;
  bootstrap.height = 1;
  bootstrap.origin_m = {ox + bx * resolution, oy + by * resolution, 0};
  std::vector<float> initial{heights[by * width + bx]};
  std::vector<LocalTerrainMeasurements> first{stats[by * width + bx]};
  if (producer.Apply({bootstrap, initial, identity, first}).status !=
      ElevationUpdateResult::Status::kApplied)
    return 3;
  if (producer.Apply({geometry, heights, identity, stats}).status ==
      ElevationUpdateResult::Status::kRejected)
    return 4;
  auto raw = producer.Snapshot();
  TraversabilityProfile profile;
  profile.planar_envelope_xy_m = wheel.footprint_xy_m;
  profile.preferred_clearance_m = wheel.minimum_clearance_m;
  profile.slope_weight = 1;
  profile.relief_weight = 1;
  auto fine = FineTraversabilityBuilder().Derive(raw, wheel, profile);
  auto evaluator = MakePlatformElevationEvaluator(wheel);
  std::vector<std::uint8_t> output(3 * heights.size());
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const double wx = ox + (x + .5) * resolution,
                   wy = oy + (y + .5) * resolution;
      const auto origin = raw->geometry().origin_m();
      GridIndex index{
          static_cast<std::int64_t>(std::floor((wx - origin.x) / resolution)),
          static_cast<std::int64_t>(std::floor((wy - origin.y) / resolution))};
      auto k = y * width + x;
      output[k] =
          static_cast<std::uint8_t>(evaluator->Evaluate(*raw, index).state);
      output[heights.size() + k] =
          static_cast<std::uint8_t>(fine->State(index));
      output[2 * heights.size() + k] = raw->ElevationRangeAt(index).has_value();
    }
  std::cout.write(reinterpret_cast<const char*>(output.data()), output.size());
}
