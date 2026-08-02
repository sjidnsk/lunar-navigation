#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lunar_path_planner/v3/contracts/base_types.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

enum class LayerKind {
  kKnownMask,
  kElevation,
  kTerrainNormal,
  kRoughness,
  kHardObstacle,
  kConfidence,
  kEsdf,
  kStaticSpeedLimit,
};

struct MapBounds final {
  Vec3 minimum_m;
  Vec3 maximum_m;
};

struct GridGeometry final {
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  Vec2 origin_m;
  std::string frame_id;

  [[nodiscard]] std::size_t CellCount() const noexcept {
    if (width == 0U || height == 0U ||
        height > std::numeric_limits<std::size_t>::max() / width) {
      return 0U;
    }
    return width * height;
  }
};

struct LayerManifestEntry final {
  LayerKind layer_kind{};
  ContentRef content_ref;
};

struct SurfaceNormalGridView final {
  std::span<const float> x;
  std::span<const float> y;
  std::span<const float> z;
};

struct MapSnapshotInput final {
  ContentRef snapshot_ref;
  std::uint32_t map_revision{};
  std::string immutable_data_handle;
  ClockStamp source_time;
  MapBounds bounds;
  GridGeometry geometry;
  std::vector<LayerManifestEntry> layer_manifest;
  std::vector<std::uint8_t> known_mask;
  std::vector<float> elevation_m;
  std::vector<float> normal_x;
  std::vector<float> normal_y;
  std::vector<float> normal_z;
  std::vector<float> roughness_m;
  std::vector<std::uint8_t> hard_obstacle_mask;
  std::vector<float> confidence;
  std::vector<float> esdf_m;
  std::vector<float> static_speed_limit_mps;
};

class ImmutableMapSnapshot final {
 public:
  [[nodiscard]] static Result<
      std::shared_ptr<const ImmutableMapSnapshot>>
  Create(const MapSnapshotInput& input);

  [[nodiscard]] const ContentRef& snapshot_ref() const noexcept;
  [[nodiscard]] std::uint32_t map_revision() const noexcept;
  [[nodiscard]] std::string_view immutable_data_handle() const noexcept;
  [[nodiscard]] std::string_view frame_id() const noexcept;
  [[nodiscard]] ClockStamp source_time() const noexcept;
  [[nodiscard]] const MapBounds& bounds() const noexcept;
  [[nodiscard]] const GridGeometry& geometry() const noexcept;
  [[nodiscard]] std::span<const LayerManifestEntry> layer_manifest()
      const noexcept;
  [[nodiscard]] std::optional<
      std::reference_wrapper<const ContentRef>>
  LayerIdentity(LayerKind kind) const noexcept;
  [[nodiscard]] std::string_view LayerManifestHash() const noexcept;
  [[nodiscard]] std::span<const std::uint8_t> KnownMask() const noexcept;
  [[nodiscard]] std::span<const float> ElevationMeters() const noexcept;
  [[nodiscard]] SurfaceNormalGridView SurfaceNormals() const noexcept;
  [[nodiscard]] std::span<const float> RoughnessMeters() const noexcept;
  [[nodiscard]] std::span<const std::uint8_t> HardObstacleMask()
      const noexcept;
  [[nodiscard]] std::span<const float> Confidence() const noexcept;
  [[nodiscard]] std::span<const float> EsdfMeters() const noexcept;
  [[nodiscard]] std::span<const float> StaticSpeedLimitMps()
      const noexcept;

 private:
  struct Storage;

  explicit ImmutableMapSnapshot(std::shared_ptr<const Storage> storage);

  std::shared_ptr<const Storage> storage_;
};

}  // namespace lunar::planning::v3
