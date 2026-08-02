#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/codec/jcs_canonicalizer.hpp"
#include "lunar_path_planner/v3/crypto/sha256.hpp"

namespace lunar::planning::v3 {

namespace {

constexpr std::size_t kLayerKindCount = 8U;
constexpr double kUnitNormalSquaredTolerance = 1.0e-4;

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool IsFinite(const Vec2& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool IsIdentifierCharacter(char value) noexcept {
  const auto byte = static_cast<unsigned char>(value);
  return std::isalnum(byte) != 0 || value == '.' || value == '_' ||
         value == ':' || value == '/' || value == '-';
}

[[nodiscard]] bool IsIdentifier(std::string_view value) noexcept {
  if (value.empty() || value.size() > 128U) {
    return false;
  }
  const auto first = static_cast<unsigned char>(value.front());
  if (std::isalnum(first) == 0) {
    return false;
  }
  return std::ranges::all_of(value, IsIdentifierCharacter);
}

[[nodiscard]] bool IsLowerSha256(std::string_view value) noexcept {
  return value.size() == 64U &&
         std::ranges::all_of(value, [](char digit) {
           return (digit >= '0' && digit <= '9') ||
                  (digit >= 'a' && digit <= 'f');
         });
}

[[nodiscard]] bool IsValidContentRef(const ContentRef& value) noexcept {
  return IsIdentifier(value.id) && value.revision >= 1U &&
         value.revision <=
             static_cast<std::uint32_t>(
                 std::numeric_limits<std::int32_t>::max()) &&
         IsLowerSha256(value.content_hash);
}

[[nodiscard]] bool IsValidLayerKind(LayerKind kind) noexcept {
  const auto value =
      static_cast<std::underlying_type_t<LayerKind>>(kind);
  return value >= 0 &&
         static_cast<std::size_t>(value) < kLayerKindCount;
}

[[nodiscard]] std::size_t LayerIndex(LayerKind kind) noexcept {
  return static_cast<std::size_t>(
      static_cast<std::underlying_type_t<LayerKind>>(kind));
}

[[nodiscard]] std::string_view LayerKindName(LayerKind kind) noexcept {
  switch (kind) {
    case LayerKind::kKnownMask:
      return "KNOWN_MASK";
    case LayerKind::kElevation:
      return "ELEVATION";
    case LayerKind::kTerrainNormal:
      return "TERRAIN_NORMAL";
    case LayerKind::kRoughness:
      return "ROUGHNESS";
    case LayerKind::kHardObstacle:
      return "HARD_OBSTACLE";
    case LayerKind::kConfidence:
      return "CONFIDENCE";
    case LayerKind::kEsdf:
      return "ESDF";
    case LayerKind::kStaticSpeedLimit:
      return "STATIC_SPEED_LIMIT";
  }
  return "";
}

template <class T>
[[nodiscard]] bool HasExactSize(const std::vector<T>& values,
                                std::size_t expected) noexcept {
  return values.size() == expected;
}

[[nodiscard]] bool AllFinite(std::span<const float> values) noexcept {
  return std::ranges::all_of(
      values, [](float value) { return std::isfinite(value); });
}

[[nodiscard]] bool AllNonNegativeFinite(
    std::span<const float> values) noexcept {
  return std::ranges::all_of(values, [](float value) {
    return std::isfinite(value) && value >= 0.0F;
  });
}

[[nodiscard]] bool IsBinaryMask(
    std::span<const std::uint8_t> values) noexcept {
  return std::ranges::all_of(
      values, [](std::uint8_t value) { return value <= 1U; });
}

[[nodiscard]] Result<std::string> ComputeManifestHash(
    const std::vector<LayerManifestEntry>& manifest) {
  std::vector<const LayerManifestEntry*> ordered;
  ordered.reserve(manifest.size());
  for (const auto& entry : manifest) {
    ordered.push_back(&entry);
  }
  std::ranges::sort(
      ordered, [](const LayerManifestEntry* left,
                  const LayerManifestEntry* right) {
        return LayerIndex(left->layer_kind) <
               LayerIndex(right->layer_kind);
      });

  nlohmann::json canonical_manifest = nlohmann::json::array();
  for (const auto* entry : ordered) {
    canonical_manifest.push_back(
        nlohmann::json{
            {"content_hash", entry->content_ref.content_hash},
            {"id", entry->content_ref.id},
            {"layer_kind", LayerKindName(entry->layer_kind)},
            {"revision", entry->content_ref.revision},
        });
  }

  auto canonical =
      JcsCanonicalizer::Canonicalize(canonical_manifest);
  if (!IsOk(canonical)) {
    return std::get<Error>(std::move(canonical));
  }
  return Sha256Hex(std::get<std::string>(std::move(canonical)));
}

}  // namespace

struct ImmutableMapSnapshot::Storage final {
  explicit Storage(const MapSnapshotInput& input,
                   std::string manifest_hash)
      : snapshot_ref(input.snapshot_ref),
        map_revision(input.map_revision),
        immutable_data_handle(input.immutable_data_handle),
        source_time(input.source_time),
        bounds(input.bounds),
        geometry(input.geometry),
        layer_manifest(input.layer_manifest),
        manifest_hash(std::move(manifest_hash)),
        known_mask(input.known_mask),
        elevation_m(input.elevation_m),
        normal_x(input.normal_x),
        normal_y(input.normal_y),
        normal_z(input.normal_z),
        roughness_m(input.roughness_m),
        hard_obstacle_mask(input.hard_obstacle_mask),
        confidence(input.confidence),
        esdf_m(input.esdf_m),
        static_speed_limit_mps(input.static_speed_limit_mps) {}

  ContentRef snapshot_ref;
  std::uint32_t map_revision{};
  std::string immutable_data_handle;
  ClockStamp source_time;
  MapBounds bounds;
  GridGeometry geometry;
  std::vector<LayerManifestEntry> layer_manifest;
  std::string manifest_hash;
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

Result<std::shared_ptr<const ImmutableMapSnapshot>>
ImmutableMapSnapshot::Create(const MapSnapshotInput& input) {
  if (!IsValidContentRef(input.snapshot_ref)) {
    return Invalid("map_snapshot.snapshot_ref",
                   "snapshot_ref must be a valid ContentRef");
  }
  if (input.map_revision < 1U ||
      input.map_revision >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::int32_t>::max())) {
    return Invalid("map_snapshot.map_revision",
                   "map_revision must be within the schema range");
  }
  if (!IsIdentifier(input.immutable_data_handle)) {
    return Invalid("map_snapshot.immutable_data_handle",
                   "immutable_data_handle must be a non-empty identifier");
  }
  if (!IsIdentifier(input.source_time.clock_id)) {
    return Invalid("map_snapshot.source_time.clock_id",
                   "source clock_id must be a valid identifier");
  }
  if (!IsFinite(input.bounds.minimum_m) ||
      !IsFinite(input.bounds.maximum_m) ||
      input.bounds.minimum_m.x > input.bounds.maximum_m.x ||
      input.bounds.minimum_m.y > input.bounds.maximum_m.y ||
      input.bounds.minimum_m.z > input.bounds.maximum_m.z) {
    return Invalid("map_snapshot.bounds",
                   "bounds must be finite and ordered");
  }
  if (input.geometry.width == 0U || input.geometry.height == 0U) {
    return Invalid("map_snapshot.geometry",
                   "grid width and height must be positive");
  }
  if (input.geometry.height >
      std::numeric_limits<std::size_t>::max() /
          input.geometry.width) {
    return Invalid("map_snapshot.geometry",
                   "grid cell count overflows size_t");
  }
  const std::size_t cell_count =
      input.geometry.width * input.geometry.height;
  if (!std::isfinite(input.geometry.resolution_m) ||
      input.geometry.resolution_m <= 0.0 ||
      !IsFinite(input.geometry.origin_m) ||
      !IsIdentifier(input.geometry.frame_id)) {
    return Invalid(
        "map_snapshot.geometry",
        "grid resolution, origin, and frame_id must be valid");
  }

  std::array<bool, kLayerKindCount> present{};
  std::set<std::tuple<std::string, std::uint32_t, std::string>>
      exact_identities;
  std::map<std::pair<std::string, std::uint32_t>, std::string>
      revisions;
  for (std::size_t index = 0U; index < input.layer_manifest.size();
       ++index) {
    const auto& entry = input.layer_manifest[index];
    const std::string field_path =
        "map_snapshot.layer_manifest[" + std::to_string(index) + "]";
    if (!IsValidLayerKind(entry.layer_kind)) {
      return Invalid(field_path + ".layer_kind",
                     "layer kind is outside the closed enum");
    }
    const auto layer_index = LayerIndex(entry.layer_kind);
    if (present[layer_index]) {
      return Invalid(field_path + ".layer_kind",
                     "layer kinds must be unique");
    }
    present[layer_index] = true;
    if (!IsValidContentRef(entry.content_ref)) {
      return Invalid(field_path + ".content_ref",
                     "layer identity must be a valid ContentRef");
    }

    const auto exact_identity =
        std::tuple{entry.content_ref.id, entry.content_ref.revision,
                   entry.content_ref.content_hash};
    if (!exact_identities.insert(exact_identity).second) {
      return Invalid(field_path + ".content_ref",
                     "layer ContentRef identities must be unique");
    }
    const auto revision_identity =
        std::pair{entry.content_ref.id, entry.content_ref.revision};
    const auto [found, inserted] = revisions.emplace(
        revision_identity, entry.content_ref.content_hash);
    if (!inserted && found->second != entry.content_ref.content_hash) {
      return Invalid(field_path + ".content_ref",
                     "one id/revision cannot name conflicting hashes");
    }
  }

  constexpr std::array<LayerKind, 6U> required_layers = {
      LayerKind::kKnownMask,     LayerKind::kElevation,
      LayerKind::kTerrainNormal, LayerKind::kRoughness,
      LayerKind::kHardObstacle,  LayerKind::kConfidence,
  };
  for (const auto kind : required_layers) {
    if (!present[LayerIndex(kind)]) {
      return Invalid("map_snapshot.layer_manifest",
                     "all mandatory map layers must be present");
    }
  }

  if (!HasExactSize(input.known_mask, cell_count) ||
      !HasExactSize(input.elevation_m, cell_count) ||
      !HasExactSize(input.normal_x, cell_count) ||
      !HasExactSize(input.normal_y, cell_count) ||
      !HasExactSize(input.normal_z, cell_count) ||
      !HasExactSize(input.roughness_m, cell_count) ||
      !HasExactSize(input.hard_obstacle_mask, cell_count) ||
      !HasExactSize(input.confidence, cell_count)) {
    return Invalid("map_snapshot.layers",
                   "each mandatory layer must match the grid cell count");
  }

  const bool has_esdf = present[LayerIndex(LayerKind::kEsdf)];
  if (has_esdf != !input.esdf_m.empty() ||
      (has_esdf && !HasExactSize(input.esdf_m, cell_count))) {
    return Invalid(
        "map_snapshot.esdf_m",
        "ESDF manifest presence and exact cell data must agree");
  }
  const bool has_static_speed =
      present[LayerIndex(LayerKind::kStaticSpeedLimit)];
  if (has_static_speed != !input.static_speed_limit_mps.empty() ||
      (has_static_speed &&
       !HasExactSize(input.static_speed_limit_mps, cell_count))) {
    return Invalid(
        "map_snapshot.static_speed_limit_mps",
        "static speed manifest presence and exact cell data must agree");
  }

  if (!IsBinaryMask(input.known_mask)) {
    return Invalid("map_snapshot.known_mask",
                   "known mask values must be binary");
  }
  if (!IsBinaryMask(input.hard_obstacle_mask)) {
    return Invalid("map_snapshot.hard_obstacle_mask",
                   "hard-obstacle mask values must be binary");
  }
  if (!AllFinite(input.elevation_m)) {
    return Invalid("map_snapshot.elevation_m",
                   "elevation values must be finite");
  }
  if (!AllNonNegativeFinite(input.roughness_m)) {
    return Invalid("map_snapshot.roughness_m",
                   "roughness values must be finite and non-negative");
  }
  if (!std::ranges::all_of(input.confidence, [](float value) {
        return std::isfinite(value) && value >= 0.0F &&
               value <= 1.0F;
      })) {
    return Invalid("map_snapshot.confidence",
                   "confidence values must be finite and within [0, 1]");
  }
  if (!AllFinite(input.esdf_m)) {
    return Invalid("map_snapshot.esdf_m",
                   "ESDF values must be finite");
  }
  if (!AllNonNegativeFinite(input.static_speed_limit_mps)) {
    return Invalid(
        "map_snapshot.static_speed_limit_mps",
        "static speed limits must be finite and non-negative");
  }

  for (std::size_t index = 0U; index < cell_count; ++index) {
    const double normal_x = static_cast<double>(input.normal_x[index]);
    const double normal_y = static_cast<double>(input.normal_y[index]);
    const double normal_z = static_cast<double>(input.normal_z[index]);
    if (!std::isfinite(normal_x) || !std::isfinite(normal_y) ||
        !std::isfinite(normal_z)) {
      return Invalid("map_snapshot.terrain_normal",
                     "surface normals must be finite");
    }
    const double squared_norm =
        normal_x * normal_x + normal_y * normal_y +
        normal_z * normal_z;
    if (std::abs(squared_norm - 1.0) >
        kUnitNormalSquaredTolerance) {
      return Invalid("map_snapshot.terrain_normal",
                     "surface normals must have unit length");
    }
  }

  auto manifest_hash = ComputeManifestHash(input.layer_manifest);
  if (!IsOk(manifest_hash)) {
    auto error = std::get<Error>(std::move(manifest_hash));
    error.field_path = "map_snapshot.layer_manifest";
    return error;
  }

  auto storage = std::make_shared<const Storage>(
      input, std::get<std::string>(std::move(manifest_hash)));
  return std::shared_ptr<const ImmutableMapSnapshot>(
      new ImmutableMapSnapshot(std::move(storage)));
}

ImmutableMapSnapshot::ImmutableMapSnapshot(
    std::shared_ptr<const Storage> storage)
    : storage_(std::move(storage)) {}

const ContentRef& ImmutableMapSnapshot::snapshot_ref() const noexcept {
  return storage_->snapshot_ref;
}

std::uint32_t ImmutableMapSnapshot::map_revision() const noexcept {
  return storage_->map_revision;
}

std::string_view ImmutableMapSnapshot::immutable_data_handle()
    const noexcept {
  return storage_->immutable_data_handle;
}

std::string_view ImmutableMapSnapshot::frame_id() const noexcept {
  return storage_->geometry.frame_id;
}

ClockStamp ImmutableMapSnapshot::source_time() const noexcept {
  return storage_->source_time;
}

const MapBounds& ImmutableMapSnapshot::bounds() const noexcept {
  return storage_->bounds;
}

const GridGeometry& ImmutableMapSnapshot::geometry() const noexcept {
  return storage_->geometry;
}

std::span<const LayerManifestEntry>
ImmutableMapSnapshot::layer_manifest() const noexcept {
  return storage_->layer_manifest;
}

std::optional<std::reference_wrapper<const ContentRef>>
ImmutableMapSnapshot::LayerIdentity(LayerKind kind) const noexcept {
  const auto found = std::ranges::find_if(
      storage_->layer_manifest,
      [kind](const LayerManifestEntry& entry) {
        return entry.layer_kind == kind;
      });
  if (found == storage_->layer_manifest.end()) {
    return std::nullopt;
  }
  return std::cref(found->content_ref);
}

std::string_view ImmutableMapSnapshot::LayerManifestHash()
    const noexcept {
  return storage_->manifest_hash;
}

std::span<const std::uint8_t>
ImmutableMapSnapshot::KnownMask() const noexcept {
  return storage_->known_mask;
}

std::span<const float>
ImmutableMapSnapshot::ElevationMeters() const noexcept {
  return storage_->elevation_m;
}

SurfaceNormalGridView
ImmutableMapSnapshot::SurfaceNormals() const noexcept {
  return {
      .x = storage_->normal_x,
      .y = storage_->normal_y,
      .z = storage_->normal_z,
  };
}

std::span<const float>
ImmutableMapSnapshot::RoughnessMeters() const noexcept {
  return storage_->roughness_m;
}

std::span<const std::uint8_t>
ImmutableMapSnapshot::HardObstacleMask() const noexcept {
  return storage_->hard_obstacle_mask;
}

std::span<const float>
ImmutableMapSnapshot::Confidence() const noexcept {
  return storage_->confidence;
}

std::span<const float>
ImmutableMapSnapshot::EsdfMeters() const noexcept {
  return storage_->esdf_m;
}

std::span<const float>
ImmutableMapSnapshot::StaticSpeedLimitMps() const noexcept {
  return storage_->static_speed_limit_mps;
}

}  // namespace lunar::planning::v3
