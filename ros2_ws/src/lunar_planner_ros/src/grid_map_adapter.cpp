#include "lunar_planner_ros/grid_map_adapter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace lunar::planning::ros {
namespace {

enum class LayerKind {
  kFloat,
  kBinary,
  kCount,
};

enum class RangeKind {
  kAnyFinite,
  kNonNegative,
  kUnitInterval,
  kBinary,
  kCount,
};

struct LayerRequirement final {
  std::string_view name;
  LayerKind kind;
  RangeKind range;
};

constexpr std::array<LayerRequirement, 10> kRequiredLayers{
    LayerRequirement{"elevation", LayerKind::kFloat, RangeKind::kAnyFinite},
    LayerRequirement{"valid_mask", LayerKind::kBinary, RangeKind::kBinary},
    LayerRequirement{"obstacle", LayerKind::kBinary, RangeKind::kBinary},
    LayerRequirement{
        "obstacle_height", LayerKind::kFloat, RangeKind::kNonNegative},
    LayerRequirement{
        "observation_age_s", LayerKind::kFloat, RangeKind::kNonNegative},
    LayerRequirement{
        "observation_quality", LayerKind::kFloat, RangeKind::kUnitInterval},
    LayerRequirement{
        "elevation_variance", LayerKind::kFloat, RangeKind::kNonNegative},
    LayerRequirement{
        "obstacle_variance", LayerKind::kFloat, RangeKind::kNonNegative},
    LayerRequirement{"observation_count", LayerKind::kCount, RangeKind::kCount},
    LayerRequirement{"forbidden", LayerKind::kBinary, RangeKind::kBinary},
};

struct Layout final {
  bool row_major{};
  std::size_t offset{};
};

class Fingerprint final {
 public:
  void AddByte(const std::uint8_t value) noexcept {
    value_ ^= value;
    value_ *= 1'099'511'628'211ULL;
  }

  template<typename Integer>
  void AddInteger(const Integer value) noexcept {
    using Unsigned = std::make_unsigned_t<Integer>;
    const Unsigned unsigned_value = static_cast<Unsigned>(value);
    for (std::size_t index = 0U; index < sizeof(Unsigned); ++index) {
      AddByte(static_cast<std::uint8_t>(
          unsigned_value >> static_cast<unsigned>(index * 8U)));
    }
  }

  void AddString(const std::string_view value) noexcept {
    AddInteger(value.size());
    for (const unsigned char character : value) {
      AddByte(character);
    }
  }

  void AddFloat(float value) noexcept {
    if (value == 0.0F) {
      value = 0.0F;
    }
    AddInteger(std::bit_cast<std::uint32_t>(value));
  }

  void AddDouble(double value) noexcept {
    if (value == 0.0) {
      value = 0.0;
    }
    AddInteger(std::bit_cast<std::uint64_t>(value));
  }

  [[nodiscard]] std::uint64_t value() const noexcept {
    return value_ == 0U ? 1U : value_;
  }

 private:
  std::uint64_t value_{14'695'981'039'346'656'037ULL};
};

[[nodiscard]] std::string Upper(std::string_view text) {
  std::string upper{text};
  std::ranges::transform(upper, upper.begin(), [](const unsigned char value) {
    if (value >= static_cast<unsigned char>('a') &&
        value <= static_cast<unsigned char>('z')) {
      return static_cast<char>(value - static_cast<unsigned char>('a') +
                               static_cast<unsigned char>('A'));
    }
    return static_cast<char>(value);
  });
  return upper;
}

[[nodiscard]] GridMapAdaptResult Failure(
    const GridMapErrorCode code,
    std::string reason_code,
    std::string layer = {}) {
  return GridMapAdaptResult{
      .map = std::nullopt,
      .error = GridMapError{
          .code = code,
          .reason_code = std::move(reason_code),
          .layer = std::move(layer),
      },
  };
}

[[nodiscard]] bool IsFinitePose(
    const geometry_msgs::msg::Pose& pose) noexcept {
  return std::isfinite(pose.position.x) &&
      std::isfinite(pose.position.y) &&
      std::isfinite(pose.position.z) &&
      std::isfinite(pose.orientation.x) &&
      std::isfinite(pose.orientation.y) &&
      std::isfinite(pose.orientation.z) &&
      std::isfinite(pose.orientation.w);
}

[[nodiscard]] bool IsIdentityOrientation(
    const geometry_msgs::msg::Quaternion& orientation) noexcept {
  constexpr double kTolerance = 1.0e-6;
  return std::abs(orientation.x) <= kTolerance &&
      std::abs(orientation.y) <= kTolerance &&
      std::abs(orientation.z) <= kTolerance &&
      std::abs(std::abs(orientation.w) - 1.0) <= kTolerance;
}

[[nodiscard]] std::optional<std::size_t> CellDimension(
    const double length,
    const double resolution) noexcept {
  if (!std::isfinite(length) || !std::isfinite(resolution) ||
      length <= 0.0 || resolution <= 0.0) {
    return std::nullopt;
  }
  const double cells = length / resolution;
  const double rounded = std::round(cells);
  const double tolerance = 1.0e-6 * std::max(1.0, std::abs(cells));
  if (!std::isfinite(cells) || rounded < 1.0 ||
      std::abs(cells - rounded) > tolerance ||
      rounded > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(rounded);
}

[[nodiscard]] std::optional<Layout> ValidateLayout(
    const std_msgs::msg::Float32MultiArray& data,
    const std::size_t width,
    const std::size_t height) noexcept {
  if (data.layout.dim.size() != 2U ||
      width > std::numeric_limits<std::size_t>::max() / height) {
    return std::nullopt;
  }
  const std::size_t cell_count = width * height;
  const auto& outer = data.layout.dim[0];
  const auto& inner = data.layout.dim[1];
  bool row_major = false;
  if (outer.label == "column_index" && inner.label == "row_index") {
    if (outer.size != height || inner.size != width) {
      return std::nullopt;
    }
  } else if (outer.label == "row_index" && inner.label == "column_index") {
    row_major = true;
    if (outer.size != width || inner.size != height) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  if (outer.stride != cell_count || inner.stride != inner.size) {
    return std::nullopt;
  }
  const std::size_t offset = data.layout.data_offset;
  if (offset > data.data.size() || data.data.size() - offset != cell_count) {
    return std::nullopt;
  }
  return Layout{.row_major = row_major, .offset = offset};
}

[[nodiscard]] std::vector<float> Unwrap(
    const std_msgs::msg::Float32MultiArray& source,
    const Layout layout,
    const std::size_t width,
    const std::size_t height,
    const std::size_t outer_start_index,
    const std::size_t inner_start_index) {
  std::vector<float> output(width * height);
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t unwrapped_row = width - 1U - x;
      const std::size_t unwrapped_column = height - 1U - y;
      const std::size_t physical_row =
          (unwrapped_row + outer_start_index) % width;
      const std::size_t physical_column =
          (unwrapped_column + inner_start_index) % height;
      const std::size_t source_index = layout.row_major
          ? physical_row * height + physical_column
          : physical_column * width + physical_row;
      output[y * width + x] = source.data[layout.offset + source_index];
    }
  }
  return output;
}

[[nodiscard]] bool ValueInRange(
    const float value, const RangeKind range) noexcept {
  if (!std::isfinite(value)) {
    return false;
  }
  switch (range) {
    case RangeKind::kAnyFinite:
      return true;
    case RangeKind::kNonNegative:
      return value >= 0.0F;
    case RangeKind::kUnitInterval:
      return value >= 0.0F && value <= 1.0F;
    case RangeKind::kBinary:
      return value == 0.0F || value == 1.0F;
    case RangeKind::kCount:
      return value >= 0.0F &&
          static_cast<double>(value) <=
              static_cast<double>(std::numeric_limits<std::uint32_t>::max()) &&
          std::trunc(value) == value;
  }
  return false;
}

[[nodiscard]] GridLayer ToTypedLayer(
    const std::vector<float>& values, const LayerKind kind) {
  if (kind == LayerKind::kBinary) {
    std::vector<std::uint8_t> typed;
    typed.reserve(values.size());
    for (const float value : values) {
      typed.push_back(static_cast<std::uint8_t>(value));
    }
    return GridLayer{.values = std::move(typed)};
  }
  if (kind == LayerKind::kCount) {
    std::vector<std::uint32_t> typed;
    typed.reserve(values.size());
    for (const float value : values) {
      typed.push_back(static_cast<std::uint32_t>(value));
    }
    return GridLayer{.values = std::move(typed)};
  }
  return GridLayer{.values = values};
}

}  // namespace

GridMapAdaptResult GridMapAdapter::Adapt(
    const grid_map_msgs::msg::GridMap& message,
    const std::string_view expected_frame) const {
  if (expected_frame.empty() || message.header.frame_id != expected_frame) {
    return Failure(GridMapErrorCode::kWrongFrame, "GRID_MAP_WRONG_FRAME");
  }
  if (message.header.stamp.sec < 0 ||
      (message.header.stamp.sec == 0 && message.header.stamp.nanosec == 0U)) {
    return Failure(GridMapErrorCode::kInvalidStamp, "GRID_MAP_STAMP_INVALID");
  }
  const auto width = CellDimension(message.info.length_x, message.info.resolution);
  const auto height = CellDimension(message.info.length_y, message.info.resolution);
  if (!width.has_value() || !height.has_value() ||
      !IsFinitePose(message.info.pose) ||
      !IsIdentityOrientation(message.info.pose.orientation) ||
      *width > std::numeric_limits<std::size_t>::max() / *height) {
    return Failure(
        GridMapErrorCode::kInvalidGeometry, "GRID_MAP_GEOMETRY_INVALID");
  }
  if (message.outer_start_index >= *width ||
      message.inner_start_index >= *height) {
    return Failure(
        GridMapErrorCode::kInvalidStartIndex,
        "GRID_MAP_START_INDEX_INVALID");
  }
  if (message.layers.size() != message.data.size()) {
    return Failure(
        GridMapErrorCode::kMalformedLayer, "GRID_MAP_LAYER_COUNT_MISMATCH");
  }

  std::map<std::string, std::size_t, std::less<>> layer_indices;
  for (std::size_t index = 0U; index < message.layers.size(); ++index) {
    const std::string& name = message.layers[index];
    if (name.empty() || !layer_indices.emplace(name, index).second) {
      return Failure(
          GridMapErrorCode::kDuplicateLayer,
          "GRID_MAP_DUPLICATE_LAYER",
          name);
    }
  }
  std::set<std::string, std::less<>> basic_layers;
  for (const std::string& name : message.basic_layers) {
    if (name.empty() || !layer_indices.contains(name) ||
        !basic_layers.insert(name).second) {
      return Failure(
          GridMapErrorCode::kMalformedLayer,
          "GRID_MAP_BASIC_LAYER_INVALID",
          name);
    }
  }

  GridMap map{
      .frame_id = message.header.frame_id,
      .stamp = TimePoint{
          .nanoseconds_since_epoch =
              static_cast<std::int64_t>(message.header.stamp.sec) *
                  1'000'000'000LL +
              static_cast<std::int64_t>(message.header.stamp.nanosec),
      },
      .width = *width,
      .height = *height,
      .resolution_m = message.info.resolution,
      .origin_m = Vec3{
          .x = message.info.pose.position.x - message.info.length_x * 0.5,
          .y = message.info.pose.position.y - message.info.length_y * 0.5,
          .z = message.info.pose.position.z,
      },
      .layers = {},
  };
  Fingerprint fingerprint;
  fingerprint.AddString("lunar-grid-map-planning-content/v1");
  fingerprint.AddString(map.frame_id);
  fingerprint.AddInteger(map.width);
  fingerprint.AddInteger(map.height);
  fingerprint.AddDouble(map.resolution_m);
  fingerprint.AddDouble(map.origin_m.x);
  fingerprint.AddDouble(map.origin_m.y);
  fingerprint.AddDouble(map.origin_m.z);

  for (const LayerRequirement& required : kRequiredLayers) {
    const auto found = layer_indices.find(required.name);
    if (found == layer_indices.end()) {
      return Failure(
          GridMapErrorCode::kMissingLayer,
          "MISSING_MAP_LAYER_" + Upper(required.name),
          std::string{required.name});
    }
    const auto& source = message.data[found->second];
    const auto layout = ValidateLayout(source, *width, *height);
    if (!layout.has_value()) {
      return Failure(
          GridMapErrorCode::kMalformedLayer,
          "GRID_MAP_LAYER_LAYOUT_" + Upper(required.name),
          std::string{required.name});
    }
    std::vector<float> values = Unwrap(
        source, *layout, *width, *height,
        message.outer_start_index, message.inner_start_index);
    for (const float value : values) {
      if (!std::isfinite(value)) {
        return Failure(
            GridMapErrorCode::kNonFiniteValue,
            "GRID_MAP_NONFINITE_" + Upper(required.name),
            std::string{required.name});
      }
      if (!ValueInRange(value, required.range)) {
        return Failure(
            GridMapErrorCode::kOutOfRangeValue,
            "GRID_MAP_RANGE_" + Upper(required.name),
            std::string{required.name});
      }
    }
    fingerprint.AddString(required.name);
    fingerprint.AddInteger(static_cast<std::uint8_t>(required.kind));
    for (const float value : values) {
      fingerprint.AddFloat(value);
    }
    map.layers.emplace(
        std::string{required.name}, ToTypedLayer(values, required.kind));
  }

  return GridMapAdaptResult{
      .map = std::move(map),
      .error = std::nullopt,
      .content_identity = fingerprint.value(),
  };
}

}  // namespace lunar::planning::ros
