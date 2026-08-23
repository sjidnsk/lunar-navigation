#include "shared/cell_area_distance_transform.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared {
namespace {

struct EnvelopeLine final {
  long double slope{};
  long double intercept{};
  long double first_x{};
};

[[nodiscard]] std::optional<std::string_view> CheckControl(
    const SearchControl& control, std::size_t* const work) {
  if (ControlCheckDue((*work)++)) {
    return StopReason(control);
  }
  return std::nullopt;
}

void AddEnvelopeLine(std::deque<EnvelopeLine>* const envelope,
                     const long double center, const double value) {
  EnvelopeLine next{
      .slope = -2.0L * center,
      .intercept = static_cast<long double>(value) + center * center,
      .first_x = -std::numeric_limits<long double>::infinity(),
  };
  while (!envelope->empty()) {
    next.first_x =
        (next.intercept - envelope->back().intercept) /
        (envelope->back().slope - next.slope);
    if (next.first_x > envelope->back().first_x) {
      break;
    }
    envelope->pop_back();
  }
  if (envelope->empty()) {
    next.first_x = -std::numeric_limits<long double>::infinity();
  }
  envelope->push_back(next);
}

[[nodiscard]] std::optional<std::string_view> TransformFromLeft(
    const std::vector<double>& input, std::vector<double>* const output,
    const SearchControl& control, std::size_t* const work) {
  // For q < x, the cell-area term is (x - (q + 0.5))^2.  Expanding
  // it leaves x^2 plus a line with a monotonically decreasing slope, so a
  // monotone lower envelope evaluates the complete one-sided transform in
  // linear time.  Reversing the line supplies the symmetric q > x side.
  output->assign(input.size(), std::numeric_limits<double>::infinity());
  std::deque<EnvelopeLine> envelope;
  for (std::size_t x = 0U; x < input.size(); ++x) {
    if (const auto stopped = CheckControl(control, work); stopped.has_value()) {
      return stopped;
    }
    if (x > 0U && std::isfinite(input[x - 1U])) {
      AddEnvelopeLine(&envelope, static_cast<long double>(x) - 0.5L,
                      input[x - 1U]);
    }
    while (envelope.size() > 1U &&
           envelope[1].first_x <= static_cast<long double>(x)) {
      envelope.pop_front();
    }
    if (!envelope.empty()) {
      const long double query = static_cast<long double>(x);
      (*output)[x] = static_cast<double>(
          query * query + envelope.front().slope * query +
          envelope.front().intercept);
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> TransformLine(
    const std::vector<double>& input, std::vector<double>* const output,
    const SearchControl& control, std::size_t* const work) {
  std::vector<double> left;
  if (const auto stopped =
          TransformFromLeft(input, &left, control, work);
      stopped.has_value()) {
    return stopped;
  }

  std::vector<double> reversed(input.rbegin(), input.rend());
  std::vector<double> reversed_left;
  if (const auto stopped =
          TransformFromLeft(reversed, &reversed_left, control, work);
      stopped.has_value()) {
    return stopped;
  }

  output->resize(input.size());
  for (std::size_t x = 0U; x < input.size(); ++x) {
    if (const auto stopped = CheckControl(control, work); stopped.has_value()) {
      return stopped;
    }
    (*output)[x] =
        std::min({input[x], left[x], reversed_left[input.size() - 1U - x]});
  }
  return std::nullopt;
}

}  // namespace

CellAreaClearanceResult BuildCellAreaClearance(
    const std::size_t width, const std::size_t height,
    const double resolution_m,
    const std::span<const std::uint8_t> hazard_mask, SearchControl control) {
  if (width == 0U || height == 0U ||
      height > std::numeric_limits<std::size_t>::max() / width ||
      hazard_mask.size() != width * height || !std::isfinite(resolution_m) ||
      resolution_m <= 0.0) {
    return {.reason_code = "CELL_AREA_DISTANCE_INVALID"};
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }

  const std::size_t count = width * height;
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> squared_distance(count, infinity);
  std::size_t work{};
  for (std::size_t index = 0U; index < count; ++index) {
    if (const auto stopped = CheckControl(control, &work); stopped.has_value()) {
      return {.reason_code = std::string{*stopped}};
    }
    if (hazard_mask[index] != 0U) {
      squared_distance[index] = 0.0;
    }
  }

  std::vector<double> temporary(count, infinity);
  std::vector<double> line(width);
  std::vector<double> transformed;
  for (std::size_t y = 0U; y < height; ++y) {
    const std::size_t begin = y * width;
    std::copy_n(squared_distance.begin() + begin, width, line.begin());
    if (const auto stopped =
            TransformLine(line, &transformed, control, &work);
        stopped.has_value()) {
      return {.reason_code = std::string{*stopped}};
    }
    std::copy(transformed.begin(), transformed.end(),
              temporary.begin() + begin);
  }

  CellAreaClearanceResult result;
  result.clearance_m.resize(count);
  line.resize(height);
  for (std::size_t x = 0U; x < width; ++x) {
    for (std::size_t y = 0U; y < height; ++y) {
      line[y] = temporary[y * width + x];
    }
    if (const auto stopped =
            TransformLine(line, &transformed, control, &work);
        stopped.has_value()) {
      return {.reason_code = std::string{*stopped}};
    }
    for (std::size_t y = 0U; y < height; ++y) {
      result.clearance_m[y * width + x] = static_cast<float>(
          std::sqrt(transformed[y]) * resolution_m);
    }
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  return result;
}

std::vector<CellAreaOffset> BuildCellAreaInflationStencil(
    const double resolution_m, const double inflation_m) {
  if (!std::isfinite(resolution_m) || resolution_m <= 0.0 ||
      !std::isfinite(inflation_m) || inflation_m < 0.0) {
    return {};
  }
  const auto radius = static_cast<std::int64_t>(
      std::ceil(inflation_m / resolution_m + 0.5));
  std::vector<CellAreaOffset> stencil;
  for (std::int64_t dy = -radius; dy <= radius; ++dy) {
    for (std::int64_t dx = -radius; dx <= radius; ++dx) {
      const double x =
          std::max(std::abs(dx) - 0.5, 0.0) * resolution_m;
      const double y =
          std::max(std::abs(dy) - 0.5, 0.0) * resolution_m;
      if (std::hypot(x, y) < inflation_m) {
        stencil.push_back(CellAreaOffset{
            .dx = static_cast<std::int32_t>(dx),
            .dy = static_cast<std::int32_t>(dy),
        });
      }
    }
  }
  std::sort(stencil.begin(), stencil.end(),
            [](const CellAreaOffset lhs, const CellAreaOffset rhs) {
              return lhs.dy < rhs.dy ||
                     (lhs.dy == rhs.dy && lhs.dx < rhs.dx);
            });
  return stencil;
}

}  // namespace lunar::pure_planning::shared
