#pragma once

#include <cmath>
#include <optional>
#include <string>

namespace luna::task3 {

template <typename T>
struct Result final {
  std::optional<T> value;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && reason_code.empty();
  }
};

struct TaskRoi final {
  double min_x_m{};
  double min_y_m{};
  double max_x_m{};
  double max_y_m{};

  [[nodiscard]] bool Valid() const noexcept {
    return std::isfinite(min_x_m) && std::isfinite(min_y_m) &&
           std::isfinite(max_x_m) && std::isfinite(max_y_m) &&
           max_x_m > min_x_m && max_y_m > min_y_m;
  }

  [[nodiscard]] double WidthM() const noexcept { return max_x_m - min_x_m; }
  [[nodiscard]] double HeightM() const noexcept { return max_y_m - min_y_m; }
};

struct SelectedGlobalLevel final {
  std::size_t level{};
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
};

}  // namespace luna::task3
