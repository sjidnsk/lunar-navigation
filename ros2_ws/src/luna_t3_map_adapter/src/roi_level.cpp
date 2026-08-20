#include "luna_t3_map_adapter/roi_level.hpp"

#include <string>

namespace luna::task3 {

Result<SelectedGlobalLevel> SelectGlobalLevel(
    const TaskRoi& roi,
    const lunar::planning::GlobalMapConfig& config) noexcept {
  if (!roi.Valid()) {
    return Result<SelectedGlobalLevel>{.value = std::nullopt,
                                       .reason_code = "TASK_ROI_INVALID"};
  }
  const auto selected = lunar::planning::hierarchical::ExpectedGlobalMapLevel(
      roi.WidthM(), roi.HeightM(), config);
  if (!selected.ok()) {
    return Result<SelectedGlobalLevel>{.value = std::nullopt,
                                       .reason_code = selected.reason_code};
  }
  return Result<SelectedGlobalLevel>{
      .value = SelectedGlobalLevel{
          .level = *selected.level,
          .width = selected.width,
          .height = selected.height,
          .resolution_m = selected.resolution_m,
      },
      .reason_code = {},
  };
}

}  // namespace luna::task3
