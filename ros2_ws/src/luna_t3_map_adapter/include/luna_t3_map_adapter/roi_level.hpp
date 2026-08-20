#pragma once

#include "luna_t3_map_adapter/types.hpp"

#include "lunar_planner_core/types/planner_config.hpp"

namespace luna::task3 {

[[nodiscard]] Result<SelectedGlobalLevel> SelectGlobalLevel(
    const TaskRoi& roi,
    const lunar::planning::GlobalMapConfig& config) noexcept;

}  // namespace luna::task3
