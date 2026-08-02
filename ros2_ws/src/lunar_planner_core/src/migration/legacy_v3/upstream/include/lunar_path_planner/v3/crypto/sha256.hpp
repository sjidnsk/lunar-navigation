#pragma once

#include <string_view>

#include "lunar_path_planner/v3/contracts/base_types.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

[[nodiscard]] Result<Sha256Digest> Sha256Hex(
    std::string_view canonical_utf8);

}  // namespace lunar::planning::v3
