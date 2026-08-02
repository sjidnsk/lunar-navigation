#pragma once

#include <string>

#include <nlohmann/json_fwd.hpp>

#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

class JcsCanonicalizer final {
 public:
  [[nodiscard]] static Result<std::string> Canonicalize(
      const nlohmann::json& value);
};

}  // namespace lunar::planning::v3
