#pragma once

#include <optional>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "lunar_unreal_tcp_bridge/protocol.hpp"

namespace lunar::unreal_tcp {

struct MetadataError final {
  std::string reason_code;
};

[[nodiscard]] std::optional<MetadataError> ValidateMetadata(
    MessageType type, const nlohmann::json& metadata);

}  // namespace lunar::unreal_tcp
