#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace lunar::planning::shared {

[[nodiscard]] std::string Sha256Hex(
    std::span<const std::uint8_t> bytes);

}  // namespace lunar::planning::shared
