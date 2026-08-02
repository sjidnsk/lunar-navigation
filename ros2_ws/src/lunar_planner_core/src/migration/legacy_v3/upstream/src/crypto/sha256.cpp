#include "lunar_path_planner/v3/crypto/sha256.hpp"

#include <string>
#include <string_view>

#include <picosha2.h>

namespace lunar::planning::v3 {

Result<Sha256Digest> Sha256Hex(
    const std::string_view canonical_utf8) {
  return picosha2::hash256_hex_string(
      canonical_utf8.begin(), canonical_utf8.end());
}

}  // namespace lunar::planning::v3
