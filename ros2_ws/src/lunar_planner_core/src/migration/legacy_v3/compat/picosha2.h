#pragma once

#include <array>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <openssl/sha.h>

namespace picosha2 {

template <typename Iterator>
std::string hash256_hex_string(Iterator first, Iterator last) {
  const std::vector<unsigned char> input(first, last);
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  const auto* data = input.empty() ? nullptr : input.data();
  if (SHA256(data, input.size(), digest.data()) == nullptr) {
    return {};
  }

  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (const unsigned char byte : digest) {
    encoded << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return encoded.str();
}

}  // namespace picosha2
