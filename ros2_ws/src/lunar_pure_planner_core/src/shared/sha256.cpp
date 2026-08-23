#include "shared/sha256.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace lunar::pure_planning::shared {
namespace {

constexpr std::array<std::uint32_t, 64U> kRoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

class Sha256 final {
 public:
  void Update(const std::span<const std::uint8_t> bytes) noexcept {
    for (const std::uint8_t byte : bytes) {
      buffer_[buffer_size_++] = byte;
      ++byte_count_;
      if (buffer_size_ == buffer_.size()) {
        Transform(buffer_);
        buffer_size_ = 0U;
      }
    }
  }

  [[nodiscard]] std::array<std::uint8_t, 32U> Finalize() noexcept {
    const std::uint64_t bit_count = byte_count_ * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56U) {
      while (buffer_size_ < buffer_.size()) {
        buffer_[buffer_size_++] = 0U;
      }
      Transform(buffer_);
      buffer_size_ = 0U;
    }
    while (buffer_size_ < 56U) {
      buffer_[buffer_size_++] = 0U;
    }
    for (std::size_t index = 0U; index < 8U; ++index) {
      buffer_[63U - index] = static_cast<std::uint8_t>(
          bit_count >> (index * 8U));
    }
    Transform(buffer_);

    std::array<std::uint8_t, 32U> digest{};
    for (std::size_t word = 0U; word < state_.size(); ++word) {
      for (std::size_t byte = 0U; byte < 4U; ++byte) {
        digest[word * 4U + byte] = static_cast<std::uint8_t>(
            state_[word] >> ((3U - byte) * 8U));
      }
    }
    return digest;
  }

 private:
  void Transform(const std::array<std::uint8_t, 64U>& block) noexcept {
    std::array<std::uint32_t, 64U> words{};
    for (std::size_t index = 0U; index < 16U; ++index) {
      const std::size_t offset = index * 4U;
      words[index] =
          (static_cast<std::uint32_t>(block[offset]) << 24U) |
          (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
          (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
          static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t sigma0 =
          std::rotr(words[index - 15U], 7) ^
          std::rotr(words[index - 15U], 18) ^
          (words[index - 15U] >> 3U);
      const std::uint32_t sigma1 =
          std::rotr(words[index - 2U], 17) ^
          std::rotr(words[index - 2U], 19) ^
          (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + sigma0 +
          words[index - 7U] + sigma1;
    }

    std::uint32_t a = state_[0U];
    std::uint32_t b = state_[1U];
    std::uint32_t c = state_[2U];
    std::uint32_t d = state_[3U];
    std::uint32_t e = state_[4U];
    std::uint32_t f = state_[5U];
    std::uint32_t g = state_[6U];
    std::uint32_t h = state_[7U];
    for (std::size_t index = 0U; index < words.size(); ++index) {
      const std::uint32_t sum1 =
          std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temporary1 =
          h + sum1 + choice + kRoundConstants[index] + words[index];
      const std::uint32_t sum0 =
          std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0U] += a;
    state_[1U] += b;
    state_[2U] += c;
    state_[3U] += d;
    state_[4U] += e;
    state_[5U] += f;
    state_[6U] += g;
    state_[7U] += h;
  }

  std::array<std::uint32_t, 8U> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };
  std::array<std::uint8_t, 64U> buffer_{};
  std::size_t buffer_size_{};
  std::uint64_t byte_count_{};
};

}  // namespace

std::string Sha256Hex(const std::span<const std::uint8_t> bytes) {
  Sha256 sha256;
  sha256.Update(bytes);
  const auto digest = sha256.Finalize();
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

}  // namespace lunar::pure_planning::shared
