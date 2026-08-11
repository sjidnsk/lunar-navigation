#include "lunar_unreal_tcp_bridge/stream_decoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace lunar::unreal_tcp {
namespace {

[[nodiscard]] std::uint16_t ReadU16(
    const std::span<const std::byte> bytes, const std::size_t offset) {
  return static_cast<std::uint16_t>(
      std::to_integer<std::uint8_t>(bytes[offset]) |
      (static_cast<std::uint16_t>(
           std::to_integer<std::uint8_t>(bytes[offset + 1U]))
       << 8U));
}

[[nodiscard]] std::uint32_t ReadU32(
    const std::span<const std::byte> bytes, const std::size_t offset) {
  std::uint32_t value{};
  for (std::size_t index = 0U; index < 4U; ++index) {
    value |= static_cast<std::uint32_t>(
                 std::to_integer<std::uint8_t>(bytes[offset + index]))
        << (index * 8U);
  }
  return value;
}

[[nodiscard]] ProtocolError Error(
    const ProtocolErrorCode code, std::string reason_code) {
  return ProtocolError{.code = code, .reason_code = std::move(reason_code)};
}

void ZeroHeaderCrc(std::array<std::byte, kHeaderSize>& header) {
  std::fill(header.begin() + 40, header.begin() + 44, std::byte{0U});
}

}  // namespace

StreamDecodeResult StreamDecoder::Push(
    const std::span<const std::byte> bytes) {
  StreamDecodeResult result;
  constexpr std::size_t kMaximumBufferedBytes = kHeaderSize + kMaximumBodyBytes;
  if (bytes.size() > kMaximumBufferedBytes - buffer_.size()) {
    buffer_.clear();
    result.error = Error(
        ProtocolErrorCode::kBodyTooLarge, "RECEIVE_BUFFER_LIMIT");
    return result;
  }
  buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());

  std::size_t consumed{};
  while (buffer_.size() - consumed >= kHeaderSize) {
    const std::span<const std::byte> available{
        buffer_.data() + consumed, buffer_.size() - consumed};
    if (!std::equal(kMagic.begin(), kMagic.end(), available.begin())) {
      result.error = Error(ProtocolErrorCode::kInvalidMagic, "MAGIC_INVALID");
      buffer_.clear();
      return result;
    }
    if (ReadU16(available, 4U) != 1U) {
      result.error = Error(
          ProtocolErrorCode::kUnsupportedVersion, "VERSION_UNSUPPORTED");
      buffer_.clear();
      return result;
    }
    if (ReadU16(available, 6U) != kHeaderSize) {
      result.error = Error(
          ProtocolErrorCode::kInvalidHeaderSize, "HEADER_SIZE_INVALID");
      buffer_.clear();
      return result;
    }
    std::array<std::byte, kHeaderSize> header{};
    std::copy_n(available.begin(), kHeaderSize, header.begin());
    const std::uint32_t expected_header_crc = ReadU32(available, 40U);
    ZeroHeaderCrc(header);
    if (Crc32(header) != expected_header_crc) {
      result.error = Error(
          ProtocolErrorCode::kHeaderCrcMismatch, "HEADER_CRC_MISMATCH");
      buffer_.clear();
      return result;
    }

    const std::uint32_t metadata_length = ReadU32(available, 28U);
    const std::uint32_t payload_length = ReadU32(available, 32U);
    if (metadata_length > kMaximumMetadataBytes) {
      result.error = Error(
          ProtocolErrorCode::kMetadataTooLarge, "METADATA_TOO_LARGE");
      buffer_.clear();
      return result;
    }
    if (payload_length > kMaximumBodyBytes - metadata_length) {
      result.error = Error(ProtocolErrorCode::kBodyTooLarge, "BODY_TOO_LARGE");
      buffer_.clear();
      return result;
    }
    const std::size_t frame_size = kHeaderSize +
        static_cast<std::size_t>(metadata_length) + payload_length;
    if (available.size() < frame_size) {
      break;
    }
    DecodeResult decoded = DecodeFrame(available.first(frame_size));
    if (!decoded.ok()) {
      result.error = std::move(decoded.error);
      buffer_.clear();
      return result;
    }
    result.frames.push_back(std::move(*decoded.frame));
    consumed += frame_size;
  }
  if (consumed != 0U) {
    buffer_.erase(
        buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(consumed));
  }
  return result;
}

void StreamDecoder::Reset() noexcept {
  buffer_.clear();
}

std::size_t StreamDecoder::buffered_bytes() const noexcept {
  return buffer_.size();
}

}  // namespace lunar::unreal_tcp
