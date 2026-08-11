#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace lunar::unreal_tcp {

inline constexpr std::size_t kHeaderSize = 48U;
inline constexpr std::size_t kMaximumMetadataBytes = 65'536U;
inline constexpr std::size_t kMaximumBodyBytes = 8U * 1024U * 1024U;
inline constexpr std::array<std::byte, 4> kMagic{
    std::byte{'L'}, std::byte{'N'}, std::byte{'T'}, std::byte{'1'}};

enum class MessageType : std::uint16_t {
  kHello = 0x0001,
  kHelloAck = 0x0002,
  kControl = 0x0010,
  kControlAck = 0x0011,
  kRobotState = 0x0020,
  kLocalElevationMap = 0x0021,
  kMotionReference = 0x0030,
  kExecutionFeedback = 0x0031,
  kHeartbeat = 0x0040,
  kError = 0x00ff,
};

enum class ProtocolErrorCode : std::uint8_t {
  kInvalidMagic,
  kUnsupportedVersion,
  kInvalidHeaderSize,
  kUnknownMessageType,
  kUnknownFlags,
  kNonzeroReserved,
  kMetadataTooLarge,
  kBodyTooLarge,
  kLengthMismatch,
  kHeaderCrcMismatch,
  kBodyCrcMismatch,
  kInvalidJson,
  kInvalidMetadata,
};

struct ProtocolError final {
  ProtocolErrorCode code{ProtocolErrorCode::kInvalidMagic};
  std::string reason_code;
};

struct FrameHeader final {
  std::uint16_t protocol_version{1U};
  std::uint16_t header_size{kHeaderSize};
  MessageType message_type{MessageType::kHello};
  std::uint16_t flags{1U};
  std::uint64_t sequence{1U};
  std::int64_t simulation_time_ns{};
  std::uint32_t metadata_length{};
  std::uint32_t payload_length{};
  std::uint32_t body_crc32{};
  std::uint32_t header_crc32{};
  std::uint32_t reserved{};
};

struct Frame final {
  FrameHeader header;
  nlohmann::json metadata{nlohmann::json::object()};
  std::vector<std::byte> payload;
};

struct EncodeResult final {
  std::optional<std::vector<std::byte>> bytes;
  std::optional<ProtocolError> error;
  [[nodiscard]] bool ok() const noexcept { return bytes.has_value(); }
};

struct DecodeResult final {
  std::optional<Frame> frame;
  std::optional<ProtocolError> error;
  [[nodiscard]] bool ok() const noexcept { return frame.has_value(); }
};

[[nodiscard]] std::uint32_t Crc32(std::span<const std::byte> bytes) noexcept;
[[nodiscard]] EncodeResult EncodeFrame(const Frame& frame);
[[nodiscard]] DecodeResult DecodeFrame(std::span<const std::byte> bytes);

}  // namespace lunar::unreal_tcp
