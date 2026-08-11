#include "lunar_unreal_tcp_bridge/protocol.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "lunar_unreal_tcp_bridge/metadata_validation.hpp"

namespace lunar::unreal_tcp {
namespace {

using Json = nlohmann::json;

template<typename Integer>
void AppendLittleEndian(std::vector<std::byte>& output, Integer value) {
  using Unsigned = std::make_unsigned_t<Integer>;
  const Unsigned bits = static_cast<Unsigned>(value);
  for (std::size_t index = 0U; index < sizeof(Integer); ++index) {
    output.push_back(static_cast<std::byte>(bits >> (index * 8U)));
  }
}

template<typename Integer>
[[nodiscard]] Integer ReadLittleEndian(
    const std::span<const std::byte> bytes, const std::size_t offset) {
  using Unsigned = std::make_unsigned_t<Integer>;
  Unsigned value{};
  for (std::size_t index = 0U; index < sizeof(Integer); ++index) {
    value |= static_cast<Unsigned>(
                 std::to_integer<std::uint8_t>(bytes[offset + index]))
        << (index * 8U);
  }
  return static_cast<Integer>(value);
}

void PutLittleEndian(
    std::span<std::byte> bytes, const std::size_t offset,
    const std::uint32_t value) {
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    bytes[offset + index] = static_cast<std::byte>(value >> (index * 8U));
  }
}

[[nodiscard]] ProtocolError Error(
    const ProtocolErrorCode code, std::string reason_code) {
  return ProtocolError{.code = code, .reason_code = std::move(reason_code)};
}

[[nodiscard]] EncodeResult EncodeFailure(
    const ProtocolErrorCode code, std::string reason_code) {
  return EncodeResult{
      .bytes = std::nullopt,
      .error = Error(code, std::move(reason_code)),
  };
}

[[nodiscard]] DecodeResult DecodeFailure(
    const ProtocolErrorCode code, std::string reason_code) {
  return DecodeResult{
      .frame = std::nullopt,
      .error = Error(code, std::move(reason_code)),
  };
}

[[nodiscard]] bool KnownMessageType(const MessageType type) noexcept {
  switch (type) {
    case MessageType::kHello:
    case MessageType::kHelloAck:
    case MessageType::kControl:
    case MessageType::kControlAck:
    case MessageType::kRobotState:
    case MessageType::kLocalElevationMap:
    case MessageType::kMotionReference:
    case MessageType::kExecutionFeedback:
    case MessageType::kHeartbeat:
    case MessageType::kError:
      return true;
  }
  return false;
}

class DuplicateKeySax final : public nlohmann::json_sax<Json> {
 public:
  bool null() override { return true; }
  bool boolean(bool) override { return true; }
  bool number_integer(number_integer_t) override { return true; }
  bool number_unsigned(number_unsigned_t) override { return true; }
  bool number_float(number_float_t, const string_t&) override { return true; }
  bool string(string_t&) override { return true; }
  bool binary(binary_t&) override { return true; }
  bool start_object(std::size_t) override {
    object_keys_.emplace_back(std::unordered_set<std::string>{});
    return true;
  }
  bool key(string_t& value) override {
    if (object_keys_.empty() || !object_keys_.back().has_value()) {
      syntax_error_ = true;
      return false;
    }
    if (!object_keys_.back()->insert(value).second) {
      duplicate_key_ = true;
      return false;
    }
    return true;
  }
  bool end_object() override {
    if (object_keys_.empty() || !object_keys_.back().has_value()) {
      syntax_error_ = true;
      return false;
    }
    object_keys_.pop_back();
    return true;
  }
  bool start_array(std::size_t) override {
    object_keys_.emplace_back(std::nullopt);
    return true;
  }
  bool end_array() override {
    if (object_keys_.empty() || object_keys_.back().has_value()) {
      syntax_error_ = true;
      return false;
    }
    object_keys_.pop_back();
    return true;
  }
  bool parse_error(
      std::size_t, const std::string&, const nlohmann::detail::exception&) override {
    syntax_error_ = true;
    return false;
  }

  [[nodiscard]] bool duplicate_key() const noexcept { return duplicate_key_; }
  [[nodiscard]] bool syntax_error() const noexcept { return syntax_error_; }

 private:
  std::vector<std::optional<std::unordered_set<std::string>>> object_keys_;
  bool duplicate_key_{};
  bool syntax_error_{};
};

[[nodiscard]] std::optional<ProtocolError> ValidateJsonText(
    const std::string& text, Json& output) {
  DuplicateKeySax sax;
  const bool valid = Json::sax_parse(text, &sax);
  if (!valid || sax.duplicate_key() || sax.syntax_error()) {
    return Error(
        ProtocolErrorCode::kInvalidJson,
        sax.duplicate_key() ? "JSON_DUPLICATE_KEY" : "JSON_INVALID");
  }
  try {
    output = Json::parse(text);
  } catch (const Json::exception&) {
    return Error(ProtocolErrorCode::kInvalidJson, "JSON_INVALID");
  }
  if (!output.is_object()) {
    return Error(ProtocolErrorCode::kInvalidJson, "JSON_TOP_LEVEL_NOT_OBJECT");
  }
  return std::nullopt;
}

}  // namespace

std::uint32_t Crc32(const std::span<const std::byte> bytes) noexcept {
  std::uint32_t crc = 0xffffffffU;
  for (const std::byte byte : bytes) {
    crc ^= std::to_integer<std::uint8_t>(byte);
    for (std::uint8_t bit = 0U; bit < 8U; ++bit) {
      crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
  }
  return crc ^ 0xffffffffU;
}

EncodeResult EncodeFrame(const Frame& frame) {
  if (!KnownMessageType(frame.header.message_type)) {
    return EncodeFailure(
        ProtocolErrorCode::kUnknownMessageType, "MESSAGE_TYPE_UNKNOWN");
  }
  if (!frame.metadata.is_object()) {
    return EncodeFailure(
        ProtocolErrorCode::kInvalidJson, "JSON_TOP_LEVEL_NOT_OBJECT");
  }
  if (const auto metadata_error =
          ValidateMetadata(frame.header.message_type, frame.metadata)) {
    return EncodeFailure(
        ProtocolErrorCode::kInvalidMetadata, metadata_error->reason_code);
  }

  std::string metadata_text;
  try {
    metadata_text = frame.metadata.dump(
        -1, ' ', false, nlohmann::json::error_handler_t::strict);
  } catch (const Json::exception&) {
    return EncodeFailure(ProtocolErrorCode::kInvalidJson, "JSON_INVALID");
  }
  if (metadata_text.size() > kMaximumMetadataBytes) {
    return EncodeFailure(
        ProtocolErrorCode::kMetadataTooLarge, "METADATA_TOO_LARGE");
  }
  if (frame.payload.size() > kMaximumBodyBytes - metadata_text.size()) {
    return EncodeFailure(ProtocolErrorCode::kBodyTooLarge, "BODY_TOO_LARGE");
  }

  std::vector<std::byte> body;
  body.reserve(metadata_text.size() + frame.payload.size());
  for (const char character : metadata_text) {
    body.push_back(static_cast<std::byte>(character));
  }
  body.insert(body.end(), frame.payload.begin(), frame.payload.end());

  const std::uint16_t flags = static_cast<std::uint16_t>(
      0x0001U | (frame.payload.empty() ? 0x0000U : 0x0002U));
  std::vector<std::byte> header;
  header.reserve(kHeaderSize);
  header.insert(header.end(), kMagic.begin(), kMagic.end());
  AppendLittleEndian<std::uint16_t>(header, 1U);
  AppendLittleEndian<std::uint16_t>(header, kHeaderSize);
  AppendLittleEndian<std::uint16_t>(
      header, static_cast<std::uint16_t>(frame.header.message_type));
  AppendLittleEndian<std::uint16_t>(header, flags);
  AppendLittleEndian<std::uint64_t>(header, frame.header.sequence);
  AppendLittleEndian<std::int64_t>(header, frame.header.simulation_time_ns);
  AppendLittleEndian<std::uint32_t>(header, metadata_text.size());
  AppendLittleEndian<std::uint32_t>(header, frame.payload.size());
  AppendLittleEndian<std::uint32_t>(header, Crc32(body));
  AppendLittleEndian<std::uint32_t>(header, 0U);
  AppendLittleEndian<std::uint32_t>(header, 0U);
  PutLittleEndian(header, 40U, Crc32(header));

  header.insert(header.end(), body.begin(), body.end());
  return EncodeResult{.bytes = std::move(header), .error = std::nullopt};
}

DecodeResult DecodeFrame(const std::span<const std::byte> bytes) {
  if (bytes.size() < kHeaderSize) {
    return DecodeFailure(
        ProtocolErrorCode::kLengthMismatch, "FRAME_HEADER_INCOMPLETE");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    return DecodeFailure(ProtocolErrorCode::kInvalidMagic, "MAGIC_INVALID");
  }
  const std::uint16_t version = ReadLittleEndian<std::uint16_t>(bytes, 4U);
  if (version != 1U) {
    return DecodeFailure(
        ProtocolErrorCode::kUnsupportedVersion, "VERSION_UNSUPPORTED");
  }
  const std::uint16_t header_size = ReadLittleEndian<std::uint16_t>(bytes, 6U);
  if (header_size != kHeaderSize) {
    return DecodeFailure(
        ProtocolErrorCode::kInvalidHeaderSize, "HEADER_SIZE_INVALID");
  }

  std::array<std::byte, kHeaderSize> header_copy{};
  std::copy_n(bytes.begin(), kHeaderSize, header_copy.begin());
  const std::uint32_t expected_header_crc =
      ReadLittleEndian<std::uint32_t>(bytes, 40U);
  PutLittleEndian(header_copy, 40U, 0U);
  if (Crc32(header_copy) != expected_header_crc) {
    return DecodeFailure(
        ProtocolErrorCode::kHeaderCrcMismatch, "HEADER_CRC_MISMATCH");
  }

  const auto message_type = static_cast<MessageType>(
      ReadLittleEndian<std::uint16_t>(bytes, 8U));
  if (!KnownMessageType(message_type)) {
    return DecodeFailure(
        ProtocolErrorCode::kUnknownMessageType, "MESSAGE_TYPE_UNKNOWN");
  }
  const std::uint16_t flags = ReadLittleEndian<std::uint16_t>(bytes, 10U);
  if ((flags & ~0x0003U) != 0U || (flags & 0x0001U) == 0U) {
    return DecodeFailure(
        ProtocolErrorCode::kUnknownFlags, "FLAGS_INVALID");
  }
  const std::uint32_t reserved = ReadLittleEndian<std::uint32_t>(bytes, 44U);
  if (reserved != 0U) {
    return DecodeFailure(
        ProtocolErrorCode::kNonzeroReserved, "RESERVED_NONZERO");
  }

  const std::uint32_t metadata_length =
      ReadLittleEndian<std::uint32_t>(bytes, 28U);
  const std::uint32_t payload_length =
      ReadLittleEndian<std::uint32_t>(bytes, 32U);
  if (metadata_length > kMaximumMetadataBytes) {
    return DecodeFailure(
        ProtocolErrorCode::kMetadataTooLarge, "METADATA_TOO_LARGE");
  }
  if (payload_length > kMaximumBodyBytes - metadata_length) {
    return DecodeFailure(ProtocolErrorCode::kBodyTooLarge, "BODY_TOO_LARGE");
  }
  const std::size_t body_size =
      static_cast<std::size_t>(metadata_length) + payload_length;
  if (bytes.size() != kHeaderSize + body_size) {
    return DecodeFailure(
        ProtocolErrorCode::kLengthMismatch, "FRAME_LENGTH_MISMATCH");
  }
  if (((flags & 0x0002U) != 0U) != (payload_length != 0U)) {
    return DecodeFailure(
        ProtocolErrorCode::kUnknownFlags, "PAYLOAD_FLAG_MISMATCH");
  }
  const std::span<const std::byte> body = bytes.subspan(kHeaderSize, body_size);
  const std::uint32_t expected_body_crc =
      ReadLittleEndian<std::uint32_t>(bytes, 36U);
  if (Crc32(body) != expected_body_crc) {
    return DecodeFailure(
        ProtocolErrorCode::kBodyCrcMismatch, "BODY_CRC_MISMATCH");
  }

  std::string metadata_text;
  metadata_text.reserve(metadata_length);
  for (const std::byte byte : body.first(metadata_length)) {
    metadata_text.push_back(static_cast<char>(std::to_integer<unsigned char>(byte)));
  }
  Json metadata;
  if (const auto json_error = ValidateJsonText(metadata_text, metadata)) {
    return DecodeResult{.frame = std::nullopt, .error = json_error};
  }
  if (const auto metadata_error = ValidateMetadata(message_type, metadata)) {
    return DecodeFailure(
        ProtocolErrorCode::kInvalidMetadata, metadata_error->reason_code);
  }

  Frame frame;
  frame.header = FrameHeader{
      .protocol_version = version,
      .header_size = header_size,
      .message_type = message_type,
      .flags = flags,
      .sequence = ReadLittleEndian<std::uint64_t>(bytes, 12U),
      .simulation_time_ns = ReadLittleEndian<std::int64_t>(bytes, 20U),
      .metadata_length = metadata_length,
      .payload_length = payload_length,
      .body_crc32 = expected_body_crc,
      .header_crc32 = expected_header_crc,
      .reserved = reserved,
  };
  frame.metadata = std::move(metadata);
  frame.payload.assign(
      body.begin() + static_cast<std::ptrdiff_t>(metadata_length), body.end());
  return DecodeResult{.frame = std::move(frame), .error = std::nullopt};
}

}  // namespace lunar::unreal_tcp
