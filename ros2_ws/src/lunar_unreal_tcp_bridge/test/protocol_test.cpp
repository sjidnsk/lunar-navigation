#include "lunar_unreal_tcp_bridge/protocol.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lunar::unreal_tcp {
namespace {

std::vector<std::byte> Hex(const std::string& text) {
  std::vector<std::byte> bytes;
  bytes.reserve(text.size() / 2U);
  for (std::size_t index = 0U; index < text.size(); index += 2U) {
    bytes.push_back(static_cast<std::byte>(
        std::stoul(text.substr(index, 2U), nullptr, 16)));
  }
  return bytes;
}

nlohmann::json Vectors() {
  std::ifstream stream{UNREAL_TCP_VECTOR_FILE};
  return nlohmann::json::parse(stream);
}

std::uint32_t TestCrc32(std::span<const std::byte> bytes) {
  std::uint32_t crc = 0xffffffffU;
  for (const std::byte byte : bytes) {
    crc ^= std::to_integer<std::uint8_t>(byte);
    for (std::uint8_t bit = 0U; bit < 8U; ++bit) {
      crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
  }
  return crc ^ 0xffffffffU;
}

void PutU32(std::vector<std::byte>& bytes, const std::size_t offset,
            const std::uint32_t value) {
  for (std::size_t index = 0U; index < 4U; ++index) {
    bytes[offset + index] = static_cast<std::byte>(value >> (8U * index));
  }
}

void RefreshHeaderCrc(std::vector<std::byte>& bytes) {
  PutU32(bytes, 40U, 0U);
  PutU32(bytes, 40U, TestCrc32(
      std::span<const std::byte>{bytes.data(), kHeaderSize}));
}

TEST(Protocol, EncodesGoldenHelloFrame) {
  const auto fixture = Vectors().at("hello");
  Frame frame;
  frame.header.message_type = MessageType::kHello;
  frame.header.sequence = 1U;
  frame.metadata = fixture.at("metadata");

  const EncodeResult result = EncodeFrame(frame);

  ASSERT_TRUE(result.ok()) << result.error->reason_code;
  EXPECT_EQ(*result.bytes, Hex(fixture.at("hex").get<std::string>()));
}

TEST(Protocol, DecodesGoldenRobotStateFrame) {
  const auto fixture = Vectors().at("robot_state");
  const auto bytes = Hex(fixture.at("hex").get<std::string>());

  const DecodeResult result = DecodeFrame(bytes);

  ASSERT_TRUE(result.ok()) << result.error->reason_code;
  EXPECT_EQ(result.frame->header.message_type, MessageType::kRobotState);
  EXPECT_EQ(result.frame->header.sequence, 7U);
  EXPECT_EQ(result.frame->header.simulation_time_ns, 123'456'789LL);
  EXPECT_EQ(result.frame->header.body_crc32,
            fixture.at("body_crc32").get<std::uint32_t>());
  EXPECT_EQ(result.frame->metadata, fixture.at("metadata"));
  EXPECT_TRUE(result.frame->payload.empty());
}

TEST(Protocol, RejectsHeaderAndBodyCrcMismatch) {
  auto header = Hex(Vectors().at("hello").at("hex").get<std::string>());
  header[12] ^= std::byte{1U};
  const DecodeResult header_result = DecodeFrame(header);
  ASSERT_TRUE(header_result.error.has_value());
  EXPECT_EQ(header_result.error->code,
            ProtocolErrorCode::kHeaderCrcMismatch);

  auto body = Hex(Vectors().at("hello").at("hex").get<std::string>());
  body.back() ^= std::byte{1U};
  const DecodeResult body_result = DecodeFrame(body);
  ASSERT_TRUE(body_result.error.has_value());
  EXPECT_EQ(body_result.error->code, ProtocolErrorCode::kBodyCrcMismatch);
}

TEST(Protocol, RejectsUnknownFlagsAndNonzeroReserved) {
  auto flags = Hex(Vectors().at("hello").at("hex").get<std::string>());
  flags[10] = std::byte{0x05U};
  RefreshHeaderCrc(flags);
  const DecodeResult flags_result = DecodeFrame(flags);
  ASSERT_TRUE(flags_result.error.has_value());
  EXPECT_EQ(flags_result.error->code, ProtocolErrorCode::kUnknownFlags);

  auto reserved = Hex(Vectors().at("hello").at("hex").get<std::string>());
  reserved[44] = std::byte{1U};
  RefreshHeaderCrc(reserved);
  const DecodeResult reserved_result = DecodeFrame(reserved);
  ASSERT_TRUE(reserved_result.error.has_value());
  EXPECT_EQ(reserved_result.error->code,
            ProtocolErrorCode::kNonzeroReserved);
}

TEST(Protocol, RejectsMetadataAndBodyLimits) {
  Frame metadata;
  metadata.header.message_type = MessageType::kError;
  metadata.metadata = {{"reason_code", std::string(kMaximumMetadataBytes, 'x')},
                       {"recoverable", false}};
  const EncodeResult metadata_result = EncodeFrame(metadata);
  ASSERT_TRUE(metadata_result.error.has_value());
  EXPECT_EQ(metadata_result.error->code,
            ProtocolErrorCode::kMetadataTooLarge);

  Frame body;
  body.header.message_type = MessageType::kError;
  body.metadata = {{"reason_code", "TEST"}, {"recoverable", false}};
  body.payload.resize(kMaximumBodyBytes);
  const EncodeResult body_result = EncodeFrame(body);
  ASSERT_TRUE(body_result.error.has_value());
  EXPECT_EQ(body_result.error->code, ProtocolErrorCode::kBodyTooLarge);
}

TEST(Protocol, RejectsDuplicateJsonKeys) {
  const std::string metadata =
      R"({"reason_code":"FIRST","reason_code":"SECOND","recoverable":false})";
  std::vector<std::byte> bytes(kHeaderSize + metadata.size(), std::byte{0U});
  std::copy(kMagic.begin(), kMagic.end(), bytes.begin());
  bytes[4] = std::byte{1U};
  bytes[6] = std::byte{48U};
  bytes[8] = std::byte{0xffU};
  bytes[10] = std::byte{1U};
  bytes[12] = std::byte{1U};
  PutU32(bytes, 28U, static_cast<std::uint32_t>(metadata.size()));
  for (std::size_t index = 0U; index < metadata.size(); ++index) {
    bytes[kHeaderSize + index] = static_cast<std::byte>(metadata[index]);
  }
  PutU32(bytes, 36U, TestCrc32(
      std::span<const std::byte>{bytes.data() + kHeaderSize, metadata.size()}));
  RefreshHeaderCrc(bytes);

  const DecodeResult result = DecodeFrame(bytes);

  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->code, ProtocolErrorCode::kInvalidJson);
  EXPECT_EQ(result.error->reason_code, "JSON_DUPLICATE_KEY");
}

}  // namespace
}  // namespace lunar::unreal_tcp
