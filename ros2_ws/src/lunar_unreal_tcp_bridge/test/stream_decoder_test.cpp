#include "lunar_unreal_tcp_bridge/stream_decoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

namespace lunar::unreal_tcp {
namespace {

std::vector<std::byte> HelloBytes(const std::uint64_t sequence) {
  Frame frame;
  frame.header.message_type = MessageType::kHello;
  frame.header.sequence = sequence;
  frame.metadata = {
      {"protocol", "lunar-unreal-tcp/v1"}, {"role", "ROS_CLIENT"},
      {"client_nonce", "nonce"}, {"platform_type", "WHEELED"},
      {"robot_count", 1}, {"state_rate_hz", 20}, {"map_rate_hz", 5},
      {"heartbeat_rate_hz", 1}, {"maximum_body_bytes", 8'388'608},
  };
  const auto result = EncodeFrame(frame);
  EXPECT_TRUE(result.ok());
  return result.ok() ? *result.bytes : std::vector<std::byte>{};
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

TEST(StreamDecoder, DecodesEveryByteSplitOfGoldenFrame) {
  const auto encoded = HelloBytes(1U);
  for (std::size_t split = 1U; split < encoded.size(); ++split) {
    StreamDecoder decoder;
    const auto first = decoder.Push(
        std::span<const std::byte>{encoded.data(), split});
    ASSERT_TRUE(first.ok()) << split;
    EXPECT_TRUE(first.frames.empty()) << split;
    const auto second = decoder.Push(std::span<const std::byte>{
        encoded.data() + split, encoded.size() - split});
    ASSERT_TRUE(second.ok()) << split;
    ASSERT_EQ(second.frames.size(), 1U) << split;
    EXPECT_EQ(second.frames.front().header.sequence, 1U);
    EXPECT_EQ(decoder.buffered_bytes(), 0U);
  }
}

TEST(StreamDecoder, DecodesCoalescedFramesAndRetainsHalfFrame) {
  const auto first = HelloBytes(1U);
  const auto second = HelloBytes(2U);
  const auto third = HelloBytes(3U);
  std::vector<std::byte> chunk = first;
  chunk.insert(chunk.end(), second.begin(), second.end());
  chunk.insert(chunk.end(), third.begin(), third.begin() + third.size() / 2U);

  StreamDecoder decoder;
  const auto result = decoder.Push(chunk);

  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result.frames.size(), 2U);
  EXPECT_EQ(result.frames[0].header.sequence, 1U);
  EXPECT_EQ(result.frames[1].header.sequence, 2U);
  EXPECT_EQ(decoder.buffered_bytes(), third.size() / 2U);

  const auto tail = decoder.Push(std::span<const std::byte>{
      third.data() + third.size() / 2U,
      third.size() - third.size() / 2U});
  ASSERT_TRUE(tail.ok());
  ASSERT_EQ(tail.frames.size(), 1U);
  EXPECT_EQ(tail.frames.front().header.sequence, 3U);
}

TEST(StreamDecoder, RejectsMaliciousLengthBeforeAllocation) {
  auto header = HelloBytes(1U);
  header.resize(kHeaderSize);
  PutU32(header, 28U, static_cast<std::uint32_t>(kMaximumMetadataBytes));
  PutU32(header, 32U, static_cast<std::uint32_t>(kMaximumBodyBytes));
  PutU32(header, 40U, 0U);
  PutU32(header, 40U, TestCrc32(header));

  StreamDecoder decoder;
  const auto result = decoder.Push(header);

  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->code, ProtocolErrorCode::kBodyTooLarge);
  EXPECT_EQ(decoder.buffered_bytes(), 0U);
}

}  // namespace
}  // namespace lunar::unreal_tcp
