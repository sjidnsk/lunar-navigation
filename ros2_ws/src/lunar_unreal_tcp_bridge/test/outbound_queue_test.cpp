#include "lunar_unreal_tcp_bridge/outbound_queue.hpp"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace lunar::unreal_tcp {
namespace {

Frame Message(const MessageType type, const std::uint64_t sequence) {
  Frame frame;
  frame.header.message_type = type;
  frame.header.sequence = sequence;
  if (type == MessageType::kControl) {
    frame.metadata = {{"command", "HOLD"}};
  }
  return frame;
}

TEST(OutboundQueue, ReplacesOnlyLatestStateAndMap) {
  OutboundQueue queue;
  EXPECT_TRUE(queue.Push(Message(MessageType::kRobotState, 1U)).accepted);
  const auto state_replacement =
      queue.Push(Message(MessageType::kRobotState, 2U));
  EXPECT_TRUE(state_replacement.accepted);
  EXPECT_TRUE(state_replacement.replaced_latest);
  EXPECT_TRUE(queue.Push(Message(MessageType::kLocalElevationMap, 3U)).accepted);
  const auto map_replacement =
      queue.Push(Message(MessageType::kLocalElevationMap, 4U));
  EXPECT_TRUE(map_replacement.replaced_latest);
  EXPECT_EQ(queue.size(), 2U);

  EXPECT_EQ(queue.Pop()->header.sequence, 2U);
  EXPECT_EQ(queue.Pop()->header.sequence, 4U);
  EXPECT_FALSE(queue.Pop().has_value());
}

TEST(OutboundQueue, PreservesControlReferenceFeedbackOrdering) {
  OutboundQueue queue;
  EXPECT_TRUE(queue.Push(Message(MessageType::kLocalElevationMap, 1U)).accepted);
  EXPECT_TRUE(queue.Push(Message(MessageType::kRobotState, 2U)).accepted);
  EXPECT_TRUE(queue.Push(Message(MessageType::kExecutionFeedback, 3U)).accepted);
  EXPECT_TRUE(queue.Push(Message(MessageType::kMotionReference, 4U)).accepted);
  EXPECT_TRUE(queue.Push(Message(MessageType::kControl, 5U)).accepted);

  std::vector<std::uint64_t> order;
  while (auto frame = queue.Pop()) {
    order.push_back(frame->header.sequence);
  }
  EXPECT_EQ(order, (std::vector<std::uint64_t>{5U, 3U, 4U, 2U, 1U}));
}

TEST(OutboundQueue, ReportsHighPriorityQueueExhaustion) {
  OutboundQueue queue{2U};
  EXPECT_TRUE(queue.Push(Message(MessageType::kControl, 1U)).accepted);
  EXPECT_TRUE(queue.Push(Message(MessageType::kError, 2U)).accepted);

  const auto result = queue.Push(Message(MessageType::kMotionReference, 3U));

  EXPECT_FALSE(result.accepted);
  EXPECT_TRUE(result.fatal);
  EXPECT_EQ(result.reason_code, "OUTBOUND_RELIABLE_QUEUE_EXHAUSTED");
  EXPECT_EQ(queue.size(), 2U);
}

}  // namespace
}  // namespace lunar::unreal_tcp
