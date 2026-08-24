#include "lunar_pure_exploration_ros/planner_timing_accumulator.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <gtest/gtest.h>

namespace lunar::pure_exploration_ros {
namespace {

using diagnostic_msgs::msg::DiagnosticArray;
using diagnostic_msgs::msg::DiagnosticStatus;
using diagnostic_msgs::msg::KeyValue;

DiagnosticArray TimingMessage(
    std::string request_id, std::string platform_type = "WHEELED",
    std::string environment_mode = "1", std::string planning_outcome = "0",
    std::string reason_code = "PLAN_FOUND",
    std::string global_elapsed_ms = "1.25",
    std::string global_call_count = "2",
    std::string local_elapsed_ms = "3.5",
    std::string local_call_count = "4",
    std::string total_elapsed_ms = "5.75") {
  DiagnosticStatus status;
  status.values = {
      KeyValue{}.set__key("request_id").set__value(std::move(request_id)),
      KeyValue{}.set__key("platform_type").set__value(
          std::move(platform_type)),
      KeyValue{}.set__key("environment_mode").set__value(
          std::move(environment_mode)),
      KeyValue{}.set__key("planning_outcome").set__value(
          std::move(planning_outcome)),
      KeyValue{}.set__key("reason_code").set__value(std::move(reason_code)),
      KeyValue{}.set__key("global_elapsed_ms")
          .set__value(std::move(global_elapsed_ms)),
      KeyValue{}.set__key("global_call_count")
          .set__value(std::move(global_call_count)),
      KeyValue{}.set__key("local_elapsed_ms")
          .set__value(std::move(local_elapsed_ms)),
      KeyValue{}.set__key("local_call_count")
          .set__value(std::move(local_call_count)),
      KeyValue{}.set__key("total_elapsed_ms")
          .set__value(std::move(total_elapsed_ms)),
  };
  DiagnosticArray message;
  message.status.push_back(std::move(status));
  return message;
}

void ExpectTiming(const PlannerTiming& actual, const std::string& request_id,
                  const double global_elapsed_ms,
                  const std::uint64_t global_call_count,
                  const double local_elapsed_ms,
                  const std::uint64_t local_call_count,
                  const double total_elapsed_ms) {
  EXPECT_EQ(actual.request_id, request_id);
  EXPECT_DOUBLE_EQ(actual.global_elapsed_ms, global_elapsed_ms);
  EXPECT_EQ(actual.global_call_count, global_call_count);
  EXPECT_DOUBLE_EQ(actual.local_elapsed_ms, local_elapsed_ms);
  EXPECT_EQ(actual.local_call_count, local_call_count);
  EXPECT_DOUBLE_EQ(actual.total_elapsed_ms, total_elapsed_ms);
}

TEST(PlannerTimingAccumulatorTest, RequiresPositiveCapacity) {
  EXPECT_THROW(PlannerTimingAccumulator{0U}, std::invalid_argument);
}

TEST(PlannerTimingAccumulatorTest,
     AcceptsTwoInterleavedIdsAndKeepsTimingQueryableAfterward) {
  PlannerTimingAccumulator accumulator{2U};
  auto first = TimingMessage("request-a", "WHEELED", "1", "3", "NO_PATH",
                             "0", "1", "2.5", "0", "2.5");
  first.header.stamp.sec = 99;
  first.status.front().name = "ignored/status/name";
  first.status.front().hardware_id = "ignored-hardware";
  auto second = TimingMessage("request-b", "HOPPER", "2", "7", "TIMEOUT",
                              "12.5", "18446744073709551615", "0", "3",
                              "13");

  EXPECT_EQ(accumulator.Ingest(first), TimingIngestResult::kAccepted);
  EXPECT_EQ(accumulator.Ingest(second), TimingIngestResult::kAccepted);

  const auto found_a = accumulator.Find("request-a");
  const auto found_b = accumulator.Find("request-b");
  ASSERT_TRUE(found_a.has_value());
  ASSERT_TRUE(found_b.has_value());
  ExpectTiming(*found_a, "request-a", 0.0, 1U, 2.5, 0U, 2.5);
  ExpectTiming(*found_b, "request-b", 12.5,
               std::numeric_limits<std::uint64_t>::max(), 0.0, 3U, 13.0);
  EXPECT_FALSE(accumulator.Find("missing").has_value());
}

TEST(PlannerTimingAccumulatorTest,
     FirstResidentRecordWinsWithoutRefreshingFifoAge) {
  PlannerTimingAccumulator accumulator{2U};
  EXPECT_EQ(accumulator.Ingest(TimingMessage("A", "WHEELED", "1", "0",
                                             "PLAN_FOUND", "1", "1", "2",
                                             "2", "3")),
            TimingIngestResult::kAccepted);
  EXPECT_EQ(accumulator.Ingest(TimingMessage("B")),
            TimingIngestResult::kAccepted);
  EXPECT_EQ(accumulator.Ingest(TimingMessage("A", "LEGGED", "2", "6",
                                             "CONFLICT", "90", "90", "90",
                                             "90", "90")),
            TimingIngestResult::kDuplicate);
  EXPECT_EQ(accumulator.Ingest(TimingMessage("C")),
            TimingIngestResult::kAccepted);

  EXPECT_FALSE(accumulator.Find("A").has_value());
  EXPECT_TRUE(accumulator.Find("B").has_value());
  EXPECT_TRUE(accumulator.Find("C").has_value());
}

TEST(PlannerTimingAccumulatorTest, EvictionLeavesNoRequestIdTombstone) {
  PlannerTimingAccumulator accumulator{1U};
  EXPECT_EQ(accumulator.Ingest(TimingMessage("A", "WHEELED", "1", "0",
                                             "A1", "1", "1", "1", "1",
                                             "1")),
            TimingIngestResult::kAccepted);
  EXPECT_EQ(accumulator.Ingest(TimingMessage("B")),
            TimingIngestResult::kAccepted);
  EXPECT_EQ(accumulator.Ingest(TimingMessage("A", "LEGGED", "2", "9", "A2",
                                             "2", "2", "2", "2", "2")),
            TimingIngestResult::kAccepted);

  const auto found_a = accumulator.Find("A");
  ASSERT_TRUE(found_a.has_value());
  ExpectTiming(*found_a, "A", 2.0, 2U, 2.0, 2U, 2.0);
  EXPECT_FALSE(accumulator.Find("B").has_value());
}

TEST(PlannerTimingAccumulatorTest,
     RejectsNonExactStatusSchemaWithoutChangingRecordsOrFifo) {
  PlannerTimingAccumulator accumulator{2U};
  EXPECT_EQ(accumulator.Ingest(TimingMessage("A")),
            TimingIngestResult::kAccepted);

  std::vector<DiagnosticArray> malformed;
  malformed.emplace_back();
  auto two_statuses = TimingMessage("bad-two");
  two_statuses.status.push_back(two_statuses.status.front());
  malformed.push_back(std::move(two_statuses));
  auto missing = TimingMessage("bad-missing");
  missing.status.front().values.pop_back();
  malformed.push_back(std::move(missing));
  auto unknown = TimingMessage("bad-unknown");
  unknown.status.front().values.back().key = "unknown";
  malformed.push_back(std::move(unknown));
  auto duplicate = TimingMessage("bad-duplicate");
  duplicate.status.front().values.back().key = "request_id";
  malformed.push_back(std::move(duplicate));

  for (const auto& message : malformed) {
    EXPECT_EQ(accumulator.Ingest(message), TimingIngestResult::kRejected);
  }

  EXPECT_EQ(accumulator.Ingest(TimingMessage("B")),
            TimingIngestResult::kAccepted);
  EXPECT_EQ(accumulator.Ingest(TimingMessage("C")),
            TimingIngestResult::kAccepted);
  EXPECT_FALSE(accumulator.Find("A").has_value());
  EXPECT_TRUE(accumulator.Find("B").has_value());
  EXPECT_TRUE(accumulator.Find("C").has_value());
}

TEST(PlannerTimingAccumulatorTest,
     ExactTenKeyPlannerRecordParsesAndIgnoresUniqueExtensionKeys) {
  PlannerTimingAccumulator exact_accumulator{1U};
  const auto exact = TimingMessage("request-exact", "WHEELED", "1", "0",
                                   "PLAN_FOUND", "11.0", "2", "7.0",
                                   "3", "18.0");
  EXPECT_EQ(exact_accumulator.Ingest(exact), TimingIngestResult::kAccepted);
  const auto parsed = exact_accumulator.Find("request-exact");
  ASSERT_TRUE(parsed.has_value());
  ExpectTiming(*parsed, "request-exact", 11.0, 2U, 7.0, 3U, 18.0);

  auto exploration_output = exact;
  exploration_output.status.front().values.push_back(
      KeyValue{}.set__key("stop_wait_elapsed_ms").set__value("5000.000000"));
  PlannerTimingAccumulator extension_accumulator{1U};
  EXPECT_EQ(extension_accumulator.Ingest(exploration_output),
            TimingIngestResult::kAccepted);
  EXPECT_TRUE(extension_accumulator.Find("request-exact").has_value());
}

TEST(PlannerTimingAccumulatorTest,
     AcceptsUniqueGridV1DiagnosticExtensionsButRejectsBrokenRequiredFields) {
  auto grid_v1 = TimingMessage("grid-v1", "WHEELED", "1", "0",
                               "PLAN_FOUND", "11.0", "2", "7.0", "3",
                               "18.0");
  grid_v1.status.front().values.insert(
      grid_v1.status.front().values.end(),
      {KeyValue{}.set__key("grid_v1_active").set__value("true"),
       KeyValue{}.set__key("global_input_sequence").set__value("9"),
       KeyValue{}.set__key("fused_tile_count").set__value("4")});
  PlannerTimingAccumulator accepted{1U};
  EXPECT_EQ(accepted.Ingest(grid_v1), TimingIngestResult::kAccepted);
  EXPECT_TRUE(accepted.Find("grid-v1").has_value());

  auto duplicate_required = grid_v1;
  duplicate_required.status.front().values.push_back(
      KeyValue{}.set__key("request_id").set__value("other"));
  PlannerTimingAccumulator duplicate{1U};
  EXPECT_EQ(duplicate.Ingest(duplicate_required), TimingIngestResult::kRejected);

  auto missing_required = grid_v1;
  missing_required.status.front().values.erase(
      missing_required.status.front().values.begin() + 4);
  PlannerTimingAccumulator missing{1U};
  EXPECT_EQ(missing.Ingest(missing_required), TimingIngestResult::kRejected);

  auto malformed_required = grid_v1;
  malformed_required.status.front().values[5].value = "-1";
  PlannerTimingAccumulator malformed{1U};
  EXPECT_EQ(malformed.Ingest(malformed_required), TimingIngestResult::kRejected);
}

TEST(PlannerTimingAccumulatorTest,
     RejectsInvalidEnumerationsIdentifiersCountsAndElapsedValues) {
  struct InvalidCase final {
    const char* label;
    DiagnosticArray message;
  };
  const std::array cases{
      InvalidCase{"empty request", TimingMessage("")},
      InvalidCase{"empty reason", TimingMessage("id", "WHEELED", "1", "0", "")},
      InvalidCase{"platform case", TimingMessage("id", "wheeled")},
      InvalidCase{"unknown platform", TimingMessage("id", "UNKNOWN")},
      InvalidCase{"environment zero", TimingMessage("id", "WHEELED", "0")},
      InvalidCase{"environment whitespace", TimingMessage("id", "WHEELED", " 1")},
      InvalidCase{"environment sign", TimingMessage("id", "WHEELED", "+1")},
      InvalidCase{"environment trailing", TimingMessage("id", "WHEELED", "1x")},
      InvalidCase{"unknown outcome", TimingMessage("id", "WHEELED", "1", "10")},
      InvalidCase{"outcome whitespace", TimingMessage("id", "WHEELED", "1", "0 ")},
      InvalidCase{"negative count", TimingMessage("id", "WHEELED", "1", "0",
                                                    "ok", "1", "-1")},
      InvalidCase{"signed count", TimingMessage("id", "WHEELED", "1", "0",
                                                  "ok", "1", "+1")},
      InvalidCase{"count trailing", TimingMessage("id", "WHEELED", "1", "0",
                                                    "ok", "1", "1x")},
      InvalidCase{"count overflow",
                  TimingMessage("id", "WHEELED", "1", "0", "ok", "1",
                                "18446744073709551616")},
      InvalidCase{"negative elapsed", TimingMessage("id", "WHEELED", "1", "0",
                                                      "ok", "-0.1")},
      InvalidCase{"nan elapsed", TimingMessage("id", "WHEELED", "1", "0", "ok",
                                                 "nan")},
      InvalidCase{"infinite elapsed",
                  TimingMessage("id", "WHEELED", "1", "0", "ok", "inf")},
      InvalidCase{"elapsed whitespace",
                  TimingMessage("id", "WHEELED", "1", "0", "ok", " 1")},
      InvalidCase{"elapsed trailing",
                  TimingMessage("id", "WHEELED", "1", "0", "ok", "1ms")},
  };

  for (const auto& invalid : cases) {
    SCOPED_TRACE(invalid.label);
    PlannerTimingAccumulator accumulator{1U};
    EXPECT_EQ(accumulator.Ingest(invalid.message), TimingIngestResult::kRejected);
    EXPECT_FALSE(accumulator.Find("id").has_value());
  }
}

TEST(PlannerTimingAccumulatorTest, AcceptsEveryDefinedOutcomeAndExactEnumStrings) {
  constexpr std::array<const char*, 3> kPlatforms{"WHEELED", "LEGGED", "HOPPER"};
  for (std::uint8_t outcome = 0U; outcome <= 9U; ++outcome) {
    for (const char* platform : kPlatforms) {
      for (const char* environment : {"1", "2"}) {
        PlannerTimingAccumulator accumulator{1U};
        EXPECT_EQ(accumulator.Ingest(TimingMessage(
                      "id", platform, environment, std::to_string(outcome),
                      "REASON", "0", "1", "0", "1", "0")),
                  TimingIngestResult::kAccepted);
      }
    }
  }
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
