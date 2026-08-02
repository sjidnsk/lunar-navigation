#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lpp = lunar::planning::v3;

namespace {

constexpr std::string_view kMapHash =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr std::string_view kCapabilityHash =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr std::string_view kConfigHash =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

std::string ReadMinimalWheeledRequest() {
  const auto fixture =
      std::filesystem::path{LPP_V3_SCHEMA_ROOT}.parent_path().parent_path() /
      "cpp" / "tests" / "fixtures" / "minimal_wheeled_request.json";
  std::ifstream input{fixture, std::ios::binary};
  EXPECT_TRUE(input.is_open()) << fixture.string();
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>{}};
}

std::string RepeatedHash(char digit) {
  return std::string(64U, digit);
}

lpp::ContentRef MakeRef(std::string id, char hash_digit) {
  return {std::move(id), 1U, RepeatedHash(hash_digit)};
}

std::shared_ptr<const lpp::ImmutableMapSnapshot>
MakeImmutableMapSnapshot() {
  lpp::MapSnapshotInput input{
      .snapshot_ref =
          {"map-snapshot-1", 1U, std::string{kMapHash}},
      .map_revision = 1U,
      .immutable_data_handle = "map-handle-1",
      .source_time = {"mission-clock", std::chrono::nanoseconds{
                                            9'007'199'254'740'990LL}},
      .bounds = {{-10.0, -10.0, -1.0}, {10.0, 10.0, 2.0}},
      .geometry =
          {.width = 1U,
           .height = 1U,
           .resolution_m = 0.25,
           .origin_m = {-10.0, -10.0},
           .frame_id = "map"},
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask,
               MakeRef("known-mask-1", '1')},
              {lpp::LayerKind::kElevation,
               MakeRef("elevation-1", '2')},
              {lpp::LayerKind::kTerrainNormal,
               MakeRef("terrain-normal-1", '3')},
              {lpp::LayerKind::kRoughness,
               MakeRef("roughness-1", '4')},
              {lpp::LayerKind::kHardObstacle,
               MakeRef("hard-obstacle-1", '5')},
              {lpp::LayerKind::kConfidence,
               MakeRef("confidence-1", '6')},
          },
      .known_mask = {1U},
      .elevation_m = {0.0F},
      .normal_x = {0.0F},
      .normal_y = {0.0F},
      .normal_z = {1.0F},
      .roughness_m = {0.0F},
      .hard_obstacle_mask = {0U},
      .confidence = {1.0F},
  };
  auto snapshot = lpp::ImmutableMapSnapshot::Create(input);
  if (!lpp::IsOk(snapshot)) {
    throw std::runtime_error{
        std::get<lpp::Error>(snapshot).message};
  }
  return std::get<
      std::shared_ptr<const lpp::ImmutableMapSnapshot>>(snapshot);
}

class TestContractObject final : public lpp::ImmutableContractObject {
 public:
  TestContractObject(lpp::ContentRef ref, lpp::ContractObjectKind kind)
      : ref_{std::move(ref)}, kind_{kind} {}

  lpp::ContentRef content_ref() const override { return ref_; }
  lpp::ContractObjectKind kind() const override { return kind_; }

 private:
  lpp::ContentRef ref_;
  lpp::ContractObjectKind kind_;
};

lpp::CubicPolynomialSegment MakeCubicSegment() {
  return {
      .start_offset = {
          std::chrono::nanoseconds{0}},
      .end_offset = {
          std::chrono::nanoseconds{1'000'000'000}},
      .coefficients = {0.0, 0.0, 0.0, 0.0},
  };
}

class FixtureRegistry final : public lpp::ContractObjectRegistry {
 public:
  FixtureRegistry()
      : map_snapshot_{MakeImmutableMapSnapshot()},
        safety_capability_{std::make_shared<lpp::SafetyCapabilityProfile>()},
        algorithm_config_{std::make_shared<lpp::PlannerAlgorithmConfig>()} {
    safety_capability_->content_ref = {
        "wheel-capability-1", 1U, std::string{kCapabilityHash}};
    algorithm_config_->content_ref = {
        "planner-config-1", 1U, std::string{kConfigHash}};
  }

  std::shared_ptr<const lpp::ImmutableMapSnapshot> FindMapSnapshot(
      const lpp::ContentRef& snapshot_ref,
      std::string_view immutable_data_handle) const override {
    map_lookup_seen = true;
    if (snapshot_ref ==
            lpp::ContentRef{"map-snapshot-1", 1U, std::string{kMapHash}} &&
        immutable_data_handle == "map-handle-1") {
      return map_snapshot_;
    }
    return nullptr;
  }

  std::shared_ptr<const lpp::SafetyCapabilityProfile> FindSafetyCapability(
      const lpp::ContentRef& ref) const override {
    capability_lookup_seen = true;
    return ref == safety_capability_->content_ref ? safety_capability_
                                                  : nullptr;
  }

  std::shared_ptr<const lpp::PlannerAlgorithmConfig> FindAlgorithmConfig(
      const lpp::ContentRef& ref) const override {
    config_lookup_seen = true;
    return ref == algorithm_config_->content_ref ? algorithm_config_
                                                 : nullptr;
  }

  std::shared_ptr<const lpp::LearnedCostSnapshot> FindLearnedCost(
      const lpp::ContentRef&, std::string_view) const override {
    return nullptr;
  }

  lpp::Result<lpp::ResolvedCapabilityBindings> ResolveCapabilityBindings(
      const lpp::SafetyCapabilityProfile&) const override {
    bindings_lookup_seen = true;
    return lpp::ResolvedCapabilityBindings{};
  }

  lpp::Result<std::shared_ptr<const lpp::ImmutableContractObject>> Resolve(
      const lpp::ContentRef& ref,
      lpp::ContractObjectKind kind) const override {
    return std::make_shared<const TestContractObject>(ref, kind);
  }

  mutable bool map_lookup_seen{};
  mutable bool capability_lookup_seen{};
  mutable bool config_lookup_seen{};
  mutable bool bindings_lookup_seen{};

 private:
  std::shared_ptr<const lpp::ImmutableMapSnapshot> map_snapshot_;
  std::shared_ptr<lpp::SafetyCapabilityProfile> safety_capability_;
  std::shared_ptr<lpp::PlannerAlgorithmConfig> algorithm_config_;
};

std::string SimpleResponseJson() {
  return R"({
    "schema_version":"path-planner-v3-planning-response/v1",
    "request_id":"request-1",
    "response_time":{"clock_id":"mission-clock","tick_ns":"-1"},
    "planning_outcome":"NO_KNOWN_SAFE_ROUTE",
    "execution_directive":"NO_SAFE_PLANNER_REFERENCE",
    "reason_code":"NO_KNOWN_SAFE_ROUTE",
    "call_diagnostics":{
      "api_latency_ns":"17",
      "termination_reason":"GRAPH_EXHAUSTED",
      "expanded_state_count":3,
      "candidate_count":2,
      "resource_limit_hit":false,
      "learned_cost_usage":"DISABLED"
    }
  })";
}

lpp::ReferenceActivationContext MakeActivationContext(
    const lpp::PlanningRequest& request,
    const lpp::ContractObjectRegistry& registry) {
  return {.request = request, .registry = registry};
}

lpp::PlanningResponse MakeWheeledActivationResponse() {
  const lpp::ContentRef certificate_ref =
      MakeRef("certificate-1", 'e');
  const lpp::ContentRef evidence_ref =
      MakeRef("evidence-1", 'd');

  lpp::SpinSegment spin{
      .segment_id = "spin-1",
      .time_interval =
          {{std::chrono::nanoseconds{0}},
           {std::chrono::nanoseconds{1'000'000'000}}},
      .fixed_position_m = {0.0, 0.0, 0.0},
      .unwrapped_yaw_rad =
          {.value_semantics = "unwrapped_yaw_rad",
           .segments = {MakeCubicSegment()}},
  };
  lpp::WheeledReference reference{
      .reference_id = "wheel-reference-1",
      .reference_hash = RepeatedHash('f'),
      .reference_time_origin =
          {"mission-clock", std::chrono::nanoseconds{10}},
      .segments = {spin},
      .safe_stop_anchor =
          {.anchor_id = "stop-1",
           .pose = {{0.0, 0.0, 0.0}, 0.0},
           .target_linear_velocity_mps = 0.0,
           .target_yaw_rate_radps = 0.0,
           .terrain_certification_ref = certificate_ref},
  };

  lpp::ReferenceBundle bundle{
      .bundle_id = "bundle-1",
      .bundle_revision = 1U,
      .bundle_hash = RepeatedHash('9'),
      .source_request_id = "request-1",
      .source_map_snapshot_ref =
          {"map-snapshot-1", 1U, std::string{kMapHash}},
      .source_safety_capability_ref =
          {"wheel-capability-1", 1U, std::string{kCapabilityHash}},
      .source_algorithm_config_ref =
          {"planner-config-1", 1U, std::string{kConfigHash}},
      .platform_type = lpp::PlatformType::kWheeled,
      .platform_reference = reference,
      .route_skeleton =
          {.component_id = "route-1",
           .component_hash = RepeatedHash('8'),
           .content =
               {.source_reference_id = reference.reference_id,
                .source_reference_hash = reference.reference_hash,
                .waypoints = {},
                .unresolved_tail = {}}},
      .committed_prefix =
          {.component_id = "committed-1",
           .component_hash = RepeatedHash('7'),
           .content =
               {.role =
                    lpp::ReferenceViewContent::Role::kCommittedPrefix,
                .source_reference_id = reference.reference_id,
                .source_reference_hash = reference.reference_hash,
                .selector = lpp::TimeViewSelector{
                    {{std::chrono::nanoseconds{0}},
                     {std::chrono::nanoseconds{500'000'000}}}}}},
      .preview =
          {.component_id = "preview-1",
           .component_hash = RepeatedHash('6'),
           .content =
               {.role = lpp::ReferenceViewContent::Role::kPreview,
                .source_reference_id = reference.reference_id,
                .source_reference_hash = reference.reference_hash,
                .selector = lpp::TimeViewSelector{
                    {{std::chrono::nanoseconds{500'000'000}},
                     {std::chrono::nanoseconds{1'000'000'000}}}}}},
      .validity =
          {.valid_from =
               {"mission-clock", std::chrono::nanoseconds{10}},
           .required_map_snapshot_ref =
               {"map-snapshot-1", 1U, std::string{kMapHash}},
           .required_capability_ref =
               {"wheel-capability-1",
                1U,
                std::string{kCapabilityHash}},
           .allowed_state_deviation =
               lpp::WheeledOrLeggedErrorBounds{},
           .invalidation_conditions =
               {lpp::InvalidationCondition::
                    kMapSafetyRevisionChanged}},
      .validation_summary =
          {.hard_constraints_passed = true,
           .continuous_validation_passed = true,
           .certificate_refs = {certificate_ref},
           .warning_codes = {}},
      .generation_evidence =
          {.selected_candidate_id = "candidate-1",
           .generation_mode =
               lpp::GenerationMode::kValidatedPrimitiveChainReference,
           .termination_reason = "TARGET_REACHED",
           .evidence_refs = {evidence_ref}},
  };

  return {
      .request_id = "request-1",
      .response_time =
          {"mission-clock", std::chrono::nanoseconds{11}},
      .planning_outcome = lpp::PlanningOutcome::kNewReferenceReady,
      .execution_directive =
          lpp::ExecutionDirective::kActivateNewBundle,
      .reason_code = "TARGET_REACHED",
      .new_reference_bundle = std::move(bundle),
      .call_diagnostics =
          {.api_latency = {std::chrono::nanoseconds{42}},
           .termination_reason = "TARGET_REACHED",
           .expanded_state_count = 5U,
           .candidate_count = 2U,
           .resource_limit_hit = false,
           .learned_cost_usage = lpp::LearnedCostUsage::kDisabled},
  };
}

}  // namespace

TEST(JsonCodec, RejectsWrongRequestSchemaVersionBeforeRegistryLookup) {
  const std::string payload = R"({
    "schema_version":"path-planner-v3-planning-request/v0"
  })";
  lpp::EmptyContractObjectRegistry registry;
  const auto result =
      lpp::JsonCodec::DecodePlanningRequest(payload, registry);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kSchemaMismatch);
}

TEST(JsonCodec, NanosecondsRoundTripAsDecimalString) {
  const lpp::DurationNanoseconds value{
      std::chrono::nanoseconds{9'007'199'254'740'993LL}};
  const auto encoded = lpp::JsonCodec::EncodeDuration(value);
  ASSERT_TRUE(lpp::IsOk(encoded));
  EXPECT_EQ(std::get<std::string>(encoded), "\"9007199254740993\"");

  const auto decoded =
      lpp::JsonCodec::DecodeDuration(std::get<std::string>(encoded));
  ASSERT_TRUE(lpp::IsOk(decoded));
  EXPECT_EQ(std::get<lpp::DurationNanoseconds>(decoded).value.count(),
            value.value.count());
}

TEST(JsonCodec, RejectsNonCanonicalOrNegativeDurationNanoseconds) {
  for (const std::string_view text :
       {"\"00\"", "\"01\"", "\"-0\"", "\"+1\"", "\"-1\"",
        "\"9223372036854775808\""}) {
    EXPECT_FALSE(lpp::IsOk(lpp::JsonCodec::DecodeDuration(text)))
        << text;
  }
}

TEST(JsonCodec, RejectsDuplicateKeysBeforeRequiredFieldChecks) {
  const std::string payload = R"({
    "schema_version":"path-planner-v3-planning-request/v1",
    "schema_version":"path-planner-v3-planning-request/v1"
  })";
  lpp::EmptyContractObjectRegistry registry;
  const auto result =
      lpp::JsonCodec::DecodePlanningRequest(payload, registry);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kSchemaMismatch);
}

TEST(JsonCodec, RejectsUnknownTopLevelRequestKey) {
  auto payload = nlohmann::json::parse(ReadMinimalWheeledRequest());
  payload["deadline_ns"] = "1000000000";

  FixtureRegistry registry;
  const auto result =
      lpp::JsonCodec::DecodePlanningRequest(payload.dump(), registry);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kSchemaMismatch);
  EXPECT_FALSE(registry.map_lookup_seen);
}

TEST(JsonCodec, RejectsUnknownPlatformEnum) {
  auto payload = nlohmann::json::parse(ReadMinimalWheeledRequest());
  payload["platform_type"] = "ACKERMANN";

  FixtureRegistry registry;
  const auto result =
      lpp::JsonCodec::DecodePlanningRequest(payload.dump(), registry);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kSchemaMismatch);
}

TEST(JsonCodec, RejectsOverflowingNonFiniteNumber) {
  std::string payload = ReadMinimalWheeledRequest();
  const std::string needle = "\"yaw_rad\": 0.0";
  const auto position = payload.find(needle);
  ASSERT_NE(position, std::string::npos);
  payload.replace(position, needle.size(), "\"yaw_rad\": 1e9999");

  FixtureRegistry registry;
  const auto result =
      lpp::JsonCodec::DecodePlanningRequest(payload, registry);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kInvalidArgument);
}

TEST(JsonCodec, MissingRegistryObjectFailsClosed) {
  lpp::EmptyContractObjectRegistry registry;
  const auto result = lpp::JsonCodec::DecodePlanningRequest(
      ReadMinimalWheeledRequest(), registry);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kMissingRegistryObject);
}

TEST(JsonCodec, DecodesMinimalWheeledFixtureAndResolvesAllRequestObjects) {
  FixtureRegistry registry;
  const auto result = lpp::JsonCodec::DecodePlanningRequest(
      ReadMinimalWheeledRequest(), registry);
  ASSERT_TRUE(lpp::IsOk(result))
      << std::get<lpp::Error>(result).message;

  const auto& request = std::get<lpp::PlanningRequest>(result);
  EXPECT_EQ(request.request_id, "request-1");
  EXPECT_EQ(request.request_time.tick.count(), 9'007'199'254'740'993LL);
  EXPECT_EQ(request.platform_type, lpp::PlatformType::kWheeled);
  ASSERT_TRUE(
      std::holds_alternative<lpp::WheeledOrLeggedState>(
          request.current_state));
  EXPECT_TRUE(registry.map_lookup_seen);
  EXPECT_TRUE(registry.capability_lookup_seen);
  EXPECT_TRUE(registry.config_lookup_seen);
  EXPECT_TRUE(registry.bindings_lookup_seen);
}

TEST(JsonCodec, PlanningRequestRoundTripsThroughImmutableMapWireView) {
  FixtureRegistry registry;
  const auto decoded = lpp::JsonCodec::DecodePlanningRequest(
      ReadMinimalWheeledRequest(), registry);
  ASSERT_TRUE(lpp::IsOk(decoded));

  const std::string encoded = lpp::JsonCodec::EncodePlanningRequest(
      std::get<lpp::PlanningRequest>(decoded));
  const auto wire = nlohmann::json::parse(encoded);
  EXPECT_EQ(wire.at("request_time").at("tick_ns"),
            "9007199254740993");
  EXPECT_EQ(wire.at("map_snapshot").at("immutable_data_handle"),
            "map-handle-1");
  EXPECT_EQ(wire.at("map_snapshot").at("layer_manifest").size(), 6U);

  const auto round_trip =
      lpp::JsonCodec::DecodePlanningRequest(encoded, registry);
  ASSERT_TRUE(lpp::IsOk(round_trip));
  EXPECT_EQ(std::get<lpp::PlanningRequest>(round_trip)
                .request_time.tick.count(),
            9'007'199'254'740'993LL);
}

TEST(JsonCodec, SignedTimePointAllowsCanonicalNegativeNanoseconds) {
  auto payload = nlohmann::json::parse(ReadMinimalWheeledRequest());
  payload["request_time"]["tick_ns"] = "-9007199254740993";

  FixtureRegistry registry;
  const auto result =
      lpp::JsonCodec::DecodePlanningRequest(payload.dump(), registry);
  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::PlanningRequest>(result).request_time.tick.count(),
            -9'007'199'254'740'993LL);
}

TEST(JsonCodec, PlanningResponseRoundTripsLosslessSignedAndDurationTime) {
  lpp::PlanningRequest request{};
  FixtureRegistry registry;
  const auto context = MakeActivationContext(request, registry);

  const auto decoded =
      lpp::JsonCodec::DecodePlanningResponse(SimpleResponseJson(), context);
  ASSERT_TRUE(lpp::IsOk(decoded))
      << std::get<lpp::Error>(decoded).message;
  const auto& response = std::get<lpp::PlanningResponse>(decoded);
  EXPECT_EQ(response.response_time.tick.count(), -1);
  EXPECT_EQ(response.call_diagnostics.api_latency.value.count(), 17);

  const std::string encoded = lpp::JsonCodec::EncodePlanningResponse(response);
  const auto round_trip =
      lpp::JsonCodec::DecodePlanningResponse(encoded, context);
  ASSERT_TRUE(lpp::IsOk(round_trip));
  const auto& decoded_again = std::get<lpp::PlanningResponse>(round_trip);
  EXPECT_EQ(decoded_again.request_id, response.request_id);
  EXPECT_EQ(decoded_again.response_time.tick, response.response_time.tick);
  EXPECT_EQ(decoded_again.call_diagnostics.api_latency.value,
            response.call_diagnostics.api_latency.value);
  EXPECT_EQ(decoded_again.planning_outcome, response.planning_outcome);
  EXPECT_EQ(decoded_again.execution_directive,
            response.execution_directive);
}

TEST(JsonCodec, RejectsWrongResponseVersionUnknownKeyAndUnknownEnum) {
  lpp::PlanningRequest request{};
  FixtureRegistry registry;
  const auto context = MakeActivationContext(request, registry);

  auto wrong_version = nlohmann::json::parse(SimpleResponseJson());
  wrong_version["schema_version"] =
      "path-planner-v3-planning-response/v0";
  auto result = lpp::JsonCodec::DecodePlanningResponse(
      wrong_version.dump(), context);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kSchemaMismatch);

  auto unknown_key = nlohmann::json::parse(SimpleResponseJson());
  unknown_key["benchmark_profile_ref"] = nlohmann::json::object();
  result =
      lpp::JsonCodec::DecodePlanningResponse(unknown_key.dump(), context);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kSchemaMismatch);

  auto unknown_enum = nlohmann::json::parse(SimpleResponseJson());
  unknown_enum["execution_directive"] = "RETRY";
  result =
      lpp::JsonCodec::DecodePlanningResponse(unknown_enum.dump(), context);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kSchemaMismatch);
}

TEST(JsonCodec, WheeledActivationBundleRoundTripsAllInlineComponents) {
  FixtureRegistry registry;
  lpp::PlanningRequest request{};
  const auto context = MakeActivationContext(request, registry);
  const auto response = MakeWheeledActivationResponse();

  const std::string encoded =
      lpp::JsonCodec::EncodePlanningResponse(response);
  const auto decoded =
      lpp::JsonCodec::DecodePlanningResponse(encoded, context);
  ASSERT_TRUE(lpp::IsOk(decoded))
      << std::get<lpp::Error>(decoded).message;
  const auto& decoded_response =
      std::get<lpp::PlanningResponse>(decoded);
  ASSERT_TRUE(decoded_response.new_reference_bundle.has_value());
  const auto& decoded_bundle =
      *decoded_response.new_reference_bundle;
  EXPECT_EQ(decoded_bundle.bundle_id, "bundle-1");
  EXPECT_TRUE(std::holds_alternative<lpp::WheeledReference>(
      decoded_bundle.platform_reference));
  EXPECT_EQ(decoded_bundle.committed_prefix.content.role,
            lpp::ReferenceViewContent::Role::kCommittedPrefix);
  EXPECT_EQ(decoded_bundle.preview.content.role,
            lpp::ReferenceViewContent::Role::kPreview);
}

TEST(JsonCodec, CanonicalReferenceHashOmitsOnlyItsOwnHashField) {
  auto response = MakeWheeledActivationResponse();
  ASSERT_TRUE(response.new_reference_bundle.has_value());
  auto& reference = std::get<lpp::WheeledReference>(
      response.new_reference_bundle->platform_reference);

  const auto first = lpp::CanonicalReferenceHash(
      lpp::PlatformReference{reference});
  ASSERT_TRUE(lpp::IsOk(first));
  EXPECT_EQ(std::get<lpp::Sha256Digest>(first).size(), 64U);

  reference.reference_hash = RepeatedHash('1');
  const auto changed_self_hash = lpp::CanonicalReferenceHash(
      lpp::PlatformReference{reference});
  ASSERT_TRUE(lpp::IsOk(changed_self_hash));
  EXPECT_EQ(std::get<lpp::Sha256Digest>(changed_self_hash),
            std::get<lpp::Sha256Digest>(first));

  reference.reference_id = "wheel-reference-2";
  const auto changed_content = lpp::CanonicalReferenceHash(
      lpp::PlatformReference{reference});
  ASSERT_TRUE(lpp::IsOk(changed_content));
  EXPECT_NE(std::get<lpp::Sha256Digest>(changed_content),
            std::get<lpp::Sha256Digest>(first));
}

TEST(JsonCodec, RejectsUnknownNestedPlatformReferenceKey) {
  FixtureRegistry registry;
  lpp::PlanningRequest request{};
  const auto context = MakeActivationContext(request, registry);
  auto wire = nlohmann::json::parse(
      lpp::JsonCodec::EncodePlanningResponse(
          MakeWheeledActivationResponse()));
  wire["new_reference_bundle"]["platform_reference"]["deadline_ns"] =
      "1";

  const auto decoded =
      lpp::JsonCodec::DecodePlanningResponse(wire.dump(), context);
  ASSERT_FALSE(lpp::IsOk(decoded));
  EXPECT_EQ(std::get<lpp::Error>(decoded).code,
            lpp::ErrorCode::kSchemaMismatch);
}

TEST(JsonCodec, FrozenSchemaIdsAndVersionsMatchCodecConstants) {
  struct SchemaExpectation final {
    std::string_view relative_path;
    std::string_view id;
    std::string_view version;
  };
  constexpr std::array<SchemaExpectation, 11> expectations{{
      {"common.schema.json", "urn:lunar-path-planner:v3:common", ""},
      {"planning-request.schema.json",
       "urn:lunar-path-planner:v3:planning-request",
       "path-planner-v3-planning-request/v1"},
      {"planning-response.schema.json",
       "urn:lunar-path-planner:v3:planning-response",
       "path-planner-v3-planning-response/v1"},
      {"reference-bundle.schema.json",
       "urn:lunar-path-planner:v3:reference-bundle",
       "path-planner-v3-reference-bundle/v1"},
      {"safety-capability-profile.schema.json",
       "urn:lunar-path-planner:v3:safety-capability-profile",
       "path-planner-v3-safety-capability-profile/v1"},
      {"planner-algorithm-config.schema.json",
       "urn:lunar-path-planner:v3:planner-algorithm-config",
       "path-planner-v3-planner-algorithm-config/v1"},
      {"benchmark-profile.schema.json",
       "urn:lunar-path-planner:v3:benchmark-profile",
       "path-planner-v3-benchmark-profile/v1"},
      {"benchmark-report.schema.json",
       "urn:lunar-path-planner:v3:benchmark-report",
       "path-planner-v3-benchmark-report/v1"},
      {"references/wheeled.schema.json",
       "urn:lunar-path-planner:v3:reference:wheeled",
       "path-planner-v3-wheeled-reference/v1"},
      {"references/legged.schema.json",
       "urn:lunar-path-planner:v3:reference:legged",
       "path-planner-v3-legged-body-reference/v1"},
      {"references/hopper.schema.json",
       "urn:lunar-path-planner:v3:reference:hopper",
       "path-planner-v3-hopper-reference/v1"},
  }};

  for (const auto& expectation : expectations) {
    const auto path =
        std::filesystem::path{LPP_V3_SCHEMA_ROOT} /
        expectation.relative_path;
    std::ifstream input{path, std::ios::binary};
    ASSERT_TRUE(input.is_open()) << path.string();
    const nlohmann::json schema =
        nlohmann::json::parse(input);
    EXPECT_EQ(schema.at("$id").get<std::string>(),
              std::string{expectation.id})
        << path.string();
    if (!expectation.version.empty()) {
      EXPECT_EQ(
          schema.at("properties")
              .at("schema_version")
              .at("const")
              .get<std::string>(),
          std::string{expectation.version})
          << path.string();
    } else {
      EXPECT_FALSE(schema.contains("schema_version"))
          << path.string();
    }
  }
}
