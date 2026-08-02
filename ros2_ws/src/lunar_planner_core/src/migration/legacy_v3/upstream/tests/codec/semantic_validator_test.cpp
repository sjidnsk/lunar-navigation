#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lpp = lunar::planning::v3;
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

lpp::ContentRef Ref(std::string id, const char hash_digit = 'a') {
  return lpp::ContentRef{
      .id = std::move(id),
      .revision = 1U,
      .content_hash = std::string(64U, hash_digit),
  };
}

lpp::AxisAlignedBox3 ZeroBox() {
  return lpp::AxisAlignedBox3{
      .center = {},
      .half_extent = {},
  };
}

lpp::HopperErrorBounds ZeroHopperError() {
  return lpp::HopperErrorBounds{
      .position_bound_m = ZeroBox(),
      .orientation_bound = {.radius_rad = 0.0},
      .linear_velocity_bound_mps = ZeroBox(),
      .angular_velocity_bound_radps = ZeroBox(),
  };
}

lpp::LandingPlane HorizontalPlane() {
  return lpp::LandingPlane{
      .origin_m = {},
      .normal = {0.0, 0.0, 1.0},
      .basis_u = {1.0, 0.0, 0.0},
      .basis_v = {0.0, 1.0, 0.0},
      .residual_bound_m = 0.0,
  };
}

lpp::ConvexPolygonUv UnitSquare() {
  return lpp::ConvexPolygonUv{
      .vertices_uv =
          {
              {-1.0, -1.0},
              {1.0, -1.0},
              {1.0, 1.0},
              {-1.0, 1.0},
          },
  };
}

lpp::CircularYawInterval FullYaw() {
  return lpp::CircularYawInterval{
      .start_rad = -kPi,
      .span_rad = 2.0 * kPi,
  };
}

lpp::ConvexPolytope3 TetrahedronEnvelope() {
  return lpp::ConvexPolytope3{
      .halfspaces =
          {
              {{1.0, 0.0, 0.0}, 10.0},
              {{0.0, 1.0, 0.0}, 10.0},
              {{0.0, 0.0, 1.0}, 10.0},
              {{-0.5773502691896258,
                -0.5773502691896258,
                -0.5773502691896258},
               10.0},
          },
  };
}

lpp::HopperReference MakeLocallyValidHopperReference() {
  const lpp::DurationNanoseconds one_second{
      std::chrono::seconds{1}};
  const lpp::HopperKinematicState hold{
      .position_m = {},
      .orientation_body_to_frame = {},
      .linear_velocity_mps = {},
      .angular_velocity_radps = {},
  };
  const lpp::HopperKinematicState launch{
      .position_m = {},
      .orientation_body_to_frame = {},
      .linear_velocity_mps = {0.0, 0.0, 1.0},
      .angular_velocity_radps = {},
  };

  return lpp::HopperReference{
      .reference_id = "hopper-reference",
      .reference_hash = std::string(64U, '1'),
      .reference_time_origin = {
          .clock_id = "mission-clock",
          .tick = std::chrono::nanoseconds{0},
      },
      .ground_hold_anchor =
          {
              .anchor_id = "ground-hold",
              .hold_state = hold,
              .allowed_hold_state_error_set = ZeroHopperError(),
              .terrain_certification_ref = Ref("ground-terrain", '2'),
          },
      .next_landing_region =
          {
              .region_id = "landing-region",
              .frame_id = "map",
              .landing_plane = HorizontalPlane(),
              .convex_polygon = UnitSquare(),
              .allowed_yaw_interval = FullYaw(),
              .terrain_certification_ref = Ref("landing-terrain", '3'),
              .inward_safety_margin_m = 0.0,
          },
      .jump_boundary =
          {
              .boundary_id = "jump-boundary",
              .nominal_launch_state = launch,
              .allowed_launch_state_error_set = ZeroHopperError(),
              .gravity_model_ref = Ref("gravity", '4'),
              .ballistic_flight_time = one_second,
              .actuator_or_impulse_profile_ref = Ref("impulse", '5'),
          },
      .predicted_landing_footprint =
          {
              .landing_plane = HorizontalPlane(),
              .convex_center_landing_polygon =
                  {
                      .vertices_uv =
                          {
                              {-0.5, -0.5},
                              {0.5, -0.5},
                              {0.5, 0.5},
                              {-0.5, 0.5},
                          },
                  },
              .landing_time_window = {one_second, one_second},
              .landing_velocity_bounds =
                  {
                      .lower = {-1.0, -1.0, -2.0},
                      .upper = {1.0, 1.0, -0.1},
                  },
              .landing_yaw_interval = FullYaw(),
              .source_error_model_ref = Ref("error-model", '6'),
              .outer_approximation_margin_m = 0.0,
          },
      .certified_flight_tube =
          {
              .frame_id = "map",
              .sections =
                  {
                      {
                          .time_interval =
                              {
                                  .start_offset =
                                      {std::chrono::nanoseconds{0}},
                                  .end_offset = one_second,
                              },
                          .envelope = TetrahedronEnvelope(),
                      },
                  },
              .source_map_snapshot_ref = Ref("map", '7'),
              .body_rotation_envelope_ref =
                  Ref("body-rotation-envelope", '8'),
              .error_model_ref = Ref("error-model", '6'),
              .minimum_certified_clearance_m = 1.0,
          },
      .attitude_boundary =
          {
              .initial_orientation_error_set = {.radius_rad = 0.0},
              .initial_angular_velocity_error_set_radps = ZeroBox(),
              .target_attitude_set =
                  {
                      .nominal_orientation_body_to_frame = {},
                      .orientation_error_set = {.radius_rad = 0.0},
                      .allowed_yaw_interval = FullYaw(),
                  },
              .landing_angular_velocity_bounds_radps = ZeroBox(),
              .settle_guard = {std::chrono::nanoseconds{0}},
              .certification_ref = Ref("attitude-certificate", '9'),
          },
      .nominal_aim_point = {.position_m = {0.0, 0.0, 0.5}},
      .physical_certification_ref = Ref("physical-certificate", 'a'),
  };
}

}  // namespace

TEST(SemanticValidator, RejectsPlatformStateVariantMismatch) {
  lpp::PlanningRequest request{};
  request.platform_type = lpp::PlatformType::kWheeled;
  request.current_state = lpp::HopperState{};

  const auto report = lpp::SemanticValidator{}.Validate(request);

  EXPECT_FALSE(report.ok());
  EXPECT_TRUE(report.Contains("current_state", "platform_state_mismatch"));
}

TEST(SemanticValidator, RejectsActivationWithoutBundle) {
  const lpp::PlanningResponse response{
      .planning_outcome = lpp::PlanningOutcome::kNewReferenceReady,
      .execution_directive = lpp::ExecutionDirective::kActivateNewBundle,
  };

  const auto report = lpp::SemanticValidator{}.Validate(response);

  EXPECT_TRUE(report.Contains(
      "new_reference_bundle", "activation_requires_validated_bundle"));
}

TEST(SemanticValidator, RejectsZeroBallisticFlightTime) {
  auto hopper = MakeLocallyValidHopperReference();
  hopper.jump_boundary.ballistic_flight_time =
      lpp::DurationNanoseconds{std::chrono::nanoseconds{0}};

  const auto report =
      lpp::SemanticValidator{}.Validate(lpp::PlatformReference{hopper});

  EXPECT_TRUE(report.Contains("jump_boundary.ballistic_flight_time",
                              "positive_flight_time_required"));
}

TEST(SemanticValidator, RejectsDiscontinuousGroundHoldToLaunch) {
  auto hopper = MakeLocallyValidHopperReference();
  hopper.jump_boundary.nominal_launch_state.position_m.x = 1.0;

  const auto report =
      lpp::SemanticValidator{}.Validate(lpp::PlatformReference{hopper});

  EXPECT_TRUE(report.Contains("jump_boundary.nominal_launch_state",
                              "ground_hold_launch_discontinuity"));
}

TEST(SemanticValidator, AccumulatesIssuesInStableFieldPathOrder) {
  auto hopper = MakeLocallyValidHopperReference();
  hopper.jump_boundary.ballistic_flight_time =
      lpp::DurationNanoseconds{std::chrono::nanoseconds{0}};
  hopper.next_landing_region.landing_plane.normal.x =
      std::numeric_limits<double>::quiet_NaN();
  hopper.predicted_landing_footprint.landing_velocity_bounds.lower.x = 2.0;
  hopper.predicted_landing_footprint.landing_velocity_bounds.upper.x = 1.0;

  const auto first =
      lpp::SemanticValidator{}.Validate(lpp::PlatformReference{hopper});
  const auto second =
      lpp::SemanticValidator{}.Validate(lpp::PlatformReference{hopper});

  ASSERT_GT(first.issues.size(), 2U);
  EXPECT_TRUE(std::is_sorted(
      first.issues.begin(), first.issues.end(),
      [](const lpp::ValidationIssue& lhs,
         const lpp::ValidationIssue& rhs) {
        return std::tie(lhs.field_path, lhs.reason_code, lhs.message) <
               std::tie(rhs.field_path, rhs.reason_code, rhs.message);
      }));
  EXPECT_EQ(first.issues.size(), second.issues.size());
  for (std::size_t index = 0; index < first.issues.size(); ++index) {
    EXPECT_EQ(first.issues[index].field_path,
              second.issues[index].field_path);
    EXPECT_EQ(first.issues[index].reason_code,
              second.issues[index].reason_code);
  }
}

namespace {

std::shared_ptr<const lpp::ImmutableMapSnapshot> MakeMapSnapshot(
    const lpp::ContentRef& snapshot_ref,
    std::string frame_id = "map",
    std::string clock_id = "mission-clock",
    const std::chrono::nanoseconds source_tick =
        std::chrono::nanoseconds{0},
    const bool unknown_static_speed_hint = false,
    const bool alternate_known_identity = false) {
  lpp::MapSnapshotInput input{
      .snapshot_ref = snapshot_ref,
      .map_revision = snapshot_ref.revision,
      .immutable_data_handle = "map-handle",
      .source_time =
          {
              .clock_id = std::move(clock_id),
              .tick = source_tick,
          },
      .bounds =
          {
              .minimum_m = {0.0, 0.0, -1.0},
              .maximum_m = {2.0, 1.0, 1.0},
          },
      .geometry =
          {
              .width = 2U,
              .height = 1U,
              .resolution_m = 1.0,
              .origin_m = {0.0, 0.0},
              .frame_id = std::move(frame_id),
          },
      .layer_manifest =
          {
              {lpp::LayerKind::kKnownMask,
               Ref(alternate_known_identity ? "known-alt" : "known",
                   '1')},
              {lpp::LayerKind::kElevation, Ref("elevation", '2')},
              {lpp::LayerKind::kTerrainNormal, Ref("normal", '3')},
              {lpp::LayerKind::kRoughness, Ref("roughness", '4')},
              {lpp::LayerKind::kHardObstacle, Ref("obstacle", '5')},
              {lpp::LayerKind::kConfidence, Ref("confidence", '6')},
          },
      .known_mask = {1U, 0U},
      .elevation_m = {0.0F, 0.0F},
      .normal_x = {0.0F, 0.0F},
      .normal_y = {0.0F, 0.0F},
      .normal_z = {1.0F, 1.0F},
      .roughness_m = {0.0F, 0.0F},
      .hard_obstacle_mask = {0U, 0U},
      .confidence = {1.0F, 0.0F},
  };
  if (unknown_static_speed_hint) {
    input.layer_manifest.push_back(
        {lpp::LayerKind::kStaticSpeedLimit,
         Ref("static-speed-limit", '7')});
    input.static_speed_limit_mps = {1.0F, 1.0F};
  }
  auto result = lpp::ImmutableMapSnapshot::Create(input);
  if (!lpp::IsOk(result)) {
    return nullptr;
  }
  return std::get<
      std::shared_ptr<const lpp::ImmutableMapSnapshot>>(
      std::move(result));
}

lpp::ReferenceBundle MakeHopperBundle() {
  lpp::HopperReference hopper = MakeLocallyValidHopperReference();
  const lpp::ContentRef map_ref =
      hopper.certified_flight_tube.source_map_snapshot_ref;
  const lpp::ContentRef capability_ref = Ref("capability", 'b');
  const lpp::ContentRef config_ref = Ref("algorithm-config", 'c');

  return lpp::ReferenceBundle{
      .bundle_id = "bundle",
      .bundle_revision = 1U,
      .bundle_hash = std::string(64U, 'd'),
      .source_request_id = "request",
      .source_map_snapshot_ref = map_ref,
      .source_safety_capability_ref = capability_ref,
      .source_algorithm_config_ref = config_ref,
      .platform_type = lpp::PlatformType::kHopper,
      .platform_reference = hopper,
      .route_skeleton =
          {
              .component_id = "route-skeleton",
              .component_hash = std::string(64U, 'e'),
              .content =
                  {
                      .source_reference_id = hopper.reference_id,
                      .source_reference_hash = hopper.reference_hash,
                      .waypoints = {},
                      .unresolved_tail = {},
                  },
          },
      .committed_prefix =
          {
              .component_id = "committed-prefix",
              .component_hash = std::string(64U, 'f'),
              .content =
                  {
                      .role =
                          lpp::ReferenceViewContent::Role::
                              kCommittedPrefix,
                      .source_reference_id = hopper.reference_id,
                      .source_reference_hash = hopper.reference_hash,
                      .selector = lpp::GroundHoldViewSelector{
                          hopper.ground_hold_anchor.anchor_id},
                  },
          },
      .preview =
          {
              .component_id = "preview",
              .component_hash = std::string(64U, '0'),
              .content =
                  {
                      .role =
                          lpp::ReferenceViewContent::Role::kPreview,
                      .source_reference_id = hopper.reference_id,
                      .source_reference_hash = hopper.reference_hash,
                      .selector = lpp::JumpViewSelector{
                          .boundary_id =
                              hopper.jump_boundary.boundary_id,
                          .scope =
                              lpp::JumpViewSelector::Scope::kNextHop,
                      },
                  },
          },
      .validity =
          {
              .valid_from = hopper.reference_time_origin,
              .required_map_snapshot_ref = map_ref,
              .required_capability_ref = capability_ref,
              .allowed_state_deviation = ZeroHopperError(),
              .invalidation_conditions =
                  {
                      lpp::InvalidationCondition::
                          kMapSafetyRevisionChanged,
                      lpp::InvalidationCondition::
                          kStateDeviationExceeded,
                      lpp::InvalidationCondition::
                          kCapabilityRevisionChanged,
                      lpp::InvalidationCondition::
                          kReferenceHorizonExhausted,
                  },
          },
      .validation_summary =
          {
              .hard_constraints_passed = true,
              .continuous_validation_passed = true,
              .certificate_refs = {Ref("validation-certificate", 'd')},
              .warning_codes = {},
          },
      .generation_evidence =
          {
              .selected_candidate_id = "candidate",
              .generation_mode =
                  lpp::GenerationMode::kCertifiedBallisticReference,
              .termination_reason = "TARGET_REACHED",
              .evidence_refs = {Ref("candidate-evidence", 'e')},
          },
  };
}

lpp::HopperCapability MakeHopperCapability(
    const lpp::HopperReference& reference) {
  return lpp::HopperCapability{
      .frame_id = "map",
      .collision_envelope =
          {
              .body_frame_halfspaces = TetrahedronEnvelope(),
          },
      .motion_model_ref = Ref("motion-model", '1'),
      .analytic_cost_model_ref = Ref("analytic-cost", '2'),
      .gravity_model_ref = reference.jump_boundary.gravity_model_ref,
      .landing_terrain_thresholds =
          {
              .maximum_slope_rad = 0.5,
              .maximum_roughness_m = 1.0,
              .maximum_plane_residual_m = 1.0,
              .minimum_overhead_clearance_m = 0.0,
              .minimum_lateral_clearance_m = 0.0,
              .minimum_landing_region_area_m2 = 0.1,
          },
      .launch_limits =
          {
              .maximum_launch_speed_mps = 10.0,
              .maximum_launch_impulse_newton_seconds = 100.0,
              .minimum_flight_time =
                  {std::chrono::milliseconds{100}},
              .maximum_flight_time = {std::chrono::seconds{2}},
              .maximum_landing_speed_mps = 10.0,
              .minimum_downward_impact_speed_mps = 0.0,
              .minimum_landing_clearance_m = 0.0,
          },
      .attitude_envelope =
          {
              .maximum_angular_speed_radps = 10.0,
              .maximum_angular_acceleration_radps2 = 10.0,
              .maximum_initial_angular_speed_radps = 10.0,
              .minimum_settle_guard = {std::chrono::nanoseconds{0}},
          },
      .certified_state_error_bounds = ZeroHopperError(),
  };
}

lpp::PlanningRequest MakeActivationRequest(
    const lpp::ReferenceBundle& bundle) {
  const auto& hopper =
      std::get<lpp::HopperReference>(bundle.platform_reference);
  auto capability = std::make_shared<lpp::SafetyCapabilityProfile>();
  capability->content_ref = bundle.source_safety_capability_ref;
  capability->content = MakeHopperCapability(hopper);
  auto config = std::make_shared<lpp::PlannerAlgorithmConfig>();
  config->content_ref = bundle.source_algorithm_config_ref;

  lpp::PlanningRequest request{
      .request_id = bundle.source_request_id,
      .request_time = hopper.reference_time_origin,
      .state_time = hopper.reference_time_origin,
      .frame_id = "map",
      .platform_type = lpp::PlatformType::kHopper,
      .current_state =
          lpp::HopperState{
              .position_m =
                  hopper.ground_hold_anchor.hold_state.position_m,
              .orientation_body_to_frame =
                  hopper.ground_hold_anchor.hold_state
                      .orientation_body_to_frame,
              .linear_velocity_mps = {},
              .angular_velocity_radps = {},
              .error_bounds = ZeroHopperError(),
          },
      .safety_capability = std::move(capability),
      .algorithm_config = std::move(config),
  };
  const auto& hopper_capability = std::get<lpp::HopperCapability>(
      request.safety_capability->content);
  request.capability_bindings.motion_model.content_ref =
      hopper_capability.motion_model_ref;
  request.capability_bindings.analytic_cost_model.content_ref =
      hopper_capability.analytic_cost_model_ref;
  request.capability_bindings.gravity_model =
      lpp::ResolvedBinding<lpp::GravityModel>{
          .content_ref = hopper.jump_boundary.gravity_model_ref};
  request.capability_bindings.error_model =
      lpp::ResolvedBinding<lpp::DeterministicErrorModel>{
          .content_ref =
              hopper.predicted_landing_footprint.source_error_model_ref};
  request.capability_bindings.actuator_or_impulse_profile =
      lpp::ResolvedBinding<lpp::ActuatorOrImpulseProfile>{
          .content_ref =
              hopper.jump_boundary.actuator_or_impulse_profile_ref};
  request.capability_bindings.body_rotation_envelope =
      lpp::ResolvedBinding<lpp::BodyRotationEnvelope>{
          .content_ref =
              hopper.certified_flight_tube.body_rotation_envelope_ref};
  return request;
}

class TestContractObject final : public lpp::ImmutableContractObject {
 public:
  TestContractObject(lpp::ContentRef ref,
                     const lpp::ContractObjectKind kind)
      : ref_(std::move(ref)), kind_(kind) {}

  [[nodiscard]] lpp::ContentRef content_ref() const override {
    return ref_;
  }

  [[nodiscard]] lpp::ContractObjectKind kind() const override {
    return kind_;
  }

 private:
  lpp::ContentRef ref_;
  lpp::ContractObjectKind kind_;
};

class TestCertificate final : public lpp::CertificationObject {
 public:
  TestCertificate(lpp::ContentRef ref,
                  lpp::CertificationProvenance provenance)
      : ref_(std::move(ref)), provenance_(std::move(provenance)) {}

  [[nodiscard]] lpp::ContentRef content_ref() const override {
    return ref_;
  }

  [[nodiscard]] lpp::ContractObjectKind kind() const override {
    return lpp::ContractObjectKind::kCertification;
  }

  [[nodiscard]] const lpp::CertificationProvenance& provenance()
      const override {
    return provenance_;
  }

 private:
  lpp::ContentRef ref_;
  lpp::CertificationProvenance provenance_;
};

class TestRegistry final : public lpp::ContractObjectRegistry {
 public:
  std::shared_ptr<const lpp::ImmutableMapSnapshot> map_snapshot;
  std::shared_ptr<const lpp::SafetyCapabilityProfile> capability;
  std::shared_ptr<const lpp::PlannerAlgorithmConfig> config;
  lpp::ResolvedCapabilityBindings bindings;
  std::vector<std::shared_ptr<const lpp::ImmutableContractObject>>
      objects;

  [[nodiscard]] std::shared_ptr<const lpp::ImmutableMapSnapshot>
  FindMapSnapshot(const lpp::ContentRef& ref,
                  const std::string_view handle) const override {
    return map_snapshot && map_snapshot->snapshot_ref() == ref &&
                   map_snapshot->immutable_data_handle() == handle
               ? map_snapshot
               : nullptr;
  }

  [[nodiscard]]
  std::shared_ptr<const lpp::SafetyCapabilityProfile>
  FindSafetyCapability(const lpp::ContentRef& ref) const override {
    return capability && capability->content_ref == ref ? capability
                                                        : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const lpp::PlannerAlgorithmConfig>
  FindAlgorithmConfig(const lpp::ContentRef& ref) const override {
    return config && config->content_ref == ref ? config : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const lpp::LearnedCostSnapshot>
  FindLearnedCost(const lpp::ContentRef&,
                  std::string_view) const override {
    return nullptr;
  }

  [[nodiscard]] lpp::Result<lpp::ResolvedCapabilityBindings>
  ResolveCapabilityBindings(
      const lpp::SafetyCapabilityProfile&) const override {
    return bindings;
  }

  [[nodiscard]] lpp::Result<
      std::shared_ptr<const lpp::ImmutableContractObject>>
  Resolve(const lpp::ContentRef& ref,
          const lpp::ContractObjectKind expected_kind) const override {
    const auto found = std::find_if(
        objects.begin(), objects.end(),
        [&](const auto& object) {
          return object && object->content_ref() == ref &&
                 object->kind() == expected_kind;
        });
    if (found == objects.end()) {
      return lpp::Error{
          lpp::ErrorCode::kMissingRegistryObject,
          "content_ref",
          "test registry object missing",
      };
    }
    return *found;
  }
};

std::vector<lpp::ContentRef> FixedHopperInputs(
    const lpp::PlanningRequest& request) {
  return {
      request.capability_bindings.gravity_model->content_ref,
      request.capability_bindings.error_model->content_ref,
      request.capability_bindings.actuator_or_impulse_profile->content_ref,
      request.capability_bindings.body_rotation_envelope->content_ref,
  };
}

void PopulateActivationRegistry(
    const lpp::ReferenceBundle& bundle,
    const lpp::PlanningRequest& request,
    TestRegistry& registry,
    const bool wrong_validation_map) {
  registry.capability = request.safety_capability;
  registry.config = request.algorithm_config;
  registry.bindings = request.capability_bindings;
  const auto& hopper =
      std::get<lpp::HopperReference>(bundle.platform_reference);
  registry.objects.push_back(std::make_shared<TestContractObject>(
      hopper.jump_boundary.gravity_model_ref,
      lpp::ContractObjectKind::kGravityModel));
  registry.objects.push_back(std::make_shared<TestContractObject>(
      hopper.jump_boundary.actuator_or_impulse_profile_ref,
      lpp::ContractObjectKind::kActuatorOrImpulseProfile));
  registry.objects.push_back(std::make_shared<TestContractObject>(
      hopper.predicted_landing_footprint.source_error_model_ref,
      lpp::ContractObjectKind::kDeterministicErrorModel));
  registry.objects.push_back(std::make_shared<TestContractObject>(
      hopper.certified_flight_tube.body_rotation_envelope_ref,
      lpp::ContractObjectKind::kBodyRotationEnvelope));

  const std::vector<lpp::ContentRef> fixed_inputs =
      FixedHopperInputs(request);
  const auto add_certificate =
      [&](const lpp::ContentRef& ref, const bool wrong_map) {
        registry.objects.push_back(std::make_shared<TestCertificate>(
            ref,
            lpp::CertificationProvenance{
                .source_map_snapshot_ref =
                    wrong_map ? Ref("wrong-map", 'f')
                              : bundle.source_map_snapshot_ref,
                .source_safety_capability_ref =
                    bundle.source_safety_capability_ref,
                .source_algorithm_config_ref =
                    bundle.source_algorithm_config_ref,
                .input_refs = fixed_inputs,
                .certification_purpose = "TEST_CERTIFICATION",
            }));
      };
  add_certificate(
      hopper.ground_hold_anchor.terrain_certification_ref, false);
  add_certificate(
      hopper.next_landing_region.terrain_certification_ref, false);
  add_certificate(hopper.attitude_boundary.certification_ref, false);
  add_certificate(hopper.physical_certification_ref, false);
  add_certificate(bundle.validation_summary.certificate_refs.front(),
                  wrong_validation_map);
}

}  // namespace

TEST(SemanticValidator, RejectsBundleProvenanceMismatch) {
  auto bundle = MakeHopperBundle();
  bundle.validity.required_map_snapshot_ref = Ref("other-map", 'f');

  const auto report = lpp::SemanticValidator{}.Validate(bundle);

  EXPECT_TRUE(report.Contains("validity.required_map_snapshot_ref",
                              "bundle_map_provenance_mismatch"));
}

TEST(SemanticValidator, RejectsFootprintOutsideLandingRegion) {
  auto hopper = MakeLocallyValidHopperReference();
  hopper.predicted_landing_footprint
      .convex_center_landing_polygon.vertices_uv[2].x = 2.0;

  const auto report =
      lpp::SemanticValidator{}.Validate(lpp::PlatformReference{hopper});

  EXPECT_TRUE(report.Contains(
      "predicted_landing_footprint.convex_center_landing_polygon",
      "landing_footprint_not_contained"));
}

TEST(SemanticValidator, RejectsYawSubsetViolation) {
  auto hopper = MakeLocallyValidHopperReference();
  hopper.predicted_landing_footprint.landing_yaw_interval = {
      .start_rad = -1.0,
      .span_rad = 2.0,
  };
  hopper.attitude_boundary.target_attitude_set.allowed_yaw_interval = {
      .start_rad = -0.5,
      .span_rad = 1.0,
  };

  const auto report =
      lpp::SemanticValidator{}.Validate(lpp::PlatformReference{hopper});

  EXPECT_TRUE(report.Contains(
      "predicted_landing_footprint.landing_yaw_interval",
      "landing_yaw_not_subset_of_target_attitude"));
}

TEST(SemanticValidator, RejectsNonCanonicalQuaternionAndPlane) {
  auto hopper = MakeLocallyValidHopperReference();
  hopper.jump_boundary.nominal_launch_state.orientation_body_to_frame =
      {-1.0, 0.0, 0.0, 0.0};
  hopper.next_landing_region.landing_plane.basis_v =
      {1.0, 0.0, 0.0};

  const auto report =
      lpp::SemanticValidator{}.Validate(lpp::PlatformReference{hopper});

  EXPECT_TRUE(report.Contains(
      "jump_boundary.nominal_launch_state.orientation_body_to_frame",
      "canonical_quaternion_sign_required"));
  EXPECT_TRUE(report.Contains("next_landing_region.landing_plane",
                              "orthogonal_plane_basis_required"));
}

TEST(SemanticValidator, RejectsDirectiveAndLearnedUsageContradictions) {
  lpp::PlanningResponse response{};
  response.request_id = "request";
  response.response_time = {"mission-clock", std::chrono::nanoseconds{0}};
  response.planning_outcome = lpp::PlanningOutcome::kNoKnownSafeRoute;
  response.execution_directive =
      lpp::ExecutionDirective::kContinueActiveBundle;
  response.reason_code = "NO_ROUTE";
  response.call_diagnostics.learned_cost_usage =
      lpp::LearnedCostUsage::kUsedBoundedSoftCost;

  const auto report = lpp::SemanticValidator{}.Validate(response);

  EXPECT_TRUE(report.Contains(
      "active_bundle_ref", "continue_requires_active_bundle_ref"));
  EXPECT_TRUE(report.Contains(
      "call_diagnostics.learned_cost_snapshot_ref",
      "learned_cost_snapshot_ref_required"));
}

TEST(SemanticValidator, RejectsCertificateProvenanceMismatch) {
  const lpp::ReferenceBundle bundle = MakeHopperBundle();
  const lpp::PlanningRequest request = MakeActivationRequest(bundle);
  TestRegistry registry;
  PopulateActivationRegistry(bundle, request, registry, true);
  const lpp::ReferenceActivationContext context{request, registry};

  const auto report =
      lpp::SemanticValidator{}.ValidateForActivation(bundle, context);

  EXPECT_TRUE(report.Contains(
      "validation_summary.certificate_refs[0].provenance."
      "source_map_snapshot_ref",
      "certificate_map_provenance_mismatch"));
}

TEST(SemanticValidator, RejectsRequestStateOutsideGroundHold) {
  const lpp::ReferenceBundle bundle = MakeHopperBundle();
  lpp::PlanningRequest request = MakeActivationRequest(bundle);
  std::get<lpp::HopperState>(request.current_state).position_m.x = 1.0;
  TestRegistry registry;
  PopulateActivationRegistry(bundle, request, registry, false);
  const lpp::ReferenceActivationContext context{request, registry};

  const auto report =
      lpp::SemanticValidator{}.ValidateForActivation(bundle, context);

  EXPECT_TRUE(report.Contains(
      "current_state", "request_state_not_contained_in_ground_hold"));
}

TEST(SemanticValidator, RejectsErrorExpandedCapabilityLimitViolation) {
  const lpp::ReferenceBundle bundle = MakeHopperBundle();
  lpp::PlanningRequest request = MakeActivationRequest(bundle);
  auto mutable_profile =
      std::make_shared<lpp::SafetyCapabilityProfile>(
          *request.safety_capability);
  auto& capability =
      std::get<lpp::HopperCapability>(mutable_profile->content);
  capability.launch_limits.maximum_launch_speed_mps = 0.5;
  request.safety_capability = std::move(mutable_profile);
  TestRegistry registry;
  PopulateActivationRegistry(bundle, request, registry, false);
  const lpp::ReferenceActivationContext context{request, registry};

  const auto report =
      lpp::SemanticValidator{}.ValidateForActivation(bundle, context);

  EXPECT_TRUE(report.Contains(
      "jump_boundary.nominal_launch_state.linear_velocity_mps",
      "launch_speed_limit_exceeded"));
}

TEST(SemanticValidator, RejectsInlineContentHashMismatch) {
  const lpp::ReferenceBundle bundle = MakeHopperBundle();

  const auto report = lpp::SemanticValidator{}.Validate(bundle);

  EXPECT_TRUE(report.Contains("route_skeleton.component_hash",
                              "component_hash_content_mismatch"));
  EXPECT_TRUE(report.Contains("platform_reference.reference_hash",
                              "reference_hash_content_mismatch"));
  EXPECT_TRUE(report.Contains("bundle_hash",
                              "bundle_hash_content_mismatch"));
}

TEST(SemanticValidator, RejectsNullMapAndCapabilityPointers) {
  lpp::PlanningRequest request{};
  request.platform_type = lpp::PlatformType::kWheeled;
  request.current_state = lpp::WheeledOrLeggedState{};

  const auto report = lpp::SemanticValidator{}.Validate(request);

  EXPECT_TRUE(report.Contains("map_snapshot", "missing_registry_object"));
  EXPECT_TRUE(
      report.Contains("safety_capability", "missing_registry_object"));
}

TEST(SemanticValidator, RejectsRequestAndMapFrameMismatch) {
  const lpp::ReferenceBundle bundle = MakeHopperBundle();
  lpp::PlanningRequest request = MakeActivationRequest(bundle);
  request.map_snapshot = MakeMapSnapshot(
      bundle.source_map_snapshot_ref, "other-map-frame");
  ASSERT_NE(request.map_snapshot, nullptr);

  const auto report = lpp::SemanticValidator{}.Validate(request);

  EXPECT_TRUE(
      report.Contains("map_snapshot.frame_id", "map_frame_mismatch"));
}

TEST(SemanticValidator, RejectsMapSourceClockMismatchAndSkew) {
  const lpp::ReferenceBundle bundle = MakeHopperBundle();
  {
    lpp::PlanningRequest request = MakeActivationRequest(bundle);
    request.map_snapshot = MakeMapSnapshot(
        bundle.source_map_snapshot_ref, "map", "other-clock");
    ASSERT_NE(request.map_snapshot, nullptr);

    const auto report = lpp::SemanticValidator{}.Validate(request);
    EXPECT_TRUE(report.Contains("map_snapshot.source_time.clock_id",
                                "clock_id_mismatch"));
  }
  {
    lpp::PlanningRequest request = MakeActivationRequest(bundle);
    request.map_snapshot = MakeMapSnapshot(
        bundle.source_map_snapshot_ref, "map", "mission-clock",
        std::chrono::nanoseconds{100});
    ASSERT_NE(request.map_snapshot, nullptr);
    auto mutable_config =
        std::make_shared<lpp::PlannerAlgorithmConfig>(
            *request.algorithm_config);
    mutable_config->max_input_skew =
        {std::chrono::nanoseconds{10}};
    request.algorithm_config = std::move(mutable_config);

    const auto report = lpp::SemanticValidator{}.Validate(request);
    EXPECT_TRUE(report.Contains("map_snapshot.source_time",
                                "input_time_skew_exceeded"));
  }
}

TEST(SemanticValidator, RejectsUnknownCellStaticFeasibilityHint) {
  const lpp::ReferenceBundle bundle = MakeHopperBundle();
  lpp::PlanningRequest request = MakeActivationRequest(bundle);
  request.map_snapshot = MakeMapSnapshot(
      bundle.source_map_snapshot_ref, "map", "mission-clock",
      std::chrono::nanoseconds{0}, true);
  ASSERT_NE(request.map_snapshot, nullptr);

  const auto report = lpp::SemanticValidator{}.Validate(request);

  EXPECT_TRUE(report.Contains(
      "map_snapshot.static_speed_limit_mps[1]",
      "unknown_cell_has_hard_feasible_hint"));
}

TEST(SemanticValidator, RejectsRegistryMapLayerIdentityConflict) {
  const lpp::ReferenceBundle bundle = MakeHopperBundle();
  lpp::PlanningRequest request = MakeActivationRequest(bundle);
  request.map_snapshot =
      MakeMapSnapshot(bundle.source_map_snapshot_ref);
  ASSERT_NE(request.map_snapshot, nullptr);
  TestRegistry registry;
  PopulateActivationRegistry(bundle, request, registry, false);
  registry.map_snapshot = MakeMapSnapshot(
      bundle.source_map_snapshot_ref, "map", "mission-clock",
      std::chrono::nanoseconds{0}, false, true);
  ASSERT_NE(registry.map_snapshot, nullptr);
  const lpp::ReferenceActivationContext context{request, registry};

  const auto report =
      lpp::SemanticValidator{}.ValidateForActivation(bundle, context);

  EXPECT_TRUE(report.Contains(
      "map_snapshot.layer_manifest.KNOWN_MASK",
      "registry_map_layer_ref_mismatch"));
  EXPECT_TRUE(report.Contains(
      "map_snapshot.layer_manifest",
      "registry_map_manifest_mismatch"));
}
