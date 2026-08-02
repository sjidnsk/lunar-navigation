#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/contracts/platform_reference.hpp"

namespace lunar::planning::v3 {

template <class Content>
struct InlineComponent final {
  ComponentId component_id;
  Sha256Digest component_hash;
  Content content;
};

struct RouteSkeletonContent final {
  ReferenceId source_reference_id;
  Sha256Digest source_reference_hash;
  std::vector<PoseXyzYaw> waypoints;
  std::vector<Vec3> unresolved_tail;
};

struct TimeViewSelector final {
  TimeInterval time_interval;
};

struct SegmentViewSelector final {
  std::size_t first_segment_index{};
  std::size_t past_last_segment_index{};
};

struct GroundHoldViewSelector final {
  Identifier anchor_id;
};

struct JumpViewSelector final {
  enum class Scope {
    kNextHop,
    kFutureMissionPreview,
  };

  JumpBoundaryId boundary_id;
  Scope scope{Scope::kNextHop};
};

using ReferenceViewSelector =
    std::variant<TimeViewSelector,
                 SegmentViewSelector,
                 GroundHoldViewSelector,
                 JumpViewSelector>;

struct ReferenceViewContent final {
  enum class Role {
    kCommittedPrefix,
    kPreview,
  };

  Role role{};
  ReferenceId source_reference_id;
  Sha256Digest source_reference_hash;
  ReferenceViewSelector selector;
};

enum class InvalidationCondition {
  kMapSafetyRevisionChanged,
  kStateDeviationExceeded,
  kCapabilityRevisionChanged,
  kReferenceHorizonExhausted,
};

struct ReferenceValidity final {
  ClockStamp valid_from;
  std::optional<ClockStamp> valid_until;
  ContentRef required_map_snapshot_ref;
  ContentRef required_capability_ref;
  std::variant<WheeledOrLeggedErrorBounds, HopperErrorBounds>
      allowed_state_deviation;
  std::vector<InvalidationCondition> invalidation_conditions;
};

struct ValidationSummary final {
  bool hard_constraints_passed{true};
  bool continuous_validation_passed{true};
  std::vector<ContentRef> certificate_refs;
  std::vector<std::string> warning_codes;
};

enum class GenerationMode {
  kSmoothedSplineReference,
  kValidatedPrimitiveChainReference,
  kCertifiedBallisticReference,
};

struct GenerationEvidence final {
  std::string selected_candidate_id;
  GenerationMode generation_mode{};
  std::string termination_reason;
  std::vector<ContentRef> evidence_refs;
  std::optional<ContentRef> learned_cost_snapshot_ref;
};

struct ReferenceBundle final {
  BundleId bundle_id;
  std::uint32_t bundle_revision{};
  Sha256Digest bundle_hash;
  std::optional<BundleId> supersedes_bundle_id;
  RequestId source_request_id;
  ContentRef source_map_snapshot_ref;
  ContentRef source_safety_capability_ref;
  ContentRef source_algorithm_config_ref;
  PlatformType platform_type{};
  PlatformReference platform_reference;
  InlineComponent<RouteSkeletonContent> route_skeleton;
  InlineComponent<ReferenceViewContent> committed_prefix;
  InlineComponent<ReferenceViewContent> preview;
  ReferenceValidity validity;
  ValidationSummary validation_summary;
  GenerationEvidence generation_evidence;
};

}  // namespace lunar::planning::v3
