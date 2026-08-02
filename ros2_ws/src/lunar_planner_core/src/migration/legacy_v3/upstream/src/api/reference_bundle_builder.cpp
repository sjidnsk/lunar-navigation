#include "lunar_path_planner/v3/api/reference_bundle_builder.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <exception>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

namespace lunar::planning::v3 {
namespace {

struct ReferenceIdentity final {
  PlatformType platform_type{};
  std::string_view reference_id;
  std::string_view reference_hash;
};

[[nodiscard]] ReferenceIdentity IdentityOf(
    const PlatformReference& reference) {
  return std::visit(
      [](const auto& concrete) {
        using ReferenceT = std::decay_t<decltype(concrete)>;
        PlatformType platform_type{};
        if constexpr (std::is_same_v<ReferenceT, WheeledReference>) {
          platform_type = PlatformType::kWheeled;
        } else if constexpr (
            std::is_same_v<ReferenceT, LeggedBodyReference>) {
          platform_type = PlatformType::kLegged;
        } else {
          platform_type = PlatformType::kHopper;
        }
        return ReferenceIdentity{
            platform_type,
            concrete.reference_id,
            concrete.reference_hash,
        };
      },
      reference);
}

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return Error{
      ErrorCode::kInvalidArgument,
      std::move(field_path),
      std::move(message),
  };
}

[[nodiscard]] Error ValidationError(
    const ValidationReport& report) {
  if (report.issues.empty()) {
    return Invalid("reference_bundle", "validation failed");
  }
  const ValidationIssue& issue = report.issues.front();
  ErrorCode code = ErrorCode::kInvalidArgument;
  if (issue.reason_code.find("missing_registry_object") !=
      std::string::npos) {
    code = ErrorCode::kMissingRegistryObject;
  } else if (issue.reason_code.find("stale") != std::string::npos) {
    code = ErrorCode::kStaleInput;
  } else if (issue.reason_code.find("resource") !=
             std::string::npos) {
    code = ErrorCode::kResourceLimit;
  } else if (issue.reason_code.find("numerical") !=
             std::string::npos) {
    code = ErrorCode::kNumericalFailure;
  }
  return Error{
      code,
      issue.field_path,
      issue.reason_code + ": " + issue.message,
  };
}

[[nodiscard]] Result<Sha256Digest> CheckedReferenceHash(
    const PlatformReference& reference) {
  const auto digest = CanonicalReferenceHash(reference);
  if (!IsOk(digest)) {
    return std::get<Error>(digest);
  }
  if (std::get<Sha256Digest>(digest) !=
      IdentityOf(reference).reference_hash) {
    return Invalid(
        "platform_reference.reference_hash",
        "reference hash does not bind the canonical platform payload");
  }
  return std::get<Sha256Digest>(digest);
}

[[nodiscard]] Result<std::array<Sha256Digest, 3U>>
ComputeComponentHashes(const BundleBuildRequest& request) {
  const auto route_hash =
      CanonicalComponentHash(request.route_skeleton.content);
  if (!IsOk(route_hash)) {
    return std::get<Error>(route_hash);
  }
  const auto committed_hash =
      CanonicalComponentHash(request.committed_prefix.content);
  if (!IsOk(committed_hash)) {
    return std::get<Error>(committed_hash);
  }
  const auto preview_hash =
      CanonicalComponentHash(request.preview.content);
  if (!IsOk(preview_hash)) {
    return std::get<Error>(preview_hash);
  }
  return std::array<Sha256Digest, 3U>{
      std::get<Sha256Digest>(route_hash),
      std::get<Sha256Digest>(committed_hash),
      std::get<Sha256Digest>(preview_hash),
  };
}

[[nodiscard]] Result<bool> CheckComponentBinding(
    const BundleBuildRequest& request,
    const ReferenceIdentity& identity) {
  if (request.route_skeleton.content.source_reference_id !=
          identity.reference_id ||
      request.route_skeleton.content.source_reference_hash !=
          identity.reference_hash) {
    return Invalid(
        "route_skeleton.content",
        "route skeleton must bind the exact platform reference");
  }
  const auto check_view =
      [&](const ComponentBuildInput<ReferenceViewContent>& component,
          const ReferenceViewContent::Role role,
          std::string_view path) -> Result<bool> {
    if (component.content.role != role ||
        component.content.source_reference_id !=
            identity.reference_id ||
        component.content.source_reference_hash !=
            identity.reference_hash) {
      return Invalid(
          std::string{path},
          "reference view role and payload identity must match its slot");
    }
    return true;
  };
  const auto committed = check_view(
      request.committed_prefix,
      ReferenceViewContent::Role::kCommittedPrefix,
      "committed_prefix.content");
  if (!IsOk(committed)) {
    return std::get<Error>(committed);
  }
  const auto preview = check_view(
      request.preview,
      ReferenceViewContent::Role::kPreview,
      "preview.content");
  if (!IsOk(preview)) {
    return std::get<Error>(preview);
  }

  if (identity.platform_type == PlatformType::kHopper) {
    const HopperReference& hopper =
        std::get<HopperReference>(request.platform_reference);
    const auto* hold = std::get_if<GroundHoldViewSelector>(
        &request.committed_prefix.content.selector);
    if (hold == nullptr ||
        hold->anchor_id != hopper.ground_hold_anchor.anchor_id) {
      return Invalid(
          "committed_prefix.content.selector",
          "a Hopper bundle must commit the same ground-hold anchor");
    }
    const auto* jump = std::get_if<JumpViewSelector>(
        &request.preview.content.selector);
    if (jump == nullptr ||
        jump->boundary_id != hopper.jump_boundary.boundary_id ||
        jump->scope != JumpViewSelector::Scope::kNextHop) {
      return Invalid(
          "preview.content.selector",
          "a Hopper bundle must preview the same next-hop boundary");
    }
  }
  return true;
}

[[nodiscard]] Result<bool> CheckRevisionAndSupersession(
    const BundleBuildRequest& request) {
  if (!request.active_bundle.has_value()) {
    if (request.supersedes_bundle_id.has_value()) {
      return Invalid(
          "supersedes_bundle_id",
          "a bundle cannot supersede an unspecified active bundle");
    }
    return true;
  }

  const ReferenceBundle& active = *request.active_bundle;
  if (request.bundle_id == active.bundle_id) {
    if (request.bundle_revision <= active.bundle_revision) {
      return Invalid(
          "bundle_revision",
          "same-ID bundle revisions must increase strictly");
    }
    if (request.supersedes_bundle_id.has_value()) {
      return Invalid(
          "supersedes_bundle_id",
          "same-ID revision does not use supersedes_bundle_id");
    }
  } else if (!request.supersedes_bundle_id.has_value() ||
             *request.supersedes_bundle_id != active.bundle_id) {
    return Invalid(
        "supersedes_bundle_id",
        "replacement bundle must supersede the exact active bundle ID");
  }
  return true;
}

enum class ComponentSlot : std::size_t {
  kRoute = 0U,
  kCommitted = 1U,
  kPreview = 2U,
};

struct ComponentIdentity final {
  ComponentId id;
  Sha256Digest stored_hash;
  Sha256Digest canonical_hash;
  ComponentSlot slot{};
};

[[nodiscard]] Result<std::array<ComponentIdentity, 3U>>
ActiveComponentIdentities(const ReferenceBundle& bundle) {
  const auto route =
      CanonicalComponentHash(bundle.route_skeleton.content);
  if (!IsOk(route)) {
    return std::get<Error>(route);
  }
  const auto committed =
      CanonicalComponentHash(bundle.committed_prefix.content);
  if (!IsOk(committed)) {
    return std::get<Error>(committed);
  }
  const auto preview =
      CanonicalComponentHash(bundle.preview.content);
  if (!IsOk(preview)) {
    return std::get<Error>(preview);
  }
  return std::array<ComponentIdentity, 3U>{
      ComponentIdentity{
          bundle.route_skeleton.component_id,
          bundle.route_skeleton.component_hash,
          std::get<Sha256Digest>(route),
          ComponentSlot::kRoute,
      },
      ComponentIdentity{
          bundle.committed_prefix.component_id,
          bundle.committed_prefix.component_hash,
          std::get<Sha256Digest>(committed),
          ComponentSlot::kCommitted,
      },
      ComponentIdentity{
          bundle.preview.component_id,
          bundle.preview.component_hash,
          std::get<Sha256Digest>(preview),
          ComponentSlot::kPreview,
      },
  };
}

[[nodiscard]] Result<bool> CheckComponentReuse(
    const BundleBuildRequest& request,
    const std::array<Sha256Digest, 3U>& new_hashes,
    const Sha256Digest& new_reference_hash) {
  if (!request.active_bundle.has_value()) {
    return true;
  }
  const ReferenceBundle& active = *request.active_bundle;
  const auto active_reference_hash =
      CheckedReferenceHash(active.platform_reference);
  if (!IsOk(active_reference_hash)) {
    return Invalid(
        "active_bundle.platform_reference.reference_hash",
        "active bundle reference hash is not canonical");
  }
  const auto active_components =
      ActiveComponentIdentities(active);
  if (!IsOk(active_components)) {
    return std::get<Error>(active_components);
  }
  const auto& identities =
      std::get<std::array<ComponentIdentity, 3U>>(active_components);
  const std::array<ComponentId, 3U> new_ids{
      request.route_skeleton.component_id,
      request.committed_prefix.component_id,
      request.preview.component_id,
  };

  for (std::size_t new_index = 0U;
       new_index < new_ids.size(); ++new_index) {
    for (const ComponentIdentity& old_component : identities) {
      if (new_ids[new_index] != old_component.id) {
        continue;
      }
      const auto new_slot =
          static_cast<ComponentSlot>(new_index);
      if (new_slot != old_component.slot ||
          old_component.stored_hash != old_component.canonical_hash ||
          new_hashes[new_index] != old_component.canonical_hash ||
          std::get<Sha256Digest>(active_reference_hash) !=
              new_reference_hash) {
        return Invalid(
            "component_id",
            "a reused component ID must preserve slot, canonical content "
            "and platform payload identity");
      }
    }
  }
  return true;
}

[[nodiscard]] Result<bool> CheckRequestRoots(
    const BundleBuildRequest& request,
    const ReferenceActivationContext& context,
    PlatformType platform_type) {
  const PlanningRequest& planning_request = context.request;
  if (request.source_request_id != planning_request.request_id) {
    return Invalid(
        "source_request_id",
        "bundle source request must equal the activation request");
  }
  if (planning_request.platform_type != platform_type) {
    return Invalid(
        "platform_reference",
        "platform reference must match the activation request platform");
  }
  if (!planning_request.map_snapshot ||
      planning_request.map_snapshot->snapshot_ref() !=
          request.source_map_snapshot_ref) {
    return Invalid(
        "source_map_snapshot_ref",
        "bundle map must equal the request-fixed immutable snapshot");
  }
  if (!planning_request.safety_capability ||
      planning_request.safety_capability->content_ref !=
          request.source_safety_capability_ref) {
    return Invalid(
        "source_safety_capability_ref",
        "bundle capability must equal the request-fixed profile");
  }
  if (!planning_request.algorithm_config ||
      planning_request.algorithm_config->content_ref !=
          request.source_algorithm_config_ref) {
    return Invalid(
        "source_algorithm_config_ref",
        "bundle algorithm config must equal the request-fixed config");
  }
  return true;
}

[[nodiscard]] Result<bool> CheckRoundTrip(
    const ReferenceBundle& bundle,
    const ReferenceActivationContext& context) {
  PlanningResponse carrier{
      .request_id = context.request.request_id,
      .response_time = context.request.request_time,
      .planning_outcome = PlanningOutcome::kNewReferenceReady,
      .execution_directive =
          ExecutionDirective::kActivateNewBundle,
      .reason_code = "REFERENCE_BUNDLE_VALIDATED",
      .new_reference_bundle = bundle,
      .call_diagnostics =
          {
              .api_latency =
                  DurationNanoseconds{std::chrono::nanoseconds{0}},
              .termination_reason =
                  bundle.generation_evidence.termination_reason,
          },
  };
  try {
    const std::string encoded =
        JsonCodec::EncodePlanningResponse(carrier);
    const auto decoded =
        JsonCodec::DecodePlanningResponse(encoded, context);
    if (!IsOk(decoded)) {
      return std::get<Error>(decoded);
    }
    const PlanningResponse& decoded_response =
        std::get<PlanningResponse>(decoded);
    if (!decoded_response.new_reference_bundle.has_value() ||
        decoded_response.new_reference_bundle->bundle_hash !=
            bundle.bundle_hash) {
      return Invalid(
          "bundle_hash",
          "typed response round-trip changed the bundle identity");
    }
    const auto round_trip_hash =
        CanonicalBundleHash(*decoded_response.new_reference_bundle);
    if (!IsOk(round_trip_hash)) {
      return std::get<Error>(round_trip_hash);
    }
    if (std::get<Sha256Digest>(round_trip_hash) !=
        bundle.bundle_hash) {
      return Invalid(
          "bundle_hash",
          "canonical bundle hash changed after typed round-trip");
    }
  } catch (const std::exception& exception) {
    return Invalid("reference_bundle", exception.what());
  }
  return true;
}

}  // namespace

Result<ReferenceBundle> BuildReferenceBundle(
    const BundleBuildRequest& request,
    const ReferenceActivationContext& activation_context) {
  const ReferenceIdentity identity =
      IdentityOf(request.platform_reference);

  const auto reference_hash =
      CheckedReferenceHash(request.platform_reference);
  if (!IsOk(reference_hash)) {
    return std::get<Error>(reference_hash);
  }
  const auto component_binding =
      CheckComponentBinding(request, identity);
  if (!IsOk(component_binding)) {
    return std::get<Error>(component_binding);
  }
  const auto revision = CheckRevisionAndSupersession(request);
  if (!IsOk(revision)) {
    return std::get<Error>(revision);
  }
  const auto roots =
      CheckRequestRoots(request, activation_context,
                        identity.platform_type);
  if (!IsOk(roots)) {
    return std::get<Error>(roots);
  }
  const auto component_hashes = ComputeComponentHashes(request);
  if (!IsOk(component_hashes)) {
    return std::get<Error>(component_hashes);
  }
  const auto& hashes =
      std::get<std::array<Sha256Digest, 3U>>(component_hashes);
  const auto reuse = CheckComponentReuse(
      request, hashes, std::get<Sha256Digest>(reference_hash));
  if (!IsOk(reuse)) {
    return std::get<Error>(reuse);
  }

  ReferenceBundle bundle{
      .bundle_id = request.bundle_id,
      .bundle_revision = request.bundle_revision,
      .bundle_hash = {},
      .supersedes_bundle_id = request.supersedes_bundle_id,
      .source_request_id = request.source_request_id,
      .source_map_snapshot_ref = request.source_map_snapshot_ref,
      .source_safety_capability_ref =
          request.source_safety_capability_ref,
      .source_algorithm_config_ref =
          request.source_algorithm_config_ref,
      .platform_type = identity.platform_type,
      .platform_reference = request.platform_reference,
      .route_skeleton =
          {
              request.route_skeleton.component_id,
              hashes[0U],
              request.route_skeleton.content,
          },
      .committed_prefix =
          {
              request.committed_prefix.component_id,
              hashes[1U],
              request.committed_prefix.content,
          },
      .preview =
          {
              request.preview.component_id,
              hashes[2U],
              request.preview.content,
          },
      .validity = request.validity,
      .validation_summary = request.validation_summary,
      .generation_evidence = request.generation_evidence,
  };

  const auto bundle_hash = CanonicalBundleHash(bundle);
  if (!IsOk(bundle_hash)) {
    return std::get<Error>(bundle_hash);
  }
  bundle.bundle_hash = std::get<Sha256Digest>(bundle_hash);

  const SemanticValidator validator;
  const ValidationReport local = validator.Validate(bundle);
  if (!local.ok()) {
    return ValidationError(local);
  }
  const ValidationReport activation =
      validator.ValidateForActivation(bundle, activation_context);
  if (!activation.ok()) {
    return ValidationError(activation);
  }
  const auto round_trip = CheckRoundTrip(bundle, activation_context);
  if (!IsOk(round_trip)) {
    return std::get<Error>(round_trip);
  }
  return bundle;
}

}  // namespace lunar::planning::v3
