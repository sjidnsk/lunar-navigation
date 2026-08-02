#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "lunar_path_planner/v3/contracts/planning_request.hpp"
#include "lunar_path_planner/v3/contracts/reference_bundle.hpp"

namespace lunar::planning::v3 {

template <class ContentT>
struct ComponentBuildInput final {
  ComponentId component_id;
  ContentT content;
};

struct BundleBuildRequest final {
  BundleId bundle_id;
  std::uint32_t bundle_revision{};
  std::optional<BundleId> supersedes_bundle_id;
  RequestId source_request_id;
  ContentRef source_map_snapshot_ref;
  ContentRef source_safety_capability_ref;
  ContentRef source_algorithm_config_ref;
  PlatformReference platform_reference;
  ComponentBuildInput<RouteSkeletonContent> route_skeleton;
  ComponentBuildInput<ReferenceViewContent> committed_prefix;
  ComponentBuildInput<ReferenceViewContent> preview;
  ReferenceValidity validity;
  ValidationSummary validation_summary;
  GenerationEvidence generation_evidence;
  std::optional<ReferenceBundle> active_bundle;
};

[[nodiscard]] Result<ReferenceBundle> BuildReferenceBundle(
    const BundleBuildRequest& request,
    const ReferenceActivationContext& activation_context);

}  // namespace lunar::planning::v3
