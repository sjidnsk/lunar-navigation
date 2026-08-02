#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/contracts/profiles.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lunar::planning::v3 {

struct CircularYawInterval final {
  enum class Representation {
    kCanonicalCcw,
  };

  Representation representation{Representation::kCanonicalCcw};
  double start_rad{};
  double span_rad{};
  bool closed{true};
};

struct LandingPlane final {
  Vec3 origin_m;
  Vec3 normal;
  Vec3 basis_u;
  Vec3 basis_v;
  double residual_bound_m{};
};

struct ConvexPolygonUv final {
  enum class Winding {
    kCcw,
  };

  std::vector<Vec2> vertices_uv;
  Winding winding{Winding::kCcw};
};

struct PointGoal final {
  Vec3 position_m;
  double position_tolerance_m{};
};

struct PlanarRegionGoal final {
  LandingPlane plane;
  ConvexPolygonUv polygon;
  double normal_tolerance_m{};
};

using GoalTarget = std::variant<PointGoal, PlanarRegionGoal>;
using MetadataValue = std::variant<std::string, double, bool>;

struct MetadataEntry final {
  std::string key;
  MetadataValue value;
};

struct GoalRegion final {
  std::string goal_id;
  GoalTarget target;
  std::optional<CircularYawInterval> optional_yaw_interval;
  std::optional<Vec3> mission_direction_hint;
  std::vector<MetadataEntry> task_metadata;
};

class LearnedCostSnapshot;
class MotionModel;
class AnalyticCostModel;
class GravityModel;
class DeterministicErrorModel;
class ActuatorOrImpulseProfile;
class BodyRotationEnvelope;
class AttitudeTighteningTable;

template <class T>
struct ResolvedBinding final {
  ContentRef content_ref;
  std::shared_ptr<const T> object;
};

struct ResolvedCapabilityBindings final {
  ResolvedBinding<MotionModel> motion_model;
  ResolvedBinding<AnalyticCostModel> analytic_cost_model;
  std::optional<ResolvedBinding<GravityModel>> gravity_model;
  std::optional<ResolvedBinding<DeterministicErrorModel>> error_model;
  std::optional<ResolvedBinding<ActuatorOrImpulseProfile>>
      actuator_or_impulse_profile;
  std::optional<ResolvedBinding<BodyRotationEnvelope>>
      body_rotation_envelope;
  std::optional<ResolvedBinding<AttitudeTighteningTable>>
      attitude_tightening_table;
};

struct TimeExecutionCursor final {
  DurationNanoseconds offset;
  std::optional<SegmentId> segment_id;
};

enum class JumpExecutionState {
  kGroundHold,
  kJumpReady,
  kJumpCommitted,
  kInFlight,
  kLandedHold,
};

struct JumpExecutionCursor final {
  JumpExecutionState jump_state{};
  std::optional<JumpBoundaryId> boundary_id;
};

using ExecutionCursor =
    std::variant<TimeExecutionCursor, JumpExecutionCursor>;

struct TimeCommitBoundary final {
  DurationNanoseconds committed_until_offset;
};

struct JumpCommitBoundary final {
  JumpBoundaryId boundary_id;
  bool locked{};
};

using CommitBoundary =
    std::variant<TimeCommitBoundary, JumpCommitBoundary>;

struct PreviousExecutionContext final {
  ContentRef active_bundle_ref;
  std::string active_bundle_handle;
  CommitBoundary commit_boundary;
  ExecutionCursor execution_cursor;
  ControllerStatus controller_status{};
  ContentRef source_map_snapshot_ref;
  ContentRef source_capability_ref;
};

struct LearnedCostSnapshotBinding final {
  ContentRef snapshot_ref;
  std::string registry_handle;
  std::shared_ptr<const LearnedCostSnapshot> resolved_snapshot;
};

struct PlanningRequest final {
  RequestId request_id;
  ClockStamp request_time;
  ClockStamp state_time;
  FrameId frame_id;
  PlatformType platform_type;
  PlatformState current_state;
  GoalRegion goal;
  std::shared_ptr<const ImmutableMapSnapshot> map_snapshot;
  std::shared_ptr<const SafetyCapabilityProfile> safety_capability;
  std::shared_ptr<const PlannerAlgorithmConfig> algorithm_config;
  ResolvedCapabilityBindings capability_bindings;
  std::optional<PreviousExecutionContext> previous_execution_context;
  std::optional<LearnedCostSnapshotBinding> learned_cost_snapshot;
};

class ContractObjectRegistry;

struct ReferenceActivationContext final {
  const PlanningRequest& request;
  const ContractObjectRegistry& registry;
};

}  // namespace lunar::planning::v3
