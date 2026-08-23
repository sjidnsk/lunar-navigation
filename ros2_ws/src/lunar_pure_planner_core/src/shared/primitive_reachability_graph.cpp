#include "shared/primitive_reachability_graph.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hopper/hopper_reachability_graph.hpp"
#include "legged/legged_lattice.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"
#include "shared/sha256.hpp"
#include "wheel/wheel_primitive_expansion.hpp"

namespace lunar::pure_planning {
namespace {

[[nodiscard]] PrimitiveReachabilityResult Failure(std::string reason_code) {
  return PrimitiveReachabilityResult{
      .snapshot = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool StateMatchesPlatform(
    const PlatformState& state, const PlatformType platform) noexcept {
  switch (platform) {
    case PlatformType::kWheeled:
      return std::holds_alternative<WheeledState>(state);
    case PlatformType::kLegged:
      return std::holds_alternative<LeggedState>(state);
    case PlatformType::kHopper:
      return std::holds_alternative<HopperState>(state);
  }
  return false;
}

[[nodiscard]] bool CloseResolution(
    const double lhs, const double rhs) noexcept {
  return std::isfinite(lhs) && std::isfinite(rhs) &&
      std::abs(lhs - rhs) <=
          1.0e-9 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

[[nodiscard]] double GroundGraphResolution(
    const PlannerInput& input, const PlatformType platform) noexcept {
  return platform == PlatformType::kWheeled
      ? input.config.wheel.xy_resolution_m
      : input.config.legged.xy_resolution_m;
}

class EvidenceBytes final {
 public:
  void Byte(const std::uint8_t value) {
    bytes_.push_back(value);
  }

  void Uint32(const std::uint32_t value) {
    for (std::size_t index = 0U; index < 4U; ++index) {
      Byte(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
  }

  void Uint64(const std::uint64_t value) {
    for (std::size_t index = 0U; index < 8U; ++index) {
      Byte(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
  }

  void Double(double value) {
    if (value == 0.0) {
      value = 0.0;
    }
    Uint64(std::bit_cast<std::uint64_t>(value));
  }

  void Float(float value) {
    if (value == 0.0F) {
      value = 0.0F;
    }
    Uint32(std::bit_cast<std::uint32_t>(value));
  }

  void String(const std::string& value) {
    Uint64(static_cast<std::uint64_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  void Grid(const GridMap& map) {
    String(map.frame_id);
    Uint64(static_cast<std::uint64_t>(map.width));
    Uint64(static_cast<std::uint64_t>(map.height));
    Double(map.resolution_m);
    Double(map.origin_m.x);
    Double(map.origin_m.y);
    Double(map.origin_m.z);
    Uint64(static_cast<std::uint64_t>(map.layers.size()));
    for (const auto& [name, layer] : map.layers) {
      String(name);
      Byte(static_cast<std::uint8_t>(layer.values.index()));
      std::visit(
          [this](const auto& values) {
            Uint64(static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
              using Value = std::remove_cvref_t<decltype(value)>;
              if constexpr (std::is_same_v<Value, float>) {
                Float(value);
              } else if constexpr (std::is_same_v<Value, std::uint8_t>) {
                Byte(value);
              } else {
                Uint32(value);
              }
            }
          },
          layer.values);
    }
  }

  void HopperLandings(const HopperLandingEvidenceGrid& evidence) {
    Uint64(static_cast<std::uint64_t>(evidence.width));
    Uint64(static_cast<std::uint64_t>(evidence.height));
    String(evidence.algorithm_id);
    Uint64(static_cast<std::uint64_t>(evidence.landings.size()));
    for (const HopperLandingEvidence& landing : evidence.landings) {
      Byte(landing.certified);
      Double(landing.aim_position_on_surface_m.x);
      Double(landing.aim_position_on_surface_m.y);
      Double(landing.aim_position_on_surface_m.z);
      for (const Vec3 vertex : landing.boundary_m) {
        Double(vertex.x);
        Double(vertex.y);
        Double(vertex.z);
      }
      Double(landing.area_m2);
    }
  }

  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return bytes_;
  }

 private:
  std::vector<std::uint8_t> bytes_;
};

[[nodiscard]] std::string WorldEvidenceSha256(
    const WorldSnapshot& world,
    const HopperLandingEvidenceGrid* const hopper_landing_evidence) {
  EvidenceBytes bytes;
  bytes.String(
      hopper_landing_evidence == nullptr
          ? "primitive-reachability-world-evidence/v1"
          : "primitive-reachability-world-evidence/v2");
  bytes.Grid(world.global_map);
  bytes.Grid(world.local_map);
  bytes.String(world.map_from_odom.parent_frame);
  bytes.String(world.map_from_odom.child_frame);
  bytes.Double(world.map_from_odom.translation_m.x);
  bytes.Double(world.map_from_odom.translation_m.y);
  bytes.Double(world.map_from_odom.translation_m.z);
  bytes.Double(world.map_from_odom.rotation.w);
  bytes.Double(world.map_from_odom.rotation.x);
  bytes.Double(world.map_from_odom.rotation.y);
  bytes.Double(world.map_from_odom.rotation.z);
  if (hopper_landing_evidence != nullptr) {
    bytes.HopperLandings(*hopper_landing_evidence);
  }
  return shared::Sha256Hex(bytes.bytes());
}

}  // namespace

class PrimitiveReachabilityEngine::Impl final {
 public:
  std::uint64_t revision{};
  bool has_previous{};
  PlatformType previous_platform{};
  std::string previous_algorithm_id;
  std::string previous_state_schema;
  std::string previous_primitive_set_sha256;
  std::string previous_world_evidence_sha256;
  std::size_t previous_edge_count{};

  [[nodiscard]] PrimitiveReachabilityResult Publish(
      PrimitiveReachabilityResult result,
      const std::string& world_evidence_sha256) {
    if (!result.ok()) {
      return result;
    }
    PrimitiveReachabilitySnapshot& snapshot = *result.snapshot;
    const bool compatible = has_previous &&
        previous_platform == snapshot.platform_type &&
        previous_algorithm_id == snapshot.algorithm_id &&
        previous_state_schema == snapshot.state_schema &&
        previous_primitive_set_sha256 == snapshot.primitive_set_sha256;
    if (!compatible) {
      revision = 1U;
      snapshot.invalidated_edge_count = 0U;
      snapshot.revalidated_edge_count = 0U;
    } else {
      ++revision;
      if (previous_world_evidence_sha256 != world_evidence_sha256) {
        snapshot.invalidated_edge_count = previous_edge_count;
        snapshot.revalidated_edge_count = snapshot.edges.size();
      } else {
        snapshot.invalidated_edge_count = 0U;
        snapshot.revalidated_edge_count = 0U;
      }
    }
    snapshot.revision = revision;
    snapshot.world_evidence_sha256 = world_evidence_sha256;
    has_previous = true;
    previous_platform = snapshot.platform_type;
    previous_algorithm_id = snapshot.algorithm_id;
    previous_state_schema = snapshot.state_schema;
    previous_primitive_set_sha256 = snapshot.primitive_set_sha256;
    previous_world_evidence_sha256 = world_evidence_sha256;
    previous_edge_count = snapshot.edges.size();
    return result;
  }

  void Reset() noexcept {
    revision = 0U;
    has_previous = false;
    previous_platform = PlatformType{};
    previous_algorithm_id.clear();
    previous_state_schema.clear();
    previous_primitive_set_sha256.clear();
    previous_world_evidence_sha256.clear();
    previous_edge_count = 0U;
  }
};

PrimitiveReachabilityEngine::PrimitiveReachabilityEngine()
    : impl_(std::make_unique<Impl>()) {}

PrimitiveReachabilityEngine::~PrimitiveReachabilityEngine() = default;

PrimitiveReachabilityEngine::PrimitiveReachabilityEngine(
    PrimitiveReachabilityEngine&&) noexcept = default;

PrimitiveReachabilityEngine& PrimitiveReachabilityEngine::operator=(
    PrimitiveReachabilityEngine&&) noexcept = default;

PrimitiveReachabilityResult PrimitiveReachabilityEngine::Update(
    const PlannerInput& input,
    const std::optional<double> maximum_action_distance_m) {
  return UpdateImpl(input, maximum_action_distance_m, nullptr);
}

PrimitiveReachabilityResult PrimitiveReachabilityEngine::Update(
    const PlannerInput& input,
    const std::optional<double> maximum_action_distance_m,
    const HopperLandingEvidenceGrid& hopper_landing_evidence) {
  return UpdateImpl(
      input, maximum_action_distance_m, &hopper_landing_evidence);
}

PrimitiveReachabilityResult PrimitiveReachabilityEngine::UpdateImpl(
    const PlannerInput& input,
    const std::optional<double> maximum_action_distance_m,
    const HopperLandingEvidenceGrid* const hopper_landing_evidence) {
  if (input.stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED");
  }
  if (maximum_action_distance_m.has_value() &&
      (!std::isfinite(*maximum_action_distance_m) ||
       *maximum_action_distance_m <= 0.0)) {
    return Failure("PRIMITIVE_REACHABILITY_DISTANCE_INVALID");
  }
  const PlatformType platform = CapabilityPlatform(input.capability);
  if (!StateMatchesPlatform(input.current_state, platform)) {
    return Failure("PLATFORM_STATE_CAPABILITY_MISMATCH");
  }
  if (hopper_landing_evidence != nullptr &&
      platform != PlatformType::kHopper) {
    return Failure("HOPPER_LANDING_EVIDENCE_PLATFORM_INVALID");
  }
  const std::string world_evidence_sha256 =
      WorldEvidenceSha256(input.world, hopper_landing_evidence);
  const shared::MapSnapshotBuildResult map =
      shared::MapSnapshot::Create(input.world.global_map);
  if (!map.ok()) {
    return Failure(map.reason_code);
  }
  const shared::SafeProjectionBuildResult safe = shared::BuildSafeProjection(
      map.snapshot, input.capability, input.config.map_safety,
      input.stop_token);
  if (!safe.ok()) {
    return Failure(safe.reason_code);
  }
  const bool ground_platform = platform == PlatformType::kWheeled ||
      platform == PlatformType::kLegged;
  const double graph_resolution_m =
      ground_platform ? GroundGraphResolution(input, platform) : 0.0;
  const bool use_local_ground_projection = ground_platform &&
      !CloseResolution(map.snapshot->resolution_m(), graph_resolution_m) &&
      CloseResolution(input.world.local_map.resolution_m, graph_resolution_m);
  shared::MapSnapshotBuildResult local_ground_map;
  shared::SafeProjectionBuildResult local_ground_safe;
  const shared::SafeProjection* ground_projection = &*safe.projection;
  if (use_local_ground_projection) {
    if (input.world.local_map.frame_id !=
        input.world.map_from_odom.child_frame) {
      return Failure("LOCAL_MAP_FRAME_INVALID");
    }
    local_ground_map = shared::MapSnapshot::Create(input.world.local_map);
    if (!local_ground_map.ok()) {
      return Failure(local_ground_map.reason_code);
    }
    local_ground_safe = shared::BuildSafeProjection(
        local_ground_map.snapshot, input.capability, input.config.map_safety,
        input.stop_token);
    if (!local_ground_safe.ok()) {
      return Failure(local_ground_safe.reason_code);
    }
    ground_projection = &*local_ground_safe.projection;
  }
  if (platform == PlatformType::kWheeled) {
    WheeledState state = std::get<WheeledState>(input.current_state);
    if (!use_local_ground_projection) {
      const auto pose_map = hierarchical::TransformPose(
          state.pose, input.world.map_from_odom,
          hierarchical::TransformDirection::kChildToParent);
      if (!pose_map.has_value()) {
        return Failure("FRAME_TRANSFORM_INVALID");
      }
      state.pose = *pose_map;
    }
    shared::PrimitiveGraphBuildResult graph =
        wheel::BuildWheelPrimitiveGraph(
            state, *ground_projection,
            std::get<WheeledCapability>(input.capability), input.config,
            input.stop_token);
    return impl_->Publish(
        shared::FinalizePrimitiveGraph(std::move(graph), input.stop_token),
        world_evidence_sha256);
  }
  if (platform == PlatformType::kLegged) {
    LeggedState state = std::get<LeggedState>(input.current_state);
    if (!use_local_ground_projection) {
      const auto pose_map = hierarchical::TransformPose(
          state.body_pose, input.world.map_from_odom,
          hierarchical::TransformDirection::kChildToParent);
      if (!pose_map.has_value()) {
        return Failure("FRAME_TRANSFORM_INVALID");
      }
      state.body_pose = *pose_map;
    }
    shared::PrimitiveGraphBuildResult graph =
        legged::BuildLeggedPrimitiveGraph(
            state, *ground_projection,
            std::get<LeggedCapability>(input.capability), input.config,
            input.stop_token);
    return impl_->Publish(
        shared::FinalizePrimitiveGraph(std::move(graph), input.stop_token),
        world_evidence_sha256);
  }
  if (platform == PlatformType::kHopper) {
    const shared::MapSnapshotBuildResult local_map =
        shared::MapSnapshot::Create(input.world.local_map);
    if (!local_map.ok()) {
      return Failure(local_map.reason_code);
    }
    shared::PrimitiveGraphBuildResult graph =
        hopper::BuildHopperPrimitiveGraph(
            input, map.snapshot, local_map.snapshot, *safe.projection,
            maximum_action_distance_m, hopper_landing_evidence);
    return impl_->Publish(
        shared::FinalizePrimitiveGraph(std::move(graph), input.stop_token),
        world_evidence_sha256);
  }
  return Failure("PRIMITIVE_REACHABILITY_PLATFORM_NOT_IMPLEMENTED");
}

void PrimitiveReachabilityEngine::Reset() noexcept {
  impl_->Reset();
}

namespace shared {
namespace {

[[nodiscard]] double NormalizedDouble(const double value) noexcept {
  return value == 0.0 ? 0.0 : value;
}

[[nodiscard]] auto StateKey(const PrimitiveReachabilityState& state) {
  return std::tuple{
      state.cell_x,
      state.cell_y,
      state.yaw_bin,
      state.motion_mode,
      NormalizedDouble(state.position_m.x),
      NormalizedDouble(state.position_m.y),
      NormalizedDouble(state.position_m.z),
      NormalizedDouble(state.yaw_rad),
      NormalizedDouble(state.body_z_m.lower),
      NormalizedDouble(state.body_z_m.upper),
  };
}

[[nodiscard]] bool StateFinite(
    const PrimitiveReachabilityState& state) noexcept {
  return std::isfinite(state.position_m.x) &&
      std::isfinite(state.position_m.y) &&
      std::isfinite(state.position_m.z) &&
      std::isfinite(state.yaw_rad) &&
      std::isfinite(state.body_z_m.lower) &&
      std::isfinite(state.body_z_m.upper) &&
      std::isfinite(state.path_cost);
}

class CanonicalBytes final {
 public:
  void Byte(const std::uint8_t value) {
    bytes_.push_back(value);
  }

  void Uint32(const std::uint32_t value) {
    for (std::size_t index = 0U; index < 4U; ++index) {
      Byte(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
  }

  void Int32(const std::int32_t value) {
    Uint32(std::bit_cast<std::uint32_t>(value));
  }

  void Uint64(const std::uint64_t value) {
    for (std::size_t index = 0U; index < 8U; ++index) {
      Byte(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
  }

  void Double(const double value) {
    Uint64(std::bit_cast<std::uint64_t>(NormalizedDouble(value)));
  }

  void String(const std::string& value) {
    Uint64(static_cast<std::uint64_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return bytes_;
  }

 private:
  std::vector<std::uint8_t> bytes_;
};

void VisitFromAnchor(
    const std::vector<std::vector<std::size_t>>& adjacency,
    const std::size_t anchor, std::vector<std::uint8_t>& visited,
    const std::stop_token stop_token) {
  std::deque<std::size_t> queue;
  visited[anchor] = 1U;
  queue.push_back(anchor);
  while (!queue.empty() && !stop_token.stop_requested()) {
    const std::size_t source = queue.front();
    queue.pop_front();
    for (const std::size_t target : adjacency[source]) {
      if (visited[target] == 0U) {
        visited[target] = 1U;
        queue.push_back(target);
      }
    }
  }
}

[[nodiscard]] std::string GraphSha256(
    const PrimitiveReachabilitySnapshot& snapshot) {
  CanonicalBytes canonical;
  canonical.String("lunar-primitive-reachability-graph/v1");
  canonical.Byte(static_cast<std::uint8_t>(snapshot.platform_type));
  canonical.Uint64(static_cast<std::uint64_t>(snapshot.width));
  canonical.Uint64(static_cast<std::uint64_t>(snapshot.height));
  canonical.String(snapshot.algorithm_id);
  canonical.String(snapshot.state_schema);
  canonical.String(snapshot.primitive_set_sha256);
  canonical.Uint64(static_cast<std::uint64_t>(snapshot.states.size()));
  for (const PrimitiveReachabilityState& state : snapshot.states) {
    canonical.Uint64(state.state_id);
    canonical.Double(state.position_m.x);
    canonical.Double(state.position_m.y);
    canonical.Double(state.position_m.z);
    canonical.Double(state.yaw_rad);
    canonical.Int32(state.cell_x);
    canonical.Int32(state.cell_y);
    canonical.Int32(state.yaw_bin);
    canonical.Int32(state.motion_mode);
    canonical.Double(state.body_z_m.lower);
    canonical.Double(state.body_z_m.upper);
    canonical.Double(state.path_cost);
    canonical.Byte(state.forward_reachable);
    canonical.Byte(state.returnable);
    canonical.Byte(state.observation_state);
    canonical.Byte(state.direct_successor);
  }
  canonical.Uint64(static_cast<std::uint64_t>(snapshot.edges.size()));
  for (const PrimitiveReachabilityEdge& edge : snapshot.edges) {
    canonical.Uint64(edge.source_state_id);
    canonical.String(edge.primitive_id);
    canonical.Uint64(edge.target_state_id);
    canonical.Uint32(edge.primitive_index);
    canonical.Double(edge.cost);
  }
  return Sha256Hex(canonical.bytes());
}

}  // namespace

PrimitiveReachabilityResult FinalizePrimitiveGraph(
    PrimitiveGraphBuildResult graph, const std::stop_token stop_token) {
  if (!graph.reason_code.empty()) {
    return Failure(std::move(graph.reason_code));
  }
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED");
  }
  if (graph.width == 0U || graph.height == 0U ||
      graph.width > std::numeric_limits<std::size_t>::max() / graph.height) {
    return Failure("PRIMITIVE_GRAPH_GEOMETRY_INVALID");
  }
  if (graph.states.empty() ||
      graph.anchor_state_index >= graph.states.size()) {
    return Failure("PRIMITIVE_GRAPH_ANCHOR_INVALID");
  }
  if (graph.algorithm_id.empty() || graph.state_schema.empty()) {
    return Failure("PRIMITIVE_GRAPH_IDENTITY_INVALID");
  }

  for (const PrimitiveReachabilityState& state : graph.states) {
    if (stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED");
    }
    if (!StateFinite(state)) {
      return Failure("PRIMITIVE_GRAPH_STATE_NONFINITE");
    }
    if (state.body_z_m.lower > state.body_z_m.upper ||
        state.path_cost < 0.0) {
      return Failure("PRIMITIVE_GRAPH_STATE_RANGE_INVALID");
    }
    if (state.cell_x < 0 || state.cell_y < 0 ||
        static_cast<std::size_t>(state.cell_x) >= graph.width ||
        static_cast<std::size_t>(state.cell_y) >= graph.height) {
      return Failure("PRIMITIVE_GRAPH_STATE_CELL_INVALID");
    }
    if (state.observation_state > 1U) {
      return Failure("PRIMITIVE_GRAPH_OBSERVATION_FLAG_INVALID");
    }
  }
  for (const PrimitiveGraphPotentialEdge& edge : graph.potential_edges) {
    if (edge.source_state_index >= graph.states.size() ||
        edge.target_state_index >= graph.states.size()) {
      return Failure("PRIMITIVE_GRAPH_EDGE_INDEX_INVALID");
    }
    if (!std::isfinite(edge.cost) || edge.cost < 0.0) {
      return Failure("PRIMITIVE_GRAPH_EDGE_COST_INVALID");
    }
    if (edge.primitive_id.empty() ||
        (!edge.certified && edge.rejection_reason.empty())) {
      return Failure("PRIMITIVE_GRAPH_EDGE_IDENTITY_INVALID");
    }
  }

  std::vector<std::size_t> order(graph.states.size());
  std::iota(order.begin(), order.end(), 0U);
  std::sort(order.begin(), order.end(), [&](const std::size_t lhs,
                                            const std::size_t rhs) {
    return StateKey(graph.states[lhs]) < StateKey(graph.states[rhs]);
  });
  for (std::size_t index = 1U; index < order.size(); ++index) {
    if (StateKey(graph.states[order[index - 1U]]) ==
        StateKey(graph.states[order[index]])) {
      return Failure("PRIMITIVE_GRAPH_STATE_KEY_DUPLICATE");
    }
  }

  std::vector<std::size_t> canonical_index(graph.states.size());
  std::vector<PrimitiveReachabilityState> states;
  states.reserve(graph.states.size());
  for (std::size_t state_id = 0U; state_id < order.size(); ++state_id) {
    const std::size_t source_index = order[state_id];
    canonical_index[source_index] = state_id;
    PrimitiveReachabilityState state = graph.states[source_index];
    state.state_id = static_cast<std::uint64_t>(state_id);
    state.position_m.x = NormalizedDouble(state.position_m.x);
    state.position_m.y = NormalizedDouble(state.position_m.y);
    state.position_m.z = NormalizedDouble(state.position_m.z);
    state.yaw_rad = NormalizedDouble(state.yaw_rad);
    state.body_z_m.lower = NormalizedDouble(state.body_z_m.lower);
    state.body_z_m.upper = NormalizedDouble(state.body_z_m.upper);
    state.path_cost = NormalizedDouble(state.path_cost);
    state.forward_reachable = 0U;
    state.returnable = 0U;
    state.direct_successor = 0U;
    states.push_back(std::move(state));
  }
  const std::size_t anchor = canonical_index[graph.anchor_state_index];

  std::vector<PrimitiveReachabilityEdge> edges;
  edges.reserve(graph.potential_edges.size());
  for (const PrimitiveGraphPotentialEdge& potential : graph.potential_edges) {
    if (!potential.certified) {
      continue;
    }
    edges.push_back(PrimitiveReachabilityEdge{
        .source_state_id = static_cast<std::uint64_t>(
            canonical_index[potential.source_state_index]),
        .target_state_id = static_cast<std::uint64_t>(
            canonical_index[potential.target_state_index]),
        .primitive_index = potential.primitive_index,
        .primitive_id = potential.primitive_id,
        .cost = NormalizedDouble(potential.cost),
    });
  }
  std::sort(edges.begin(), edges.end(),
            [](const PrimitiveReachabilityEdge& lhs,
               const PrimitiveReachabilityEdge& rhs) {
    return std::tie(lhs.source_state_id, lhs.primitive_id,
                    lhs.target_state_id, lhs.primitive_index, lhs.cost) <
        std::tie(rhs.source_state_id, rhs.primitive_id,
                 rhs.target_state_id, rhs.primitive_index, rhs.cost);
  });
  for (std::size_t index = 1U; index < edges.size(); ++index) {
    const auto& previous = edges[index - 1U];
    const auto& current = edges[index];
    if (previous.source_state_id == current.source_state_id &&
        previous.primitive_id == current.primitive_id &&
        previous.target_state_id == current.target_state_id) {
      return Failure("PRIMITIVE_GRAPH_EDGE_KEY_DUPLICATE");
    }
  }

  std::vector<std::vector<std::size_t>> adjacency(states.size());
  std::vector<std::vector<std::size_t>> reverse_adjacency(states.size());
  for (const PrimitiveReachabilityEdge& edge : edges) {
    const std::size_t source = static_cast<std::size_t>(edge.source_state_id);
    const std::size_t target = static_cast<std::size_t>(edge.target_state_id);
    adjacency[source].push_back(target);
    reverse_adjacency[target].push_back(source);
    if (source == anchor) {
      states[target].direct_successor = 1U;
    }
  }
  for (auto& successors : adjacency) {
    std::sort(successors.begin(), successors.end());
  }
  for (auto& predecessors : reverse_adjacency) {
    std::sort(predecessors.begin(), predecessors.end());
  }

  std::vector<std::uint8_t> forward(states.size(), 0U);
  std::vector<std::uint8_t> returnable(states.size(), 0U);
  VisitFromAnchor(adjacency, anchor, forward, stop_token);
  VisitFromAnchor(reverse_adjacency, anchor, returnable, stop_token);
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED");
  }

  std::vector<std::uint8_t> reachable(graph.width * graph.height, 0U);
  for (std::size_t index = 0U; index < states.size(); ++index) {
    states[index].forward_reachable = forward[index];
    states[index].returnable = returnable[index];
    if (forward[index] != 0U && returnable[index] != 0U &&
        states[index].observation_state != 0U) {
      const std::size_t cell =
          static_cast<std::size_t>(states[index].cell_y) * graph.width +
          static_cast<std::size_t>(states[index].cell_x);
      reachable[cell] = 1U;
    }
  }

  PrimitiveReachabilitySnapshot snapshot{
      .platform_type = graph.platform_type,
      .width = graph.width,
      .height = graph.height,
      .states = std::move(states),
      .edges = std::move(edges),
      .reachable = std::move(reachable),
      .algorithm_id = std::move(graph.algorithm_id),
      .state_schema = std::move(graph.state_schema),
      .primitive_set_sha256 = Sha256Hex(
          graph.primitive_set_canonical_bytes),
      .revision = graph.revision,
      .invalidated_edge_count = graph.invalidated_edge_count,
      .revalidated_edge_count = graph.revalidated_edge_count,
  };
  snapshot.graph_sha256 = GraphSha256(snapshot);
  return PrimitiveReachabilityResult{
      .snapshot = std::move(snapshot),
      .reason_code = {},
  };
}

}  // namespace shared
}  // namespace lunar::pure_planning
