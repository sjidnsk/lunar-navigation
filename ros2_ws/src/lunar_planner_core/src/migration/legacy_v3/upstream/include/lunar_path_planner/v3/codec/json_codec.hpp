#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lunar_path_planner/v3/contracts/planning_response.hpp"

namespace lunar::planning::v3 {

enum class ContractObjectKind {
  kMotionModel,
  kAnalyticCostModel,
  kGravityModel,
  kDeterministicErrorModel,
  kActuatorOrImpulseProfile,
  kBodyRotationEnvelope,
  kAttitudeTighteningTable,
  kCertification,
};

struct CertificationProvenance final {
  ContentRef source_map_snapshot_ref;
  ContentRef source_safety_capability_ref;
  ContentRef source_algorithm_config_ref;
  std::vector<ContentRef> input_refs;
  std::string certification_purpose;
};

class ImmutableContractObject {
 public:
  virtual ~ImmutableContractObject() = default;

  [[nodiscard]] virtual ContentRef content_ref() const = 0;
  [[nodiscard]] virtual ContractObjectKind kind() const = 0;
};

class CertificationObject : public ImmutableContractObject {
 public:
  [[nodiscard]] virtual const CertificationProvenance& provenance()
      const = 0;
};

class ContractObjectRegistry {
 public:
  virtual ~ContractObjectRegistry() = default;

  [[nodiscard]] virtual std::shared_ptr<const ImmutableMapSnapshot>
  FindMapSnapshot(const ContentRef& snapshot_ref,
                  std::string_view immutable_data_handle) const = 0;

  [[nodiscard]] virtual std::shared_ptr<const SafetyCapabilityProfile>
  FindSafetyCapability(const ContentRef& ref) const = 0;

  [[nodiscard]] virtual std::shared_ptr<const PlannerAlgorithmConfig>
  FindAlgorithmConfig(const ContentRef& ref) const = 0;

  [[nodiscard]] virtual std::shared_ptr<const LearnedCostSnapshot>
  FindLearnedCost(const ContentRef& snapshot_ref,
                  std::string_view registry_handle) const = 0;

  [[nodiscard]] virtual Result<ResolvedCapabilityBindings>
  ResolveCapabilityBindings(
      const SafetyCapabilityProfile& profile) const = 0;

  [[nodiscard]] virtual
      Result<std::shared_ptr<const ImmutableContractObject>>
      Resolve(const ContentRef& ref,
              ContractObjectKind expected_kind) const = 0;
};

class EmptyContractObjectRegistry final : public ContractObjectRegistry {
 public:
  [[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot>
  FindMapSnapshot(const ContentRef&,
                  std::string_view) const override {
    return nullptr;
  }

  [[nodiscard]] std::shared_ptr<const SafetyCapabilityProfile>
  FindSafetyCapability(const ContentRef&) const override {
    return nullptr;
  }

  [[nodiscard]] std::shared_ptr<const PlannerAlgorithmConfig>
  FindAlgorithmConfig(const ContentRef&) const override {
    return nullptr;
  }

  [[nodiscard]] std::shared_ptr<const LearnedCostSnapshot>
  FindLearnedCost(const ContentRef&, std::string_view) const override {
    return nullptr;
  }

  [[nodiscard]] Result<ResolvedCapabilityBindings>
  ResolveCapabilityBindings(
      const SafetyCapabilityProfile&) const override {
    return Error{ErrorCode::kMissingRegistryObject, "safety_capability_ref",
                 "capability bindings are absent from the registry"};
  }

  [[nodiscard]] Result<std::shared_ptr<const ImmutableContractObject>>
  Resolve(const ContentRef&, ContractObjectKind) const override {
    return Error{ErrorCode::kMissingRegistryObject, "content_ref",
                 "contract object is absent from the registry"};
  }
};

class JsonCodec final {
 public:
  [[nodiscard]] static Result<PlanningRequest> DecodePlanningRequest(
      std::string_view payload,
      const ContractObjectRegistry& registry);

  [[nodiscard]] static Result<PlanningResponse> DecodePlanningResponse(
      std::string_view payload,
      const ReferenceActivationContext& activation_context);

  [[nodiscard]] static std::string EncodePlanningRequest(
      const PlanningRequest& request);

  [[nodiscard]] static std::string EncodePlanningResponse(
      const PlanningResponse& response);

  [[nodiscard]] static Result<std::string> EncodeDuration(
      DurationNanoseconds value);

  [[nodiscard]] static Result<DurationNanoseconds> DecodeDuration(
      std::string_view json_string);
};

[[nodiscard]] Result<Sha256Digest> CanonicalReferenceHash(
    const PlatformReference& reference);

[[nodiscard]] Result<Sha256Digest> CanonicalComponentHash(
    const RouteSkeletonContent& content);

[[nodiscard]] Result<Sha256Digest> CanonicalComponentHash(
    const ReferenceViewContent& content);

[[nodiscard]] Result<Sha256Digest> CanonicalBundleHash(
    const ReferenceBundle& bundle);

}  // namespace lunar::planning::v3
