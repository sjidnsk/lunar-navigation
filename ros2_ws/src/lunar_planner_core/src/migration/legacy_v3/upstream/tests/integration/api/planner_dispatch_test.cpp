#include <memory>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/api/planner_v3.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

namespace lpp = lunar::planning::v3;
namespace {

TEST(PlannerDispatchTest,
     InvalidRequestFailsBeforeProjectionOrPlatformDispatch) {
  const lpp::SemanticValidator validator;
  const lpp::EmptyContractObjectRegistry registry;
  lpp::SafeProjectionCache projection_cache{4U};
  const std::unique_ptr<lpp::PlannerV3> planner =
      lpp::MakeDefaultPlannerV3(
          validator, registry, projection_cache);

  lpp::PlanningRequest request;
  request.request_id = "invalid-request";
  request.request_time.clock_id = "mission";
  request.state_time.clock_id = "mission";
  request.frame_id = "map";

  const lpp::PlanningResponse response = planner->Plan(request);

  EXPECT_EQ(
      response.planning_outcome,
      lpp::PlanningOutcome::kInvalidRequest);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
  EXPECT_TRUE(projection_cache.Keys().empty());
}

}  // namespace
