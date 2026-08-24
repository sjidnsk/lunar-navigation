#pragma once

#include <functional>
#include <memory>
#include <rclcpp/executor.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

#include "lunar_pure_planner_core/planner.hpp"

namespace lunar::pure_planner_ros {

using PlannerFn = std::function<lunar::pure_planning::PlanningResult(
    const lunar::pure_planning::PlanningRequest&)>;
using LocalPlannerFn = std::function<lunar::pure_planning::LocalStageResult(
    const lunar::pure_planning::PlanningRequest&,
    const lunar::pure_planning::LocalGoalSet&,
    lunar::pure_planning::SearchControl)>;

[[nodiscard]] std::unique_ptr<rclcpp::Executor> MakePurePlannerExecutor();

[[nodiscard]] PlannerFn RealPlannerFn();
[[nodiscard]] LocalPlannerFn RealLocalPlannerFn();

class PurePlanMotionServer final : public rclcpp::Node {
 public:
  explicit PurePlanMotionServer(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{},
      PlannerFn planner = RealPlannerFn(),
      LocalPlannerFn local_planner = RealLocalPlannerFn());
  ~PurePlanMotionServer() override;

  PurePlanMotionServer(const PurePlanMotionServer&) = delete;
  PurePlanMotionServer& operator=(const PurePlanMotionServer&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::pure_planner_ros
