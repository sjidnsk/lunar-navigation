#pragma once

#include <functional>
#include <memory>
#include <rclcpp/executor.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planner_ros {

using PlannerFn = std::function<lunar::pure_planning::PlanningResult(
    const lunar::pure_planning::PlanningRequest&)>;

[[nodiscard]] std::unique_ptr<rclcpp::Executor> MakePurePlannerExecutor();

[[nodiscard]] PlannerFn RealPlannerFn();

class PurePlanMotionServer final : public rclcpp::Node {
 public:
  explicit PurePlanMotionServer(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions{},
      PlannerFn planner = RealPlannerFn());
  ~PurePlanMotionServer() override;

  PurePlanMotionServer(const PurePlanMotionServer&) = delete;
  PurePlanMotionServer& operator=(const PurePlanMotionServer&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::pure_planner_ros
