#pragma once

#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <nav2_core/global_planner.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

namespace lunar::planning::nav2 {

class LunarGlobalPlanner final : public nav2_core::GlobalPlanner {
 public:
  LunarGlobalPlanner();
  ~LunarGlobalPlanner() override;

  LunarGlobalPlanner(const LunarGlobalPlanner&) = delete;
  LunarGlobalPlanner& operator=(const LunarGlobalPlanner&) = delete;

  void configure(
      const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
      std::string name,
      std::shared_ptr<tf2_ros::Buffer> tf,
      std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;
  void cleanup() override;
  void activate() override;
  void deactivate() override;

  [[nodiscard]] nav_msgs::msg::Path createPlan(
      const geometry_msgs::msg::PoseStamped& start,
      const geometry_msgs::msg::PoseStamped& goal) override;

  [[nodiscard]] nav_msgs::msg::Path ConvertResult(
      const lunar_planning_msgs::action::PlanMotion::Result& result) const;
  void SetLatestOdometryForTesting(const nav_msgs::msg::Odometry& odometry);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::planning::nav2
