#include <string>

#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>

#include "lunar_pure_exploration_sim/run_recorder.hpp"

int main() {
  using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;
  return lunar::pure_exploration_sim::ExplorationStateName(Status::COMPLETED) ==
                 std::string{"COMPLETED"}
             ? 0
             : 1;
}
