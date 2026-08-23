#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/local_planning_problem.hpp"
#include "lunar_pure_planner_core/types/planner_io.hpp"
#include "wheel/wheel_planner.hpp"

namespace {
using namespace lunar::pure_planning;
constexpr std::size_t W = 12, H = 8;
constexpr double P = 52, M = 42;
struct Scenario { std::string name; std::vector<std::size_t> obstacles; };

Quaternion Yaw(double yaw) { return {.w = std::cos(yaw / 2), .z = std::sin(yaw / 2)}; }

GridMap Map(const Scenario& scenario) {
  const std::size_t count = W * H;
  std::vector<std::uint8_t> obstacle(count, 0);
  for (auto index : scenario.obstacles) { if (index >= count) throw std::runtime_error("bad obstacle"); obstacle[index] = 1; }
  return {.frame_id = "odom", .stamp = {.nanoseconds_since_epoch = 1'000'000'000}, .width = W, .height = H, .resolution_m = 1.0,
    .layers = {{"elevation", {.values = std::vector<float>(count, 0)}}, {"valid_mask", {.values = std::vector<std::uint8_t>(count, 1)}},
      {"obstacle", {.values = std::move(obstacle)}}, {"obstacle_height", {.values = std::vector<float>(count, 0)}},
      {"observation_age_s", {.values = std::vector<float>(count, 0)}}, {"observation_quality", {.values = std::vector<float>(count, 1)}},
      {"elevation_variance", {.values = std::vector<float>(count, 0)}}, {"obstacle_variance", {.values = std::vector<float>(count, 0)}},
      {"observation_count", {.values = std::vector<std::uint32_t>(count, 1)}}, {"forbidden", {.values = std::vector<std::uint8_t>(count, 0)}}}};
}

WheeledCapability Capability() {
  return {.footprint_xy_m = {{-.591,-.409},{.591,-.409},{.591,.409},{-.591,.409}}, .body_extent_m = {1.182,.818,1.29996},
    .wheel_diameter_m=.319, .wheel_width_m=.148, .wheelbase_m=.8175, .track_width_m=.67, .minimum_underbody_clearance_m=.21,
    .maximum_local_obstacle_relief_m=.2, .minimum_body_z_m=-.1, .maximum_body_z_m=.5, .maximum_forward_speed_mps=1, .maximum_reverse_speed_mps=.8,
    .maximum_spin_rate_radps=1, .maximum_acceleration_mps2=1, .maximum_braking_deceleration_mps2=1, .maximum_yaw_acceleration_radps2=1,
    .maximum_lateral_acceleration_mps2=1, .maximum_curvature_per_m=10, .maximum_slope_rad=.5,
    .motion_primitives = {{.primitive_id="forward", .kind=WheelPrimitiveKind::kForward, .relative_end_pose={.position_m={1,0,0}}},
      {.primitive_id="left", .kind=WheelPrimitiveKind::kForwardArc, .relative_end_pose={.position_m={1,1,0},.orientation=Yaw(1.5707963267948966)}},
      {.primitive_id="right", .kind=WheelPrimitiveKind::kForwardArc, .relative_end_pose={.position_m={1,-1,0},.orientation=Yaw(-1.5707963267948966)}},
      {.primitive_id="reverse", .kind=WheelPrimitiveKind::kReverse, .relative_end_pose={.position_m={-1,0,0}}},
      {.primitive_id="spin-left", .kind=WheelPrimitiveKind::kSpinCounterclockwise, .relative_end_pose={.orientation=Yaw(1.5707963267948966)}},
      {.primitive_id="spin-right", .kind=WheelPrimitiveKind::kSpinClockwise, .relative_end_pose={.orientation=Yaw(-1.5707963267948966)}}}};
}

hierarchical::LocalPlanningProblem Problem(const Scenario& s) {
  PlannerConfig config; config.wheel.xy_resolution_m = 1; config.optimization.maximum_iterations = 0;
  return {.request_id="offline-"+s.name, .platform_id="wheel", .capability_version="offline-v1", .local_map_generation=1,
    .state_time={.nanoseconds_since_epoch=1'000'000'000}, .current_state=WheeledState{.pose={.position_m={2.5,3.5,0}}},
    .goal_odom={.goal_id="goal",.target=PointGoal{.position_m={4.5,3.5,0},.tolerance_m=.2}}, .local_map=Map(s),
    .search_domain=hierarchical::LocalSearchDomain(W,H,std::vector<std::uint8_t>(W*H,1)), .capability=Capability(), .config=std::move(config)};
}

std::pair<double,double> Pt(const Vec3& p) { return {M+p.x*P, M+(H-p.y)*P}; }
void Draw(const std::filesystem::path& file, const Scenario& s, const PlannerOutput& o) {
  std::ofstream f(file); if (!f) throw std::runtime_error("cannot write svg"); const double width=2*M+W*P, height=2*M+H*P+38;
  f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""<<width<<"\" height=\""<<height<<"\"><title>Offline pure planner "<<s.name<<"</title><rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
  for(std::size_t y=0;y<H;++y) for(std::size_t x=0;x<W;++x) { bool hit=std::find(s.obstacles.begin(),s.obstacles.end(),y*W+x)!=s.obstacles.end(); f<<"<rect x=\""<<M+x*P<<"\" y=\""<<M+(H-1-y)*P<<"\" width=\""<<P<<"\" height=\""<<P<<"\" fill=\""<<(hit?"#111827":"#f8fafc")<<"\" stroke=\"#cbd5e1\"/>\n"; }
  if(o.reference) if(const auto* path=std::get_if<TrajectoryReference>(&o.reference->data)){ f<<"<polyline points=\""; for(const auto& q:path->points){auto [x,y]=Pt(q.pose.position_m);f<<x<<','<<y<<' ';} f<<"\" fill=\"none\" stroke=\"#1d4ed8\" stroke-width=\"5\"/>\n"; }
  auto[sx,sy]=Pt({2.5,3.5,0});auto[gx,gy]=Pt({4.5,3.5,0}); f<<"<circle cx=\""<<sx<<"\" cy=\""<<sy<<"\" r=\"9\" fill=\"#16a34a\"/><circle cx=\""<<gx<<"\" cy=\""<<gy<<"\" r=\"9\" fill=\"#dc2626\"/><text x=\"42\" y=\"24\" font-size=\"18\">"<<s.name<<" — "<<o.reason_code<<"</text>"; if(!o.reference) f<<"<text x=\"42\" y=\""<<height-12<<"\" fill=\"#b91c1c\" font-size=\"16\">No trajectory returned</text>"; f<<"</svg>\n";
}
}
int main(int argc,char**argv) try { if(argc!=2){std::cerr<<"usage: offline_wheel_demo OUTPUT_DIRECTORY\n";return 2;} std::filesystem::create_directories(argv[1]); wheel::WheelPlanner planner; for(const Scenario&s:std::vector<Scenario>{{"direct",{}},{"detour",{3*W+3}},{"no-path",{2*W+3,3*W+3,4*W+3}}}){auto o=planner.Plan(Problem(s));Draw(std::filesystem::path(argv[1])/(s.name+".svg"),s,o);std::cout<<s.name<<": "<<o.reason_code<<'\n';}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
