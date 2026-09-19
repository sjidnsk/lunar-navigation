// Offline calibration harness. Calls production terrain derivation and planner.
// Binary input: uint32 width,height; doubles resolution,origin_x,origin_y; float32 heights.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"
using namespace lunar::incremental_navigation;
template<class T> void Read(std::ifstream& f,T& x){f.read(reinterpret_cast<char*>(&x),sizeof(x));}
static double Ms(SteadyClock::time_point t){return std::chrono::duration<double,std::milli>(SteadyClock::now()-t).count();}
int main(int argc,char** argv){
 if(argc!=3)return 2;
 std::ifstream f(argv[1],std::ios::binary);std::uint32_t w,h;double res,ox,oy;
 Read(f,w);Read(f,h);Read(f,res);Read(f,ox);Read(f,oy);
 std::vector<float> z(w*h);f.read(reinterpret_cast<char*>(z.data()),z.size()*sizeof(float));if(!f)return 3;
 WheeledCapability cap{.footprint_xy_m={{.591,.409},{.591,-.409},{-.591,-.409},{-.591,.409}},
 .minimum_underbody_clearance_m=.21,.maximum_local_obstacle_relief_m=.2,
 .maximum_forward_speed_mps=.2,.maximum_reverse_speed_mps=.2,.maximum_spin_rate_radps=1.,
 .maximum_slope_rad=.3490658503988659,.minimum_clearance_m=.2};
 TraversabilityProfile profile{.planar_envelope_xy_m=cap.footprint_xy_m,.preferred_clearance_m=.2,.slope_weight=1.,.relief_weight=1.};
 PersistentElevationMap map;
 auto applied=map.Apply({.geometry={.frame_id="map",.width=w,.height=h,.resolution_m=res,.origin_m={ox,oy,0}},.elevation_m=z,.map_from_source={.parent_frame="map",.child_frame="map"}});
 if(applied.status!=ElevationUpdateResult::Status::kApplied)return 4;
 auto raw=map.Snapshot();auto begin=SteadyClock::now();auto zero=FineTraversabilityBuilder{}.Derive(raw,cap,profile);double zero_ms=Ms(begin);
 profile.clearance_weight=1.;begin=SteadyClock::now();auto one=FineTraversabilityBuilder{}.Derive(raw,cap,profile);double one_ms=Ms(begin);
 const auto& g=zero->geometry();auto lo=g.min_inclusive(),hi=g.max_exclusive();
 std::ofstream grid(argv[2],std::ios::binary);
 for(auto y=lo.y;y<hi.y;++y)for(auto x=lo.x;x<hi.x;++x){GridIndex i{x,y};auto state=static_cast<std::uint8_t>(zero->State(i));if(one->State(i)!=zero->State(i))return 5;grid.write(reinterpret_cast<char*>(&state),1);}
 for(int layer=0;layer<2;++layer)for(auto y=lo.y;y<hi.y;++y)for(auto x=lo.x;x<hi.x;++x){GridIndex i{x,y};double c=layer==0?zero->TraversalCost(i):one->TraversalCost(i)-zero->TraversalCost(i);grid.write(reinterpret_cast<char*>(&c),sizeof(c));}
 grid.close();std::cout<<std::setprecision(17)<<"{\"width\":"<<g.width()<<",\"height\":"<<g.height()<<",\"res\":"<<res<<",\"origin\":["<<g.origin_m().x+lo.x*res<<","<<g.origin_m().y+lo.y*res<<"],\"derive_zero_ms\":"<<zero_ms<<",\"derive_nonzero_ms\":"<<one_ms<<",\"hard_radius_m\":"<<zero->hard_inflation_radius_m()<<"}"<<std::endl;
 // Exact linearity of the production cost in weight: keep hard states fixed.
 std::map<double,std::shared_ptr<const FineTraversabilitySnapshot>> cache{{0.,zero},{1.,one}};
 WheelLocalPlanner planner(cap);double weight,sx,sy,tx,ty;
 while(std::cin>>weight>>sx>>sy>>tx>>ty){
  if(!cache.contains(weight)){
   FineTraversabilityTileDirectory dir(g);
   for(auto key:zero->tile_indices()){
    auto a=zero->FindTile(key),b=one->FindTile(key);auto costs=a->traversal_costs();
    for(std::size_t j=0;j<costs.size();++j)costs[j]+=weight*std::max(0.,b->TraversalCost(j)-a->TraversalCost(j));
    dir=dir.WithTile(key,std::make_shared<const FineTraversabilityTile>(a->states(),std::move(costs)));
   }
   cache[weight]=std::make_shared<const FineTraversabilitySnapshot>(g,raw->raw_elevation_revision(),1U,"calibration",zero->hard_inflation_radius_m(),.2,TraversalCostWeights{1.,1.,weight},raw,std::move(dir),std::vector<TileIndex>{},std::vector<TileIndex>{},FineSnapshotMetrics{});
  }
  Pose2 start{{sx,sy},0};RequestLocalPlanningView view(cache.at(weight),start,0.,{});
  LocalTarget target{.center={tx,ty},.position_tolerance_m=.01,.is_final_goal=true};
  std::vector<double> times;LocalPlanResult result;
  for(int repeat=0;repeat<3;++repeat){begin=SteadyClock::now();result=planner.Plan(view,start,target,SteadyClock::now()+std::chrono::seconds(5),{});times.push_back(Ms(begin));}
  std::sort(times.begin(),times.end());
  std::cout<<"{\"weight\":"<<weight<<",\"status\":"<<static_cast<int>(result.status)<<",\"median_ms\":"<<times[1]<<",\"max_ms\":"<<times[2]<<",\"expanded\":"<<result.statistics.expanded_states<<",\"postprocess_ms\":"<<std::chrono::duration<double,std::milli>(result.postprocess_elapsed).count();
  for(int k=0;k<2;++k){std::cout<<(k==0?",\"raw\":[":",\"path\":[");const auto& path=k==0?result.raw_path:result.path;for(std::size_t j=0;j<path.size();++j){if(j)std::cout<<",";std::cout<<"["<<path[j].pose.position_m.x<<","<<path[j].pose.position_m.y<<"]";}std::cout<<"]";}
  std::cout<<"}"<<std::endl;
 }
}
