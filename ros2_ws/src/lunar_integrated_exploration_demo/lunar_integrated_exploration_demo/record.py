"""Record progress of the real integrated stack without driving or resetting it."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from diagnostic_msgs.msg import DiagnosticArray
from lunar_planning_msgs.msg import PathReference, TrackingStatus
from lunar_pure_exploration_msgs.msg import PureExplorationStatus

P='/lunar_demo/integrated'


class Recorder(Node):
    def __init__(self):
        super().__init__('integrated_progress_recorder')
        self.plant={};self.status={};self.references={};self.completed=set();self.plan_found={}
        self.events=[];self.map_revisions=set();self.coarse_res=None;self.fine_res=None;self.status_first=None;self.bootstrap_ids=set()
        self.bootstrap_receipt_seen=False
        q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String,P+'/bootstrap_sessions',self.on_bootstrap_sessions,q)
        self.create_subscription(String,P+'/plant_state',lambda m:setattr(self,'plant',json.loads(m.data)),10)
        self.create_subscription(PureExplorationStatus,P+'/exploration/status',self.on_status,10)
        self.create_subscription(PathReference,P+'/path_reference',self.on_reference,q)
        self.create_subscription(TrackingStatus,P+'/tracking_status',self.on_tracking,10)
        self.create_subscription(DiagnosticArray,P+'/planning/diagnostics',self.on_diag,10)
        self.create_subscription(OccupancyGrid,P+'/exploration_map',lambda m:setattr(self,'coarse_res',m.info.resolution),q)
        self.create_subscription(OccupancyGrid,P+'/planning/fine_state',lambda m:setattr(self,'fine_res',m.info.resolution),q)

    def on_bootstrap_sessions(self,m):
        self.bootstrap_ids=set(json.loads(m.data))
        self.bootstrap_receipt_seen=True

    def on_status(self,m):
        self.status={k:getattr(m,k)for k in ['task_id','state','reason_code','coverage_ratio','frontier_cluster_count','candidate_count','reachable_candidate_count','failed_candidate_count','completed_goal_count','known_free_area_m2','known_occupied_area_m2','unknown_area_m2','outside_map_area_m2']}
        q=m.current_goal.orientation
        self.status['goal_xyyaw']=[m.current_goal.position.x,m.current_goal.position.y,
                                  math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
        if m.task_id and self.status_first is None:self.status_first=dict(self.status)
        item={k:self.status[k]for k in ['state','reason_code','completed_goal_count','candidate_count','reachable_candidate_count','coverage_ratio','goal_xyyaw']}
        if not self.events or item!=self.events[-1]:self.events.append(item)

    def on_reference(self,m):
        if m.state==PathReference.ACTIVE and m.path.poses:
            key=(bytes(m.session_id.uuid).hex(),int(m.segment_revision))
            self.references[key]={'final':m.reaches_final_goal,'points':len(m.path.poses),'fine_revision':int(m.traversability_revision)}

    def on_tracking(self,m):
        if m.state==TrackingStatus.COMPLETED and all(math.isfinite(v)for v in [m.linear_speed_mps,m.angular_speed_radps]) and abs(m.linear_speed_mps)<=.01 and abs(m.angular_speed_radps)<=.03:
            self.completed.add((bytes(m.session_id.uuid).hex(),int(m.segment_revision)))

    def on_diag(self,m):
        for status in m.status:
            d={x.key:x.value for x in status.values}
            if d.get('cycle_result')=='PLAN_FOUND' and d.get('path_state')=='ACTIVE':
                self.plan_found[(d['session_id'],int(d['segment_revision']))]=int(d['fine_traversability_revision'])
            if 'fine_traversability_revision'in d:self.map_revisions.add(int(d['fine_traversability_revision']))

    def gates(self,minimum_goals,terminal):
        confirmed={k for k,v in self.references.items() if v['final'] and k[0] not in self.bootstrap_ids and self.plan_found.get(k)==v['fine_revision']}
        stopped_completions={k[0] for k in confirmed&self.completed}
        p=self.plant
        raw=p.get('raw_limit_violations',p.get('raw_command_limit_violations',1))
        return dict(bootstrap_receipt_received=self.bootstrap_receipt_seen,
                    multiple_exploration_goals=self.status.get('completed_goal_count',0)>=minimum_goals,
                    formal_plan_and_stopped_feedback=len(stopped_completions)>=minimum_goals,
                    coarse_1m=self.coarse_res is not None and abs(self.coarse_res-1)<1e-5,
                    fine_02m=self.fine_res is not None and abs(self.fine_res-.2)<1e-5,
                    map_revisions_advanced=len(self.map_revisions)>=5,
                    no_raw_speed_violation_or_collision=bool(p) and raw==0 and p.get('raw_invalid_commands',1)==0 and p.get('raw_angular_limit_violations',1)==0 and p.get('collisions',1)==0 and all(math.isfinite(p.get(k,float('nan'))) and p[k]<=.2+1e-9 for k in ['max_actual_forward','max_actual_reverse']),
                    requested_terminal_condition=(not terminal or self.status.get('state')==PureExplorationStatus.COMPLETED))


def main(args=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True);parser.add_argument('--timeout',type=float,default=180.)
    parser.add_argument('--minimum-goals',type=int,default=5);parser.add_argument('--require-completed',action='store_true')
    opts,rosargs=parser.parse_known_args(args)
    rclpy.init(args=rosargs);node=Recorder();started=time.monotonic();passed=False
    try:
        while rclpy.ok() and time.monotonic()-started<opts.timeout:
            rclpy.spin_once(node,timeout_sec=.1)
            if all(node.gates(opts.minimum_goals,opts.require_completed).values()):passed=True;break
    except KeyboardInterrupt:pass
    finally:
        output=dict(schema='integrated-demo-progress/v1',passed=passed,wall_duration_s=time.monotonic()-started,
                    gates=node.gates(opts.minimum_goals,opts.require_completed),plant=node.plant,exploration=node.status,
                    initial_exploration=node.status_first,coarse_resolution=node.coarse_res,fine_resolution=node.fine_res,
                    references=[dict(session_id=k[0],revision=k[1],**v)for k,v in node.references.items()],
                    bootstrap_sessions=sorted(node.bootstrap_ids), terminal_required=opts.require_completed,
                    actual_terminal=node.status.get('state')==PureExplorationStatus.COMPLETED,
                    matching_stopped_completions=[list(k)for k in set(node.references)&set(node.plan_found)&node.completed if k[0] not in node.bootstrap_ids and node.references[k]['final'] and node.references[k]['fine_revision']==node.plan_found[k]],
                    distinct_map_revisions=len(node.map_revisions),events=node.events,
                    runtime=f"{os.environ.get('ROS_DISTRO', 'unknown')} command-driven simulation",
                    boundaries={'native_Orin':'NOT_RUN','vehicle':'NOT_RUN'})
        p=Path(opts.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({'output':str(p),'passed':passed,'gates':output['gates'],'exploration':node.status}))
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    raise SystemExit(0 if passed else 1)
