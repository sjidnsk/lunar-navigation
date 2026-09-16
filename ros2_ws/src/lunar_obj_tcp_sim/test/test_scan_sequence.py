from concurrent.futures import Future
from types import SimpleNamespace as NS
import math
from nav_msgs.msg import Odometry
from lunar_obj_tcp_sim.scan_sequence import ScanSequence


def pose(x=0.,yaw=0.,w=0.):
    o=Odometry();o.pose.pose.position.x=x
    o.pose.pose.orientation.z=math.sin(yaw/2);o.pose.pose.orientation.w=math.cos(yaw/2)
    o.twist.twist.angular.z=w
    return o


class Client:
    def __init__(self): self.goals=[];self.results=[];self.canceled=0
    def send_goal_async(self,goal):
        self.goals.append(goal);result=Future();self.results.append(result)
        def cancel():
            self.canceled+=1;f=Future();f.set_result(NS());return f
        handle=NS(accepted=True,get_result_async=lambda:result,cancel_goal_async=cancel)
        accepted=Future();accepted.set_result(handle);return accepted


def tick_ready(scan,now=0.,x=0.):
    scan.advance(pose(x),True,now);scan.advance(pose(x),True,now+.4)
    scan.advance(pose(x),True,now+.5)


def test_slip_waits_for_cancel_terminal_and_stop_then_reanchors_same_yaw():
    client=Client();scan=ScanSequence(client)
    tick_ready(scan)
    yaw=client.goals[0].target_yaw_rad
    scan.advance(pose(.15,w=.1),False,1.)
    assert client.canceled==1 and len(client.goals)==1
    scan.advance(pose(.16),True,1.5)
    assert len(client.goals)==1 # no overlapping navigation before Action terminal
    client.results[0].set_result(NS(result=NS(outcome=5,reason_code='CANCELED')))
    scan.advance(pose(.17,w=.1),False,1.6)
    assert len(client.goals)==1
    scan.advance(pose(.18),True,2.)
    scan.advance(pose(.18),True,2.4)
    assert len(client.goals)==2
    assert client.goals[1].target_x_m==.18
    assert client.goals[1].target_yaw_rad==yaw
    assert scan.reanchors==1


def test_new_anchor_rejected_by_formal_planner_does_not_continue():
    client=Client();scan=ScanSequence(client);tick_ready(scan)
    client.results[0].set_result(NS(result=NS(outcome=1,reason_code='START_BLIND_ZONE_UNRESOLVED')))
    assert scan.advance(pose(),True,1.)=='BOOTSTRAP_FAILED'
    scan.advance(pose(.3),True,2.)
    assert len(client.goals)==1


def test_next_segment_uses_measured_position_not_original_start():
    client=Client();scan=ScanSequence(client);tick_ready(scan)
    client.results[0].set_result(NS(result=NS(outcome=0,reason_code='GOAL_REACHED')))
    scan.advance(pose(.05,yaw=.52),True,1.)
    assert len(client.goals)==2 and client.goals[1].target_x_m==.05
    assert client.goals[1].target_yaw_rad>client.goals[0].target_yaw_rad
