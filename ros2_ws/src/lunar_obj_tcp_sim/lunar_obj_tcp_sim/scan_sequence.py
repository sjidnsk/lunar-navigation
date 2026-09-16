"""Observation bootstrap using bounded, formally planned rotations at live positions."""
from collections import deque
import math
from lunar_planning_msgs.action import NavigateToPose
from .observation_state import DeferredGoalCancellation


class ScanSequence:
    def __init__(self, client, *, enabled=True, step_deg=30., drift_m=.1, timeout_s=45.):
        if not math.isfinite(step_deg) or not 5<=step_deg<=90:
            raise ValueError('scan_step_deg must be between 5 and 90 degrees')
        if not all(math.isfinite(v) and v>0 for v in (drift_m,timeout_s)):
            raise ValueError('scan drift threshold and timeout must be finite and positive')
        self.client=client
        self.enabled=enabled
        self.step_deg=step_deg
        self.drift_m=drift_m
        self.timeout_s=timeout_s
        self.phase='OBSERVING'
        self.reason=''
        self.targets=None
        self.future=None
        self.handle=None
        self.cancel_token=None
        self.anchor=None
        self.reanchoring=False
        self.reanchors=0
        self.heading_since=None
        self.stopped_since=None

    def fail(self, reason):
        self.reason=reason
        self.phase='BOOTSTRAP_FAILED'
        if self.cancel_token is not None:
            self.cancel_token.cancel()
        return self.phase

    def advance(self, odometry, command_stopped, now):
        if self.phase=='BOOTSTRAP_FAILED':
            return self.phase
        if not self.enabled:
            self.phase='SCAN_COMPLETE'
            return self.phase
        p=odometry.pose.pose.position
        q=odometry.pose.pose.orientation
        twist=odometry.twist.twist
        stopped=(command_stopped and math.hypot(twist.linear.x,twist.linear.y)<.02
                 and abs(twist.angular.z)<.03)
        if not stopped:
            self.stopped_since=None
        elif self.stopped_since is None:
            self.stopped_since=now
        settled=self.stopped_since is not None and now-self.stopped_since>=.3
        if self.heading_since is not None and now-self.heading_since>self.timeout_s:
            return self.fail('INITIAL_SCAN_TIMEOUT')
        if self.targets is None:
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            count=math.ceil(360/self.step_deg)
            self.targets=deque(yaw+2*math.pi*i/count for i in range(1,count+1))
        if self.future is not None:
            if self.handle is None:
                if not self.future.done():
                    return self.phase
                response=self.future.result()
                if not response.accepted:
                    return self.fail('INITIAL_SCAN_REJECTED')
                self.handle=response
                self.future=response.get_result_async()
                self.phase='EXECUTING_SCAN'
                return self.phase
            if not self.reanchoring and math.hypot(p.x-self.anchor[0],p.y-self.anchor[1])>self.drift_m:
                self.reanchoring=True
                self.cancel_token.cancel()
                self.phase='STOPPING_SCAN'
            if not self.future.done():
                return self.phase
            result=self.future.result().result
            if self.reanchoring:
                if result.outcome not in (NavigateToPose.Result.GOAL_REACHED,NavigateToPose.Result.CANCELED):
                    return self.fail(result.reason_code)
            elif result.outcome!=0 or result.reason_code!='GOAL_REACHED':
                return self.fail(result.reason_code)
            if not settled:
                self.phase='STOPPING_SCAN'
                return self.phase
            self.future=self.handle=None
            if self.reanchoring:
                self.reanchors+=1
                self.reanchoring=False
                # Keep this target and deadline. Slippage must not reset the
                # timeout forever or be counted as a completed observation turn.
            else:
                self.targets.popleft()
                self.heading_since=None
        if not self.targets:
            self.phase='SCAN_COMPLETE'
            return self.phase
        if not settled:
            self.phase='STOPPING_SCAN'
            return self.phase
        goal=NavigateToPose.Goal(target_x_m=p.x,target_y_m=p.y,
                                has_target_yaw=True,target_yaw_rad=self.targets[0])
        # Navigation must certify this new start/goal from its fine map. There is
        # no direct Twist publisher or assumed-FREE patch in the scan adapter.
        self.future=self.client.send_goal_async(goal)
        self.cancel_token=DeferredGoalCancellation(self.future)
        self.anchor=(p.x,p.y)
        if self.heading_since is None:
            self.heading_since=now
        self.phase='WAITING_SCAN_ACCEPTANCE'
        return self.phase
