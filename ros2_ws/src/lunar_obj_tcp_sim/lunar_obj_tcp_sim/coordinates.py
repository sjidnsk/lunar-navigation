"""World: LH Z-up. VehicleActor local axes: X back, Y down, Z left.

Raw velocity axes are uncalibrated and are not used as ROS body twist.
Body velocities are derived from the converted pose.
"""
from dataclasses import replace
from collections import deque
import math
import numpy as np
from .geometry import rotation_matrix


def world_position(value):
    x, y, z = value
    return x, -y, z


def orientation(value):
    """R_ros = W R_wire B.T, W=diag(1,-1,1).

    B maps Actor vectors to ROS: (-x, z, -y). Quaternion sign is immaterial.
    """
    x, y, z, w = value
    scale = 1.0 / math.sqrt(2.0 * sum(v*v for v in value))
    return tuple(scale*v for v in (y+z, w+x, w-x, z-y))


def wire_orientation(value):
    """Inverse pose basis, used only by the local simulated TCP vehicle."""
    x, y, z, w = value
    scale = 1.0 / math.sqrt(2.0 * sum(v*v for v in value))
    return tuple(scale*v for v in (y-z, x-w, x+w, y+z))


def convert_feedback(feedback):
    """Convert pose only; velocity fields remain raw, not ROS twist."""
    return replace(feedback, position_m=world_position(feedback.position_m),
                   orientation_xyzw=orientation(feedback.orientation_xyzw))


class PoseRates:
    """Receive-time finite differences, expressed in current ROS body axes.

No remote timestamp exists. TCP batching/jitter affects this estimate; it is
commissioning evidence, not a calibrated replacement for physical feedback.
"""
    def __init__(self):
        self.previous = None
        self.linear_history = deque()

    def update(self, position, quaternion, now):
        p = np.asarray(position, dtype=float)
        r = rotation_matrix(quaternion)
        previous, self.previous = self.previous, (p, r, now)
        self.linear_history.append((p, now))
        if previous is None:
            return None
        old_p, old_r, old_time = previous
        dt = now-old_time
        if not 1e-4 < dt <= .5:
            self.linear_history.clear()
            self.linear_history.append((p, now))
            return None
        relative = old_r.T @ r
        skew = np.array([relative[2,1]-relative[1,2], relative[0,2]-relative[2,0], relative[1,0]-relative[0,1]])/2
        sine = np.linalg.norm(skew)
        angle = math.atan2(sine, np.clip((np.trace(relative)-1)/2, -1., 1.))
        if angle > math.pi-.01:
            return None  # axis is ambiguous at a half turn
        omega = skew * (angle/sine if sine > 1e-10 else 1.)/dt
        # A short displacement window avoids dividing TCP-bunched samples
        # by sub-millisecond receive intervals. Retain the bracketing sample.
        while len(self.linear_history) > 2 and self.linear_history[1][1] <= now-.2:
            self.linear_history.popleft()
        linear_p, linear_time = self.linear_history[0]
        return r.T @ ((p-linear_p)/(now-linear_time)), relative.T @ omega
