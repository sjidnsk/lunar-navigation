"""UE path wire payloads and the same stationary P4/P3 coordinate anchor.
No sockets or ROS nodes; used inside the existing feedback adapter.
"""
import math
import struct
import numpy as np

MAX_POINTS = 8192

def decode_request(data):
    if len(data) != 16:
        raise ValueError('PATH_REQUEST must be 16 bytes')
    request_id, *xyz = struct.unpack('<I3f', bytes(data))
    if not all(math.isfinite(v) for v in xyz):
        raise ValueError('Nonfinite UE target')
    return request_id, tuple(xyz)

def encode_response(request_id, status, points):
    if not 0 <= request_id <= 0xffffffff or status not in (0, 1, 2):
        raise ValueError('Invalid response header')
    if len(points) > MAX_POINTS or (status == 0) != bool(points):
        raise ValueError('Invalid response points/status')
    result = bytearray(struct.pack('<IB3xI', request_id, status, len(points)))
    for p in points:
        if len(p) != 3 or not all(math.isfinite(v) and abs(v) <= 3.4028234e38 for v in p):
            raise ValueError('Invalid path point')
        result.extend(struct.pack('<3f', *p))
    return bytes(result)

class Coordinates:
    def __init__(self, theta, offset, map_translation, map_quaternion):
        values = (theta, *offset, *map_translation, *map_quaternion)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite transform')
        q = np.asarray(map_quaternion, dtype=float)
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            raise ValueError('Invalid map transform quaternion')
        x,y,z,w = q/norm
        self.map_r = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        c,s = math.cos(theta), math.sin(theta)
        self.anchor_r = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        self.offset = np.array(offset)
        self.map_t = np.array(map_translation)

    def to_map(self, ue_cm):
        # UE world LH -> metre RH -> anchored odom -> map.
        p = np.asarray(ue_cm) * np.array([.01,-.01,.01])
        return tuple(self.map_r @ (self.anchor_r @ p + self.offset) + self.map_t)

    def to_ue(self, map_m):
        p = self.anchor_r.T @ (self.map_r.T @ (np.asarray(map_m)-self.map_t)-self.offset)
        return tuple(p * np.array([100.,-100.,100.]))

class ReferenceGate:
    def __init__(self, session):
        self.session, self.revision, self.state = session, -1, None

    def accept(self, session, revision, state):
        if session != self.session or state not in (0,1) or revision < self.revision:
            return False
        if revision == self.revision and (state == self.state or self.state == 1):
            return False
        self.revision, self.state = revision, state
        return True

def height_at(message, x, y):
    """Read P3 GridMap without treating NaN as zero or changing traversability."""
    try:
        if message.header.frame_id != 'map': return None
        info = message.info
        q = info.pose.orientation
        if max(abs(q.x),abs(q.y),abs(q.z)) > 1e-6 or abs(abs(q.w)-1) > 1e-6: return None
        r = info.resolution
        if not math.isfinite(r) or r <= 0: return None
        w,h = round(info.length_x/r),round(info.length_y/r)
        ix = math.floor((x-(info.pose.position.x-info.length_x/2))/r)
        iy = math.floor((y-(info.pose.position.y-info.length_y/2))/r)
        if not (0 <= ix < w and 0 <= iy < h): return None
        layer = message.data[message.layers.index('elevation')]
        a,b = layer.layout.dim
        row = (w-1-ix+message.outer_start_index)%w
        col = (h-1-iy+message.inner_start_index)%h
        if (a.label,b.label)==('column_index','row_index') and (a.size,b.size)==(h,w):
            index = col*w+row
        elif (a.label,b.label)==('row_index','column_index') and (a.size,b.size)==(w,h):
            index = row*h+col
        else: return None
        if a.stride != w*h or b.stride != b.size: return None
        value = float(layer.data[layer.layout.data_offset+index])
        return value if math.isfinite(value) else None
    except (ValueError,IndexError,AttributeError,OverflowError):
        return None
