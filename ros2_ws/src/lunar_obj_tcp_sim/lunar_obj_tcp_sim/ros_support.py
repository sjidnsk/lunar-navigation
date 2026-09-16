"""Message adaptation for the existing elevation and exploration contracts."""
from array import array
import math

import numpy as np
from geometry_msgs.msg import Point32
from grid_map_msgs.msg import GridMap
from lunar_pure_exploration_msgs.msg import PureExplorationTask
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


def observation_grid_map(observation, stamp):
    values = np.asarray(observation.elevation, dtype=np.float32)
    resolution = float(observation.resolution)
    x, y = map(float, observation.origin_xy)
    if (values.ndim != 2 or not values.size or np.isinf(values).any()
            or not all(math.isfinite(v) for v in (x, y, resolution)) or resolution <= 0):
        raise ValueError('elevation must be a nonempty matrix of finite heights or NaN with valid geometry')
    height, width = values.shape
    msg = GridMap()
    msg.header.stamp, msg.header.frame_id = stamp, 'odom'
    msg.info.resolution = resolution
    msg.info.length_x, msg.info.length_y = width * resolution, height * resolution
    msg.info.pose.position.x = x + msg.info.length_x / 2
    msg.info.pose.position.y = y + msg.info.length_y / 2
    msg.info.pose.orientation.w = 1.0
    msg.layers, msg.basic_layers = ['elevation'], ['elevation']
    layer = Float32MultiArray()
    layer.layout.dim = [
        MultiArrayDimension(label='column_index', size=height, stride=width * height),
        MultiArrayDimension(label='row_index', size=width, stride=width)]
    # GridMap indices increase towards negative x/y, unlike our [y,x] raster.
    layer.data = array('f', values[::-1, ::-1].ravel())
    msg.data = [layer]
    return msg


def exploration_task(bounds, stamp, command=PureExplorationTask.START):
    xmin, ymin, xmax, ymax = map(float, bounds)
    msg = PureExplorationTask()
    msg.header.stamp, msg.header.frame_id = stamp, 'map'
    msg.task_id, msg.command = 'obj-tcp-exploration', command
    msg.boundary.points = [Point32(x=x, y=y, z=0.) for x, y in
                           ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))]
    return msg
