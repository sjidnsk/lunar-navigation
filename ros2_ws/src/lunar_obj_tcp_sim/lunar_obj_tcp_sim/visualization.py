"""Display-only geometry. Nothing produced here is a planning input."""
import math
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from .geometry import rotation_matrix


def marker(identity, kind, stamp, namespace='simulation'):
    value=Marker()
    value.header.frame_id, value.header.stamp='map',stamp
    value.ns,value.id,value.type,value.action=namespace,identity,kind,Marker.ADD
    value.pose.orientation.w=1.
    return value


def scene_markers(odometry, settings, bounds):
    stamp=odometry.header.stamp
    p,q=odometry.pose.pose.position,odometry.pose.pose.orientation
    rotation=rotation_matrix((q.x,q.y,q.z,q.w))
    sensor=np.array((p.x,p.y,p.z))+rotation@np.asarray(settings['sensor_offset_xyz_m'])
    yaw=math.atan2(rotation[1,0],rotation[0,0])
    body=marker(0,Marker.ARROW,stamp,'vehicle')
    body.pose=odometry.pose.pose
    body.scale.x,body.scale.y,body.scale.z=1.,.25,.25
    body.color.r,body.color.g,body.color.b,body.color.a=.2,.7,1.,1.
    origin=marker(0,Marker.SPHERE,stamp,'sensor_origin')
    origin.pose.position=Point(x=float(sensor[0]),y=float(sensor[1]),z=float(sensor[2]))
    origin.scale.x=origin.scale.y=origin.scale.z=.2
    origin.color.g=origin.color.a=1.
    fov=marker(0,Marker.LINE_STRIP,stamp,'fov_limit')
    fov.scale.x=.05;fov.color.g=fov.color.a=1.
    half=math.radians(settings['sensor_fov_deg'])/2
    radius=settings['sensor_range_m']
    fov.points=[origin.pose.position]
    fov.points += [Point(x=float(sensor[0]+radius*math.cos(a)),y=float(sensor[1]+radius*math.sin(a)),z=float(sensor[2]))
                   for a in np.linspace(yaw-half,yaw+half,49)]
    fov.points.append(origin.pose.position)
    def rectangle(namespace,extent,color):
        value=marker(0,Marker.LINE_STRIP,stamp,namespace)
        value.scale.x=.1
        value.color.r,value.color.g,value.color.b,value.color.a=color
        xmin,ymin,xmax,ymax=extent
        value.points=[Point(x=float(x),y=float(y),z=p.z+.02) for x,y in
                      ((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax),(xmin,ymin))]
        return value
    half_window=settings['local_window_size_m']/2
    window=rectangle('planning_window',(p.x-half_window,p.y-half_window,p.x+half_window,p.y+half_window),(.4,.5,1.,.8))
    boundary=rectangle('task_boundary',bounds,(1.,.15,.8,1.))
    xmin,ymin,xmax,ymax=bounds
    size=max(xmax-xmin,ymax-ymin)
    boundary.scale.x=max(.15,size/180)
    for point in boundary.points:
        point.z=p.z+2.
    label=marker(0,Marker.TEXT_VIEW_FACING,stamp,'task_label')
    label.pose.position=Point(x=(xmin+xmax)/2,y=ymax+size*.035,z=p.z+2.)
    label.scale.z=max(.4,size*.023)
    label.color.r,label.color.g,label.color.b,label.color.a=1.,.4,.9,1.
    label.text=f'EXPLORATION AREA  {xmax-xmin:g} m x {ymax-ymin:g} m'
    return MarkerArray(markers=[body,origin,fov,window,boundary,label])


def truth_preview(terrain, stamp):
    """Downsampled upper-surface preview; published only if explicitly displayed."""
    stride=max(1,math.ceil(max(terrain.elevation.shape)/150))
    sample=np.asarray(terrain.elevation[::stride,::stride])
    ys,xs=np.nonzero(np.isfinite(sample))
    value=marker(0,Marker.POINTS,stamp,'terrain_truth')
    value.scale.x=value.scale.y=terrain.resolution*stride
    value.color.r,value.color.g,value.color.b,value.color.a=.55,.35,.18,.45
    value.points=[Point(x=float(terrain.origin_xy[0]+(x*stride+.5)*terrain.resolution),
                        y=float(terrain.origin_xy[1]+(y*stride+.5)*terrain.resolution),
                        z=float(sample[y,x])) for y,x in zip(ys,xs)]
    return MarkerArray(markers=[value])
