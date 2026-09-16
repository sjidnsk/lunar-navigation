"""Memory-mapped terrain and exact triangle line-of-sight for sampled surfaces.

The output is a single upper sampled surface, not a volumetric occupancy map.
Every finite observation corresponds to an actual retained triangle sample.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Observation:
    elevation: np.ndarray
    origin_xy: tuple
    resolution: float


def rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError('finite nonzero xyzw quaternion required')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class TerrainMap:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory/'metadata.json').read_text(encoding='utf-8'))
        if self.metadata['format_version'] != 1:
            raise ValueError('unsupported terrain format')
        self.resolution = float(self.metadata['resolution'])
        self.origin_xy = np.asarray(self.metadata['origin_xy'], dtype=float)
        self.elevation = np.load(self.directory/'elevation.npy', mmap_mode='r')
        self.sample_xy = np.load(self.directory/'sample_xy.npy', mmap_mode='r')
        self.triangles = np.memmap(self.directory/'triangles.bin', dtype='float32', mode='r',
                                   shape=(self.metadata['triangle_count'], 3, 3))
        self.index_offsets = np.load(self.directory/'index_offsets.npy', mmap_mode='r')
        self.index_ids = np.load(self.directory/'index_ids.npy', mmap_mode='r')

    def height_at(self, x, y):
        ix, iy = np.floor((np.array((x, y))-self.origin_xy)/self.resolution).astype(int)
        if 0 <= iy < self.elevation.shape[0] and 0 <= ix < self.elevation.shape[1]:
            return float(self.elevation[iy, ix])
        return float('nan')

    def iter_known_chunks(self, chunk_cells=512, center_xy=None):
        if chunk_cells < 1:
            raise ValueError('chunk_cells must be positive')
        center = (np.asarray(center_xy)-self.origin_xy)/self.resolution if center_xy is not None else None
        def boundaries(length, axis):
            anchor = 0 if center is None else int(np.floor(center[axis]))-chunk_cells//2
            return sorted(set([0, length, *range(anchor % chunk_cells, length, chunk_cells)]))
        xs, ys = boundaries(self.elevation.shape[1],0), boundaries(self.elevation.shape[0],1)
        blocks = [(x,y,x1,y1) for y,y1 in zip(ys,ys[1:]) for x,x1 in zip(xs,xs[1:])]
        if center is not None:
            blocks.sort(key=lambda b: ((b[0]+b[2])/2-center[0])**2+((b[1]+b[3])/2-center[1])**2)
        for x,y,x1,y1 in blocks:
            block = np.array(self.elevation[y:y1, x:x1])
            if np.isfinite(block).any():
                yield Observation(block, tuple(self.origin_xy+np.array((x,y))*self.resolution), self.resolution)

    def _local_triangles(self, lower, upper):
        index = self.metadata['index']
        ny, nx = index['shape']
        lo = np.maximum(0, np.floor((lower-self.origin_xy)/index['tile_size_m']).astype(int))
        hi = np.minimum((nx-1, ny-1), np.floor((upper-self.origin_xy)/index['tile_size_m']).astype(int))
        parts = []
        for y in range(lo[1], hi[1]+1):
            for x in range(lo[0], hi[0]+1):
                cell = y*nx+x
                parts.append(self.index_ids[self.index_offsets[cell]:self.index_offsets[cell+1]])
        if not parts:
            return np.empty((0, 3, 3), dtype=float)
        return np.asarray(self.triangles[np.unique(np.concatenate(parts))], dtype=float)

    def visible(self, sensor_origin, targets):
        """Segment/triangle intersections; endpoint contact is the observed surface.

        Exact Moller-Trumbore intersections after spatial-index and ray-AABB pruning.
        Coplanar grazing is intentionally unresolved; no free cells are synthesized.
        """
        sensor_origin = np.asarray(sensor_origin, dtype=float)
        targets = np.asarray(targets, dtype=float)
        visible = np.ones(len(targets), dtype=bool)
        if not len(targets):
            return visible
        directions = targets-sensor_origin
        lower = np.minimum(sensor_origin, targets.min(axis=0))
        upper = np.maximum(sensor_origin, targets.max(axis=0))
        triangles = self._local_triangles(lower[:2], upper[:2])
        mins, maxs = triangles.min(axis=1), triangles.max(axis=1)
        keep = np.all(maxs >= lower, axis=1) & np.all(mins <= upper, axis=1)
        triangles, mins, maxs = triangles[keep], mins[keep], maxs[keep]
        # Azimuth intervals reduce the candidate pairs before exact 3-D intersection.
        # Intervals crossing the branch cut fall back to the whole circle; this is
        # deliberately conservative, including triangles surrounding the sensor.
        azimuth = np.arctan2(directions[:, 1], directions[:, 0])
        ray_order = np.argsort(azimuth)
        sorted_angles = azimuth[ray_order]
        relative = triangles-sensor_origin
        angles = np.arctan2(relative[:, :, 1], relative[:, :, 0])
        angle_min, angle_max = angles.min(axis=1)-1e-8, angles.max(axis=1)+1e-8
        wrap = angle_max-angle_min >= np.pi
        angle_min[wrap], angle_max[wrap] = -np.pi-1e-8, np.pi+1e-8
        begin = np.searchsorted(sorted_angles, angle_min, side='left')
        end = np.searchsorted(sorted_angles, angle_max, side='right')
        ray_min = np.minimum(sensor_origin, targets)
        ray_max = np.maximum(sensor_origin, targets)
        for start in range(0, len(triangles), 128):
            stop = min(start+128, len(triangles))
            counts = end[start:stop]-begin[start:stop]
            cumulative = np.cumsum(counts)
            if not cumulative[-1]:
                continue
            ti = np.repeat(np.arange(start, stop), counts)
            ri = ray_order[np.repeat(begin[start:stop], counts)+np.arange(cumulative[-1])-np.repeat(cumulative-counts, counts)]
            valid = visible[ri] & np.all(ray_max[ri] >= mins[ti]-1e-8, axis=1) & np.all(ray_min[ri] <= maxs[ti]+1e-8, axis=1)
            ti, ri = ti[valid], ri[valid]
            if not len(ti):
                continue
            a = triangles[ti, 0]
            e1, e2 = triangles[ti, 1]-a, triangles[ti, 2]-a
            d = directions[ri]
            p = np.cross(d, e2)
            det = np.einsum('ij,ij->i', p, e1)
            valid = np.abs(det) > 1e-12
            inv = np.zeros(len(det))
            inv[valid] = 1/det[valid]
            offset = sensor_origin-a
            u = np.einsum('ij,ij->i', p, offset)*inv
            q = np.cross(offset, e1)
            v = np.einsum('ij,ij->i', d, q)*inv
            t = np.einsum('ij,ij->i', e2, q)*inv
            hit = valid & (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8) & (t > 1e-7) & (t < 1-1e-5)
            visible[ri[hit]] = False
        return visible

    def observe(self, position_xyz, orientation_xyzw, *, sensor_range_m=12.,
                sensor_fov_deg=120., near_field_radius_m=2.1,
                sensor_offset_xyz_m=(0.591, 0., 1.0), observation_window_m=28.):
        position = np.asarray(position_xyz, dtype=float)
        offset = np.asarray(sensor_offset_xyz_m, dtype=float)
        if position.shape != (3,) or offset.shape != (3,) or not np.all(np.isfinite([*position, *offset])):
            raise ValueError('finite XYZ position and sensor offset required')
        if not np.all(np.isfinite([sensor_range_m, sensor_fov_deg, near_field_radius_m, observation_window_m])) or sensor_range_m <= 0 or not 0 < sensor_fov_deg <= 360 or near_field_radius_m < 0 or observation_window_m <= 0:
            raise ValueError('invalid sensor configuration')
        rotation = rotation_matrix(orientation_xyzw)
        sensor = position+rotation @ offset
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        n = int(math.ceil(observation_window_m/self.resolution))
        # Quantize robot first. A sub-cell move never changes cell identity.
        center = np.floor((position[:2]-self.origin_xy)/self.resolution).astype(int)
        start = center-n//2
        origin = self.origin_xy+start*self.resolution
        output = np.full((n, n), np.nan, dtype='float32')
        low = np.maximum(start, 0)
        high = np.minimum(start+n, self.elevation.shape[::-1])
        if np.any(high <= low):
            return Observation(output, tuple(origin), self.resolution)
        values = np.asarray(self.elevation[low[1]:high[1], low[0]:high[0]])
        sy, sx = np.nonzero(np.isfinite(values))
        xy = np.asarray(self.sample_xy[low[1]+sy, low[0]+sx], dtype=float)
        z = values[sy, sx]
        delta = xy-sensor[:2]
        distance = np.linalg.norm(delta, axis=1)
        bearing = np.arctan2(delta[:, 1], delta[:, 0])-yaw
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        candidate = (distance <= sensor_range_m) & ((np.abs(bearing) <= math.radians(sensor_fov_deg/2)) | (distance <= near_field_radius_m))
        sx, sy, xy, z = sx[candidate], sy[candidate], xy[candidate], z[candidate]
        good = self.visible(sensor, np.column_stack((xy, z)))
        output[low[1]-start[1]+sy[good], low[0]-start[0]+sx[good]] = z[good]
        return Observation(output, tuple(origin), self.resolution)
