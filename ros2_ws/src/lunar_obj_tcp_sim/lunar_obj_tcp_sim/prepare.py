"""Streaming triangular OBJ conversion; triangulate mesh before importing polygons."""
import argparse
from collections import defaultdict
import fnmatch
import json
import math
from pathlib import Path
import time

import numpy as np

DEFAULT_EXCLUDES = ('*sky*', '*camera*', '*cam_*', '*matineecam*', '*vehicle*',
                    '*lunarcar*', '*sensor*', '*lidar*', '*rover*', '*depthintensity*', '*eventactor*')


def axis_transform(axes, scale, offset):
    names = axes.split(',')
    if len(names) != 3 or sorted(n.lstrip('-+') for n in names) != ['x', 'y', 'z']:
        raise ValueError('axes must be a signed permutation, e.g. x,-z,y')
    if not np.isfinite(scale) or scale <= 0 or not np.all(np.isfinite(offset)):
        raise ValueError('scale must be positive and transform finite')
    matrix = np.zeros((3, 3))
    for row, name in enumerate(names):
        matrix[row, 'xyz'.index(name.lstrip('-+'))] = scale * (-1 if name.startswith('-') else 1)
    return matrix


def _intersects(triangles, bounds):
    """Triangle/rectangle SAT, including vertical projected segments."""
    center = (np.array(bounds[:2]) + bounds[2:]) / 2
    half = (np.array(bounds[2:]) - bounds[:2]) / 2
    p = triangles[:, :, :2] - center
    valid = np.all(p.min(axis=1) <= half, axis=1) & np.all(p.max(axis=1) >= -half, axis=1)
    for k in range(3):
        edge = p[:, (k+1) % 3] - p[:, k]
        axis = np.stack((-edge[:, 1], edge[:, 0]), axis=1)
        projection = np.einsum('nvi,ni->nv', p, axis)
        radius = np.abs(axis) @ half
        valid &= (projection.min(axis=1) <= radius) & (projection.max(axis=1) >= -radius)
    return valid


def _rasterize(triangles, output, shape, origin, resolution):
    height = np.lib.format.open_memmap(output / 'elevation.npy', mode='w+', dtype='float32', shape=shape)
    # XY samples preserve real boundary/vertical-face points rather than inventing cell-center surfaces.
    sample_xy = np.lib.format.open_memmap(output / 'sample_xy.npy', mode='w+', dtype='float32', shape=(*shape, 2))
    height[:] = np.nan
    sample_xy[:] = np.nan
    multilayer = 0
    ny, nx = shape

    def write(ix, iy, xyz):
        nonlocal multilayer
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, xyz = ix[valid], iy[valid], xyz[valid]
        old = height[iy, ix]
        multilayer += int(np.count_nonzero(np.isfinite(old) & (np.abs(old-xyz[:, 2]) > resolution)))
        use = ~np.isfinite(old) | (xyz[:, 2] > old)
        # Duplicates are sorted by height so the final write is the highest sampled surface.
        ix, iy, xyz = ix[use], iy[use], xyz[use]
        order = np.argsort(xyz[:, 2], kind='stable')
        height[iy[order], ix[order]] = xyz[order, 2]
        sample_xy[iy[order], ix[order]] = xyz[order, :2]

    for index, tri in enumerate(triangles):
        tri = np.asarray(tri, dtype='float64')
        lower = np.maximum(0, np.floor((tri[:, :2].min(axis=0)-origin)/resolution).astype(int))
        upper = np.minimum((nx-1, ny-1), np.floor((tri[:, :2].max(axis=0)-origin)/resolution).astype(int))
        a, b, c = tri
        den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den) > 1e-12:
            # Rows bounded independently: a giant triangle never allocates a whole-map temporary.
            for y0 in range(lower[1], upper[1]+1, 32):
                yy, xx = np.mgrid[y0:min(y0+32, upper[1]+1), lower[0]:upper[0]+1]
                x = origin[0]+(xx+.5)*resolution
                y = origin[1]+(yy+.5)*resolution
                u = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
                v = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
                valid = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
                z = u*a[2]+v*b[2]+(1-u-v)*c[2]
                write(xx[valid], yy[valid], np.stack((x[valid], y[valid], z[valid]), axis=1))
        # Edges are sampled at <= half-cell XY spacing. Include zero-projection vertical faces.
        # Only boundary cells without a valid cell-center intersection receive edge samples;
        # vertical triangles instead retain the highest real sampled surface in their cells.
        for start, end in ((a, b), (b, c), (c, a)):
            delta = end-start
            t0, t1 = 0., 1.
            for axis in range(2):
                if abs(delta[axis]) > 1e-15:
                    limits = sorted(((origin[axis]-start[axis])/delta[axis],
                                     (origin[axis]+shape[1-axis]*resolution-start[axis])/delta[axis]))
                    t0, t1 = max(t0, limits[0]), min(t1, limits[1])
            if t1 < t0:
                continue
            count = max(2, int(np.linalg.norm(delta[:2])*(t1-t0)/resolution*2)+2)
            points = start + np.linspace(t0, t1, count)[:, None]*delta
            ij = np.floor((points[:, :2]-origin)/resolution).astype(int)
            if abs(den) > 1e-12:
                centers = origin + (ij+.5)*resolution
                u = ((b[1]-c[1])*(centers[:, 0]-c[0])+(c[0]-b[0])*(centers[:, 1]-c[1]))/den
                v = ((c[1]-a[1])*(centers[:, 0]-c[0])+(a[0]-c[0])*(centers[:, 1]-c[1]))/den
                keep = (u < -1e-9) | (v < -1e-9) | (u+v > 1+1e-9)
                points, ij = points[keep], ij[keep]
            write(ij[:, 0], ij[:, 1], points)
        if index and index % 100000 == 0:
            print(f'rasterized {index} triangles', flush=True)
    height.flush()
    sample_xy.flush()
    return height, multilayer


def _build_index(triangles, output, origin, shape, resolution, tile_size=8.):
    nx = math.ceil(shape[1]*resolution/tile_size)
    ny = math.ceil(shape[0]*resolution/tile_size)
    counts = np.zeros(nx*ny, dtype=np.int64)
    lows = np.maximum(0, np.floor((triangles[:, :, :2].min(axis=1)-origin)/tile_size).astype(int))
    highs = np.minimum((nx-1, ny-1), np.floor((triangles[:, :, :2].max(axis=1)-origin)/tile_size).astype(int))
    for lo, hi in zip(lows, highs):
        for y in range(lo[1], hi[1]+1):
            counts[y*nx+lo[0]:y*nx+hi[0]+1] += 1
    offsets = np.concatenate(([0], np.cumsum(counts)))
    ids = np.lib.format.open_memmap(output/'index_ids.npy', mode='w+', dtype='int32', shape=(int(offsets[-1]),))
    cursor = offsets[:-1].copy()
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        for y in range(lo[1], hi[1]+1):
            bins = np.arange(y*nx+lo[0], y*nx+hi[0]+1)
            ids[cursor[bins]] = i
            cursor[bins] += 1
    ids.flush()
    np.save(output/'index_offsets.npy', offsets)
    return dict(tile_size_m=tile_size, shape=[ny, nx])


def prepare_map(obj, output, *, center, size=1000., resolution=.2, halo=12.,
                scale=.01, axes='x,-z,y', offset=(0., 0., 0.),
                excludes=DEFAULT_EXCLUDES, includes=(), preset='explicit-transform'):
    obj, output = Path(obj), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty; existing maps are never overwritten')
    if not np.all(np.isfinite([*center, size, resolution, halo])) or size <= 0 or resolution <= 0 or halo < 0:
        raise ValueError('finite center, positive size/resolution and nonnegative halo required')
    matrix = axis_transform(axes, scale, offset)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    origin = np.floor((np.asarray(center)-size/2-halo)/resolution)*resolution
    shape = tuple(np.ceil((np.asarray(center)+size/2+halo-origin)/resolution).astype(int)[::-1])
    bounds = [*origin, *(origin+np.array(shape[::-1])*resolution)]
    vertex_file, tri_file = output/'vertices.tmp', output/'triangles.bin'
    count = 0
    buffer = []
    with obj.open('r', encoding='utf-8', errors='strict') as src, vertex_file.open('wb') as dst:
        for line in src:
            if line.startswith('v '):
                buffer.append(line[2:])
                count += 1
                if len(buffer) >= 65536:
                    values = np.array([row.split()[:3] for row in buffer], dtype=np.float64)
                    ((values @ matrix.T)+offset).astype('float32').tofile(dst)
                    buffer.clear()
        if buffer:
            values = np.array([row.split()[:3] for row in buffer], dtype=np.float64)
            ((values @ matrix.T)+offset).astype('float32').tofile(dst)
    if not count:
        raise ValueError('OBJ has no vertices')
    vertices = np.memmap(vertex_file, mode='r', dtype='float32', shape=(count, 3))
    groups = defaultdict(lambda: dict(kept=True, faces=0, retained_triangles=0))
    def selected(group_name, parent_name):
        names = [group_name.lower(), parent_name.lower()]
        included = not includes or any(fnmatch.fnmatch(name, p.lower()) for name in names for p in includes)
        excluded = any(fnmatch.fnmatch(name, p.lower()) for name in names for p in excludes)
        return included and not excluded

    group, seen_vertices, kept_count = 'default', 0, 0
    object_name = ''
    keep = selected(group, object_name)
    groups[group]['kept'] = keep
    face_buffer = []
    def flush(dst):
        nonlocal kept_count
        if not face_buffer:
            return
        triangles = vertices[np.asarray(face_buffer, dtype=np.int64)]
        valid = _intersects(triangles, bounds)
        triangles[valid].astype('float32').tofile(dst)
        n = int(valid.sum())
        kept_count += n
        groups[group]['retained_triangles'] += n
        face_buffer.clear()
    with obj.open('r', encoding='utf-8') as src, tri_file.open('wb') as dst:
        for line in src:
            if line.startswith('v '):
                seen_vertices += 1
            elif line.startswith(('g ', 'o ')):
                flush(dst)
                group = line[2:].strip()
                if line.startswith('o '):
                    object_name = group
                keep = selected(group, object_name)
                groups[group]['kept'] = keep
            elif line.startswith('f '):
                groups[group]['faces'] += 1
                if not keep:
                    continue
                fields = line[2:].split('#', 1)[0].split()
                if len(fields) != 3:
                    raise ValueError(f'unsupported OBJ face with {len(fields)} vertices in {group!r}; triangulate mesh before importing')
                indices = [int(p.split('/')[0]) for p in fields]
                if any(i == 0 or i > count or i < -seen_vertices for i in indices):
                    raise ValueError('invalid OBJ vertex index')
                indices = [i-1 if i > 0 else seen_vertices+i for i in indices]
                face_buffer.append(tuple(indices))
                if len(face_buffer) >= 65536:
                    flush(dst)
        flush(dst)
    del vertices
    vertex_file.unlink()
    if not kept_count:
        raise ValueError('no retained triangles intersect ROI; check explicit transform and center')
    triangles = np.memmap(tri_file, mode='r', dtype='float32', shape=(kept_count, 3, 3))
    print(f'kept {kept_count} triangles from {count} vertices; rasterizing', flush=True)
    height, multilayer = _rasterize(triangles, output, shape, origin, resolution)
    index = _build_index(triangles, output, origin, shape, resolution)
    metadata = dict(format_version=1, source=str(obj.resolve()), source_bytes=obj.stat().st_size,
                    origin_xy=origin.tolist(), shape=[int(v) for v in shape], resolution=resolution,
                    task_bounds_xy=[center[0]-size/2, center[1]-size/2, center[0]+size/2, center[1]+size/2],
                    halo_m=halo, triangle_count=kept_count, source_vertices=count, groups=dict(groups),
                    filter_excludes=list(excludes), filter_includes=list(includes),
                    transform=dict(axes=axes, scale=scale, offset_xyz=list(offset), preset=preset),
                    alignment_status='NOT_EXTERNALLY_ALIGNED', index=index,
                    multiple_height_sample_conflicts=multilayer,
                    representation='upper sampled surface; single elevation cannot represent underpasses or overhangs',
                    elapsed_s=time.monotonic()-started, valid_cells=int(np.count_nonzero(np.isfinite(height))))
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    # Lightweight portable preview, holes black; no imaging dependency.
    preview = np.asarray(height[::max(1, shape[0]//512), ::max(1, shape[1]//512)])
    finite = np.isfinite(preview)
    grey = np.zeros(preview.shape, dtype=np.uint8)
    if finite.any():
        lo, hi = np.nanmin(preview), np.nanmax(preview)
        grey[finite] = 32+(223*(preview[finite]-lo)/max(float(hi-lo), 1e-6)).astype(np.uint8)
    with (output/'preview.pgm').open('wb') as dst:
        dst.write(f'P5\n{grey.shape[1]} {grey.shape[0]}\n255\n'.encode())
        dst.write(grey.tobytes())
    print(json.dumps({k: metadata[k] for k in ('triangle_count', 'valid_cells', 'elapsed_s', 'alignment_status')}), flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obj', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--center-x', type=float)
    parser.add_argument('--center-y', type=float)
    parser.add_argument('--vehicle-id', type=int, default=0)
    parser.add_argument('--auto-center', action='store_true')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=6668)
    parser.add_argument('--timeout', type=float, default=30.)
    parser.add_argument('--size', type=float, default=1000.)
    parser.add_argument('--resolution', type=float, default=.2)
    parser.add_argument('--halo', type=float, default=12.)
    parser.add_argument('--scale', type=float, default=.01)
    parser.add_argument('--axes', help='signed permutation, e.g. x,-z,y')
    parser.add_argument('--preset', choices=['obj-y-up-hypothesis', 'obj-y-up-ros-hypothesis'])
    parser.add_argument('--offset', type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument('--exclude', action='append', default=[])
    parser.add_argument('--include', action='append', default=[])
    parser.add_argument('--exclude-file', help='JSON array of source-specific excluded group glob patterns')
    args = parser.parse_args(argv)
    if not args.axes and not args.preset:
        parser.error('supply --axes or --preset obj-y-up-hypothesis; alignment is not externally verified')
    if args.auto_center:
        if not args.host or args.center_x is not None or args.center_y is not None:
            parser.error('--auto-center requires --host and no explicit center')
        from .transport import read_first_feedback
        feedback = read_first_feedback(args.host, args.port, args.timeout, args.vehicle_id)
        from .coordinates import world_position
        args.center_x, args.center_y = world_position(feedback.position_m)[:2]
    elif args.center_x is None or args.center_y is None:
        parser.error('supply both --center-x and --center-y, or --auto-center')
    if args.exclude_file:
        patterns = json.loads(Path(args.exclude_file).read_text(encoding='utf-8'))
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            parser.error('--exclude-file must contain a JSON array of strings')
        args.exclude.extend(patterns)
    prepare_map(args.obj, args.output, center=(args.center_x, args.center_y), size=args.size,
                resolution=args.resolution, halo=args.halo, scale=args.scale,
                axes=args.axes or ('x,z,y' if args.preset == 'obj-y-up-ros-hypothesis' else 'x,-z,y'), offset=args.offset,
                includes=args.include, excludes=(*DEFAULT_EXCLUDES, *args.exclude),
                preset=args.preset or 'explicit-transform')


if __name__ == '__main__':
    main()
