"""Crop an already processed terrain without rescanning OBJ or resampling cells."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .geometry import TerrainMap
from .prepare import _intersects, _build_index


def crop_map(source, output, size=500., center=None, halo=12.):
    started = time.monotonic()
    terrain = TerrainMap(source)
    bounds = terrain.metadata['task_bounds_xy']
    center = np.asarray(center if center is not None else ((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2))
    if not np.isfinite([*center, size, halo]).all() or size <= 0 or halo < 0:
        raise ValueError('center and size must be finite; size positive, halo nonnegative')
    lo = np.floor((center-size/2-halo-terrain.origin_xy)/terrain.resolution).astype(int)
    hi = np.ceil((center+size/2+halo-terrain.origin_xy)/terrain.resolution).astype(int)
    if np.any(lo < 0) or np.any(hi > terrain.elevation.shape[::-1]):
        raise ValueError('crop including halo must fit within source map')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    origin = terrain.origin_xy+lo*terrain.resolution
    selection = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
    elevation = np.asarray(terrain.elevation[selection])
    np.save(output/'elevation.npy', elevation)
    np.save(output/'sample_xy.npy', terrain.sample_xy[selection])
    crop_bounds = [*origin, *(terrain.origin_xy+hi*terrain.resolution)]
    triangles = np.asarray(terrain.triangles[_intersects(terrain.triangles, crop_bounds)])
    triangles.tofile(output/'triangles.bin')
    index = _build_index(triangles, output, origin, elevation.shape, terrain.resolution)
    metadata = {k:v for k,v in terrain.metadata.items() if k not in ('groups','multiple_height_sample_conflicts')}
    metadata.update(source_map=str(Path(source).resolve()), origin_xy=origin.tolist(),
                    shape=list(elevation.shape), task_bounds_xy=[*(center-size/2), *(center+size/2)],
                    halo_m=halo, triangle_count=len(triangles), index=index,
                    valid_cells=int(np.isfinite(elevation).sum()), elapsed_s=time.monotonic()-started)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--size',type=float,default=500.)
    parser.add_argument('--center',type=float,nargs=2)
    args=parser.parse_args()
    metadata=crop_map(**vars(args))
    print(json.dumps({k:metadata[k] for k in ('shape','triangle_count','elapsed_s')}))


if __name__ == '__main__':
    main()
