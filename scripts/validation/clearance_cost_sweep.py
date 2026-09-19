"""Paired offline sweep using the production C++ terrain builder and A*.

Run under the DRL Jazzy Python overlay to use its procedural Scene generator.
Artifacts go to --output. No running navigation or training state is modified.
"""
import argparse
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage

WEIGHTS = [0., .5, 1., 2., 5., 10., 20.]


def boundary_segments(free, origin, res):
    padded = np.pad(free, 1)
    segments = []
    # Merge exposed unit cell faces into maximal collinear segments.
    for horizontal, faces in [(True, padded[1:, 1:-1] != padded[:-1, 1:-1]),
                              (False, (padded[1:-1, 1:] != padded[1:-1, :-1]).T)]:
        for fixed, row in enumerate(faces):
            transitions = np.flatnonzero(np.diff(np.r_[False, row, False]))
            for a, b in zip(transitions[::2], transitions[1::2]):
                segments.append([[a, fixed], [b, fixed]] if horizontal else [[fixed, a], [fixed, b]])
    return np.asarray(segments, float) * res + origin


def point_distances(points, segments):
    # Exhaustive geometry, batched to bound memory; independent of production simplifier.
    a, b = segments[:, 0], segments[:, 1]
    d = b - a
    l2 = (d*d).sum(axis=1)
    out = []
    for p in np.array_split(points, max(1, int(np.ceil(len(points) / 128)))):
        t = np.clip(((p[:, None] - a) * d).sum(axis=2) / l2, 0, 1)
        out.append(np.linalg.norm(p[:, None] - a - t[:, :, None]*d, axis=2).min(axis=1))
    return np.concatenate(out)


def exact_minimum(path, boundary):
    a, b = boundary[:, 0], boundary[:, 1]
    bd = b-a
    result = float('inf')
    for p, q in zip(path, path[1:]):
        pd = q-p
        l2 = pd @ pd
        t = np.clip(((boundary-p)*pd).sum(axis=2) / max(l2, 1e-30), 0, 1)
        result = min(result, float(np.linalg.norm(boundary-p-t[:, :, None]*pd, axis=2).min()))
        result = min(result, float(point_distances(np.array([p,q]), boundary).min()))
        den = pd[0]*bd[:,1]-pd[1]*bd[:,0]
        nz = np.abs(den)>1e-15
        v = a[nz]-p
        u = (v[:,0]*bd[nz,1]-v[:,1]*bd[nz,0])/den[nz]
        t2 = (v[:,0]*pd[1]-v[:,1]*pd[0])/den[nz]
        if np.any((u>=0)&(u<=1)&(t2>=0)&(t2<=1)):
            return 0.
    return result


def metrics(path, boundary):
    p = np.asarray(path)
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0., np.cumsum(ds)]
    if s[-1] < 1e-10:
        return dict(length_m=0., min_clearance_m=0., p10_clearance_m=0., near_boundary_fraction=0.)
    sample_s = np.arange(.0125, s[-1], .025)
    samples = np.column_stack([np.interp(sample_s,s,p[:,k]) for k in [0,1]])
    distance = point_distances(samples, boundary)
    interior = distance[(sample_s>=1.)&(sample_s<=s[-1]-1.)]
    if not len(interior): interior = distance
    return dict(length_m=float(s[-1]), min_clearance_m=exact_minimum(p,boundary),
                p10_clearance_m=float(np.quantile(interior,.1)),
                near_boundary_fraction=float(np.mean(interior<.15-1e-9)),
                mean_clearance_m=float(interior.mean()), points=len(path))


def synthetic(name, res):
    w,h = int(round(32/res)),int(round(24/res))
    x,y = np.meshgrid((np.arange(w)+.5)*res,(np.arange(h)+.5)*res)
    z = np.zeros((h,w),np.float32)
    if name == 'unknown_edge': z[y>13] = np.nan
    elif name == 'block': z[(x>13)&(x<19)&(y>5)&(y<19)] = 2
    elif name == 'concave': z[((x>12)&(x<15)&(y>4)&(y<18))|((x>12)&(x<24)&(y>15)&(y<18))]=2
    elif name == 'two_routes':
        z[(x>14)&(x<17)&(y>2)&(y<22)]=2
        z[(x>14)&(x<17)&(y>10.5)&(y<13.5)]=0
    elif name == 'narrow': z[np.abs(y-12)>1.4]=2
    elif name == 'rooms':
        z[(x>14)&(x<17)]=2
        z[(x>14)&(x<17)&(y>9.8)&(y<14.2)]=0
    elif name == 'clutter':
        rng=np.random.default_rng(9271)
        for cx,cy,r in zip(rng.uniform(4,28,18),rng.uniform(4,20,18),rng.uniform(.35,1.4,18)):
            z[(x-cx)**2+(y-cy)**2<r*r]=2
    elif name != 'open': raise ValueError(name)
    return z,np.zeros(2),res


def fixtures():
    for res in [.1,.2,.4]:
        for name in ['open','unknown_edge','block','concave','two_routes','narrow','rooms','clutter']:
            yield f'{name}-{res}', 'synthetic', synthetic(name,res)
    from lunar_drl_exploration.scene import Scene
    for extent in [40.,80.,300.]:
        for family in ['moon','cave']:
            for seed in [2026091901,2026091902]:
                scene=Scene(seed,family,extent)
                # A 32m local snapshot, including the primary cave loop or moon terrain.
                center=np.array([-.5*extent-3.,.12*extent]) if family=='cave' else np.array([-.1*extent,0.])
                start=np.floor((center-16-np.array(scene.origin))/.2).astype(int)
                start=np.maximum(0,np.minimum(start,np.array(scene.shape[::-1])-160))
                z=scene.height_tile(int(start[0]),int(start[1]),160,160).astype(np.float32)
                yield f'{family}-{int(extent)}-{seed}', family, (z,np.array(scene.origin)+start*.2,.2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--core-library-dir',type=Path,required=True);parser.add_argument('--resolution',type=float);args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    rows=[]; maps=[]
    if (root/'results.json').exists():
        saved=json.loads((root/'results.json').read_text());rows=saved['trials'];maps=saved['maps']
    completed={m['name'] for m in maps}
    for name,family,(z,origin,res) in fixtures():
        if name in completed or (args.resolution is not None and res!=args.resolution):continue
        folder=root/name;folder.mkdir(exist_ok=True)
        terrain=folder/'terrain.bin'
        with terrain.open('wb') as f:
            f.write(struct.pack('<IIddd',z.shape[1],z.shape[0],res,*origin));f.write(z.astype('<f4').tobytes())
        proc=subprocess.Popen([str(args.probe),str(terrain),str(folder/'grid.bin')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True, env={**os.environ,'LD_LIBRARY_PATH':str(args.core_library_dir)+':'+os.environ.get('LD_LIBRARY_PATH','')})
        line=proc.stdout.readline()
        if not line:raise RuntimeError(f'{name}: native probe exited {proc.wait()}')
        meta=json.loads(line);meta.update(name=name,family=family);maps.append(meta)
        meta['loaded_core']=[line.split()[-1] for line in Path(f'/proc/{proc.pid}/maps').read_text().splitlines() if 'liblunar_incremental_navigation_core.so' in line]
        assert meta['loaded_core'] and all(str(args.core_library_dir) in line for line in meta['loaded_core'])
        data=(folder/'grid.bin').read_bytes();n=meta['width']*meta['height'];shape=(meta['height'],meta['width'])
        nav=np.frombuffer(data[:n],np.uint8).reshape(shape);base=np.frombuffer(data[n:n+8*n],'<f8').reshape(shape);unit=np.frombuffer(data[n+8*n:],'<f8').reshape(shape)
        origin=np.array(meta['origin']);free=nav==1;labels,count=ndimage.label(free) # 4-connected avoids diagonal pinch connectivity.
        sizes=np.bincount(labels.ravel());sizes[0]=0
        if family=='cave':
            assert z.shape==nav.shape
            sizes=np.bincount(labels[(z<.5)&free].ravel(),minlength=count+1);sizes[0]=0
        component=labels==np.argmax(sizes)
        meta['selected_component_cells']=int(component.sum())
        cells=np.argwhere(component)[:,::-1];rng=np.random.default_rng(913)
        pairs=[]
        for i in range(8):
            for attempt in range(1000):
                a,b=cells[rng.integers(len(cells),size=2)]
                dist=np.linalg.norm(a-b)*res
                if 4<=dist<=28:break
            else:raise RuntimeError(f'no usable pair in {name}')
            pairs.append((origin+(a+.5)*res,origin+(b+.5)*res))
        boundary=boundary_segments(free,origin,res)
        for pair_id,(a,b) in enumerate(pairs):
            for weight in WEIGHTS:
                proc.stdin.write(' '.join(map(str,[weight,*a,*b]))+'\n');proc.stdin.flush()
                result=json.loads(proc.stdout.readline());result.update(map=name,family=family,pair=pair_id,start=a.tolist(),goal=b.tolist(),resolution_m=res)
                if result['status']==0:
                    result.update(metrics(result['path'],boundary));result['raw_min_clearance_m']=exact_minimum(np.asarray(result['raw']),boundary)
                    assert result['min_clearance_m']+1e-9>=result['raw_min_clearance_m'],result
                rows.append(result)
        proc.stdin.close();assert proc.wait()==0
        np.savez_compressed(folder/'fixture.npz',navigation=nav,base_cost=base,unit_cost=unit,origin=origin,resolution=res,heights=z)
        terrain.unlink();(folder/'grid.bin').unlink()
        (root/'results.json').write_text(json.dumps({'weights':WEIGHTS,'maps':maps,'trials':rows},separators=(',',':')))
        print(name,'plans',len(rows),'derive_ms',round(meta['derive_nonzero_ms'],1),flush=True)
    print('COMPLETE',len(maps),len(rows),flush=True)

if __name__=='__main__':main()
