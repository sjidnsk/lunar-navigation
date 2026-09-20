"""Frozen measured-workspace replay; truth checks never enter TaskAnalyzer.

Input NPZ: known/intrinsic/navigation/reachable/task_mask plus JSON metadata
with bounds, origin, resolution, available, sensor, extra. These are diagnostic
snapshots, not a model/checkpoint format. No policy, replay or training updates.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from lunar_drl_exploration.contracts import SensorSpec
from lunar_drl_exploration.scene import Scene,TerrainGrid
from lunar_drl_exploration.reference import CoverageReference
from lunar_drl_exploration.task_analysis import TaskAnalyzer,MeasuredWorkspace


def load_workspace(path):
    with np.load(path) as f:
        m=json.loads(str(f['metadata']))
        w=MeasuredWorkspace(tuple(m['bounds']),tuple(m['origin']),
            *(f[k].copy() for k in ('known','intrinsic','navigation','reachable','task_mask')),m['available'])
    return w,m


def audit(directory,seed,extent):
    results=[];truth={}
    for path in sorted(directory.glob('*.npz')):
        w,metadata=load_workspace(path);res=metadata['resolution'];sensor=SensorSpec(**metadata['sensor'])
        family=path.name.split('-')[0]
        key=(family,res,tuple(sorted(metadata['sensor'].items())))
        if key not in truth:
            scene=Scene(seed,family,extent,resolution_m=res)
            terrain=TerrainGrid.from_scene(scene)
            reference=CoverageReference.build(terrain,scene.initial_pose(terrain),scene.task,sensor)
            truth[key]=(scene,terrain,reference)
        scene,terrain,reference=truth[key]
        # Reconstruct the frozen cumulative meter; only map-read dependency is
        # replaced with its captured value. All opportunity logic runs normally.
        analyzer=TaskAnalyzer(scene.task,sensor)
        analyzer.coverage=SimpleNamespace(consume=lambda _:None,identity='frozen',
            known_area_m2=float((w.known&w.task_mask).sum()*res**2))
        start=time.perf_counter()
        with patch('lunar_drl_exploration.task_analysis.measured_workspace',return_value=w):
            report=analyzer.update(SimpleNamespace(resolution_m=res,revision=1))
        elapsed=(time.perf_counter()-start)*1000
        # Independently align the truth reference to this measured workspace.
        mapped=np.zeros(w.known.shape,bool)
        x,y=np.rint((np.asarray(terrain.origin[:2])-w.origin)/res).astype(int)
        x0,y0=max(0,x),max(0,y)
        x1,y1=min(w.known.shape[1],x+terrain.shape[1]),min(w.known.shape[0],y+terrain.shape[0])
        mapped[y0:y1,x0:x1]=reference.mask()[y0-y:y1-y,x0-x:x1-x]
        remaining=mapped&~w.known
        covered=float((mapped&w.known).sum()*res**2)
        coverage=covered/reference.area_m2
        row=dict(snapshot=path.name,coverage=coverage,reference_area_m2=reference.area_m2,
            known_area_m2=report.known_area_m2,reference_remaining_m2=float(remaining.sum()*res**2),
            completed=report.completed,exhausted=report.exhausted,available=report.available,
            reached_80=coverage>=.8,reached_99=coverage>=.99,frontiers=len(report.frontier_cells),
            elapsed_ms=elapsed,old=metadata['extra'])
        results.append(row);print(json.dumps(row),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026091901)
    parser.add_argument('--extent',type=float,default=40.)
    args=parser.parse_args()
    rows=audit(args.snapshot_dir,args.seed,args.extent)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
