"""Replay recorded goals through the native navigator and public controller.

No Actor/learner/replay updates. Inputs are the earlier diagnostic JSON goal
sequences, not routes obtained from the new completion bound. One isolated
Jazzy environment runs at a time and every result remains in the output JSON.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
from lunar_drl_exploration.cli import load_config
from lunar_drl_exploration.ros_env import RosExplorationEnv


def assess(env):
    report=env.report;w=env.core.analyzer.workspace;res=env.terrain.resolution_m
    mapped=np.zeros(w.known.shape,bool)
    x,y=np.rint((np.asarray(env.terrain.origin[:2])-w.origin)/res).astype(int)
    x0,y0=max(0,x),max(0,y)
    x1,y1=min(w.known.shape[1],x+env.terrain.shape[1]),min(w.known.shape[0],y+env.terrain.shape[0])
    mapped[y0:y1,x0:x1]=env.reference.mask()[y0-y:y1-y,x0-x:x1-x]
    remaining=mapped&~w.known
    missing=remaining&~env.core.analyzer.remaining_mask
    coverage=env.reference.coverage_ratio(env.privileged.observed)
    lower=report.coverage_lower_bound
    return dict(coverage=coverage,coverage_lower_bound=lower,
        remaining_area_upper_m2=report.remaining_area_upper_m2,
        reference_remaining_m2=float(remaining.sum()*res**2),
        missed_true_remaining_cells=int(missing.sum()),
        bound_valid=not bool(missing.any()) and (lower is None or lower<=coverage+1e-12),
        completed=report.completed,exhausted=report.exhausted,available=report.available,
        frontiers=len(report.frontier_cells),distance_m=env.plant.distance_m)


def run(args):
    config=load_config(SimpleNamespace(config=args.config,sensor_range=None,sensor_fov=None,
                                      command='evaluate'))
    config=replace(config,coverage_target=args.target)
    captures=json.loads((args.recorded_dir/'capture-results.json').read_text(encoding='utf-8'))
    sequences={row['family']:row for row in captures}
    sequences['moon']=json.loads((args.recorded_dir/'route-results.json').read_text(encoding='utf-8'))
    results=[]
    for family in args.families:
        env=RosExplorationEnv(config,0,domain_base=args.domain)
        record=dict(family=family,seed=args.seed,extent_m=args.extent,coverage_target=args.target,
                    mode='recorded_goal_replay',steps=[],error=None)
        start=time.monotonic()
        try:
            env.reset(args.seed,family,args.extent,episode_budget=512)
            record['initial']=assess(env)
            for i,old in enumerate(sequences[family]['steps']):
                if env.report.completed:break
                env.adapter.planning_evidence.clear();env.adapter.references.clear()
                result=env._execute_goal(tuple(old['goal']))
                snap=env._snapshot()
                env.observation,env.report=env.core.observe(snap,env.adapter.velocity)
                env.privileged=env._privileged();env.steps+=1
                row=dict(step=i+1,goal=old['goal'],reason=result.reason_code,outcome=result.outcome,
                    last_segment_revision=result.last_segment_revision,
                    planning_evidence=list(env.adapter.planning_evidence),
                    active_references=[r for r in env.adapter.references if r['active']],**assess(env))
                record['steps'].append(row)
                print(json.dumps(dict(family=family,**{k:v for k,v in row.items()
                    if k not in ('planning_evidence','active_references')})),flush=True)
                if result.reason_code!='GOAL_REACHED' or not row['available'] or not row['bound_valid']:break
            record['final']=assess(env)
        except Exception:
            record['error']=traceback.format_exc();print(record['error'],flush=True)
        finally:
            env.close();record['seconds']=time.monotonic()-start;results.append(record)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--recorded-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--families',nargs='+',choices=['moon','cave'],default=['moon','cave'])
    p.add_argument('--seed',type=int,default=2026091901)
    p.add_argument('--extent',type=float,default=40.)
    p.add_argument('--target',type=float,default=.99)
    p.add_argument('--domain',type=int,default=225)
    rows=run(p.parse_args())
    raise SystemExit(int(any(r['error'] or any(not s['bound_valid'] for s in r['steps']) for r in rows)))
