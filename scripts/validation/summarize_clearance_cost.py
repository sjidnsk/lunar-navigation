"""Summarize paired sweeps. Report means per query, not pooled sample lengths."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root):
    a=json.loads((root/'results.json').read_text());ref={(t['map'],t['pair']):t for t in a['trials'] if t['weight']==0}
    out={'maps':len(a['maps']),'plans':len(a['trials']),'groups':{}}
    for group in ['all','resolution_0.2','moon','cave','boundary_cases']:
        def select(t):
            if group=='all':return True
            if group=='resolution_0.2':return t['resolution_m']==.2
            if group=='boundary_cases':return ref[t['map'],t['pair']].get('min_clearance_m',float('inf'))<=.2+1e-8
            return t['family']==group
        table=[]
        for weight in a['weights']:
            candidates=[t for t in a['trials'] if t['weight']==weight and select(t)]
            rows=[t for t in candidates if t['status']==0 and ref[t['map'],t['pair']]['status']==0]
            detour=[100*(t['length_m']/ref[t['map'],t['pair']]['length_m']-1) for t in rows]
            table.append(dict(weight=weight,queries=len(candidates),success=len(rows),
                near_boundary_percent=100*float(np.mean([t['near_boundary_fraction'] for t in rows])),
                mean_min_clearance_m=float(np.mean([t['min_clearance_m'] for t in rows])),
                mean_p10_clearance_m=float(np.mean([t['p10_clearance_m'] for t in rows])),
                mean_length_change_percent=float(np.mean(detour)),max_length_change_percent=float(max(detour)),
                planning_p50_ms=float(np.median([t['median_ms'] for t in rows])),planning_p95_ms=float(np.quantile([t['median_ms'] for t in rows],.95)),
                expanded_p95=float(np.quantile([t['expanded'] for t in rows],.95))))
        out['groups'][group]=table
    maps=[m for m in a['maps'] if m['res']==.2]
    out['cold_derivation_0.2']={k:float(np.mean([m[k] for m in maps])) for k in ['derive_zero_ms','derive_nonzero_ms']}
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();root=args.output
    out={name:summarize(root/name) for name in ['verified-sweep','application-sweep']}
    loop=json.loads((root/'closed-loop/closed-loop.json').read_text());assert loop['closed']
    out['closed_loop']={str(w):dict(trials=0,reached=0,collision=0) for w in [0.,.5,1.,2.,5.,10.,20.]}
    for t in loop['trials']:
        assert 'error' not in t and all(p['result']['reason_code']=='GOAL_REACHED' for p in t['prefix'])
        assert any(e['reason_code']=='PLAN_FOUND' and int(e['path_points'])>0 for e in t['planning_evidence'])
        r=out['closed_loop'][str(t['weight'])];r['trials']+=1;r['reached']+=t['result']['reason_code']=='GOAL_REACHED';r['collision']+=t['collision']
    (root/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
