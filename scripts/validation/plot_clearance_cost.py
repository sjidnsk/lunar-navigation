"""Standalone comparison figure from saved calibration evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();root=args.output
summary=json.loads((root/'summary.json').read_text())
a=json.loads((root/'application-sweep/results.json').read_text())
name='clutter-0.2';rows=[t for t in a['trials'] if t['map']==name and t['pair']==0]
with np.load(root/'application-sweep'/name/'fixture.npz') as z:
 nav=z['navigation'];origin=z['origin'];res=float(z['resolution'])
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
ax=axes[0];ax.imshow(nav,origin='lower',extent=[origin[0],origin[0]+nav.shape[1]*res,origin[1],origin[1]+nav.shape[0]*res],cmap=matplotlib.colors.ListedColormap(['#b9bec7','#f4f6f8','#59616e']),vmin=0,vmax=2)
colors={0:'#ca5553',1:'#d69933',2:'#27856b',20:'#656dc4'}
for t in rows:
 if t['weight'] not in colors:continue
 p=np.array(t['path']);ax.plot(p[:,0],p[:,1],'.-',c=colors[t['weight']],label=f"w={t['weight']:g}, min={t['min_clearance_m']:.3f} m",lw=2)
xy=np.vstack([t['path'] for t in rows]);lo=xy.min(axis=0)-1;hi=xy.max(axis=0)+1
ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_title('Same terrain and endpoints | new simplifier');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.legend(fontsize=8)
t=summary['application-sweep']['groups']['resolution_0.2'];ax=axes[1]
for r in t:
 w=r['weight'];ax.scatter(r['max_length_change_percent'],r['near_boundary_percent'],c=colors.get(w,'#888888'),s=80 if w==2 else 40)
 ax.annotate(f'w={w:g}',(r['max_length_change_percent'],r['near_boundary_percent']),xytext=(7,3 if w!=10 else -12),textcoords='offset points',fontsize=9)
ax.set_xlabel('Largest path-length increase vs w=0 (%)');ax.set_ylabel('Mean path fraction within 0.15 m of boundary (%)');ax.set_title('Application core | 160 paired queries | 0.2 m cells');ax.grid(alpha=.2)
fig.savefig(root/'comparison.png',dpi=160)
