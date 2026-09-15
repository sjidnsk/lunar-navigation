"""Task-owned cumulative effective observation K, independent of current B/M.

Consumes only applied native snapshots. No terrain, simulated sensor, reference
or predicted visibility enters this spatial history. One packed bit per cell.
"""
import numpy as np
from .geometry import polygon_mask


class CoverageHistory:
    def __init__(self,task):
        self.task=task;self.identity=None;self.revision=-1
        self._tiles={};self._sources={};self._task_bits={}
        self._known_count=0;self.bounds=None

    @property
    def known_area_m2(self):
        return self._known_count*self.identity[1]**2 if self.identity is not None else 0.

    def consume(self,snapshot):
        identity=(snapshot.epoch,snapshot.resolution_m,tuple(snapshot.origin))
        if self.identity!=identity:
            self.identity=identity;self.revision=-1
            self._tiles.clear();self._sources.clear();self._task_bits.clear()
            self._known_count=0;self.bounds=None
        if snapshot.revision<self.revision:raise ValueError('stale coverage snapshot')
        if snapshot.revision==self.revision:return
        self.revision=snapshot.revision
        for key,tile in snapshot.tiles.items():
            if self._sources.get(key) is tile:continue
            self._sources[key]=tile
            incoming=np.packbits(tile.observed!=0,bitorder='little')
            previous=self._tiles.get(key)
            new=incoming if previous is None else incoming & ~previous
            if not np.any(new):continue
            self._tiles[key]=incoming.copy() if previous is None else previous|incoming
            if key not in self._task_bits:
                origin=(snapshot.origin[0]+key[0]*256*snapshot.resolution_m,
                        snapshot.origin[1]+key[1]*256*snapshot.resolution_m)
                self._task_bits[key]=np.packbits(polygon_mask((256,256),origin,
                    snapshot.resolution_m,self.task.polygon).ravel(),bitorder='little')
            self._known_count+=int(np.unpackbits(new & self._task_bits[key]).sum())
            ids=np.flatnonzero(np.unpackbits(new,bitorder='little'))
            xs=key[0]*256+ids%256;ys=key[1]*256+ids//256
            box=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
            self.bounds=box if self.bounds is None else (
                min(box[0],self.bounds[0]),min(box[1],self.bounds[1]),
                max(box[2],self.bounds[2]),max(box[3],self.bounds[3]))

    def mask(self,bounds):
        """Bounded projection at decision/report time; never changes current B/M."""
        x0,y0,x1,y1=map(int,bounds)
        if x1<x0 or y1<y0:raise ValueError('inverted projection bounds')
        result=np.zeros((y1-y0,x1-x0),bool)
        for (tx,ty),bits in self._tiles.items():
            left,right=max(x0,tx*256),min(x1,(tx+1)*256)
            top,bottom=max(y0,ty*256),min(y1,(ty+1)*256)
            if right<=left or bottom<=top:continue
            tile=np.unpackbits(bits,bitorder='little').reshape(256,256)
            result[top-y0:bottom-y0,left-x0:right-x0]=tile[top-ty*256:bottom-ty*256,left-tx*256:right-tx*256]
        return result
