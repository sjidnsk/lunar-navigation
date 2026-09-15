"""Independent continuous closed-square intersections for finite center-tip beams."""

from functools import lru_cache
import math
import numpy as np


@lru_cache(maxsize=16)
def beams(radius):
    result = []
    d = math.ceil(radius)
    for ty in range(-d, d + 1):
        for tx in range(-d, d + 1):
            if (tx == 0 and ty == 0) or math.hypot(tx, ty) > radius + 1e-10:
                continue
            events = {}
            for y in range(min(0, ty), max(0, ty) + 1):
                for x in range(min(0, tx), max(0, tx) + 1):
                    if (x, y) == (0, 0):
                        continue
                    lo, hi = 0.0, 1.0
                    for delta, lower in ((tx, x - 0.5), (ty, y - 0.5)):
                        if delta == 0:
                            if not lower <= 0 <= lower + 1:
                                hi = -1
                                break
                        else:
                            a, b = lower / delta, (lower + 1) / delta
                            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
                    if lo <= hi + 1e-12 and hi > 0:
                        # Side-touch contact events precede diagonal interior
                        # entry at the very same continuous parameter.
                        key = (round(lo, 12), 0 if abs(lo - hi) < 1e-12 else 1)
                        events.setdefault(key, []).append((x, y))
            result.append(
                (math.atan2(ty, tx), tuple(tuple(events[k]) for k in sorted(events)))
            )
    return tuple(result)


def observe(b, source, radius, yaw=0.0, fov=2 * math.pi):
    x, y = source
    out = np.zeros(b.shape, bool)
    out[y, x] = True
    if b[y, x] == 2:
        return out
    for angle, events in beams(radius):
        if abs(math.remainder(angle - yaw, 2 * math.pi)) > fov / 2 + 1e-12:
            continue
        for event in events:
            blocked = False
            outside = False
            for dx, dy in event:
                xx, yy = x + dx, y + dy
                if not (0 <= xx < b.shape[1] and 0 <= yy < b.shape[0]):
                    outside = True
                    continue
                out[yy, xx] = True
                blocked |= b[yy, xx] == 2
            if blocked or outside:
                break
    return out
