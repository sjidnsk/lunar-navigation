"""Bounded observation statistics and pacing using existing map diagnostics."""
import numpy as np


def exploration_bounds(map_bounds, size_m):
    """Fixed task square about the map centre, clipped for smaller test maps."""
    if not np.isfinite(size_m) or size_m <= 0:
        raise ValueError('exploration_size_m must be finite and positive')
    xmin, ymin, xmax, ymax = map_bounds
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    half = size_m/2
    return [max(xmin,cx-half), max(ymin,cy-half),
            min(xmax,cx+half), min(ymax,cy+half)]


class EvidenceCoverage:
    """Count measured cells inside the task once, independently of traversability."""

    def __init__(self, metadata):
        self.resolution = float(metadata['resolution'])
        self.origin = np.asarray(metadata['origin_xy'], dtype=float)
        self.mask = np.zeros(metadata['shape'], dtype=bool)
        xmin, ymin, xmax, ymax = metadata['task_bounds_xy']
        # Cell-centre inclusion is independent of the observation window's origin.
        low = np.ceil((np.array([xmin, ymin]) - self.origin) / self.resolution - .5).astype(int)
        high = np.ceil((np.array([xmax, ymax]) - self.origin) / self.resolution - .5).astype(int)
        self.low = np.maximum(low, 0)
        self.high = np.minimum(high, self.mask.shape[::-1])
        self.total_cells = int(np.prod(np.maximum(self.high - self.low, 0)))
        self.observed_cells = 0

    def add(self, observation):
        if not np.isclose(observation.resolution, self.resolution):
            raise ValueError('observation resolution differs from map')
        offset = np.rint((np.asarray(observation.origin_xy) - self.origin) / self.resolution).astype(int)
        values = np.asarray(observation.elevation)
        low = np.maximum(offset, self.low)
        high = np.minimum(offset + values.shape[::-1], self.high)
        if np.any(high <= low):
            return 0
        target = self.mask[low[1]:high[1], low[0]:high[0]]
        source = values[low[1]-offset[1]:high[1]-offset[1], low[0]-offset[0]:high[0]-offset[0]]
        visible = np.isfinite(source)
        added = int(np.count_nonzero(visible & ~target))
        target |= visible
        self.observed_cells += added
        return added


class MapReceipt:
    """Pace known-map chunks from mapper diagnostics, without a demo ACK protocol."""

    def __init__(self):
        self.current = (0, 0, 0, 0, 0)
        self.waiting = None

    def update(self, received, lag, fine, guidance, duplicates=0):
        self.current = (received, lag, fine, guidance, duplicates)

    def sent(self):
        self.waiting = self.current

    @property
    def ready(self):
        if self.waiting is None:
            return True
        received, lag, fine, guidance, duplicates = self.current
        old_received, _, old_fine, old_guidance, old_duplicates = self.waiting
        return (received > old_received and lag == 0
                and ((fine > old_fine and guidance > old_guidance)
                     or duplicates > old_duplicates))


class DeferredGoalCancellation:
    """Remember cancellation across the asynchronous goal-acceptance boundary."""

    def __init__(self, acceptance):
        self.acceptance = acceptance
        self.handle = None
        self.requested = False
        self.cancel_future = None
        acceptance.add_done_callback(self._accepted)

    def _accepted(self, future):
        if future.cancelled() or future.exception() is not None:
            return
        response = future.result()
        if response.accepted:
            self.handle = response
            if self.requested:
                self.cancel()

    def cancel(self):
        self.requested = True
        if self.handle is not None and self.cancel_future is None:
            self.cancel_future = self.handle.cancel_goal_async()
        return self.cancel_future if self.cancel_future is not None else self.acceptance
