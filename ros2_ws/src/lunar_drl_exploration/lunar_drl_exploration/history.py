"""Actual observation directions at executed world poses, independent of graph IDs."""

import math
import numpy as np
from scipy.spatial import cKDTree
from .sensor import validate_sensor

HEADINGS = np.arange(8, dtype=np.float64) * (math.pi / 4)


def heading_bins(angles):
    """Nearest optical-world sector, with identical wrapping for record/lookup."""
    return np.floor((np.asarray(angles) % (2 * math.pi)) / (math.pi / 4) + 0.5).astype(np.int64) % 8


class DirectionHistory:
    def __init__(self, tolerance_m=0.1):
        if not math.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError("positive world-position tolerance required")
        self.tolerance_m = float(tolerance_m)
        self._records = []

    def clear(self):
        self._records.clear()

    def record(self, pose, observed_cells, sensor):
        validate_sensor(sensor)
        if not np.asarray(observed_cells).size:
            return
        if not np.all(np.isfinite([pose.x, pose.y, pose.yaw])):
            raise ValueError("finite executed pose required")
        # Store actual optical direction, including mounting yaw. A visit bit
        # denotes the nearest world heading that was physically observed.
        heading = int(heading_bins(pose.yaw + sensor.offset_yaw_rad))
        self._records.append((float(pose.x), float(pose.y), heading))

    def bits(self, positions):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        result = np.zeros((len(positions), 8), np.float32)
        if self._records and len(positions):
            records = np.asarray(self._records)
            tree = cKDTree(records[:, :2])
            for i, indices in enumerate(
                tree.query_ball_point(positions, self.tolerance_m)
            ):
                if indices:
                    result[i, records[indices, 2].astype(int)] = 1
        return result

    def action_bits(self, positions, sensor):
        """Project physical views into the current vehicle-action heading columns."""
        validate_sensor(sensor)
        return self.bits(positions)[:, heading_bins(HEADINGS + sensor.offset_yaw_rad)]
