"""Deterministic terrain and a local synthetic elevation sensor, without ROS.

Only ``observe`` is a sensor input. ``obstacles`` and ``preview`` expose world
truth for visualisation and the command-driven vehicle's collision check; they
must never be supplied to a planner. No full-resolution global grid is stored.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    """Oriented solid outcrop; height is relative to the smooth ground."""

    x: float
    y: float
    length: float
    width: float
    yaw: float
    height: float
    kind: str = 'rock'


class Terrain:
    """A centred 300 m scene with safe corridors, rocks, ridges and depressions.

    The synthetic sensor uses one local discrete scene for height and occlusion.
    Every cell intersecting a solid represents that solid's surface; grid rays
    reveal the first such cell and stop. This conservatively expands silhouettes
    by at most one cell diagonal (0.283 m at the default 0.2 m resolution).
    Each surface height comes from a fixed point in the cell/solid intersection,
    independently of the eye position. Continuous collision geometry is separate.
    This is not an optical/LiDAR model or sloped-ground self-occlusion. The optional
    1.4 m near field uses the same cell occlusion as the forward sector.
    """

    FOOTPRINT_LENGTH_M = 1.182
    FOOTPRINT_WIDTH_M = 0.818
    COLLISION_MARGIN_M = 0.04

    def __init__(self, size_m=300.0, seed=20260910):
        if not math.isfinite(size_m) or size_m < 20.0:
            raise ValueError('size_m must be finite and at least 20 m')
        self.size_m = float(size_m)
        self.seed = int(seed)
        self.obstacles = self._make_obstacles()
        # Broad-phase AABBs and OBB axes are cached once; scalar vehicle steps
        # and local observations then examine only nearby geometry.
        geometry = np.array([
            (o.x, o.y, o.length / 2, o.width / 2, math.cos(o.yaw),
             math.sin(o.yaw), o.height) for o in self.obstacles
        ], dtype=np.float64).reshape((-1, 7))
        self._geometry = geometry
        self._extent_x = geometry[:, 2] * np.abs(geometry[:, 4]) + \
            geometry[:, 3] * np.abs(geometry[:, 5])
        self._extent_y = geometry[:, 2] * np.abs(geometry[:, 5]) + \
            geometry[:, 3] * np.abs(geometry[:, 4])

    @staticmethod
    def _crosses_corridor(obstacle):
        """Keep at least 6.2 m between solids along two axes and two diagonals."""
        cs, sn = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
        for nx, ny, offset in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                               (-0.55, 1.0, 34.0), (0.75, 1.0, -42.0)):
            normal_length = math.hypot(nx, ny)
            projected_half_size = (
                obstacle.length / 2 * abs(nx * cs + ny * sn) +
                obstacle.width / 2 * abs(-nx * sn + ny * cs))
            if abs(nx * obstacle.x + ny * obstacle.y - offset) < \
                    projected_half_size + 3.1 * normal_length:
                return True
        return False

    def _make_obstacles(self):
        rng = np.random.default_rng(self.seed)
        half = self.size_m / 2
        result = []

        def add(obstacle):
            radius = math.hypot(obstacle.length, obstacle.width) / 2
            if (abs(obstacle.x) + radius < half - 1.0 and
                    abs(obstacle.y) + radius < half - 1.0 and
                    not self._crosses_corridor(obstacle)):
                result.append(obstacle)

        # The near scene is intentionally composed, with a repeatable shadow
        # behind the first rock, several clusters, and a one-metre narrow gap.
        near_rocks = (
            (10.0, 5.0, 3.0, 2.4, 0.0, 1.3),
            (9.0, -5.4, 2.5, 3.0, -0.18, 1.8),
            (18.0, 7.5, 3.8, 2.8, 0.35, 1.1),
            (20.5, -8.0, 4.0, 3.2, -0.4, 1.5),
            (27.0, 11.8, 5.0, 3.4, 0.0, 1.7),
            (27.0, 16.2, 5.0, 3.4, 0.0, 1.4),
            (31.0, -14.5, 4.2, 2.6, 0.5, 1.9),
            (12.0, 17.0, 3.5, 2.4, -0.25, 1.2),
            (18.0, 23.0, 2.5, 2.0, 0.9, 0.9),
            (7.0, 25.0, 3.8, 3.0, 0.2, 1.6),
            (-10.0, 7.5, 3.5, 2.7, -0.3, 1.4),
            (-14.0, 16.0, 3.0, 4.0, 0.4, 1.8),
            (-22.0, 10.0, 4.8, 2.7, -0.5, 1.1),
            (-29.0, 9.0, 3.8, 2.8, 0.2, 1.6),
            (-9.0, -9.0, 3.0, 2.4, 0.15, 1.7),
            (-16.0, -15.0, 3.2, 2.8, -0.4, 1.1),
            (-26.0, -8.5, 4.2, 3.4, 0.3, 1.4),
            (-31.0, -20.0, 3.2, 2.2, -0.1, 1.6),
            (13.0, -21.0, 4.4, 2.7, -0.35, 1.2),
            (23.0, -25.0, 3.0, 2.5, 0.45, 1.8),
        )
        for values in near_rocks:
            add(Obstacle(*values))

        # Discontinuous escarpments create multiple large-scale passage
        # choices. Corridor clipping opens wide crossings rather than relying
        # on sub-footprint gaps between adjacent segments.
        for ridge_x in (-88.0, -44.0, 38.0, 82.0):
            for ridge_y in np.arange(-half + 10.0, half - 9.0, 9.0):
                add(Obstacle(ridge_x, float(ridge_y), 8.6, 2.5,
                             math.pi / 2, 1.5 + 0.3 * math.sin(ridge_y / 19),
                             'ridge'))
        for ridge_y in (-67.0, 71.0):
            for ridge_x in np.arange(-half + 10.0, half - 9.0, 11.0):
                add(Obstacle(float(ridge_x), ridge_y, 10.6, 2.2, 0.0,
                             1.6 + 0.25 * math.cos(ridge_x / 17), 'ridge'))

        # One jittered cluster per tile keeps local query work bounded while
        # distributing hundreds of distinct outcrops throughout the scene.
        for tile_x in np.arange(-half + 12.0, half - 10.0, 14.0):
            for tile_y in np.arange(-half + 12.0, half - 10.0, 14.0):
                x, y = float(tile_x + rng.uniform(-4, 4)), \
                    float(tile_y + rng.uniform(-4, 4))
                if math.hypot(x, y) < 34.0:
                    continue
                add(Obstacle(x, y, float(rng.uniform(1.6, 4.2)),
                             float(rng.uniform(1.3, 3.4)),
                             float(rng.uniform(-math.pi, math.pi)),
                             float(rng.uniform(0.8, 2.2))))
                if rng.random() < 0.3:
                    add(Obstacle(x + float(rng.uniform(3.0, 5.0)),
                                 y + float(rng.uniform(-3.0, 3.0)),
                                 float(rng.uniform(1.0, 2.1)),
                                 float(rng.uniform(0.9, 1.8)),
                                 float(rng.uniform(-math.pi, math.pi)),
                                 float(rng.uniform(0.8, 1.5))))
        return tuple(result)

    @staticmethod
    def _ground_elevation(x, y):
        """Analytic smooth hills, broad ridges and depressions; flat start."""
        height = (0.16 * np.sin(x / 19) * np.cos(y / 23) +
                  0.22 * np.sin((x + y) / 37))
        for cx, cy, sx, sy, amplitude in (
                (43.0, 27.0, 24.0, 20.0, 2.0),
                (-61.0, -29.0, 28.0, 19.0, 2.3),
                (12.0, -79.0, 37.0, 15.0, 1.4),
                (93.0, -55.0, 20.0, 27.0, -1.4),
                (-32.0, 72.0, 18.0, 23.0, -1.3),
                (3.0, 117.0, 36.0, 16.0, 1.7)):
            height = height + amplitude * np.exp(
                -0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        # C1 transition avoids an artificial step at the launch pad edge.
        blend = np.clip((np.hypot(x, y) - 5.0) / 6.0, 0.0, 1.0)
        return height * blend * blend * (3.0 - 2.0 * blend)

    def _nearby(self, min_x, max_x, min_y, max_y):
        g = self._geometry
        return np.flatnonzero(
            (g[:, 0] + self._extent_x >= min_x) &
            (g[:, 0] - self._extent_x <= max_x) &
            (g[:, 1] + self._extent_y >= min_y) &
            (g[:, 1] - self._extent_y <= max_y))

    def elevation(self, x, y):
        """Evaluate world truth on demand, returning NaN outside the world.

        Inputs follow NumPy broadcasting; a pair of scalars returns a float.
        This method does not change or accumulate sensor knowledge.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(y, dtype=np.float64))
        shape = x.shape
        xf, yf = x.ravel(), y.ravel()
        inside = np.isfinite(xf) & np.isfinite(yf) & \
            (np.abs(xf) <= self.size_m / 2) & (np.abs(yf) <= self.size_m / 2)
        values = np.full(xf.shape, np.nan, dtype=np.float64)
        if inside.any():
            valid_x, valid_y = xf[inside], yf[inside]
            ground = self._ground_elevation(valid_x, valid_y)
            rock_height = np.zeros(valid_x.shape, dtype=np.float64)
            # Roundoff in a world-coordinate subtraction/rotation must not
            # exclude a point on a solid face. This is <= 5.4e-13 m in the
            # 300 m scene, confined to elevation membership; stored geometry,
            # collision margins and sensor visibility are unchanged.
            tolerance = 8.0 * np.finfo(np.float64).eps * self.size_m
            for index in self._nearby(valid_x.min() - tolerance, valid_x.max() + tolerance,
                                      valid_y.min() - tolerance, valid_y.max() + tolerance):
                cx, cy, hl, hw, cs, sn, height = self._geometry[index]
                nearby = (np.abs(valid_x - cx) <= self._extent_x[index] + tolerance) & \
                    (np.abs(valid_y - cy) <= self._extent_y[index] + tolerance)
                indices = np.flatnonzero(nearby)
                dx, dy = valid_x[indices] - cx, valid_y[indices] - cy
                u, v = cs * dx + sn * dy, -sn * dx + cs * dy
                covered = (np.abs(u) <= hl + tolerance) & (np.abs(v) <= hw + tolerance)
                indices = indices[covered]
                top = height + 0.06 * np.sin(u[covered] * 2.1) * np.sin(v[covered] * 1.7)
                rock_height[indices] = np.maximum(rock_height[indices], top)
            values[inside] = ground + rock_height
        result = values.reshape(shape)
        return float(result) if result.ndim == 0 else result

    def collides(self, x, y, yaw):
        """Conservative OBB collision against solid rocks, cliffs and bounds.

        The full wheeled footprint plus a 4 cm margin is checked with the
        separating-axis theorem. Smooth ground is handled by elevation-based
        planner certification; this callback is not a traversability oracle.
        """
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return True
        cs, sn = math.cos(yaw), math.sin(yaw)
        hl = self.FOOTPRINT_LENGTH_M / 2 + self.COLLISION_MARGIN_M
        hw = self.FOOTPRINT_WIDTH_M / 2 + self.COLLISION_MARGIN_M
        ex, ey = hl * abs(cs) + hw * abs(sn), hl * abs(sn) + hw * abs(cs)
        if abs(x) + ex >= self.size_m / 2 or abs(y) + ey >= self.size_m / 2:
            return True
        nearby = self._nearby(x - ex, x + ex, y - ey, y + ey)
        if nearby.size == 0:
            return False
        g = self._geometry[nearby]
        dx, dy, acs, asn = g[:, 0] - x, g[:, 1] - y, g[:, 4], g[:, 5]
        parallel = np.abs(cs * acs + sn * asn)
        perpendicular = np.abs(-sn * acs + cs * asn)
        overlaps = (
            (np.abs(dx * cs + dy * sn) <= hl + g[:, 2] * parallel + g[:, 3] * perpendicular) &
            (np.abs(-dx * sn + dy * cs) <= hw + g[:, 2] * perpendicular + g[:, 3] * parallel) &
            (np.abs(dx * acs + dy * asn) <= g[:, 2] + hl * parallel + hw * perpendicular) &
            (np.abs(-dx * asn + dy * acs) <= g[:, 3] + hl * perpendicular + hw * parallel))
        return bool(overlaps.any())

    @staticmethod
    def _cell_surface_points(cx, cy, xmin, xmax, ymin, ymax, geometry, tolerance):
        """Select a fixed representative inside each cell/solid intersection."""
        ox, oy, hl, hw, cs, sn, _ = geometry
        u, v = cs * (cx - ox) + sn * (cy - oy), -sn * (cx - ox) + cs * (cy - oy)
        centre_inside = (np.abs(u) <= hl + tolerance) & (np.abs(v) <= hw + tolerance)
        px, py = cx.copy(), cy.copy()
        outside = np.flatnonzero(~centre_inside)
        if outside.size == 0:
            return px, py
        left, right = xmin[outside], xmax[outside]
        bottom, top = ymin[outside], ymax[outside]
        sum_x, sum_y = np.zeros(outside.size), np.zeros(outside.size)
        count = np.zeros(outside.size, dtype=np.int64)

        def add(x, y, valid):
            nonlocal sum_x, sum_y, count
            sum_x += np.where(valid, x, 0.0)
            sum_y += np.where(valid, y, 0.0)
            count += valid

        # Cell corners, solid corners and edge crossings are vertices of the
        # convex intersection. Their fixed-order average remains inside it.
        for x, y in ((left, bottom), (right, bottom), (right, top), (left, top)):
            du, dv = cs * (x - ox) + sn * (y - oy), -sn * (x - ox) + cs * (y - oy)
            add(x, y, (np.abs(du) <= hl + tolerance) & (np.abs(dv) <= hw + tolerance))
        corners = [(ox + cs * u - sn * v, oy + sn * u + cs * v)
                   for u, v in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]
        for x, y in corners:
            add(x, y, (x >= left - tolerance) & (x <= right + tolerance) &
                (y >= bottom - tolerance) & (y <= top + tolerance))
        for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
            ex, ey = x1 - x0, y1 - y0
            fraction_tolerance = tolerance / math.hypot(ex, ey)
            if abs(ex) > tolerance:
                for x in (left, right):
                    fraction = (x - x0) / ex
                    y = y0 + fraction * ey
                    add(x, y, (fraction >= -fraction_tolerance) &
                        (fraction <= 1 + fraction_tolerance) &
                        (y >= bottom - tolerance) & (y <= top + tolerance))
            if abs(ey) > tolerance:
                for y in (bottom, top):
                    fraction = (y - y0) / ey
                    x = x0 + fraction * ex
                    add(x, y, (fraction >= -fraction_tolerance) &
                        (fraction <= 1 + fraction_tolerance) &
                        (x >= left - tolerance) & (x <= right + tolerance))
        if np.any(count == 0):
            raise ValueError('solid cell intersection has no surface representative')
        px[outside], py[outside] = sum_x / count, sum_y / count
        return px, py

    def _discrete_scene(self, world_x, world_y, origin_ix, origin_iy, resolution):
        """Build bounded local truth; visibility decides which cells are returned."""
        n = world_x.shape[0]
        origin_x, origin_y = origin_ix * resolution, origin_iy * resolution
        solid = np.zeros((n, n), dtype=bool)
        height = self._ground_elevation(world_x, world_y)
        xf, yf, sf, hf = world_x.ravel(), world_y.ravel(), solid.ravel(), height.ravel()
        tolerance = 8 * np.finfo(np.float64).eps * self.size_m
        half = resolution / 2
        nearby = self._nearby(origin_x - tolerance, origin_x + n * resolution + tolerance,
                              origin_y - tolerance, origin_y + n * resolution + tolerance)
        for index in nearby:
            geometry = self._geometry[index]
            ox, oy, hl, hw, cs, sn, obstacle_height = geometry
            ex, ey = self._extent_x[index], self._extent_y[index]
            left = max(0, math.floor((ox - ex - origin_x) / resolution) - 1)
            right = min(n, math.ceil((ox + ex - origin_x) / resolution) + 1)
            bottom = max(0, math.floor((oy - ey - origin_y) / resolution) - 1)
            top = min(n, math.ceil((oy + ey - origin_y) / resolution) + 1)
            ids = (np.arange(bottom, top)[:, None] * n +
                   np.arange(left, right)[None, :]).ravel()
            dx, dy = xf[ids] - ox, yf[ids] - oy
            projected_half = half * (abs(cs) + abs(sn))
            overlaps = (
                (np.abs(dx) <= ex + half + tolerance) &
                (np.abs(dy) <= ey + half + tolerance) &
                (np.abs(cs * dx + sn * dy) <= hl + projected_half + tolerance) &
                (np.abs(-sn * dx + cs * dy) <= hw + projected_half + tolerance))
            ids = ids[overlaps]
            gx, gy = origin_ix + ids % n, origin_iy + ids // n
            px, py = self._cell_surface_points(
                xf[ids], yf[ids], gx * resolution, (gx + 1) * resolution,
                gy * resolution, (gy + 1) * resolution, geometry, tolerance)
            u, v = cs * (px - ox) + sn * (py - oy), -sn * (px - ox) + cs * (py - oy)
            representative = (self._ground_elevation(px, py) + obstacle_height +
                              0.06 * np.sin(u * 2.1) * np.sin(v * 1.7))
            # Highest representative wins where solids overlap the same cell.
            # The rule and its inputs are independent of which ray sees it.
            hf[ids] = np.where(sf[ids], np.maximum(hf[ids], representative), representative)
            sf[ids] = True
        return solid, height.astype(np.float32)

    def _visible_cells(self, solid, dx, dy, candidate, cx, cy,
                       origin_ix, origin_iy, resolution):
        """Trace exact centre rays through cells, including grid-corner contact."""
        n = solid.shape[0]
        ids = np.flatnonzero(candidate)
        visible = np.zeros(solid.shape, dtype=bool)
        if ids.size == 0:
            return visible
        target_x, target_y = ids % n, ids // n
        start_gx, start_gy = math.floor(cx / resolution), math.floor(cy / resolution)
        ix = np.full(ids.size, start_gx - origin_ix, dtype=np.int64)
        iy = np.full(ids.size, start_gy - origin_iy, dtype=np.int64)
        vx, vy = dx.ravel()[ids], dy.ravel()[ids]
        sx, sy = np.sign(vx).astype(np.int64), np.sign(vy).astype(np.int64)
        delta_x = np.divide(resolution, np.abs(vx),
                            out=np.full(vx.shape, np.inf), where=vx != 0)
        delta_y = np.divide(resolution, np.abs(vy),
                            out=np.full(vy.shape, np.inf), where=vy != 0)
        boundary_x = (start_gx + (sx > 0)) * resolution
        boundary_y = (start_gy + (sy > 0)) * resolution
        next_x = np.divide(boundary_x - cx, vx,
                           out=np.full(vx.shape, np.inf), where=vx != 0)
        next_y = np.divide(boundary_y - cy, vy,
                           out=np.full(vy.shape, np.inf), where=vy != 0)
        ray_length = np.hypot(vx, vy)
        tolerance = 8 * np.finfo(np.float64).eps * self.size_m
        tie_tolerance = np.divide(tolerance, ray_length,
                                  out=np.zeros(vx.shape), where=ray_length > 0)
        active = np.ones(ids.size, dtype=bool)
        for _ in range(2 * n + 2):
            working = np.flatnonzero(active)
            if working.size == 0:
                break
            reached = (ix[working] == target_x[working]) & (iy[working] == target_y[working])
            visible.ravel()[ids[working[reached]]] = True
            stopped = reached | solid[iy[working], ix[working]]
            active[working[stopped]] = False
            working = working[~stopped]
            if working.size == 0:
                continue
            move_x = next_x[working] <= next_y[working] + tie_tolerance[working]
            move_y = next_y[working] <= next_x[working] + tie_tolerance[working]
            corners = np.flatnonzero(move_x & move_y)
            if corners.size:
                rays = working[corners]
                corner_blocked = (solid[iy[rays], ix[rays] + sx[rays]] |
                                  solid[iy[rays] + sy[rays], ix[rays]])
                blocked_positions = corners[corner_blocked]
                active[working[blocked_positions]] = False
                keep = np.ones(working.size, dtype=bool)
                keep[blocked_positions] = False
                working, move_x, move_y = working[keep], move_x[keep], move_y[keep]
            step_x, step_y = working[move_x], working[move_y]
            ix[step_x] += sx[step_x]
            iy[step_y] += sy[step_y]
            next_x[step_x] += delta_x[step_x]
            next_y[step_y] += delta_y[step_y]
        if np.any(active):
            raise ValueError('sensor ray traversal exceeded the local window')
        return visible

    def observe(self, cx, cy, yaw, resolution=0.2, range_m=12.0,
                fov_deg=120.0, window_m=28.0, near_field_radius_m=1.4):
        """Return a local sensor snapshot with logical ``values[y, x]``.

        ``x`` and ``y`` are ascending one-dimensional cell-centre coordinates;
        ``origin_x/y`` are the lower-left cell boundaries snapped to resolution.
        Cell centres are calculated from global integer indices, so a shared
        cell has identical coordinates in every rolling window.
        ``center_x/y`` and ``length_m`` describe the actual rounded window.
        A solid cell carries a fixed surface representative from its geometric
        intersection, which can differ from the continuous height at its centre.
        Masked cells are float32 NaN, including outside-range/FOV/world cells
        and cells beyond the first solid cell. This function retains no history.
        """
        if not all(math.isfinite(v) for v in
                   (cx, cy, yaw, resolution, range_m, fov_deg, window_m,
                    near_field_radius_m)):
            raise ValueError('sensor pose and geometry must be finite')
        if (resolution <= 0 or range_m <= 0 or window_m < resolution or
                not 0 < fov_deg <= 360 or not 0 <= near_field_radius_m <= range_m):
            raise ValueError('invalid sensor resolution, range, FOV, window or near field')
        n = max(1, int(round(window_m / resolution)))
        length = n * resolution
        origin_ix = math.floor(cx / resolution - n / 2)
        origin_iy = math.floor(cy / resolution - n / 2)
        if n == 1:
            # A one-cell public window still contains the ray origin, including
            # when the eye lies exactly on a grid boundary.
            origin_ix, origin_iy = math.floor(cx / resolution), math.floor(cy / resolution)
        origin_x, origin_y = origin_ix * resolution, origin_iy * resolution
        # Adding a floating window origin to local offsets can round the same
        # world cell differently after movement, changing obstacle membership.
        offsets = np.arange(n, dtype=np.int64)
        x = (origin_ix + offsets + 0.5) * resolution
        y = (origin_iy + offsets + 0.5) * resolution
        xx, yy = np.meshgrid(x, y)
        dx, dy = xx - cx, yy - cy
        distance = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx)
        relative = np.arctan2(np.sin(angle - yaw), np.cos(angle - yaw))
        candidate = (distance <= range_m) & \
            ((np.abs(relative) <= math.radians(fov_deg) / 2) |
             (distance <= near_field_radius_m))
        candidate &= (np.abs(xx) <= self.size_m / 2) & (np.abs(yy) <= self.size_m / 2)
        solid, heights = self._discrete_scene(xx, yy, origin_ix, origin_iy, resolution)
        mask = self._visible_cells(solid, dx, dy, candidate, cx, cy,
                                   origin_ix, origin_iy, resolution)
        values = np.full((n, n), np.nan, dtype=np.float32)
        values[mask] = heights[mask]
        return {
            'center_x': (origin_ix + n / 2) * resolution,
            'center_y': (origin_iy + n / 2) * resolution,
            'origin_x': origin_x, 'origin_y': origin_y,
            'resolution': resolution, 'length_m': length,
            'x': x, 'y': y, 'values': values, 'mask': mask,
        }

    def preview(self, resolution=2.0):
        """Sample a coarse display-only mesh; never use it as sensor evidence."""
        if not math.isfinite(resolution) or resolution < 0.5:
            raise ValueError('preview resolution must be finite and at least 0.5 m')
        n = max(1, int(math.ceil(self.size_m / resolution)))
        actual_resolution = self.size_m / n
        axis = -self.size_m / 2 + (np.arange(n) + 0.5) * actual_resolution
        x, y = np.meshgrid(axis, axis)
        return {'x': axis, 'y': axis.copy(),
                'values': self.elevation(x, y).astype(np.float32),
                'resolution': actual_resolution, 'origin_x': -self.size_m / 2,
                'origin_y': -self.size_m / 2}
