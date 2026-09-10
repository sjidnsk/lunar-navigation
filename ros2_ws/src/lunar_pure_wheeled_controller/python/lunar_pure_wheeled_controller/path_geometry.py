"""Immutable metric polyline; projection never depends on vertex density."""

from dataclasses import dataclass
import math


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class Projection:
    index: int
    s: float
    x: float
    y: float
    error: float


class Polyline:
    def __init__(self, points):
        values = tuple(tuple(float(v) for v in p) for p in points)
        if not values or any(
            len(p) != 3 or not all(math.isfinite(v) for v in p) for p in values
        ):
            raise ValueError("path must contain finite x/y/yaw triples")
        self.original = values
        self.final_yaw = values[-1][2]
        vertices = [values[0][:2]]
        for p in values[1:]:
            if math.dist(vertices[-1], p[:2]) > 1e-8:
                vertices.append(p[:2])
        self.points = tuple(vertices)
        self.lengths = tuple(math.dist(a, b) for a, b in zip(vertices, vertices[1:]))
        arcs = [0.0]
        for length in self.lengths:
            arcs.append(arcs[-1] + length)
        self.arcs = tuple(arcs)
        self.length = arcs[-1]
        self.headings = tuple(
            math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(vertices, vertices[1:])
        )

    def project(self, x, y, index=None, reach=None):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("finite position required")
        if not self.lengths:
            px, py = self.points[0]
            return Projection(0, 0.0, px, py, math.hypot(x - px, y - py))
        candidates = []
        first = 0 if index is None else max(0, index - 1)
        for i in range(first, len(self.lengths)):
            if reach is not None and self.arcs[i] > reach:
                break
            a, b = self.points[i : i + 2]
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = max(
                0.0,
                min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / self.lengths[i] ** 2),
            )
            px, py = a[0] + t * dx, a[1] + t * dy
            candidates.append(
                Projection(
                    i,
                    self.arcs[i] + t * self.lengths[i],
                    px,
                    py,
                    math.hypot(x - px, y - py),
                )
            )
        return min(candidates, key=lambda p: (p.error, p.s))

    def point_at(self, s):
        s = max(0.0, min(self.length, s))
        for i, length in enumerate(self.lengths):
            if s <= self.arcs[i + 1]:
                t = (s - self.arcs[i]) / length
                a, b = self.points[i : i + 2]
                return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        return self.points[-1]

    def stop_arcs(self, threshold):
        return tuple(
            self.arcs[i]
            for i in range(1, len(self.headings))
            if abs(angle(self.headings[i] - self.headings[i - 1])) >= threshold
        ) + (self.length,)
