#include <algorithm>
#include <limits>
#include <numbers>
#include <utility>

#include "grid_buffer.hpp"
namespace {
// Integer center-to-center supercover, including BOTH cells at corner
// crossings. Only the target may be a first-hit blocker. Missing map borders
// are not walls.
bool line(const std::uint8_t *b, int w, int x, int y, int tx, int ty,
          const std::uint8_t *known = nullptr, std::int64_t *first = nullptr) {
  const int dx = std::abs(tx - x), dy = std::abs(ty - y), sx = tx > x ? 1 : -1,
            sy = ty > y ? 1 : -1;
  int ix = 0, iy = 0;
  const int source_x = x, source_y = y;
  if (first)
    *first = known[static_cast<std::size_t>(y) * w + x]
                 ? -1
                 : static_cast<std::int64_t>(y) * w + x;
  const auto blocked = [&](int cx, int cy) {
    const auto index = static_cast<std::int64_t>(cy) * w + cx;
    // A traversal contact is not necessarily a center measurement. Verify
    // its own center ray and keep scanning if it is occluded; the visible
    // pending demand itself remains a guaranteed final candidate.
    if (first && *first < 0 && !known[index] &&
        line(b, w, source_x, source_y, cx, cy))
      *first = index;
    return !(cx == tx && cy == ty) &&
           b[static_cast<std::size_t>(cy) * w + cx] == 2;
  };
  while (ix < dx || iy < dy) {
    const auto ax = static_cast<std::int64_t>(1 + 2 * ix) * dy;
    const auto ay = static_cast<std::int64_t>(1 + 2 * iy) * dx;
    if (ax == ay) {
      if (blocked(x + sx, y) || blocked(x, y + sy))
        return false;
      x += sx;
      y += sy;
      ++ix;
      ++iy;
    } else if (ax < ay) {
      x += sx;
      ++ix;
    } else {
      y += sy;
      ++iy;
    }
    if (blocked(x, y))
      return false;
  }
  return true;
}
void dimensions(const Buffer &b, int &h, int &w) {
  if (b.view.shape[0] <= 0 || b.view.shape[1] <= 0 ||
      b.view.shape[0] > 100000 || b.view.shape[1] > 100000)
    throw std::invalid_argument("invalid visibility grid shape");
  h = b.view.shape[0];
  w = b.view.shape[1];
}
void range_check(double r) {
  if (!std::isfinite(r) || r < 0 || r > 100000)
    throw std::invalid_argument("invalid sensor range");
}
} // namespace
PyObject *observe(PyObject *, PyObject *args) {
  PyObject *bo, *oo;
  int x, y;
  double r, yaw, fov;
  if (!PyArg_ParseTuple(args, "OiidddO", &bo, &x, &y, &r, &yaw, &fov, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), o(oo, "B", 2, true);
    int h, w;
    dimensions(b, h, w);
    o.shape(h, w);
    range_check(r);
    if (x < 0 || x >= w || y < 0 || y >= h || !std::isfinite(yaw) ||
        !std::isfinite(fov) || fov <= 0 || fov > 2 * std::numbers::pi + 1e-10)
      throw std::invalid_argument("invalid sensor pose/FOV");
    {
      ReleasedGIL release;
      std::fill_n(o.data<std::uint8_t>(), static_cast<std::size_t>(h) * w, 0);
      const int d = static_cast<int>(std::ceil(r));
      for (int ty = std::max(0, y - d); ty <= std::min(h - 1, y + d); ++ty)
        for (int tx = std::max(0, x - d); tx <= std::min(w - 1, x + d); ++tx) {
          if (std::hypot(double(tx - x), double(ty - y)) > r + 1e-10)
            continue;
          if ((tx != x || ty != y) &&
              std::abs(std::remainder(std::atan2(ty - y, tx - x) - yaw,
                                      2 * std::numbers::pi)) > fov * .5 + 1e-12)
            continue;
          if (line(b.data<std::uint8_t>(), w, x, y, tx, ty))
            o.data<std::uint8_t>()[static_cast<std::size_t>(ty) * w + tx] = 1;
        }
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}
PyObject *reachable(PyObject *, PyObject *args) {
  PyObject *mo, *oo;
  int x, y;
  if (!PyArg_ParseTuple(args, "OiiO", &mo, &x, &y, &oo))
    return nullptr;
  try {
    Buffer m(mo, "B", 2), o(oo, "B", 2, true);
    int h, w;
    dimensions(m, h, w);
    o.shape(h, w);
    if (x < 0 || x >= w || y < 0 || y >= h)
      throw std::invalid_argument("start outside terrain");
    {
      ReleasedGIL release;
      auto *out = o.data<std::uint8_t>();
      auto *state = m.data<std::uint8_t>();
      std::fill_n(out, static_cast<std::size_t>(h) * w, 0);
      // Frontier deque drops visited entries, avoiding an O(component) queue.
      std::vector<std::size_t> current, next;
      if (state[static_cast<std::size_t>(y) * w + x] == 1) {
        current.push_back(static_cast<std::size_t>(y) * w + x);
        out[current[0]] = 1;
      }
      while (!current.empty()) {
        next.clear();
        for (auto k : current) {
          int cy = k / w, cx = k % w;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              int nx = cx + dx, ny = cy + dy;
              if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                continue;
              auto q = static_cast<std::size_t>(ny) * w + nx;
              if (out[q] || state[q] != 1 ||
                  state[static_cast<std::size_t>(cy) * w + nx] != 1 ||
                  state[static_cast<std::size_t>(ny) * w + cx] != 1)
                continue;
              out[q] = 1;
              next.push_back(q);
            }
        }
        current.swap(next);
      }
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}
PyObject *visible_union(PyObject *, PyObject *args) {
  PyObject *bo, *ro, *co, *oo;
  double r;
  if (!PyArg_ParseTuple(args, "OOOdO", &bo, &ro, &co, &r, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), reach(ro, "B", 2), candidate(co, "B", 2),
        o(oo, "B", 2, true);
    int h, w;
    dimensions(b, h, w);
    reach.shape(h, w);
    candidate.shape(h, w);
    o.shape(h, w);
    range_check(r);
    {
      ReleasedGIL release;
      auto *block = b.data<std::uint8_t>();
      auto *rr = reach.data<std::uint8_t>();
      auto *out = o.data<std::uint8_t>();
      std::fill_n(out, static_cast<std::size_t>(h) * w, 0);
      std::vector<std::pair<int, int>> offsets;
      const int d = static_cast<int>(std::ceil(r));
      for (int dy = -std::min(d, h - 1); dy <= std::min(d, h - 1); ++dy)
        for (int dx = -std::min(d, w - 1); dx <= std::min(d, w - 1); ++dx)
          if (std::hypot(double(dx), double(dy)) <= r + 1e-10)
            offsets.emplace_back(dx, dy);
      std::sort(offsets.begin(), offsets.end(), [](auto a, auto b) {
        return std::int64_t(a.first) * a.first +
                   std::int64_t(a.second) * a.second <
               std::int64_t(b.first) * b.first +
                   std::int64_t(b.second) * b.second;
      });
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const auto k = static_cast<std::size_t>(y) * w + x;
          if (!candidate.data<std::uint8_t>()[k])
            continue;
          if (rr[k]) {
            out[k] = 1;
            continue;
          }
          // An interior blocker cannot be a first hit. This necessary condition
          // avoids scanning every source for the solid interiors of cave walls.
          if (block[k] == 2) {
            bool boundary = false;
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && ny >= 0 && nx < w && ny < h &&
                    block[static_cast<std::size_t>(ny) * w + nx] != 2)
                  boundary = true;
              }
            if (!boundary)
              continue;
          }
          for (auto [dx, dy] : offsets) {
            int sx = x + dx, sy = y + dy;
            if (sx < 0 || sy < 0 || sx >= w || sy >= h ||
                !rr[static_cast<std::size_t>(sy) * w + sx])
              continue;
            if (line(block, w, sx, sy, x, y)) {
              out[k] = 1;
              break;
            }
          }
        }
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}

// Sparse targets share exactly observe()'s center-ray and first-hit semantics.
// Targets are int64 [N,2] in x,y order; output is uint8 [N,1].
PyObject *visible_targets(PyObject *, PyObject *args) {
  PyObject *bo, *to, *oo;
  int x, y;
  double r;
  if (!PyArg_ParseTuple(args, "OiidOO", &bo, &x, &y, &r, &to, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), t(to, "l", 2), o(oo, "B", 2, true);
    int h, w;
    dimensions(b, h, w);
    range_check(r);
    if (x < 0 || x >= w || y < 0 || y >= h || t.view.shape[1] != 2)
      throw std::invalid_argument("invalid visibility source/targets");
    o.shape(t.view.shape[0], 1);
    // NumPy int64 is 'l' on supported Linux LP64 targets.
    static_assert(sizeof(long) == sizeof(std::int64_t));
    for (Py_ssize_t i = 0; i < t.view.shape[0]; ++i) {
      const auto tx = t.data<std::int64_t>()[2 * i],
                 ty = t.data<std::int64_t>()[2 * i + 1];
      if (tx < 0 || tx >= w || ty < 0 || ty >= h)
        throw std::invalid_argument("visibility target outside grid");
    }
    {
      ReleasedGIL release;
      for (Py_ssize_t i = 0; i < t.view.shape[0]; ++i) {
        const int tx = t.data<std::int64_t>()[2 * i],
                  ty = t.data<std::int64_t>()[2 * i + 1];
        o.data<std::uint8_t>()[i] =
            std::hypot(double(tx - x), double(ty - y)) <= r + 1e-10 &&
            line(b.data<std::uint8_t>(), w, x, y, tx, ty);
      }
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}

// Identify the pending interface on an already visible demand ray using the
// SAME supercover traversal, including both cells at diagonal corners.
PyObject *first_pending(PyObject *, PyObject *args) {
  PyObject *bo, *ko, *to, *oo;
  int x, y;
  if (!PyArg_ParseTuple(args, "OOiiOO", &bo, &ko, &x, &y, &to, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), known(ko, "B", 2), t(to, "l", 2), o(oo, "l", 2, true);
    int h, w;
    dimensions(b, h, w);
    known.shape(h, w);
    if (x < 0 || x >= w || y < 0 || y >= h || t.view.shape[1] != 2)
      throw std::invalid_argument("invalid pending source/targets");
    o.shape(t.view.shape[0], 1);
    for (Py_ssize_t i = 0; i < t.view.shape[0]; ++i) {
      const auto tx = t.data<std::int64_t>()[2 * i],
                 ty = t.data<std::int64_t>()[2 * i + 1];
      if (tx < 0 || tx >= w || ty < 0 || ty >= h)
        throw std::invalid_argument("pending target outside grid");
    }
    {
      ReleasedGIL release;
      for (Py_ssize_t i = 0; i < t.view.shape[0]; ++i) {
        const int tx = t.data<std::int64_t>()[2 * i],
                  ty = t.data<std::int64_t>()[2 * i + 1];
        auto *first = o.data<std::int64_t>() + i;
        if (!line(b.data<std::uint8_t>(), w, x, y, tx, ty,
                  known.data<std::uint8_t>(), first))
          *first = -1;
      }
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}

// Sparse direct-observation witnesses, independent of movement-component labels.
// Reuse the exact center ray and stop at the first real R source for each demand.
PyObject* visible_witnesses(PyObject*, PyObject* args) {
  PyObject *bo, *ro, *to, *oo;
  double radius;
  if (!PyArg_ParseTuple(args, "OOOdO", &bo, &ro, &to, &radius, &oo)) return nullptr;
  try {
    Buffer b(bo, "B", 2), reachable(ro, "B", 2), targets(to, "l", 2),
        out(oo, "l", 2, true);
    int h, w;
    dimensions(b, h, w);
    reachable.shape(h, w);
    range_check(radius);
    if (targets.view.shape[1] != 2)
      throw std::invalid_argument("invalid witness targets");
    out.shape(targets.view.shape[0], 2);
    for (Py_ssize_t i = 0; i < targets.view.shape[0]; ++i) {
      const auto x = targets.data<std::int64_t>()[2*i];
      const auto y = targets.data<std::int64_t>()[2*i+1];
      if (x < 0 || x >= w || y < 0 || y >= h)
        throw std::invalid_argument("witness target outside grid");
    }
    {
      ReleasedGIL release;
      std::vector<std::pair<int, int>> offsets;
      const int d = static_cast<int>(std::ceil(radius));
      for (int dy = -std::min(d, h-1); dy <= std::min(d, h-1); ++dy)
        for (int dx = -std::min(d, w-1); dx <= std::min(d, w-1); ++dx)
          if (std::hypot(double(dx), double(dy)) <= radius + 1e-10)
            offsets.emplace_back(dx, dy);
      std::sort(offsets.begin(), offsets.end(), [](auto a, auto b) {
        const auto aa = std::int64_t(a.first)*a.first + std::int64_t(a.second)*a.second;
        const auto bb = std::int64_t(b.first)*b.first + std::int64_t(b.second)*b.second;
        return aa == bb ? a < b : aa < bb;
      });
      std::fill_n(out.data<std::int64_t>(), targets.view.shape[0]*2, -1);
      for (Py_ssize_t i = 0; i < targets.view.shape[0]; ++i) {
        const int tx = targets.data<std::int64_t>()[2*i];
        const int ty = targets.data<std::int64_t>()[2*i+1];
        for (const auto& [dx, dy] : offsets) {
          const int x = tx + dx, y = ty + dy;
          if (x < 0 || x >= w || y < 0 || y >= h ||
              !reachable.data<std::uint8_t>()[static_cast<std::size_t>(y)*w+x]) continue;
          if (line(b.data<std::uint8_t>(), w, x, y, tx, ty)) {
            out.data<std::int64_t>()[2*i] = x;
            out.data<std::int64_t>()[2*i+1] = y;
            break;
          }
        }
      }
    }
    Py_RETURN_NONE;
  } catch (const std::exception& e) { return error(e); }
}
