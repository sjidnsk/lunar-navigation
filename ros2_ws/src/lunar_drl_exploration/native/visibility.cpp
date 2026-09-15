#include <algorithm>
#include <limits>
#include <numbers>
#include <utility>

#include "grid_buffer.hpp"
namespace {
// Integer center-to-center supercover, including BOTH cells at corner
// crossings. Only the target may be a first-hit blocker. Missing map borders
// are not walls.
bool line(const std::uint8_t* b, int w, int x, int y, int tx, int ty) {
  const int dx = std::abs(tx - x), dy = std::abs(ty - y), sx = tx > x ? 1 : -1,
            sy = ty > y ? 1 : -1;
  int ix = 0, iy = 0;
  const auto blocked = [&](int cx, int cy) {
    return !(cx == tx && cy == ty) &&
           b[static_cast<std::size_t>(cy) * w + cx] == 2;
  };
  while (ix < dx || iy < dy) {
    const auto ax = static_cast<std::int64_t>(1 + 2 * ix) * dy;
    const auto ay = static_cast<std::int64_t>(1 + 2 * iy) * dx;
    if (ax == ay) {
      if (blocked(x + sx, y) || blocked(x, y + sy)) return false;
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
    if (blocked(x, y)) return false;
  }
  return true;
}
void dimensions(const Buffer& b, int& h, int& w) {
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
}  // namespace
PyObject* observe(PyObject*, PyObject* args) {
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
          if (std::hypot(double(tx - x), double(ty - y)) > r + 1e-10) continue;
          if ((tx != x || ty != y) &&
              std::abs(std::remainder(std::atan2(ty - y, tx - x) - yaw,
                                      2 * std::numbers::pi)) > fov * .5 + 1e-12)
            continue;
          if (line(b.data<std::uint8_t>(), w, x, y, tx, ty))
            o.data<std::uint8_t>()[static_cast<std::size_t>(ty) * w + tx] = 1;
        }
    }
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    return error(e);
  }
}
PyObject* reachable(PyObject*, PyObject* args) {
  PyObject *mo, *oo;
  int x, y;
  if (!PyArg_ParseTuple(args, "OiiO", &mo, &x, &y, &oo)) return nullptr;
  try {
    Buffer m(mo, "B", 2), o(oo, "B", 2, true);
    int h, w;
    dimensions(m, h, w);
    o.shape(h, w);
    if (x < 0 || x >= w || y < 0 || y >= h)
      throw std::invalid_argument("start outside terrain");
    {
      ReleasedGIL release;
      auto* out = o.data<std::uint8_t>();
      auto* state = m.data<std::uint8_t>();
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
              if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
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
  } catch (const std::exception& e) {
    return error(e);
  }
}
PyObject* visible_union(PyObject*, PyObject* args) {
  PyObject *bo, *ro, *co, *oo;
  double r;
  if (!PyArg_ParseTuple(args, "OOOdO", &bo, &ro, &co, &r, &oo)) return nullptr;
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
      auto* block = b.data<std::uint8_t>();
      auto* rr = reach.data<std::uint8_t>();
      auto* out = o.data<std::uint8_t>();
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
          if (!candidate.data<std::uint8_t>()[k]) continue;
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
            if (!boundary) continue;
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
  } catch (const std::exception& e) {
    return error(e);
  }
}
