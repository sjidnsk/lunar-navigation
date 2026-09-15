#include <algorithm>
#include <array>
#include <list>
#include <memory>
#include <mutex>
#include <numbers>
#include <numeric>
#include <tuple>
#include <utility>

#include "grid_buffer.hpp"
namespace {
constexpr double tau = 2 * std::numbers::pi;
constexpr std::size_t cache_limit = 128 * 1024 * 1024;
struct Cell {
  int x, y;
  bool operator==(const Cell &) const = default;
};
// A node is one simultaneous event and its unique prefix. Sorted beam support
// is stored as contiguous runs; a wrap or an intervening shorter beam starts
// a distinct node, retaining exact support without filling any angular gaps.
// Exact corner contacts have singleton angular support, before diagonal entry.
struct Node {
  Cell a{}, b{};
  int count = 1, parent = -1, child = -1, next = -1, first = 0, last = 0;
};
struct Fan {
  int d, side;
  std::vector<Node> nodes;
  std::vector<double> angles;
  std::vector<std::vector<int>> reverse;
  std::vector<Cell> offsets;
  explicit Fan(double radius, int width, int height)
      : d(static_cast<int>(std::ceil(radius))), side(2 * d + 1) {
    // Explicit resource failure, never silently alter range or beam density.
    if (d > 512)
      throw std::invalid_argument("exact ray template exceeds resource budget");
    reverse.resize(static_cast<std::size_t>(side) * side);
    nodes.emplace_back();
    struct Tip {
      Cell cell;
      double angle;
    };
    std::vector<Tip> tips;
    for (int y = -d; y <= d; ++y)
      for (int x = -d; x <= d; ++x) {
        if (std::hypot(double(x), double(y)) > radius + 1e-10)
          continue;
        offsets.push_back({x, y});
        if (!x && !y)
          continue;
        const int gcd = std::gcd(std::abs(x), std::abs(y));
        // Only the farthest collinear tip is needed; its events contain every
        // shorter tip's prefix, and its angle is identical.
        if (std::hypot(double(x + x / gcd), double(y + y / gcd)) <=
            radius + 1e-10)
          continue;
        tips.push_back({{x, y}, std::atan2(y, x)});
      }
    std::sort(offsets.begin(), offsets.end(), [](Cell a, Cell b) {
      const int aa = a.x * a.x + a.y * a.y, bb = b.x * b.x + b.y * b.y;
      return aa == bb ? std::pair(a.x, a.y) < std::pair(b.x, b.y) : aa < bb;
    });
    std::sort(tips.begin(), tips.end(),
              [](const Tip &a, const Tip &b) { return a.angle < b.angle; });
    for (const auto &tip : tips) {
      const int beam = angles.size();
      angles.push_back(tip.angle);
      int parent = 0;
      bool clipped = false;
      const int cap_x = std::min(d, width - 1), cap_y = std::min(d, height - 1);
      auto append = [&](Cell a, Cell b, int count) {
        const auto outside = [&](Cell c) {
          return std::abs(c.x) > cap_x || std::abs(c.y) > cap_y;
        };
        clipped = outside(a) || (count == 2 && outside(b));
        if (outside(a) && (count == 1 || outside(b)))
          return;
        int n = nodes[parent].child;
        while (n >= 0 &&
               !(nodes[n].a == a && nodes[n].b == b &&
                 nodes[n].count == count && nodes[n].last == beam - 1))
          n = nodes[n].next;
        if (n < 0) {
          n = nodes.size();
          Node node;
          node.a = a;
          node.b = b;
          node.count = count;
          node.parent = parent;
          node.next = nodes[parent].child;
          node.first = beam;
          node.last = beam;
          nodes.push_back(node);
          nodes[parent].child = n;
          reverse[index(a)].push_back(n);
          if (count == 2)
            reverse[index(b)].push_back(n);
          if (nodes.size() * sizeof(Node) > cache_limit / 2)
            throw std::invalid_argument(
                "exact ray prefix template exceeds resource budget");
        } else {
          nodes[n].last = beam;
        }
        parent = n;
      };
      const int dx = std::abs(tip.cell.x), dy = std::abs(tip.cell.y);
      const int sx = tip.cell.x > 0 ? 1 : -1, sy = tip.cell.y > 0 ? 1 : -1;
      int x = 0, y = 0, ix = 0, iy = 0;
      while (!clipped && (ix < dx || iy < dy)) {
        const auto ax = std::int64_t(1 + 2 * ix) * dy,
                   ay = std::int64_t(1 + 2 * iy) * dx;
        if (ax == ay) {
          append({x + sx, y}, {x, y + sy}, 2);
          if (clipped)
            break;
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
        append({x, y}, {}, 1);
      }
    }
    // Prefer prefixes supported by the target's center beam; every reverse
    // entry is still retained and tested when that fast positive path fails.
    for (Cell c : offsets) {
      if (!c.x && !c.y)
        continue;
      const auto angle = std::atan2(c.y, c.x);
      auto &entries = reverse[index(c)];
      std::stable_partition(entries.begin(), entries.end(), [&](int n) {
        return angles[nodes[n].first] <= angle + 1e-14 &&
               angles[nodes[n].last] >= angle - 1e-14;
      });
    }
    if (bytes() > cache_limit)
      throw std::invalid_argument("exact ray template exceeds cache budget");
  }
  std::size_t index(Cell c) const {
    return static_cast<std::size_t>(c.y + d) * side + c.x + d;
  }
  const std::vector<int> *entries(Cell c) const {
    if (std::abs(c.x) > d || std::abs(c.y) > d)
      return nullptr;
    return &reverse[index(c)];
  }
  std::size_t bytes() const {
    std::size_t n = nodes.capacity() * sizeof(Node) +
                    angles.capacity() * sizeof(double) +
                    reverse.capacity() * sizeof(std::vector<int>) +
                    offsets.capacity() * sizeof(Cell);
    for (const auto &v : reverse)
      n += v.capacity() * sizeof(int);
    return n;
  }
};
// Process-owned immutable LRU, serialized construction, shared ownership keeps
// evicted templates alive only for active calls. Queries run without a lock.
std::mutex fan_mutex;
using FanKey = std::tuple<double, int, int>;
std::list<std::pair<FanKey, std::shared_ptr<const Fan>>> fan_cache;
std::shared_ptr<const Fan> fan(double radius, int width, int height) {
  const int d = static_cast<int>(std::ceil(radius));
  const FanKey key{radius, std::min(width, d + 1), std::min(height, d + 1)};
  std::lock_guard lock(fan_mutex);
  for (auto i = fan_cache.begin(); i != fan_cache.end(); ++i)
    if (i->first == key) {
      auto result = i->second;
      fan_cache.splice(fan_cache.begin(), fan_cache, i);
      return result;
    }
  auto result = std::make_shared<Fan>(radius, width, height);
  std::size_t bytes = result->bytes();
  for (const auto &item : fan_cache)
    bytes += item.second->bytes();
  while (!fan_cache.empty() && (bytes > cache_limit || fan_cache.size() >= 8)) {
    bytes -= fan_cache.back().second->bytes();
    fan_cache.pop_back();
  }
  fan_cache.emplace_front(key, result);
  return result;
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
void pose_check(int x, int y, int w, int h) {
  if (x < 0 || y < 0 || x >= w || y >= h)
    throw std::invalid_argument("visibility cell outside grid");
}
void target_check(Buffer &t, int w, int h) {
  static_assert(sizeof(long) == sizeof(std::int64_t));
  if (t.view.shape[1] != 2)
    throw std::invalid_argument("invalid visibility targets");
  for (Py_ssize_t i = 0; i < t.view.shape[0] * 2; i += 2) {
    const auto x = t.data<std::int64_t>()[i], y = t.data<std::int64_t>()[i + 1];
    if (x < 0 || y < 0 || x >= w || y >= h)
      throw std::invalid_argument("visibility cell outside grid");
  }
}
struct Grid {
  const std::uint8_t *b;
  int w, h, x, y;
  std::int64_t index(Cell c) const {
    const int xx = x + c.x, yy = y + c.y;
    return xx < 0 || yy < 0 || xx >= w || yy >= h
               ? -1
               : static_cast<std::int64_t>(yy) * w + xx;
  }
  bool stops(const Node &n) const {
    auto k = index(n.a);
    if (k < 0 || b[k] == 2)
      return true;
    if (n.count == 2) {
      k = index(n.b);
      if (k < 0 || b[k] == 2)
        return true;
    }
    return false;
  }
};
// All STRICT predecessors must transmit; the target event is emitted whole
// even when its sibling blocks. Check backwards for cheap negative rejection.
bool prefix_clear(const Fan &f, const Grid &g, int n) {
  if (g.b[static_cast<std::size_t>(g.y) * g.w + g.x] == 2)
    return false;
  for (n = f.nodes[n].parent; n > 0; n = f.nodes[n].parent)
    if (g.stops(f.nodes[n]))
      return false;
  return true;
}
int witness(const Fan &f, const Grid &g, Cell target) {
  const Cell offset{target.x - g.x, target.y - g.y};
  if (!offset.x && !offset.y)
    return 0;
  if (const auto *entries = f.entries(offset))
    for (int n : *entries)
      if (prefix_clear(f, g, n))
        return n;
  return -1;
}
struct Selection {
  std::array<std::pair<int, int>, 2> intervals{};
  int count = 0;
  Selection(const Fan &f, double yaw, double fov) {
    if (!std::isfinite(yaw) || !std::isfinite(fov) || fov <= 0 ||
        fov > tau + 1e-10)
      throw std::invalid_argument("invalid sensor yaw/FOV");
    auto add = [&](double low, double high) {
      int a = std::lower_bound(f.angles.begin(), f.angles.end(), low - 1e-12) -
              f.angles.begin();
      int b = std::upper_bound(f.angles.begin(), f.angles.end(), high + 1e-12) -
              f.angles.begin();
      if (a < b)
        intervals[count++] = {a, b - 1};
    };
    if (fov >= tau)
      add(-std::numbers::pi, std::numbers::pi);
    else {
      yaw = std::remainder(yaw, tau);
      const double low = yaw - fov / 2, high = yaw + fov / 2;
      if (low <= -std::numbers::pi + 1e-12) {
        add(low + tau, std::numbers::pi);
        add(-std::numbers::pi, high);
      } else if (high >= std::numbers::pi - 1e-12) {
        add(low, std::numbers::pi);
        add(-std::numbers::pi, high - tau);
      } else
        add(low, high);
    }
  }
  bool includes(const Node &n) const {
    for (int i = 0; i < count; ++i)
      if (n.first <= intervals[i].second && n.last >= intervals[i].first)
        return true;
    return false;
  }
};
} // namespace
PyObject *observe(PyObject *, PyObject *args) {
  PyObject *bo, *oo;
  int x, y;
  double r, yaw, fov;
  if (!PyArg_ParseTuple(args, "OiidddO", &bo, &x, &y, &r, &yaw, &fov, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), out(oo, "B", 2, true);
    int h, w;
    dimensions(b, h, w);
    out.shape(h, w);
    range_check(r);
    pose_check(x, y, w, h);
    {
      ReleasedGIL release;
      auto f = fan(r, w, h);
      Selection selected(*f, yaw, fov);
      Grid g{b.data<std::uint8_t>(), w, h, x, y};
      auto *o = out.data<std::uint8_t>();
      std::fill_n(o, static_cast<std::size_t>(h) * w, 0);
      o[static_cast<std::size_t>(y) * w + x] = 1;
      if (b.data<std::uint8_t>()[static_cast<std::size_t>(y) * w + x] != 2) {
        std::vector<int> stack;
        for (int n = f->nodes[0].child; n >= 0; n = f->nodes[n].next)
          stack.push_back(n);
        while (!stack.empty()) {
          int i = stack.back();
          stack.pop_back();
          const auto &n = f->nodes[i];
          if (!selected.includes(n))
            continue;
          const auto a = g.index(n.a);
          if (a >= 0)
            o[a] = 1;
          if (n.count == 2) {
            const auto bb = g.index(n.b);
            if (bb >= 0)
              o[bb] = 1;
          }
          if (!g.stops(n))
            for (int child = n.child; child >= 0; child = f->nodes[child].next)
              stack.push_back(child);
        }
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
    Buffer b(bo, "B", 2), rr(ro, "B", 2), c(co, "B", 2), o(oo, "B", 2, true);
    int h, w;
    dimensions(b, h, w);
    rr.shape(h, w);
    c.shape(h, w);
    o.shape(h, w);
    range_check(r);
    {
      ReleasedGIL release;
      auto f = fan(r, w, h);
      auto *out = o.data<std::uint8_t>();
      auto *reach = rr.data<std::uint8_t>();
      std::fill_n(out, static_cast<std::size_t>(h) * w, 0);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          auto k = static_cast<std::size_t>(y) * w + x;
          if (!c.data<std::uint8_t>()[k])
            continue;
          if (reach[k]) {
            out[k] = 1;
            continue;
          }
          for (Cell delta : f->offsets) {
            const int sx = x - delta.x, sy = y - delta.y;
            if (sx < 0 || sy < 0 || sx >= w || sy >= h ||
                !reach[static_cast<std::size_t>(sy) * w + sx])
              continue;
            if (witness(*f, {b.data<std::uint8_t>(), w, h, sx, sy}, {x, y}) >=
                0) {
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
    pose_check(x, y, w, h);
    target_check(t, w, h);
    o.shape(t.view.shape[0], 1);
    range_check(r);
    {
      ReleasedGIL release;
      auto f = fan(r, w, h);
      Grid g{b.data<std::uint8_t>(), w, h, x, y};
      for (Py_ssize_t i = 0; i < t.view.shape[0]; ++i)
        o.data<std::uint8_t>()[i] =
            witness(*f, g,
                    {int(t.data<std::int64_t>()[2 * i]),
                     int(t.data<std::int64_t>()[2 * i + 1])}) >= 0;
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}
// FORWARD source-array -> fixed target relation; beam visibility is directed.
PyObject *visible_sources(PyObject *, PyObject *args) {
  PyObject *bo, *so, *oo;
  int x, y;
  double r;
  if (!PyArg_ParseTuple(args, "OiidOO", &bo, &x, &y, &r, &so, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), s(so, "l", 2), o(oo, "B", 2, true);
    int h, w;
    dimensions(b, h, w);
    pose_check(x, y, w, h);
    target_check(s, w, h);
    o.shape(s.view.shape[0], 1);
    range_check(r);
    {
      ReleasedGIL release;
      auto f = fan(r, w, h);
      for (Py_ssize_t i = 0; i < s.view.shape[0]; ++i) {
        Grid g{b.data<std::uint8_t>(), w, h, int(s.data<std::int64_t>()[2 * i]),
               int(s.data<std::int64_t>()[2 * i + 1])};
        o.data<std::uint8_t>()[i] = witness(*f, g, {x, y}) >= 0;
      }
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}
PyObject *directional_targets(PyObject *, PyObject *args) {
  PyObject *bo, *to, *ho, *oo;
  int x, y;
  double r, fov;
  if (!PyArg_ParseTuple(args, "OiidOOdO", &bo, &x, &y, &r, &to, &ho, &fov, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), t(to, "l", 2), headings(ho, "d", 2),
        o(oo, "B", 2, true);
    int h, w;
    dimensions(b, h, w);
    pose_check(x, y, w, h);
    target_check(t, w, h);
    range_check(r);
    headings.shape(8, 1);
    o.shape(t.view.shape[0], 8);
    {
      ReleasedGIL release;
      auto f = fan(r, w, h);
      Grid g{b.data<std::uint8_t>(), w, h, x, y};
      std::vector<Selection> selected;
      for (int j = 0; j < 8; ++j)
        selected.emplace_back(*f, headings.data<double>()[j], fov);
      std::fill_n(o.data<std::uint8_t>(), t.view.shape[0] * 8, 0);
      for (Py_ssize_t i = 0; i < t.view.shape[0]; ++i) {
        Cell delta{int(t.data<std::int64_t>()[2 * i]) - x,
                   int(t.data<std::int64_t>()[2 * i + 1]) - y};
        auto *row = o.data<std::uint8_t>() + 8 * i;
        if (!delta.x && !delta.y) {
          std::fill_n(row, 8, 1);
          continue;
        }
        if (const auto *entries = f->entries(delta))
          for (int n : *entries) {
            std::uint8_t mask = 0;
            for (int j = 0; j < 8; ++j)
              if (!row[j] && selected[j].includes(f->nodes[n]))
                mask |= 1 << j;
            if (mask && prefix_clear(*f, g, n))
              for (int j = 0; j < 8; ++j)
                if (mask & (1 << j))
                  row[j] = 1;
            if (std::all_of(row, row + 8,
                            [](auto value) { return value != 0; }))
              break;
          }
      }
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}
PyObject *first_pending(PyObject *, PyObject *args) {
  PyObject *bo, *ko, *to, *oo;
  int x, y;
  double r;
  if (!PyArg_ParseTuple(args, "OOiidOO", &bo, &ko, &x, &y, &r, &to, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), known(ko, "B", 2), t(to, "l", 2), o(oo, "l", 2, true);
    int h, w;
    dimensions(b, h, w);
    pose_check(x, y, w, h);
    target_check(t, w, h);
    range_check(r);
    known.shape(h, w);
    o.shape(t.view.shape[0], 1);
    {
      ReleasedGIL release;
      auto f = fan(r, w, h);
      Grid g{b.data<std::uint8_t>(), w, h, x, y};
      std::vector<int> path;
      for (Py_ssize_t i = 0; i < t.view.shape[0]; ++i) {
        auto &out = o.data<std::int64_t>()[i];
        out = -1;
        int n = witness(*f, g,
                        {int(t.data<std::int64_t>()[2 * i]),
                         int(t.data<std::int64_t>()[2 * i + 1])});
        if (n < 0)
          continue;
        auto source = static_cast<std::int64_t>(y) * w + x;
        if (!known.data<std::uint8_t>()[source]) {
          out = source;
          continue;
        }
        path.clear();
        for (; n > 0; n = f->nodes[n].parent)
          path.push_back(n);
        for (auto it = path.rbegin(); it != path.rend() && out < 0; ++it) {
          const auto &node = f->nodes[*it];
          const auto a = g.index(node.a),
                     bb = node.count == 2 ? g.index(node.b) : -1;
          if (a >= 0 && !known.data<std::uint8_t>()[a])
            out = a;
          else if (bb >= 0 && !known.data<std::uint8_t>()[bb])
            out = bb;
        }
      }
    }
    Py_RETURN_NONE;
  } catch (const std::exception &e) {
    return error(e);
  }
}
PyObject *visible_witnesses(PyObject *, PyObject *args) {
  PyObject *bo, *ro, *to, *oo;
  double r;
  if (!PyArg_ParseTuple(args, "OOOdO", &bo, &ro, &to, &r, &oo))
    return nullptr;
  try {
    Buffer b(bo, "B", 2), rr(ro, "B", 2), t(to, "l", 2), o(oo, "l", 2, true);
    int h, w;
    dimensions(b, h, w);
    rr.shape(h, w);
    target_check(t, w, h);
    o.shape(t.view.shape[0], 2);
    range_check(r);
    {
      ReleasedGIL release;
      auto f = fan(r, w, h);
      std::fill_n(o.data<std::int64_t>(), t.view.shape[0] * 2, -1);
      for (Py_ssize_t i = 0; i < t.view.shape[0]; ++i) {
        const int x = t.data<std::int64_t>()[2 * i],
                  y = t.data<std::int64_t>()[2 * i + 1];
        for (Cell delta : f->offsets) {
          const int sx = x - delta.x, sy = y - delta.y;
          if (sx < 0 || sy < 0 || sx >= w || sy >= h ||
              !rr.data<std::uint8_t>()[static_cast<std::size_t>(sy) * w + sx])
            continue;
          if (witness(*f, {b.data<std::uint8_t>(), w, h, sx, sy}, {x, y}) >=
              0) {
            o.data<std::int64_t>()[2 * i] = sx;
            o.data<std::int64_t>()[2 * i + 1] = sy;
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
PyObject *visibility_cache_info(PyObject *, PyObject *) {
  std::lock_guard lock(fan_mutex);
  std::size_t bytes = 0, nodes = 0, beams = 0;
  for (const auto &item : fan_cache) {
    bytes += item.second->bytes();
    nodes += item.second->nodes.size();
    beams += item.second->angles.size();
  }
  return Py_BuildValue("{s:k,s:k,s:k,s:k,s:k}", "bytes", bytes, "limit_bytes",
                       cache_limit, "templates", fan_cache.size(),
                       "prefix_nodes", nodes, "beams", beams);
}
