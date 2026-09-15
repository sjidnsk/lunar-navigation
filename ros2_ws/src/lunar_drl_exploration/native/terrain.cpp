#include <algorithm>
#include <optional>

#include "grid_buffer.hpp"
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
namespace nav = lunar::incremental_navigation;
namespace {
class GridView final : public nav::ElevationRangeView {
 public:
  GridView(float* h, float* s, int w, int height, double r, double ox,
           double oy)
      : heights(h), stats(s), width(w) {
    // PersistentElevationMap bounds expand only around finite centers. A
    // requested buffer's NaN padding does not enlarge the producer geometry.
    int minx = w, miny = height, maxx = 0, maxy = 0;
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < w; ++x)
        if (std::isfinite(h[static_cast<std::size_t>(y) * w + x])) {
          minx = std::min(minx, x);
          miny = std::min(miny, y);
          maxx = std::max(maxx, x + 1);
          maxy = std::max(maxy, y + 1);
        }
    if (minx == w) minx = miny = maxx = maxy = 0;
    geometry_ = nav::SparseGridGeometry("map", r, {ox, oy, 0}, {minx, miny},
                                        {maxx, maxy});
  }
  const nav::SparseGridGeometry& geometry() const noexcept override {
    return geometry_;
  }
  std::optional<nav::ElevationRange> ElevationRangeAt(
      nav::GridIndex i) const noexcept override {
    if (!geometry_.Contains(i)) return std::nullopt;
    float h = heights[i.y * width + i.x];
    if (!std::isfinite(h)) return std::nullopt;
    return nav::ElevationRange{h, h};
  }
  std::optional<nav::LocalTerrainMeasurements> TerrainMeasurementsAt(
      nav::GridIndex i) const noexcept override {
    if (!stats || !geometry_.Contains(i)) return std::nullopt;
    const auto k = i.y * width + i.x;
    const float* s = stats + 4 * k;
    return nav::LocalTerrainMeasurements{std::isfinite(heights[k]), s[3] == 1.,
                                         s[0], s[1], s[2]};
  }
  float* heights;
  float* stats;
  int width;
  nav::SparseGridGeometry geometry_;
};
double number(PyObject* d, const char* key) {
  PyObject* v = PyDict_GetItemString(d, key);
  if (!v)
    throw std::invalid_argument(std::string("missing capability: ") + key);
  double value = PyFloat_AsDouble(v);
  if (PyErr_Occurred() || !std::isfinite(value) || value < 0)
    throw std::invalid_argument("invalid capability scalar");
  return value;
}
}  // namespace
PyObject* derive(PyObject*, PyObject* args) {
  PyObject *ho, *so, *bo, *mo, *cap;
  double r, ox, oy;
  int supplied;
  if (!PyArg_ParseTuple(args, "OdddOOOOi", &ho, &r, &ox, &oy, &cap, &so, &bo,
                        &mo, &supplied))
    return nullptr;
  try {
    Buffer h(ho, "f", 2), s(so, "f", 3, true), b(bo, "B", 2, true),
        m(mo, "B", 2, true);
    const int height = h.view.shape[0], width = h.view.shape[1];
    if (height <= 0 || width <= 0 || !std::isfinite(r) || r <= 0 ||
        !std::isfinite(ox) || !std::isfinite(oy))
      throw std::invalid_argument("invalid grid geometry");
    s.shape(height, width);
    b.shape(height, width);
    m.shape(height, width);
    if (s.view.shape[2] != 4 || !PyDict_Check(cap))
      throw std::invalid_argument("four stats and capability dict required");
    nav::WheeledCapability wheel;
    wheel.maximum_slope_rad = number(cap, "maximum_slope_rad");
    wheel.maximum_local_obstacle_relief_m =
        number(cap, "maximum_local_obstacle_relief_m");
    wheel.minimum_underbody_clearance_m =
        number(cap, "minimum_underbody_clearance_m");
    wheel.minimum_clearance_m = number(cap, "minimum_clearance_m");
    PyObject* footprint = PyDict_GetItemString(cap, "footprint_xy_m");
    if (!footprint || !PySequence_Check(footprint))
      throw std::invalid_argument("footprint required");
    for (Py_ssize_t i = 0; i < PySequence_Size(footprint); ++i) {
      PyObject* pair = PySequence_GetItem(footprint, i);
      if (!pair || PySequence_Size(pair) != 2) {
        Py_XDECREF(pair);
        throw std::invalid_argument("invalid footprint");
      }
      PyObject *x = PySequence_GetItem(pair, 0),
               *y = PySequence_GetItem(pair, 1);
      double px = PyFloat_AsDouble(x), py = PyFloat_AsDouble(y);
      Py_DECREF(x);
      Py_DECREF(y);
      Py_DECREF(pair);
      if (!std::isfinite(px) || !std::isfinite(py) || PyErr_Occurred())
        throw std::invalid_argument("invalid footprint");
      wheel.footprint_xy_m.push_back({px, py});
    }
    if (wheel.footprint_xy_m.size() < 3)
      throw std::invalid_argument("footprint requires polygon");
    nav::PlatformCapability capability = wheel;
    nav::TraversabilityProfile profile;
    profile.planar_envelope_xy_m = wheel.footprint_xy_m;
    profile.preferred_clearance_m = wheel.minimum_clearance_m;
    profile.slope_weight = 1.;
    profile.relief_weight = 1.;
    profile.clearance_weight = 0.;
    {
      ReleasedGIL release;
      GridView grid(h.data<float>(), nullptr, width, height, r, ox, oy);
      if (!supplied) {
        for (int y = 0; y < height; ++y)
          for (int x = 0; x < width; ++x) {
            auto v = nav::MeasureLocalTerrain(grid, {x, y});
            float* out =
                s.data<float>() + 4 * (static_cast<std::size_t>(y) * width + x);
            out[0] = static_cast<float>(v.slope_rad);
            out[1] = static_cast<float>(v.relief_m);
            out[2] = static_cast<float>(v.positive_rise_m);
            out[3] = v.neighborhood_complete ? 1.f : 0.f;
          }
      }
      // Classify only after transport float32 rounding, including
      // FineCellEvaluator's cache.
      grid.stats = s.data<float>();
      auto evaluator = nav::MakePlatformElevationEvaluator(capability);
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
          b.data<std::uint8_t>()[static_cast<std::size_t>(y) * width + x] =
              static_cast<std::uint8_t>(
                  evaluator->Evaluate(grid, {x, y}).state);
      // Bound native intrinsic cache lifetime to one real 256-cell tile plus
      // halo.
      for (int by = 0; by < height; by += nav::kGridTileWidthCells)
        for (int bx = 0; bx < width; bx += nav::kGridTileWidthCells) {
          nav::FineCellEvaluator fine(grid, capability, profile);
          for (int y = by;
               y < std::min(height, by + int(nav::kGridTileWidthCells)); ++y)
            for (int x = bx;
                 x < std::min(width, bx + int(nav::kGridTileWidthCells)); ++x)
              m.data<std::uint8_t>()[static_cast<std::size_t>(y) * width + x] =
                  static_cast<std::uint8_t>(fine.Evaluate({x, y}).state);
        }
    }
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    return error(e);
  }
}
static PyMethodDef methods[] = {
    {"derive", derive, METH_VARARGS,
     "Measure float32 terrain then classify B and native M."},
    {"visible_sources", visible_sources, METH_VARARGS, "Forward source array to target visibility."},
    {"directional_targets", directional_targets, METH_VARARGS, "Exact eight-heading beam target membership."},
    {"visibility_cache_info", visibility_cache_info, METH_NOARGS, "Bounded process ray geometry cache."},
    {"observe", observe, METH_VARARGS, "Finite beam prefix and simultaneous first-hit observations."},
    {"visible_witnesses", visible_witnesses, METH_VARARGS,
     "Direct native R beam-observation witnesses."},
    {"first_pending", first_pending, METH_VARARGS,
     "First pending hit on an exact successful beam."},
    {"visible_targets", visible_targets, METH_VARARGS,
     "Sparse directed finite-beam targets."},
    {"reachable", reachable, METH_VARARGS,
     "Native M component with no corner cuts."},
    {"visible_union", visible_union, METH_VARARGS,
     "Exact union with first-source early exit."},
    {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT,
                             "lunar_drl_terrain_native",
                             nullptr,
                             -1,
                             methods,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr};
PyMODINIT_FUNC PyInit_lunar_drl_terrain_native() {
  return PyModule_Create(&module);
}
