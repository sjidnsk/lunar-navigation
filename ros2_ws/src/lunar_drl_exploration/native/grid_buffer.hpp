#pragma once
#include <Python.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Buffer protocol only: no NumPy ABI, ROS messages or privileged construction
// required to import and use deployment geometry/visibility.
class Buffer {
 public:
  Py_buffer view{};
  Buffer(PyObject* object, const char* format, int ndim,
         bool writable = false) {
    if (PyObject_GetBuffer(object, &view,
                           PyBUF_FORMAT | PyBUF_C_CONTIGUOUS |
                               (writable ? PyBUF_WRITABLE : 0)) < 0)
      throw std::invalid_argument("contiguous buffer required");
    if (view.ndim != ndim || std::string(view.format) != format) {
      PyBuffer_Release(&view);
      view.obj = nullptr;
      throw std::invalid_argument("buffer dtype or dimensions mismatch");
    }
  }
  ~Buffer() {
    if (view.obj) PyBuffer_Release(&view);
  }
  template <class T>
  T* data() {
    return static_cast<T*>(view.buf);
  }
  void shape(Py_ssize_t h, Py_ssize_t w) const {
    if (view.shape[0] != h || view.shape[1] != w)
      throw std::invalid_argument("grid shape mismatch");
  }
};
struct ReleasedGIL {
  PyThreadState* state = PyEval_SaveThread();
  ~ReleasedGIL() { PyEval_RestoreThread(state); }
};
inline PyObject* error(const std::exception& e) {
  PyErr_SetString(PyExc_ValueError, e.what());
  return nullptr;
}
PyObject* derive(PyObject*, PyObject*);
PyObject* observe(PyObject*, PyObject*);
PyObject* reachable(PyObject*, PyObject*);
PyObject* visible_union(PyObject*, PyObject*);
PyObject* visible_targets(PyObject*, PyObject*);
PyObject* first_pending(PyObject*, PyObject*);
PyObject* visible_witnesses(PyObject*, PyObject*);
PyObject* visible_sources(PyObject*, PyObject*);
PyObject* directional_targets(PyObject*, PyObject*);
PyObject* visibility_cache_info(PyObject*, PyObject*);
