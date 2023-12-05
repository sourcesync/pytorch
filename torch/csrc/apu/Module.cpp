#include <ATen/ATen.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
    
// pthread.h is included for tracking bad forks
#ifndef WIN32
#include <pthread.h>
#endif

namespace torch {
namespace apu {

namespace {
// True for children forked after apu init
static bool in_bad_fork = false;

// Called in the forked child if apu has already been initialized
static void forked_apu_child() {
  in_bad_fork = true;
}

// Should be called before the first apu call.
static void track_bad_apu_fork() {
#ifndef WIN32
  static c10::once_flag flag;
  c10::call_once(
      flag, [] { pthread_atfork(nullptr, nullptr, forked_apu_child); });
#endif
}
} // namespace

static PyObject* APUModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_getDefaultAPUGenerator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  track_bad_apu_fork();
  /* GW
  return THPGenerator_initDefaultGenerator(
      at::detail::getAPUHooks().getDefaultAPUGenerator());
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_isAvailable(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  track_bad_apu_fork();
  /* GW
  if (at::detail::getAPUHooks().hasAPU()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  */
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_isMacOS13orNewer(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(args), "invalid argument to isOnMacOS13orNewer()");
  /* GW
  auto minor = THPUtils_unpackUInt32(args);
  if (at::detail::getAPUHooks().isOnMacOS13orNewer(minor)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  */
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_deviceSynchronize(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  /* GW
  at::detail::getAPUHooks().deviceSynchronize();
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  /* GW
  at::detail::getAPUHooks().emptyCache();
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_setMemoryFraction(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkDouble(args), "invalid argument to setMemoryFraction()");
  /* GW
  double fraction = THPUtils_unpackDouble(args);
  at::detail::getAPUHooks().setMemoryFraction(fraction);
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_currentAllocatedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  /* GW
  return THPUtils_packUInt64(
      at::detail::getAPUHooks().getCurrentAllocatedMemory());
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_driverAllocatedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  /* GW
  return THPUtils_packUInt64(
      at::detail::getAPUHooks().getDriverAllocatedMemory());
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_profilerStartTrace(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* mode_string_o = nullptr;
  PyObject* wait_until_completed_string_o = nullptr;
  if (!PyArg_ParseTuple(
          args, "OO", &mode_string_o, &wait_until_completed_string_o)) {
    return nullptr;
  }
  /* GW
  const std::string mode = THPUtils_unpackString(mode_string_o);
  const bool waitUntilCompleted =
      THPUtils_unpackBool(wait_until_completed_string_o);
  at::detail::getAPUHooks().profilerStartTrace(mode, waitUntilCompleted);
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_profilerStopTrace(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  /* GW
  at::detail::getAPUHooks().profilerStopTrace();
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_acquireEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  /* GW
  const bool enable_timing = THPUtils_unpackBool(args);
  return THPUtils_packUInt32(
      at::detail::getAPUHooks().acquireEvent(enable_timing));
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_releaseEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  /* GW
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getAPUHooks().releaseEvent(event_id);
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_recordEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  /* GW
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getAPUHooks().recordEvent(event_id);
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_waitForEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  /* GW
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getAPUHooks().waitForEvent(event_id);
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_synchronizeEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  /* GW
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getAPUHooks().synchronizeEvent(event_id);
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_queryEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  /* GW
  if (at::detail::getAPUHooks().queryEvent(event_id)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  */
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS
}

static PyObject* APUModule_elapsedTimeOfEvents(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* start_event_o = nullptr;
  PyObject* end_event_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &start_event_o, &end_event_o)) {
    return nullptr;
  }
  /* GW
  const uint32_t start_event_id = THPUtils_unpackUInt32(start_event_o);
  const uint32_t end_event_id = THPUtils_unpackUInt32(end_event_o);
  return PyFloat_FromDouble(at::detail::getAPUHooks().elapsedTimeOfEvents(
      start_event_id, end_event_id));
  */
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyMethodDef _APUModule_methods[] = {
    {"_apu_deviceSynchronize",
     APUModule_deviceSynchronize,
     METH_NOARGS,
     nullptr},
    {"_apu_is_in_bad_fork", APUModule_isInBadFork, METH_NOARGS, nullptr},
    {"_apu_is_available", APUModule_isAvailable, METH_NOARGS, nullptr},
    {"_apu_is_on_macos_13_or_newer",
     APUModule_isMacOS13orNewer,
     METH_O,
     nullptr},
    {"_apu_get_default_generator",
     APUModule_getDefaultAPUGenerator,
     METH_NOARGS,
     nullptr},
    {"_apu_emptyCache", APUModule_emptyCache, METH_NOARGS, nullptr},
    {"_apu_setMemoryFraction", APUModule_setMemoryFraction, METH_O, nullptr},
    {"_apu_currentAllocatedMemory",
     APUModule_currentAllocatedMemory,
     METH_NOARGS,
     nullptr},
    {"_apu_driverAllocatedMemory",
     APUModule_driverAllocatedMemory,
     METH_NOARGS,
     nullptr},
    {"_apu_profilerStartTrace",
     APUModule_profilerStartTrace,
     METH_VARARGS,
     nullptr},
    {"_apu_profilerStopTrace",
     APUModule_profilerStopTrace,
     METH_NOARGS,
     nullptr},
    {"_apu_acquireEvent", APUModule_acquireEvent, METH_O, nullptr},
    {"_apu_releaseEvent", APUModule_releaseEvent, METH_O, nullptr},
    {"_apu_recordEvent", APUModule_recordEvent, METH_O, nullptr},
    {"_apu_waitForEvent", APUModule_waitForEvent, METH_O, nullptr},
    {"_apu_synchronizeEvent", APUModule_synchronizeEvent, METH_O, nullptr},
    {"_apu_queryEvent", APUModule_queryEvent, METH_O, nullptr},
    {"_apu_elapsedTimeOfEvents",
     APUModule_elapsedTimeOfEvents,
     METH_VARARGS,
     nullptr},
    {nullptr}};

PyMethodDef* python_functions() {
  return _APUModule_methods;
}

} // namespace apu
} // namespace torch
