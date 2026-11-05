"""Load-once, synchronous in-process access to the independent C++ interpreter."""

import ctypes
from pathlib import Path
import threading
import time
import json

from .plan import verify_artifact


def validate_carrier(value, elements):
    import numpy as np

    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
        raise ValueError("Runtime input must be a float32 numpy tensor")
    if value.shape != (1, elements):
        raise ValueError("Runtime input shape must be [1,{}]".format(elements))
    if not np.all(np.isfinite(value)) or not np.all(value == np.round(value)):
        raise ValueError("Runtime input must contain finite, exact INT4 integers")
    if np.any(value < -8) or np.any(value > 7):
        raise ValueError("Runtime input is outside signed INT4 [-8,7]")
    return np.ascontiguousarray(value.reshape(-1), dtype=np.int32)


class PlanSession:
    """One immutable loaded program; calls serialize per session.

    The session belongs to its creating thread, matching the C ABI. A run resets
    simulated device state, so no weight residency across runs is
    claimed. The C ABI owns its JSON buffer until the next call; it is copied
    before releasing the session lock. Call close() to release the C++ handle.
    """

    def __init__(self, folder, library):
        start = time.perf_counter_ns()
        self.folder = Path(folder)
        self.manifest = verify_artifact(self.folder)
        self._lock = threading.RLock()
        self._owner_thread = threading.get_ident()
        self._handle = None
        self.last_cpp_call_ns = 0
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        self._library.tlpu_plan_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._library.tlpu_plan_create.restype = ctypes.c_void_p
        self._library.tlpu_plan_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        self._library.tlpu_plan_run.restype = ctypes.c_char_p
        self._library.tlpu_plan_last_error.argtypes = [ctypes.c_void_p]
        self._library.tlpu_plan_last_error.restype = ctypes.c_char_p
        self._library.tlpu_plan_destroy.argtypes = [ctypes.c_void_p]
        self._library.tlpu_plan_destroy.restype = None
        self._handle = self._library.tlpu_plan_create(
            str((self.folder/"program.tlp").resolve()).encode("utf-8"),
            str((self.folder/"weights.bin").resolve()).encode("utf-8"))
        if not self._handle:
            self._raise_error()
        self.load_wall_ns = time.perf_counter_ns()-start

    def _raise_error(self):
        error = self._library.tlpu_plan_last_error(self._handle)
        raise RuntimeError(error.decode("utf-8") if error else "C++ runtime failed without an error message")

    def run(self, value):
        self._check_thread()
        values = validate_carrier(value, self.manifest["input_contract"]["elements"])
        with self._lock:
            if not self._handle:
                raise RuntimeError("Plan session is closed")
            start = time.perf_counter_ns()
            data = self._library.tlpu_plan_run(self._handle,
                    values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), values.size)
            self.last_cpp_call_ns = time.perf_counter_ns()-start
            if data is None:
                self._raise_error()
            return json.loads(data.decode("utf-8"))

    def close(self):
        self._check_thread()
        with self._lock:
            if self._handle:
                self._library.tlpu_plan_destroy(self._handle)
                self._handle = None

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Plan session must be used from its creating thread")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
