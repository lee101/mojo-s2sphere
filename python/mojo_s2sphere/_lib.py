"""Build and load the Mojo S2 kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_S2SPHERE_LIB") or os.path.join(ROOT, "dist", "libmojo-s2sphere.so")
I = ctypes.c_ssize_t
P = ctypes.c_void_p

_SIGNATURES = {
    "ms2_cellids_from_latlng_radians": ([P, P, I, I, P], None),
    "ms2_cellids_parent": ([P, I, I, P], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    if override := os.environ.get("MOJO_S2SPHERE_MOJO"):
        return override.split()
    if found := shutil.which("mojo"):
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_S2SPHERE_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + ["build", "--emit", "shared-lib", source, "-o", LIB]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
    return _loaded


def f64(values) -> np.ndarray:
    """Return a finite, one-dimensional C-contiguous float64 buffer.

    The native ABI receives only an address and element count, so validation
    belongs here while this Python object keeps the underlying buffer alive.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.dtype.kind not in "fiu" or array.dtype.itemsize > 8:
        raise TypeError("values must have a real numeric dtype representable as float64")
    if array.dtype.kind in "iu" and array.size and np.max(np.abs(array.astype(object))) > 2**53:
        raise ValueError("integer values must be exactly representable as float64")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return array


def u64(values) -> np.ndarray:
    """Return a one-dimensional C-contiguous uint64 buffer without truncation."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("ids must be one-dimensional")
    if array.dtype.kind not in "iu" or array.dtype.itemsize > 8:
        raise TypeError("ids must have an integer dtype representable as uint64")
    if array.dtype.kind == "i" and array.size and np.any(array < 0):
        raise ValueError("ids must be non-negative")
    return np.ascontiguousarray(array, dtype=np.uint64)


def addr(values: np.ndarray) -> int:
    return values.ctypes.data
