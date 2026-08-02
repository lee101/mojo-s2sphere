"""Bulk Mojo CellId construction against upstream s2sphere."""

import math
import os
import platform
import sys
import time

import numpy as np
from s2sphere import CellId as ReferenceCellId
from s2sphere import LatLng as ReferenceLatLng

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojo_s2sphere import CellId  # noqa: E402


def timed(fn, reps=3):
    best = float("inf")
    result = None
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def upstream_cellids(lat, lng, level):
    return np.array([
        ReferenceCellId.from_lat_lng(ReferenceLatLng.from_radians(float(a), float(b))).parent(level).id()
        for a, b in zip(lat, lng, strict=True)
    ], dtype=np.uint64)


def row(kernel, mojo, reference, basis):
    speedup = reference / mojo
    print(f"| {kernel} | {mojo * 1e3:.2f} ms | {reference * 1e3:.2f} ms | {speedup:.2f}x | {basis} |")


def main():
    rng = np.random.default_rng(0)
    n = 500_000
    lat = np.arcsin(rng.uniform(-1, 1, n))
    lng = rng.uniform(-math.pi, math.pi, n)
    print(f"Machine: {platform.processor() or platform.machine()}, Python {platform.python_version()}")
    print("| kernel | Mojo | upstream | speedup | comparison |")
    print("| --- | ---: | ---: | ---: | --- |")

    mojo, got = timed(lambda: CellId.from_lat_lngs(lat, lng, 30))
    reference, expected = timed(lambda: upstream_cellids(lat, lng, 30), reps=1)
    assert np.array_equal(got, expected)
    row("lat/lng radians to leaf CellId, 500k", mojo, reference, "s2sphere scalar loop")

    leaves = got
    mojo, got = timed(lambda: CellId.parents(leaves, 12))
    reference, expected = timed(lambda: np.array([ReferenceCellId(int(v)).parent(12).id() for v in leaves], dtype=np.uint64), reps=1)
    assert np.array_equal(got, expected)
    row("CellId parent(level=12), 500k", mojo, reference, "s2sphere scalar loop")


if __name__ == "__main__":
    main()
