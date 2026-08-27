# mojo-s2sphere

`mojo-s2sphere` is a standalone Mojo port of the hot path in
[s2sphere](https://pypi.org/project/s2sphere/): converting latitude/longitude
coordinates into S2 `CellId` values and coarsening those IDs to a parent level.
It keeps a Python API shaped like the upstream `Angle`, `Point`, `LatLng`, and
`CellId` objects, while moving bulk indexing to compiled Mojo.

## Covered subset

- `Angle`, `Point`, and `LatLng` construction, conversion, normalization, and
  great-circle distance.
- `CellId` construction from latitude/longitude, points, face/IJ coordinates,
  and face/position values; hierarchy traversal and containment; tokens;
  face/IJ decoding; and center conversion.
- Bulk latitude/longitude conversion and bulk parent conversion.

`Cell`, `CellUnion`, `Cap`, rectangles, region covering, and neighbor search
are not implemented yet. The import name is `mojo_s2sphere` so it can be
installed beside the upstream `s2sphere` package for parity testing.

## Install and use

```bash
pixi install
pixi run build
```

```python
import numpy as np
from mojo_s2sphere import CellId, LatLng

nyc = CellId.from_lat_lng(LatLng.from_degrees(40.7128, -74.006))
print(nyc.to_token())  # 89c25a220cf80969

lat = np.deg2rad([40.7128, 51.5074, -33.8688])
lng = np.deg2rad([-74.006, -0.1278, 151.2093])
leaves = CellId.from_lat_lngs(lat, lng)
parents_at_12 = CellId.parents(leaves, 12)
```

## Benchmarks

Measured by `pixi run bench` on this machine: x86_64, Python 3.13.14.
Inputs are uniformly distributed points. Mojo times are the best of three;
the upstream scalar reference is one full pass.

| kernel | Mojo | upstream | speedup | comparison |
| --- | ---: | ---: | ---: | --- |
| lat/lng radians to leaf `CellId`, 500k | 174.26 ms | 6208.29 ms | 35.63x | `s2sphere` scalar loop |
| `CellId` parent(level=12), 500k | 2.74 ms | 812.45 ms | 296.22x | `s2sphere` scalar loop |

Both measured kernels exceed the 5x optimization cutoff. No SIMD, threading,
or GPU path was added: bulk parent conversion has too little arithmetic per
byte to benefit from GPU offload, while the compute-heavy latitude/longitude
kernel is already 35.63x faster than the upstream implementation and is not an
optimization target.

## How it works

The Python layer validates finite real coordinate inputs, creates contiguous
double-precision buffers, and keeps references to those buffers for the whole
native call. It rejects lossy CellId conversions and invalid parent requests.
It passes addresses through `ctypes` to one Mojo shared library,
`dist/libmojo-s2sphere.so`; the library allocates nothing and writes directly
into the caller's output buffer. The kernel implements the quadratic S2
projection and the orientation-changing Hilbert traversal bit by bit, then
applies the CellId parent-bit mask in bulk.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The tests assert exact CellId, parent, face/IJ, center, constructor, token,
and input-validation behavior against the upstream package where applicable.

MIT.
