"""Hot S2 CellId operations over caller-owned contiguous buffers."""

from std.math import cos, floor, sin, sqrt

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]


def uv_to_st(u: Float64) -> Float64:
    if u >= 0.0:
        return 0.5 * sqrt(1.0 + 3.0 * u)
    return 1.0 - 0.5 * sqrt(1.0 - 3.0 * u)


def st_to_ij(s: Float64) -> Int:
    var ij = Int(floor(1073741824.0 * s))
    if ij < 0:
        return 0
    if ij >= 1073741824:
        return 1073741823
    return ij


def face_uv(lat: Float64, lng: Float64) -> Tuple[Int, Float64, Float64]:
    var cp = cos(lat)
    var x = cos(lng) * cp
    var y = sin(lng) * cp
    var z = sin(lat)
    var ax = abs(x)
    var ay = abs(y)
    var az = abs(z)
    var face = 2
    if ax > ay:
        if ax > az:
            face = 0
    elif ay > az:
        face = 1
    if (face == 0 and x < 0.0) or (face == 1 and y < 0.0) or (face == 2 and z < 0.0):
        face += 3
    if face == 0:
        return (face, y / x, z / x)
    if face == 1:
        return (face, -x / y, z / y)
    if face == 2:
        return (face, -x / z, -y / z)
    if face == 3:
        return (face, z / x, y / x)
    if face == 4:
        return (face, z / y, -x / y)
    return (face, -y / z, -x / z)


def ij_to_pos(orientation: Int, ij: Int) -> Int:
    if orientation == 0:
        if ij == 0:
            return 0
        if ij == 1:
            return 1
        if ij == 3:
            return 2
        return 3
    if orientation == 1:
        if ij == 0:
            return 0
        if ij == 2:
            return 1
        if ij == 3:
            return 2
        return 3
    if orientation == 2:
        if ij == 3:
            return 0
        if ij == 2:
            return 1
        if ij == 0:
            return 2
        return 3
    if ij == 3:
        return 0
    if ij == 1:
        return 1
    if ij == 0:
        return 2
    return 3


def hilbert_pos(i: Int, j: Int, initial_orientation: Int) -> UInt64:
    var orientation = initial_orientation
    var result = UInt64(0)
    for bit in range(29, -1, -1):
        var ij = (((i >> bit) & 1) << 1) | ((j >> bit) & 1)
        var pos = ij_to_pos(orientation, ij)
        result = (result << 2) | UInt64(pos)
        if pos == 0:
            orientation ^= 1
        elif pos == 3:
            orientation ^= 3
    return result


def cell_id(lat: Float64, lng: Float64, level: Int) -> UInt64:
    var face, u, v = face_uv(lat, lng)
    var i = st_to_ij(uv_to_st(u))
    var j = st_to_ij(uv_to_st(v))
    var leaf = (UInt64(face) << 61) | (hilbert_pos(i, j, face & 1) << 1) | UInt64(1)
    var lsb = UInt64(1) << UInt64(2 * (30 - level))
    return (leaf & ~(lsb - UInt64(1))) | lsb


@export("ms2_cellids_from_latlng_radians")
def ms2_cellids_from_latlng_radians(lat_addr: Int, lng_addr: Int, n: Int, level: Int, ids_addr: Int) abi("C"):
    if n <= 0 or level < 0 or level > 30 or lat_addr == 0 or lng_addr == 0 or ids_addr == 0:
        return
    var lat = FPtr(unsafe_from_address=lat_addr)
    var lng = FPtr(unsafe_from_address=lng_addr)
    var ids = UPtr(unsafe_from_address=ids_addr)
    for index in range(n):
        ids[index] = cell_id(lat[index], lng[index], level)


@export("ms2_cellids_parent")
def ms2_cellids_parent(ids_addr: Int, n: Int, level: Int, dst_addr: Int) abi("C"):
    if n <= 0 or level < 0 or level > 30 or ids_addr == 0 or dst_addr == 0:
        return
    var ids = UPtr(unsafe_from_address=ids_addr)
    var dst = UPtr(unsafe_from_address=dst_addr)
    var lsb = UInt64(1) << UInt64(2 * (30 - level))
    for index in range(n):
        dst[index] = (ids[index] & ~(lsb - UInt64(1))) | lsb
