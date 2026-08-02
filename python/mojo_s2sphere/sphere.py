"""A compact S2 spherical-cell API with Mojo-accelerated bulk CellId creation."""

from __future__ import annotations

import math
from functools import total_ordering

import numpy as np

from ._lib import addr, f64, lib, u64

MAX_LEVEL = 30
MAX_SIZE = 1 << MAX_LEVEL
POS_BITS = 2 * MAX_LEVEL + 1
NUM_FACES = 6
WRAP_OFFSET = NUM_FACES << POS_BITS
SWAP_MASK = 0x01
INVERT_MASK = 0x02
POS_TO_IJ = ((0, 1, 3, 2), (0, 2, 3, 1), (3, 2, 0, 1), (3, 1, 0, 2))
POS_TO_ORIENTATION = (SWAP_MASK, 0, 0, INVERT_MASK | SWAP_MASK)


@total_ordering
class Angle:
    def __init__(self, radians=0):
        if not isinstance(radians, (float, int)):
            raise ValueError()
        self.__radians = radians

    @classmethod
    def from_degrees(cls, degrees):
        return cls(math.radians(degrees))

    @classmethod
    def from_radians(cls, radians):
        return cls(radians)

    @property
    def radians(self):
        return self.__radians

    @property
    def degrees(self):
        return math.degrees(self.__radians)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__radians == other.__radians

    def __lt__(self, other):
        return self.__radians < other.__radians

    def __add__(self, other):
        return type(self).from_radians(self.__radians + other.__radians)

    def __repr__(self):
        return f"{type(self).__name__}: {self.__radians}"


class Point:
    def __init__(self, x, y, z):
        self.__point = (x, y, z)

    def __getitem__(self, index):
        return self.__point[index]

    def __iter__(self):
        return iter(self.__point)

    def __neg__(self):
        return type(self)(-self[0], -self[1], -self[2])

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__point == other.__point

    def __hash__(self):
        return hash(self.__point)

    def __repr__(self):
        return f"Point: {self.__point}"

    def __add__(self, other):
        return type(self)(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __sub__(self, other):
        return type(self)(self[0] - other[0], self[1] - other[1], self[2] - other[2])

    def __mul__(self, other):
        return type(self)(self[0] * other, self[1] * other, self[2] * other)

    __rmul__ = __mul__

    def abs(self):
        return type(self)(abs(self[0]), abs(self[1]), abs(self[2]))

    def largest_abs_component(self):
        p = self.abs()
        if p[0] > p[1]:
            return 0 if p[0] > p[2] else 2
        return 1 if p[1] > p[2] else 2

    def cross_prod(self, other):
        return type(self)(self[1] * other[2] - self[2] * other[1],
                          self[2] * other[0] - self[0] * other[2],
                          self[0] * other[1] - self[1] * other[0])

    def dot_prod(self, other):
        return self[0] * other[0] + self[1] * other[1] + self[2] * other[2]

    def norm2(self):
        return self.dot_prod(self)

    def norm(self):
        return math.sqrt(self.norm2())

    def normalize(self):
        n = self.norm()
        return self * (1.0 / n if n else 0.0)

    def angle(self, other):
        return math.atan2(self.cross_prod(other).norm(), self.dot_prod(other))


class LatLng:
    @classmethod
    def from_degrees(cls, lat, lng):
        return cls(math.radians(lat), math.radians(lng))

    @classmethod
    def from_radians(cls, lat, lng):
        return cls(lat, lng)

    @classmethod
    def from_point(cls, point):
        return cls(math.atan2(point[2], math.sqrt(point[0] ** 2 + point[1] ** 2)),
                   math.atan2(point[1], point[0]))

    @classmethod
    def from_angles(cls, lat, lng):
        return cls(lat.radians, lng.radians)

    @classmethod
    def default(cls):
        return cls(0, 0)

    @classmethod
    def invalid(cls):
        return cls(math.pi, 2 * math.pi)

    def __init__(self, lat, lng):
        self.__coords = (lat, lng)

    def __eq__(self, other):
        return isinstance(other, LatLng) and self.__coords == other.__coords

    def __hash__(self):
        return hash(self.__coords)

    def __repr__(self):
        return f"LatLng: {math.degrees(self.__coords[0])},{math.degrees(self.__coords[1])}"

    def __add__(self, other):
        return type(self)(self.lat().radians + other.lat().radians,
                          self.lng().radians + other.lng().radians)

    def __sub__(self, other):
        return type(self)(self.lat().radians - other.lat().radians,
                          self.lng().radians - other.lng().radians)

    def __rmul__(self, other):
        return type(self)(other * self.lat().radians, other * self.lng().radians)

    def lat(self):
        return Angle.from_radians(self.__coords[0])

    def lng(self):
        return Angle.from_radians(self.__coords[1])

    def is_valid(self):
        return abs(self.lat().radians) <= math.pi / 2 and abs(self.lng().radians) <= math.pi

    def to_point(self):
        phi, theta = self.__coords
        cosphi = math.cos(phi)
        return Point(math.cos(theta) * cosphi, math.sin(theta) * cosphi, math.sin(phi))

    def normalized(self):
        lng = math.remainder(self.lng().radians, 2 * math.pi)
        return type(self)(max(-math.pi / 2, min(math.pi / 2, self.lat().radians)), lng)

    def approx_equals(self, other, max_error=1e-15):
        return abs(self.lat().radians - other.lat().radians) < max_error and abs(self.lng().radians - other.lng().radians) < max_error

    def get_distance(self, other):
        assert self.is_valid() and other.is_valid()
        dlat = math.sin(0.5 * (other.lat().radians - self.lat().radians))
        dlng = math.sin(0.5 * (other.lng().radians - self.lng().radians))
        x = dlat * dlat + dlng * dlng * math.cos(self.lat().radians) * math.cos(other.lat().radians)
        return Angle.from_radians(2 * math.atan2(math.sqrt(x), math.sqrt(max(0.0, 1.0 - x))))


def face_uv_to_xyz(face, u, v):
    if face == 0:
        return Point(1, u, v)
    if face == 1:
        return Point(-u, 1, v)
    if face == 2:
        return Point(-u, -v, 1)
    if face == 3:
        return Point(-1, -v, -u)
    if face == 4:
        return Point(v, -1, -u)
    return Point(v, u, -1)


def xyz_to_face_uv(point):
    face = point.largest_abs_component()
    if point[face] < 0:
        face += 3
    x, y, z = point
    if face == 0:
        return face, y / x, z / x
    if face == 1:
        return face, -x / y, z / y
    if face == 2:
        return face, -x / z, -y / z
    if face == 3:
        return face, z / x, y / x
    if face == 4:
        return face, z / y, -x / y
    return face, -y / z, -x / z


def _uv_to_st(u):
    return 0.5 * math.sqrt(1 + 3 * u) if u >= 0 else 1 - 0.5 * math.sqrt(1 - 3 * u)


def _st_to_uv(s):
    return (4 * s * s - 1) / 3 if s >= 0.5 else (1 - 4 * (1 - s) ** 2) / 3


def _hilbert_pos(i, j, orientation):
    result = 0
    for bit in range(MAX_LEVEL - 1, -1, -1):
        ij = (((i >> bit) & 1) << 1) | ((j >> bit) & 1)
        pos = POS_TO_IJ[orientation].index(ij)
        result = (result << 2) | pos
        orientation ^= POS_TO_ORIENTATION[pos]
    return result


@total_ordering
class CellId:
    LINEAR_PROJECTION = 0
    TAN_PROJECTION = 1
    QUADRATIC_PROJECTION = 2
    PROJECTION = QUADRATIC_PROJECTION
    FACE_BITS = 3
    NUM_FACES = NUM_FACES
    MAX_LEVEL = MAX_LEVEL
    POS_BITS = POS_BITS
    MAX_SIZE = MAX_SIZE
    WRAP_OFFSET = WRAP_OFFSET

    def __init__(self, id_=0):
        self.__id = int(id_) & ((1 << 64) - 1)

    def __repr__(self):
        return f"CellId: {self.id():016x}"

    def __hash__(self):
        return hash(self.id())

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.id() == other.id()

    def __lt__(self, other):
        return self.id() < other.id()

    @classmethod
    def from_lat_lng(cls, ll):
        return cls(int(cls.from_lat_lngs([ll.lat().radians], [ll.lng().radians], MAX_LEVEL)[0]))

    @classmethod
    def from_point(cls, point):
        face, u, v = xyz_to_face_uv(point)
        return cls.from_face_ij(face, cls.st_to_ij(cls.uv_to_st(u)), cls.st_to_ij(cls.uv_to_st(v)))

    @classmethod
    def from_lat_lngs(cls, lat, lng, level=MAX_LEVEL):
        lat, lng = f64(lat), f64(lng)
        if lat.size != lng.size:
            raise ValueError("lat and lng must be one-dimensional arrays of equal length")
        if not 0 <= level <= MAX_LEVEL:
            raise ValueError("level must be in [0, 30]")
        ids = np.empty(lat.size, dtype=np.uint64)
        lib().ms2_cellids_from_latlng_radians(addr(lat), addr(lng), lat.size, level, addr(ids))
        return ids

    @classmethod
    def from_face_pos_level(cls, face, pos, level):
        return cls((face << POS_BITS) + (pos | 1)).parent(level)

    @classmethod
    def from_face_ij(cls, face, i, j):
        pos = _hilbert_pos(i, j, face & SWAP_MASK)
        return cls((face << POS_BITS) | (pos << 1) | 1)

    @classmethod
    def st_to_ij(cls, s):
        return max(0, min(MAX_SIZE - 1, int(math.floor(MAX_SIZE * s))))

    @classmethod
    def lsb_for_level(cls, level):
        return 1 << (2 * (MAX_LEVEL - level))

    @classmethod
    def from_token(cls, token):
        return cls(int(token.ljust(16, "0"), 16))

    @classmethod
    def st_to_uv(cls, s):
        return _st_to_uv(s)

    @classmethod
    def uv_to_st(cls, u):
        return _uv_to_st(u)

    def id(self):
        return self.__id

    def lsb(self):
        return self.id() & -self.id()

    def face(self):
        return self.id() >> POS_BITS

    def pos(self):
        return self.id() & ((1 << 61) - 1)

    def is_valid(self):
        return self.face() < NUM_FACES and bool(self.lsb() & 0x1555555555555555)

    def is_leaf(self):
        return bool(self.id() & 1)

    def is_face(self):
        return (self.id() & (self.lsb_for_level(0) - 1)) == 0

    def level(self):
        assert self.is_valid()
        return MAX_LEVEL if self.is_leaf() else MAX_LEVEL - ((self.lsb().bit_length() - 1) // 2)

    def parent(self, *args):
        assert self.is_valid()
        if not args:
            assert not self.is_face()
            new_lsb = self.lsb() << 2
        elif len(args) == 1:
            assert 0 <= args[0] <= self.level()
            new_lsb = self.lsb_for_level(args[0])
        else:
            raise ValueError("no args or level arg")
        return type(self)((self.id() & -new_lsb) | new_lsb)

    @classmethod
    def parents(cls, ids, level):
        ids = u64(ids)
        if not 0 <= level <= MAX_LEVEL:
            raise ValueError("ids must be one-dimensional and level in [0, 30]")
        lsb = ids & -ids
        target_lsb = np.uint64(cls.lsb_for_level(level))
        valid = ((ids >> POS_BITS) < NUM_FACES) & ((lsb & np.uint64(0x1555555555555555)) != 0)
        if not np.all(valid) or np.any(lsb > target_lsb):
            raise ValueError("ids must be valid CellIds at or below the requested level")
        dst = np.empty_like(ids)
        lib().ms2_cellids_parent(addr(ids), ids.size, level, addr(dst))
        return dst

    def child(self, pos):
        assert self.is_valid() and not self.is_leaf()
        return type(self)(self.id() + (2 * pos - 3) * (self.lsb() >> 2))

    def range_min(self):
        return type(self)(self.id() - (self.lsb() - 1))

    def range_max(self):
        return type(self)(self.id() + (self.lsb() - 1))

    def contains(self, other):
        assert self.is_valid() and other.is_valid()
        return self.range_min() <= other <= self.range_max()

    def intersects(self, other):
        assert self.is_valid() and other.is_valid()
        return other.range_min() <= self.range_max() and other.range_max() >= self.range_min()

    def prev(self):
        return type(self)(self.id() - (self.lsb() << 1))

    def next(self):
        return type(self)(self.id() + (self.lsb() << 1))

    def child_begin(self, *args):
        assert self.is_valid()
        if not args:
            assert not self.is_leaf()
            return type(self)(self.id() - self.lsb() + (self.lsb() >> 2))
        level = args[0]
        assert self.level() <= level <= MAX_LEVEL
        return type(self)(self.id() - self.lsb() + self.lsb_for_level(level))

    def child_end(self, *args):
        assert self.is_valid()
        if not args:
            assert not self.is_leaf()
            return type(self)(self.id() + self.lsb() + (self.lsb() >> 2))
        level = args[0]
        assert self.level() <= level <= MAX_LEVEL
        return type(self)(self.id() + self.lsb() + self.lsb_for_level(level))

    def children(self, *args):
        child, end = self.child_begin(*args), self.child_end(*args)
        while child != end:
            yield child
            child = child.next()

    def to_face_ij_orientation(self):
        assert self.is_valid()
        path = (self.id() >> 1) & ((1 << 60) - 1)
        face, orientation, i, j = self.face(), self.face() & SWAP_MASK, 0, 0
        for bit in range(MAX_LEVEL - 1, -1, -1):
            pos = (path >> (2 * bit)) & 3
            ij = POS_TO_IJ[orientation][pos]
            i |= (ij >> 1) << bit
            j |= (ij & 1) << bit
            orientation ^= POS_TO_ORIENTATION[pos]
        if self.lsb() & 0x1111111111111110:
            orientation ^= SWAP_MASK
        return face, i, j, orientation

    def get_center_si_ti(self):
        face, i, j, _ = self.to_face_ij_orientation()
        delta = 1 if self.is_leaf() else (2 if ((i ^ (self.id() >> 2)) & 1) else 0)
        return face, 2 * i + delta, 2 * j + delta

    def get_center_uv(self):
        face, si, ti = self.get_center_si_ti()
        return self.st_to_uv(si / (2 * MAX_SIZE)), self.st_to_uv(ti / (2 * MAX_SIZE))

    def to_point_raw(self):
        face, si, ti = self.get_center_si_ti()
        return face_uv_to_xyz(face, self.st_to_uv(si / (2 * MAX_SIZE)), self.st_to_uv(ti / (2 * MAX_SIZE)))

    def to_point(self):
        return self.to_point_raw().normalize()

    def to_lat_lng(self):
        return LatLng.from_point(self.to_point_raw())

    def get_size_ij(self, *args):
        return 1 << (MAX_LEVEL - (self.level() if not args else args[0]))

    def to_token(self):
        return f"{self.id():016x}".rstrip("0")


def cellids_from_latlng_radians(lat, lng, level=MAX_LEVEL):
    return CellId.from_lat_lngs(lat, lng, level)


def cellids_from_latlng_degrees(lat, lng, level=MAX_LEVEL):
    return CellId.from_lat_lngs(np.deg2rad(lat), np.deg2rad(lng), level)
