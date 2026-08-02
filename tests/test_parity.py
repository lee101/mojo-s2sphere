import math

import numpy as np
import pytest
from s2sphere import CellId as ReferenceCellId
from s2sphere import LatLng as ReferenceLatLng

from mojo_s2sphere import Angle, CellId, LatLng, Point, cellids_from_latlng_degrees, cellids_from_latlng_radians


def reference_ids(lat, lng, level):
    return np.array([
        ReferenceCellId.from_lat_lng(ReferenceLatLng.from_radians(float(a), float(b))).parent(level).id()
        for a, b in zip(lat, lng, strict=True)
    ], dtype=np.uint64)


def test_bulk_latlng_ids_match_upstream_on_random_sphere():
    rng = np.random.default_rng(41)
    lat = np.arcsin(rng.uniform(-1, 1, 2_000))
    lng = rng.uniform(-math.pi, math.pi, 2_000)
    for level in (0, 1, 7, 18, 30):
        got = CellId.from_lat_lngs(lat, lng, level)
        assert np.array_equal(got, reference_ids(lat, lng, level))


def test_known_locations_and_batch_degrees_match_upstream():
    degrees = np.array([[0.0, 0.0], [40.7128, -74.006], [-33.8688, 151.2093], [90.0, 0.0], [-90.0, 90.0]])
    got = cellids_from_latlng_degrees(degrees[:, 0], degrees[:, 1])
    expected = reference_ids(np.deg2rad(degrees[:, 0]), np.deg2rad(degrees[:, 1]), 30)
    assert np.array_equal(got, expected)
    assert CellId.from_lat_lng(LatLng.from_degrees(40.7128, -74.006)).to_token() == "89c25a220cf80969"


def test_cell_hierarchy_and_center_conversion_match_upstream():
    upstream = ReferenceCellId.from_lat_lng(ReferenceLatLng.from_degrees(51.5074, -0.1278))
    cell = CellId(upstream.id())
    assert cell.is_valid() and cell.is_leaf() and cell.level() == upstream.level()
    for level in (0, 1, 9, 20, 30):
        got, expected = cell.parent(level), upstream.parent(level)
        assert got.id() == expected.id()
        assert got.to_face_ij_orientation() == expected.to_face_ij_orientation()
        assert got.get_center_si_ti() == expected.get_center_si_ti()
        assert got.get_center_uv() == pytest.approx(expected.get_center_uv())
        assert got.to_lat_lng().lat().radians == pytest.approx(expected.to_lat_lng().lat().radians)
        assert got.to_lat_lng().lng().radians == pytest.approx(expected.to_lat_lng().lng().radians)
    parent = cell.parent(12)
    assert [c.id() for c in parent.children()] == [c.id() for c in upstream.parent(12).children()]
    assert parent.contains(cell) and parent.intersects(cell)


def test_bulk_parent_kernel_matches_upstream():
    rng = np.random.default_rng(7)
    lat = rng.uniform(-1.5, 1.5, 1_000)
    lng = rng.uniform(-3.1, 3.1, 1_000)
    leaves = CellId.from_lat_lngs(lat, lng)
    for level in (0, 10, 29):
        expected = reference_ids(lat, lng, level)
        assert np.array_equal(CellId.parents(leaves, level), expected)


def test_point_latlng_and_angle_compatibility():
    ll = LatLng.from_degrees(45, 90)
    point = ll.to_point()
    assert tuple(point) == pytest.approx((0.0, math.sqrt(0.5), math.sqrt(0.5)))
    assert LatLng.from_point(point).lat().degrees == pytest.approx(45)
    assert LatLng.from_point(point).lng().degrees == pytest.approx(90)
    assert Point(1, 0, 0).angle(Point(0, 1, 0)) == pytest.approx(math.pi / 2)
    assert Angle.from_degrees(180).radians == pytest.approx(math.pi)
    assert ll.get_distance(LatLng.from_degrees(45, 90)).radians == 0


def test_input_validation():
    with pytest.raises(ValueError):
        CellId.from_lat_lngs([0.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        CellId.from_lat_lngs([0.0], [0.0], 31)
    with pytest.raises(ValueError):
        CellId.parents(np.array([[1]], dtype=np.uint64), 1)
    with pytest.raises(TypeError):
        CellId.from_lat_lngs(np.array([0], dtype=np.float128), [0.0])
    with pytest.raises(ValueError):
        CellId.from_lat_lngs([math.nan], [0.0])
    with pytest.raises(TypeError):
        CellId.parents(np.array([1.0]), 1)
    with pytest.raises(ValueError):
        CellId.parents(np.array([-1], dtype=np.int64), 1)
    with pytest.raises(ValueError):
        CellId.parents(np.array([0], dtype=np.uint64), 0)


def test_documented_constructors_tokens_and_normalization_match_upstream():
    ll = LatLng.from_degrees(20, 30)
    upstream = ReferenceCellId.from_lat_lng(ReferenceLatLng.from_degrees(20, 30))
    assert CellId.from_point(ll.to_point()).id() == upstream.id()
    face, i, j, _ = upstream.to_face_ij_orientation()
    assert CellId.from_face_ij(face, i, j).id() == upstream.id()
    assert CellId.from_face_pos_level(upstream.face(), upstream.pos(), upstream.level()).id() == upstream.id()
    assert CellId.from_token(upstream.to_token()).id() == upstream.id()
    assert LatLng.from_degrees(100, 540).normalized() == LatLng.from_degrees(90, -180)
    radians = cellids_from_latlng_radians([ll.lat().radians], [ll.lng().radians])
    assert int(radians[0]) == upstream.id()
