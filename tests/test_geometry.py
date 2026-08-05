import numpy as np

from tennis_vision.geometry import canonical_corners, order_quad, rectify, transform_point
from tennis_vision.models import Point


def test_quad_order_is_stable_for_shuffled_points():
    shuffled = np.array([[90, 80], [10, 10], [5, 75], [100, 15]], dtype=np.float32)

    ordered = order_quad(shuffled)

    np.testing.assert_allclose(ordered, [[10, 10], [100, 15], [90, 80], [5, 75]])


def test_rectification_round_trip_maps_center():
    frame = np.zeros((100, 140, 3), dtype=np.uint8)
    source = np.array([[20, 10], [120, 15], [135, 90], [5, 85]], dtype=np.float32)

    rectified, source_to_court, court_to_source = rectify(frame, source, 200, 120)
    source_center = transform_point(Point(70, 50), source_to_court)
    recovered = transform_point(source_center, court_to_source)

    assert rectified.shape == (120, 200, 3)
    np.testing.assert_allclose([recovered.x, recovered.y], [70, 50], atol=1e-3)
    assert canonical_corners(200, 120).shape == (4, 2)
