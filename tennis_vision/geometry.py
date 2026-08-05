"""Perspective and coordinate helpers."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from tennis_vision.models import Point

FloatArray = NDArray[np.float32]


def order_quad(points: NDArray[np.generic]) -> FloatArray:
    """Return four points as top-left, top-right, bottom-right, bottom-left."""

    quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = quad.sum(axis=1)
    coordinate_delta = np.diff(quad, axis=1).reshape(-1)
    ordered[0] = quad[np.argmin(coordinate_sum)]
    ordered[2] = quad[np.argmax(coordinate_sum)]
    ordered[1] = quad[np.argmin(coordinate_delta)]
    ordered[3] = quad[np.argmax(coordinate_delta)]

    if len({tuple(map(float, point)) for point in ordered}) != 4:
        raise ValueError("Court quadrilateral contains duplicate or ambiguous points")
    return ordered


def canonical_corners(width: int, height: int) -> FloatArray:
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def rectify(
    frame: NDArray[np.uint8], corners: NDArray[np.generic], width: int, height: int
) -> tuple[NDArray[np.uint8], FloatArray, FloatArray]:
    """Warp a source quadrilateral to the canonical court and return both transforms."""

    source = order_quad(corners)
    destination = canonical_corners(width, height)
    source_to_court = cv2.getPerspectiveTransform(source, destination)
    court_to_source = cv2.getPerspectiveTransform(destination, source)
    rectified = cv2.warpPerspective(frame, source_to_court, (width, height))
    return rectified, source_to_court, court_to_source


def transform_point(point: Point, matrix: NDArray[np.generic]) -> Point:
    """Transform one point through a 3x3 perspective matrix."""

    source = np.array([[[point.x, point.y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(source, np.asarray(matrix, dtype=np.float32))[0, 0]
    return Point(float(transformed[0]), float(transformed[1]))


def normalized_to_pixel(point: Point, width: int, height: int) -> Point:
    return Point(point.x * max(width - 1, 1), point.y * max(height - 1, 1))


def pixel_to_normalized(point: Point, width: int, height: int) -> Point:
    x = float(np.clip(point.x / max(width - 1, 1), 0.0, 1.0))
    y = float(np.clip(point.y / max(height - 1, 1), 0.0, 1.0))
    return Point(x, y)
