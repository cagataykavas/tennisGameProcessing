"""Deterministic synthetic scene used for onboarding and integration tests."""

from __future__ import annotations

import math
from collections.abc import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from tennis_vision.geometry import canonical_corners, transform_point
from tennis_vision.models import Point


def _triangle_wave(value: float) -> float:
    fractional = value % 1.0
    return 2 * fractional if fractional < 0.5 else 2 * (1 - fractional)


def generate_synthetic_frames(
    frame_count: int = 90, *, width: int = 960, height: int = 540
) -> Iterator[NDArray[np.uint8]]:
    """Yield a blue perspective court with moving geometric actors."""

    if frame_count < 8:
        raise ValueError("Synthetic demo requires at least eight frames")
    if width < 320 or height < 240:
        raise ValueError("Synthetic frame dimensions must be at least 320 x 240")

    source_court = np.array(
        [
            [0.19 * width, 0.14 * height],
            [0.81 * width, 0.14 * height],
            [0.96 * width, 0.94 * height],
            [0.04 * width, 0.94 * height],
        ],
        dtype=np.float32,
    )
    canonical_width, canonical_height = 600, 360
    canonical = canonical_corners(canonical_width, canonical_height)
    court_to_source = cv2.getPerspectiveTransform(canonical, source_court)

    def project(x: float, y: float) -> tuple[int, int]:
        point = transform_point(
            Point(x * (canonical_width - 1), y * (canonical_height - 1)),
            court_to_source,
        )
        return int(round(point.x)), int(round(point.y))

    for index in range(frame_count):
        frame = np.full((height, width, 3), (35, 62, 35), dtype=np.uint8)
        cv2.fillConvexPoly(frame, source_court.astype(np.int32), (190, 95, 30))

        boundary = np.array(
            [project(0.06, 0.04), project(0.94, 0.04), project(0.94, 0.96), project(0.06, 0.96)],
            dtype=np.int32,
        )
        cv2.polylines(frame, [boundary], True, (245, 245, 245), 2, cv2.LINE_AA)
        for first, second in (
            ((0.06, 0.50), (0.94, 0.50)),
            ((0.22, 0.04), (0.22, 0.96)),
            ((0.78, 0.04), (0.78, 0.96)),
            ((0.22, 0.25), (0.78, 0.25)),
            ((0.22, 0.75), (0.78, 0.75)),
            ((0.50, 0.25), (0.50, 0.75)),
        ):
            cv2.line(frame, project(*first), project(*second), (238, 238, 238), 2, cv2.LINE_AA)

        if index >= 5:
            phase = (index - 5) / max(frame_count - 5, 1)
            far_x = 0.5 + 0.18 * math.sin(phase * math.tau * 1.3)
            near_x = 0.5 + 0.22 * math.sin(phase * math.tau * 1.1 + 1.4)
            ball_x = 0.18 + 0.64 * _triangle_wave(phase * 2.2)
            ball_y = 0.22 + 0.58 * _triangle_wave(phase * 2.8 + 0.15)

            for (x, y), color in (
                ((far_x, 0.30), (40, 50, 225)),
                ((near_x, 0.76), (40, 215, 230)),
            ):
                center_x, center_y = project(x, y)
                player_height = max(22, int(23 + 34 * y))
                player_width = max(10, int(player_height * 0.38))
                cv2.rectangle(
                    frame,
                    (center_x - player_width // 2, center_y - player_height),
                    (center_x + player_width // 2, center_y),
                    color,
                    -1,
                    cv2.LINE_AA,
                )
            cv2.circle(frame, project(ball_x, ball_y), 6, (25, 245, 245), -1, cv2.LINE_AA)

        cv2.putText(
            frame,
            "SYNTHETIC DEMO - NOT REAL MATCH FOOTAGE",
            (18, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        yield frame
