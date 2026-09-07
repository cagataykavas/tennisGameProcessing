"""Explainable court and motion-object detectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from tennis_vision.config import VisionConfig
from tennis_vision.geometry import order_quad, pixel_to_normalized
from tennis_vision.models import BoundingBox, CourtEstimate, Detection, Point


def _find_contours(mask: NDArray[np.uint8]) -> list[NDArray[np.int32]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


class CourtDetector:
    """Locate a blue court surface and cache its last usable quadrilateral."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._last_corners: tuple[Point, ...] = ()
        self._missing_frames = 0

    def detect(self, frame: NDArray[np.uint8]) -> tuple[CourtEstimate, NDArray[np.uint8]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.asarray(self.config.court_lower_hsv, dtype=np.uint8)
        upper = np.asarray(self.config.court_upper_hsv, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours = _find_contours(mask)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            frame_area = float(frame.shape[0] * frame.shape[1])
            area_ratio = contour_area / frame_area if frame_area else 0.0
            if area_ratio >= self.config.court_min_area_ratio:
                perimeter = cv2.arcLength(contour, True)
                approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
                if len(approximation) == 4:
                    raw_corners = approximation.reshape(4, 2)
                    shape_source = "polygon"
                else:
                    raw_corners = cv2.boxPoints(cv2.minAreaRect(contour))
                    shape_source = "minimum_area_rectangle"

                try:
                    ordered = order_quad(raw_corners)
                except ValueError:
                    ordered = np.empty((0, 2), dtype=np.float32)
                if ordered.shape == (4, 2):
                    corners = tuple(Point(float(x), float(y)) for x, y in ordered)
                    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
                    solidity = min(contour_area / hull_area, 1.0)
                    area_score = min(
                        area_ratio / max(self.config.court_min_area_ratio * 2, 0.01), 1.0
                    )
                    confidence = 0.55 * area_score + 0.45 * solidity
                    self._last_corners = corners
                    self._missing_frames = 0
                    estimate = CourtEstimate(
                        visible=True,
                        source="observed",
                        confidence=confidence,
                        corners=corners,
                        evidence={
                            "decision_rule": "largest_hsv_surface",
                            "area_ratio": round(area_ratio, 4),
                            "minimum_area_ratio": self.config.court_min_area_ratio,
                            "solidity": round(solidity, 4),
                            "corner_method": shape_source,
                        },
                    )
                    return estimate, mask

        if self._last_corners and self._missing_frames < self.config.court_cache_frames:
            self._missing_frames += 1
            confidence = max(
                0.15, 0.75 * (1 - self._missing_frames / (self.config.court_cache_frames + 1))
            )
            return (
                CourtEstimate(
                    visible=False,
                    source="cached",
                    confidence=confidence,
                    corners=self._last_corners,
                    evidence={
                        "decision_rule": "last_observed_court",
                        "frames_since_observation": self._missing_frames,
                        "cache_limit": self.config.court_cache_frames,
                    },
                ),
                mask,
            )

        self._missing_frames += 1
        return (
            CourtEstimate(
                visible=False,
                source="none",
                confidence=0.0,
                evidence={
                    "decision_rule": "no_surface_above_threshold",
                    "minimum_area_ratio": self.config.court_min_area_ratio,
                },
            ),
            mask,
        )


@dataclass(slots=True)
class MotionDiagnostics:
    foreground_ratio: float
    contour_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "foreground_ratio": round(self.foreground_ratio, 4),
            "contour_count": self.contour_count,
        }


class MotionObjectDetector:
    """Classify foreground components using inspectable geometry rules."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.motion_history,
            varThreshold=config.motion_threshold,
            detectShadows=False,
        )

    def detect(
        self, rectified_frame: NDArray[np.uint8]
    ) -> tuple[list[Detection], NDArray[np.uint8], MotionDiagnostics]:
        foreground = self._subtractor.apply(rectified_frame)
        kernel_size = self.config.morphology_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground[:2, :] = 0
        foreground[-2:, :] = 0
        foreground[:, :2] = 0
        foreground[:, -2:] = 0
        candidates = self.extract_candidates(rectified_frame, foreground)
        ratio = cv2.countNonZero(foreground) / float(foreground.size)
        diagnostics = MotionDiagnostics(ratio, len(_find_contours(foreground)))
        return candidates, foreground, diagnostics

    def extract_candidates(
        self, rectified_frame: NDArray[np.uint8], foreground: NDArray[np.uint8]
    ) -> list[Detection]:
        """Turn a binary foreground mask into explainable player/ball candidates."""

        height, width = rectified_frame.shape[:2]
        frame_area = float(height * width)
        candidates: list[Detection] = []

        for contour in _find_contours(foreground):
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 0 or box_height <= 0:
                continue

            bbox = BoundingBox(x, y, box_width, box_height)
            centroid = bbox.centroid
            normalized = pixel_to_normalized(centroid, width, height)
            area_ratio = contour_area / frame_area
            aspect_ratio = box_width / float(box_height)
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4 * math.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0.0
            )
            circularity = float(np.clip(circularity, 0.0, 1.0))
            height_ratio = box_height / float(height)

            player_match = (
                self.config.player_min_area_ratio <= area_ratio <= self.config.player_max_area_ratio
                and height_ratio >= self.config.player_min_height_ratio
                and self.config.player_min_aspect_ratio
                <= aspect_ratio
                <= self.config.player_max_aspect_ratio
            )
            ball_match = (
                self.config.ball_min_area_ratio <= area_ratio <= self.config.ball_max_area_ratio
                and self.config.ball_min_aspect_ratio
                <= aspect_ratio
                <= self.config.ball_max_aspect_ratio
                and circularity >= self.config.ball_min_circularity
            )

            common_evidence = {
                "area_px": round(contour_area, 2),
                "area_ratio": round(area_ratio, 6),
                "aspect_ratio": round(aspect_ratio, 4),
                "height_ratio": round(height_ratio, 4),
                "circularity": round(circularity, 4),
            }

            if player_match:
                upright_score = max(0.0, 1.0 - abs(aspect_ratio - 0.55) / 1.25)
                size_score = min(height_ratio / 0.18, 1.0)
                confidence = min(0.98, 0.45 + 0.3 * upright_score + 0.25 * size_score)
                candidates.append(
                    Detection(
                        label="player",
                        confidence=confidence,
                        canonical_bbox=bbox,
                        court_position=normalized,
                        evidence={
                            **common_evidence,
                            "decision_rule": "upright_motion_component",
                            "thresholds": {
                                "area_ratio": [
                                    self.config.player_min_area_ratio,
                                    self.config.player_max_area_ratio,
                                ],
                                "minimum_height_ratio": self.config.player_min_height_ratio,
                                "aspect_ratio": [
                                    self.config.player_min_aspect_ratio,
                                    self.config.player_max_aspect_ratio,
                                ],
                            },
                        },
                    )
                )
            elif ball_match:
                roundness_score = min(
                    1.0,
                    (circularity - self.config.ball_min_circularity)
                    / max(1.0 - self.config.ball_min_circularity, 0.01),
                )
                aspect_score = max(0.0, 1.0 - abs(1.0 - aspect_ratio))
                confidence = min(0.98, 0.5 + 0.3 * roundness_score + 0.2 * aspect_score)
                candidates.append(
                    Detection(
                        label="ball",
                        confidence=confidence,
                        canonical_bbox=bbox,
                        court_position=normalized,
                        evidence={
                            **common_evidence,
                            "decision_rule": "small_compact_motion",
                            "thresholds": {
                                "area_ratio": [
                                    self.config.ball_min_area_ratio,
                                    self.config.ball_max_area_ratio,
                                ],
                                "minimum_circularity": self.config.ball_min_circularity,
                                "aspect_ratio": [
                                    self.config.ball_min_aspect_ratio,
                                    self.config.ball_max_aspect_ratio,
                                ],
                            },
                        },
                    )
                )

        return sorted(candidates, key=lambda item: item.confidence, reverse=True)
