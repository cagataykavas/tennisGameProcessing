"""Small serializable domain models shared by detection, tracking, and output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _rounded(value: float) -> float:
    return round(float(value), 4)


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": _rounded(self.x), "y": _rounded(self.y)}


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def centroid(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class CourtEstimate:
    visible: bool
    source: str
    confidence: float
    corners: tuple[Point, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible": self.visible,
            "source": self.source,
            "confidence": _rounded(self.confidence),
            "corners": [point.to_dict() for point in self.corners],
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    canonical_bbox: BoundingBox
    court_position: Point
    image_position: Point | None = None
    track_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": _rounded(self.confidence),
            "canonical_bbox": self.canonical_bbox.to_dict(),
            "court_position": self.court_position.to_dict(),
            "image_position": self.image_position.to_dict() if self.image_position else None,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    frame_index: int
    timestamp_ms: float
    court: CourtEstimate
    detections: tuple[Detection, ...]
    processing_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_index": self.frame_index,
            "timestamp_ms": _rounded(self.timestamp_ms),
            "processing_ms": _rounded(self.processing_ms),
            "court": self.court.to_dict(),
            "detections": [item.to_dict() for item in self.detections],
            "diagnostics": self.diagnostics,
        }
