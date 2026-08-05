"""Validated configuration for the vision pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Runtime thresholds with conservative defaults for a blue hard court."""

    court_lower_hsv: tuple[int, int, int] = (90, 55, 45)
    court_upper_hsv: tuple[int, int, int] = (145, 255, 255)
    court_min_area_ratio: float = 0.12
    court_cache_frames: int = 8
    canonical_width: int = 600
    canonical_height: int = 360

    motion_history: int = 90
    motion_threshold: int = 24
    morphology_kernel: int = 3

    player_min_area_ratio: float = 0.0008
    player_max_area_ratio: float = 0.08
    player_min_height_ratio: float = 0.045
    player_min_aspect_ratio: float = 0.12
    player_max_aspect_ratio: float = 1.8

    ball_min_area_ratio: float = 0.00001
    ball_max_area_ratio: float = 0.0012
    ball_min_aspect_ratio: float = 0.42
    ball_max_aspect_ratio: float = 2.2
    ball_min_circularity: float = 0.32

    track_smoothing: float = 0.35
    track_max_jump: float = 0.35
    track_max_missing: int = 12

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VisionConfig:
        """Create a config and reject keys that would otherwise be silently ignored."""

        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")

        prepared = dict(values)
        for key in ("court_lower_hsv", "court_upper_hsv"):
            if key in prepared:
                value = prepared[key]
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise ValueError(f"{key} must contain exactly three integers")
                prepared[key] = tuple(int(item) for item in value)

        config = cls(**prepared)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> VisionConfig:
        """Load configuration from a UTF-8 JSON object."""

        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Configuration root must be a JSON object")
        return cls.from_mapping(payload)

    def validate(self) -> None:
        """Fail fast on invalid dimensions, ranges, or HSV bounds."""

        if self.canonical_width < 64 or self.canonical_height < 64:
            raise ValueError("Canonical court dimensions must both be at least 64 pixels")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer")
        if self.motion_history < 2 or self.track_max_missing < 0:
            raise ValueError("motion_history must be >= 2 and track_max_missing must be >= 0")

        unit_interval_fields = (
            "court_min_area_ratio",
            "player_min_area_ratio",
            "player_max_area_ratio",
            "player_min_height_ratio",
            "ball_min_area_ratio",
            "ball_max_area_ratio",
            "ball_min_circularity",
            "track_smoothing",
            "track_max_jump",
        )
        for name in unit_interval_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if self.player_min_area_ratio >= self.player_max_area_ratio:
            raise ValueError("player_min_area_ratio must be smaller than player_max_area_ratio")
        if self.ball_min_area_ratio >= self.ball_max_area_ratio:
            raise ValueError("ball_min_area_ratio must be smaller than ball_max_area_ratio")
        if self.player_min_aspect_ratio >= self.player_max_aspect_ratio:
            raise ValueError("player aspect-ratio bounds are reversed")
        if self.ball_min_aspect_ratio >= self.ball_max_aspect_ratio:
            raise ValueError("ball aspect-ratio bounds are reversed")

        for name in ("court_lower_hsv", "court_upper_hsv"):
            hue, saturation, value = getattr(self, name)
            if not (0 <= hue <= 179 and 0 <= saturation <= 255 and 0 <= value <= 255):
                raise ValueError(f"{name} contains a value outside OpenCV HSV bounds")
        if any(
            low > high for low, high in zip(self.court_lower_hsv, self.court_upper_hsv, strict=True)
        ):
            raise ValueError("court_lower_hsv must not exceed court_upper_hsv")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        result = asdict(self)
        result["court_lower_hsv"] = list(self.court_lower_hsv)
        result["court_upper_hsv"] = list(self.court_upper_hsv)
        return result
