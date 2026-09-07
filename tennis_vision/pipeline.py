"""Frame processing, artifact generation, and video I/O."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from tennis_vision.config import VisionConfig
from tennis_vision.detection import CourtDetector, MotionObjectDetector
from tennis_vision.geometry import normalized_to_pixel, rectify, transform_point
from tennis_vision.models import Detection, FrameAnalysis
from tennis_vision.tracking import TrackManager


class FrameProcessor:
    """Stateful court detection, rectification, object detection, and tracking."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self.config.validate()
        self.court_detector = CourtDetector(self.config)
        self.object_detector = MotionObjectDetector(self.config)
        self.track_manager = TrackManager(self.config)

    def process(self, frame: NDArray[np.uint8], frame_index: int, fps: float) -> FrameAnalysis:
        """Analyze one BGR frame without opening a GUI or writing global files."""

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty BGR image with shape (height, width, 3)")
        started = time.perf_counter()
        court, _ = self.court_detector.detect(frame)
        detections: list[Detection] = []
        diagnostics: dict[str, Any] = {
            "candidate_count": 0,
            "tracked_count": 0,
            "foreground_ratio": 0.0,
        }

        if len(court.corners) == 4:
            corner_array = np.array(
                [[point.x, point.y] for point in court.corners], dtype=np.float32
            )
            rectified, _, court_to_source = rectify(
                frame,
                corner_array,
                self.config.canonical_width,
                self.config.canonical_height,
            )
            candidates, _, motion = self.object_detector.detect(rectified)
            tracked = self.track_manager.update(candidates)
            for item in tracked:
                canonical_point = normalized_to_pixel(
                    item.court_position,
                    self.config.canonical_width,
                    self.config.canonical_height,
                )
                image_point = transform_point(canonical_point, court_to_source)
                detections.append(replace(item, image_position=image_point))
            diagnostics = {
                "candidate_count": len(candidates),
                "tracked_count": len(detections),
                **motion.to_dict(),
            }

        processing_ms = (time.perf_counter() - started) * 1000
        return FrameAnalysis(
            frame_index=frame_index,
            timestamp_ms=frame_index * 1000.0 / max(fps, 0.001),
            court=court,
            detections=tuple(detections),
            processing_ms=processing_ms,
            diagnostics=diagnostics,
        )

    @staticmethod
    def annotate(frame: NDArray[np.uint8], analysis: FrameAnalysis) -> NDArray[np.uint8]:
        """Draw the public JSON result back onto a copy of the source frame."""

        canvas = frame.copy()
        if len(analysis.court.corners) == 4:
            polygon = np.array(
                [[point.x, point.y] for point in analysis.court.corners], dtype=np.int32
            ).reshape((-1, 1, 2))
            court_color = (70, 220, 70) if analysis.court.visible else (0, 190, 255)
            cv2.polylines(canvas, [polygon], True, court_color, 2, cv2.LINE_AA)

        colors = {
            "player_far": (80, 230, 80),
            "player_near": (60, 220, 255),
            "ball_1": (40, 40, 255),
        }
        for detection in analysis.detections:
            if detection.image_position is None:
                continue
            x = int(round(detection.image_position.x))
            y = int(round(detection.image_position.y))
            color = colors.get(detection.track_id or "", (255, 255, 255))
            cv2.circle(canvas, (x, y), 8 if detection.label == "player" else 6, color, -1)
            label = f"{detection.track_id} {detection.confidence:.2f}"
            cv2.putText(
                canvas,
                label,
                (x + 10, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        status = (
            f"frame={analysis.frame_index} court={analysis.court.source} "
            f"tracks={len(analysis.detections)} {analysis.processing_ms:.1f}ms"
        )
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 630), 32), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            status,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        return canvas


def analyze_frames(
    frames: Iterable[NDArray[np.uint8]],
    *,
    fps: float,
    output_dir: str | Path,
    config: VisionConfig | None = None,
    max_frames: int | None = None,
    write_video: bool = True,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze an iterable and write a reproducible artifact bundle."""

    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero when provided")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    events_path = destination / "events.jsonl"
    summary_path = destination / "summary.json"
    preview_path = destination / "preview.jpg"
    video_path = destination / "annotated.mp4"

    processor = FrameProcessor(config)
    writer: cv2.VideoWriter | None = None
    counts: Counter[str] = Counter()
    observed_court_frames = 0
    cached_court_frames = 0
    frame_count = 0
    processing_total_ms = 0.0
    last_annotated: NDArray[np.uint8] | None = None
    wall_started = time.perf_counter()

    with events_path.open("w", encoding="utf-8") as event_file:
        for frame_index, frame in enumerate(frames):
            if max_frames is not None and frame_index >= max_frames:
                break
            analysis = processor.process(frame, frame_index, fps)
            annotated = processor.annotate(frame, analysis)
            event_file.write(json.dumps(analysis.to_dict(), sort_keys=True) + "\n")

            if write_video and writer is None:
                height, width = annotated.shape[:2]
                candidate = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if candidate.isOpened():
                    writer = candidate
                else:
                    candidate.release()
            if writer is not None:
                writer.write(annotated)

            for detection in analysis.detections:
                counts[detection.track_id or detection.label] += 1
            if analysis.court.source == "observed":
                observed_court_frames += 1
            elif analysis.court.source == "cached":
                cached_court_frames += 1
            processing_total_ms += analysis.processing_ms
            frame_count += 1
            last_annotated = annotated

    if writer is not None:
        writer.release()
    elif video_path.exists():
        video_path.unlink()

    if frame_count == 0 or last_annotated is None:
        raise ValueError("No frames were available for analysis")
    if not cv2.imwrite(str(preview_path), last_annotated):
        raise OSError(f"Could not write preview image to {preview_path}")

    wall_seconds = time.perf_counter() - wall_started
    video_written = video_path.exists() and video_path.stat().st_size > 0
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "frames_processed": frame_count,
        "fps_reported": round(float(fps), 4),
        "observed_court_frames": observed_court_frames,
        "cached_court_frames": cached_court_frames,
        "detection_observations": dict(sorted(counts.items())),
        "mean_processing_ms": round(processing_total_ms / frame_count, 4),
        "wall_seconds": round(wall_seconds, 4),
        "throughput_fps": round(frame_count / max(wall_seconds, 0.0001), 4),
        "configuration": processor.config.to_dict(),
        "input": input_metadata or {"kind": "frame_iterable"},
        "artifacts": {
            "events": events_path.name,
            "summary": summary_path.name,
            "preview": preview_path.name,
            "video": video_path.name if video_written else None,
        },
        "metric_note": (
            "Detection counts are observations from this heuristic baseline; "
            "they are not accuracy measurements."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _capture_frames(capture: cv2.VideoCapture) -> Iterator[NDArray[np.uint8]]:
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame


def analyze_video(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    config: VisionConfig | None = None,
    max_frames: int | None = None,
    write_video: bool = True,
) -> dict[str, Any]:
    """Analyze a video file and always release the OpenCV capture."""

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open input video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    metadata = {
        "kind": "video",
        "path": str(source),
        "reported_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    try:
        return analyze_frames(
            _capture_frames(capture),
            fps=fps,
            output_dir=output_dir,
            config=config,
            max_frames=max_frames,
            write_video=write_video,
            input_metadata=metadata,
        )
    finally:
        capture.release()
