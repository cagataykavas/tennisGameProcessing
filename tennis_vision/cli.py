"""Command-line interface for real videos and the self-contained demo."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tennis_vision.config import VisionConfig
from tennis_vision.demo import generate_synthetic_frames
from tennis_vision.pipeline import analyze_frames, analyze_video


def json_contract() -> dict[str, Any]:
    """Return a compact contract; the detailed field guide lives in docs/."""

    return {
        "schema_version": "1.0",
        "event": {
            "frame_index": "non-negative integer",
            "timestamp_ms": "number",
            "processing_ms": "number",
            "court": {
                "visible": "boolean: observed in the current frame",
                "source": "observed | cached | none",
                "confidence": "heuristic score from 0 to 1",
                "corners": "top-left, top-right, bottom-right, bottom-left pixels",
                "evidence": "decision inputs and thresholds",
            },
            "detections": [
                {
                    "track_id": "player_far | player_near | ball_1",
                    "label": "player | ball",
                    "confidence": "heuristic score from 0 to 1",
                    "canonical_bbox": "box in 600 x 360 rectified court pixels",
                    "court_position": "normalized x/y in [0, 1]",
                    "image_position": "source-frame pixel x/y",
                    "evidence": "classification and identity-assignment reasons",
                }
            ],
            "diagnostics": "candidate, track, contour, and foreground counts",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tennis-vision",
        description="Explainable tennis court, player, and ball tracking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze a real video")
    analyze.add_argument("--input", required=True, type=Path, help="source video")
    analyze.add_argument("--output", required=True, type=Path, help="artifact directory")
    analyze.add_argument("--config", type=Path, help="optional JSON configuration")
    analyze.add_argument("--max-frames", type=int, help="optional processing limit")
    analyze.add_argument("--no-video", action="store_true", help="skip annotated MP4 output")

    demo = subparsers.add_parser("demo", help="run the generated integration demo")
    demo.add_argument("--output", required=True, type=Path, help="artifact directory")
    demo.add_argument("--frames", type=int, default=90, help="number of generated frames")
    demo.add_argument("--fps", type=float, default=30.0, help="synthetic frame rate")
    demo.add_argument("--config", type=Path, help="optional JSON configuration")
    demo.add_argument("--no-video", action="store_true", help="skip annotated MP4 output")

    subparsers.add_parser("explain-schema", help="print the event JSON contract")
    return parser


def _load_config(path: Path | None) -> VisionConfig:
    config = VisionConfig.from_json(path) if path else VisionConfig()
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "explain-schema":
            print(json.dumps(json_contract(), indent=2, sort_keys=True))
            return 0

        config = _load_config(args.config)
        if args.command == "analyze":
            summary = analyze_video(
                args.input,
                output_dir=args.output,
                config=config,
                max_frames=args.max_frames,
                write_video=not args.no_video,
            )
        else:
            summary = analyze_frames(
                generate_synthetic_frames(args.frames),
                fps=args.fps,
                output_dir=args.output,
                config=config,
                write_video=not args.no_video,
                input_metadata={
                    "kind": "synthetic",
                    "generator": "tennis_vision.demo.generate_synthetic_frames",
                    "frames_requested": args.frames,
                    "ground_truth_metrics": False,
                },
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
