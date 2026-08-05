# Tennis Vision Lab

[![CI](https://github.com/cagataykavas/tennisGameProcessing/actions/workflows/ci.yml/badge.svg)](https://github.com/cagataykavas/tennisGameProcessing/actions/workflows/ci.yml)

An explainable, classical-computer-vision baseline for locating a tennis court,
tracking two players and a ball, and exporting every decision as machine-readable
JSON. It runs headlessly, includes a deterministic synthetic demo, and does not
require a bundled match video or a pretrained model.

This repository is a production-minded repair of an earlier university prototype.
The original 1,352-line script is preserved in
[`legacy/tennis_tracker_monolith.py`](legacy/tennis_tracker_monolith.py) for
provenance; it is no longer imported by the application.

## What it demonstrates

- HSV court segmentation with cached-court recovery during brief occlusions
- Perspective rectification into a canonical 600 x 360 court
- Motion-based player and ball candidates with explicit heuristic evidence
- Stable `player_far`, `player_near`, and `ball_1` track identities
- Annotated video, a preview image, JSONL frame events, and a run summary
- Import-safe modules, typed configuration, tests, and headless CI

The detector is intentionally a transparent baseline, not a claim of modern
player/ball detection accuracy. Reported run metrics are processing and detection
counts—not fabricated precision or recall.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Generates its own tennis scene and exercises the full pipeline.
tennis-vision demo --output artifacts/demo --frames 90

# Analyze a real recording.
tennis-vision analyze --input tennis.mp4 --output artifacts/match

# Print the JSON contract without running video processing.
tennis-vision explain-schema
```

The historical entry point remains useful:

```bash
python project.py demo --output artifacts/demo
```

## Outputs

Each run creates:

| File | Purpose |
|---|---|
| `events.jsonl` | One explainable event per processed frame |
| `summary.json` | Input metadata, timing, and observed detection counts |
| `annotated.mp4` | Court polygon, track IDs, and current positions |
| `preview.jpg` | Final annotated frame for quick inspection |

Example event fragment:

```json
{
  "schema_version": "1.0",
  "frame_index": 18,
  "court": {
    "visible": true,
    "source": "observed",
    "confidence": 0.94
  },
  "detections": [
    {
      "track_id": "ball_1",
      "label": "ball",
      "confidence": 0.81,
      "court_position": {"x": 0.52, "y": 0.43},
      "evidence": {
        "decision_rule": "small_compact_motion",
        "area_px": 96.0,
        "circularity": 0.79
      }
    }
  ]
}
```

See [the architecture guide](docs/architecture.md) and
[the JSON contract](docs/json-contract.md) for details.

## Configuration

Copy [`config.example.json`](config.example.json), change only the thresholds you
need, and pass it to either command:

```bash
tennis-vision analyze \
  --input tennis.mp4 \
  --output artifacts/match \
  --config config.example.json \
  --max-frames 500
```

Unknown configuration keys and invalid ranges fail fast. This keeps experiments
reproducible and catches misspelled options.

## Development

```bash
pip install -e ".[dev]"
pytest
python -m compileall tennis_vision project.py
```

## Limitations and next experiments

- Blue-court HSV defaults need tuning for clay, grass, and indoor lighting.
- Motion segmentation struggles with a moving camera and stationary players.
- Very small balls can disappear after video compression.
- A learned detector can replace `MotionObjectDetector` while preserving the same
  JSON contract, tracker, CLI, and evaluation harness.

No private footage, proprietary model weights, or third-party dataset is included.

## License

MIT
