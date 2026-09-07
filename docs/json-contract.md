# Explainable JSON contract

The event stream is newline-delimited JSON (`events.jsonl`). Every line is a complete
frame result, so consumers can stream it without loading the entire match. Additive
fields may appear within schema version `1.x`; removals or semantic changes require a
new major version.

## Frame event

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Currently `1.0` |
| `frame_index` | integer | Zero-based processed-frame index |
| `timestamp_ms` | number | Derived from frame index and reported FPS |
| `processing_ms` | number | Wall-clock processing time for this frame |
| `court` | object | Court observation and the reasons for it |
| `detections` | array | Selected semantic tracks observed this frame |
| `diagnostics` | object | Candidate, foreground, contour, and track counts |

## Court object

| Field | Type | Meaning |
|---|---|---|
| `visible` | boolean | Court surface passed thresholds in the current frame |
| `source` | enum | `observed`, `cached`, or `none` |
| `confidence` | number | Inspectable heuristic score in `[0, 1]` |
| `corners` | array | Source-pixel corners in TL, TR, BR, BL order |
| `evidence` | object | Rule, measurements, thresholds, and cache state |

A cached court deliberately has `visible: false` while still retaining four corners.
This lets a consumer distinguish current visual evidence from short-term continuity.

## Detection object

| Field | Type | Meaning |
|---|---|---|
| `track_id` | string | `player_far`, `player_near`, or `ball_1` |
| `label` | string | `player` or `ball` |
| `confidence` | number | Heuristic candidate ranking score in `[0, 1]` |
| `canonical_bbox` | object | Component box in rectified-court pixels |
| `court_position` | object | Smoothed, normalized court position |
| `image_position` | object | Projected location in source-frame pixels |
| `evidence` | object | Geometry, thresholds, identity rule, and smoothing state |

`confidence` is not a probability. It has not been calibrated against labeled match
footage. The evidence fields are the authoritative explanation of why a component
passed the baseline rules.

## Complete example

```json
{
  "schema_version": "1.0",
  "frame_index": 18,
  "timestamp_ms": 600.0,
  "processing_ms": 6.12,
  "court": {
    "visible": true,
    "source": "observed",
    "confidence": 0.94,
    "corners": [
      {"x": 182.0, "y": 76.0},
      {"x": 778.0, "y": 76.0},
      {"x": 920.0, "y": 507.0},
      {"x": 39.0, "y": 507.0}
    ],
    "evidence": {
      "decision_rule": "largest_hsv_surface",
      "area_ratio": 0.56,
      "minimum_area_ratio": 0.12,
      "solidity": 0.99,
      "corner_method": "polygon"
    }
  },
  "detections": [
    {
      "track_id": "ball_1",
      "label": "ball",
      "confidence": 0.81,
      "canonical_bbox": {"x": 307, "y": 149, "width": 11, "height": 12},
      "court_position": {"x": 0.52, "y": 0.43},
      "image_position": {"x": 501.8, "y": 256.3},
      "evidence": {
        "decision_rule": "small_compact_motion",
        "area_px": 96.0,
        "area_ratio": 0.000444,
        "aspect_ratio": 0.9167,
        "height_ratio": 0.0333,
        "circularity": 0.79,
        "identity_assignment": "nearest_previous_position",
        "tracking_status": "smoothed",
        "raw_court_position": {"x": 0.53, "y": 0.42}
      }
    }
  ],
  "diagnostics": {
    "candidate_count": 3,
    "tracked_count": 3,
    "foreground_ratio": 0.026,
    "contour_count": 5
  }
}
```

The example demonstrates field shape; its numbers are illustrative and are not a
published benchmark result.

## Summary contract

`summary.json` records frames processed, input metadata, configuration, mean frame
processing time, wall-clock throughput, court-source counts, per-track observation
counts, and artifact filenames. It always includes `metric_note` to prevent raw
detection counts from being mistaken for model accuracy.

## Consumer guidance

- Branch on `schema_version` before parsing.
- Treat unknown additive fields as optional.
- Use `court.source == "observed"` when current-frame evidence is mandatory.
- Use normalized `court_position` for analytics and `image_position` for overlays.
- Do not interpret missing detections as proof that no player or ball exists.
