# Tennis Match Video Analysis with Classical Computer Vision

A video-processing project that detects the tennis court, estimates player and ball locations, rectifies the camera view, and projects the tracked objects onto a top-down tennis-court representation.

The implementation uses **OpenCV, NumPy and SciPy** with handcrafted image-processing and geometric heuristics rather than a pretrained object detector.

> This repository is an academic computer-vision project. The thresholds and assumptions are tuned to the project footage and should be understood as a classical-CV experiment rather than a production sports-tracking system.

## Project goals

The pipeline attempts to answer several problems from a single tennis video:

- determine whether the match/court is currently visible;
- locate the tennis court in the camera image;
- rectify the perspective view of the court;
- detect and distinguish player-sized and ball-sized moving regions;
- maintain short temporal histories for tracked objects;
- map detected positions from camera coordinates to a top-down court representation;
- visualise player and ball positions on `tennis_court.png`;
- optionally display movement paths and detection attributes;
- save processed output video.

## Processing pipeline

At a high level:

```text
Input tennis video
        │
        ▼
Court / scene analysis
        │
        ▼
Court geometry estimation
        │
        ▼
Perspective rectification
        │
        ▼
Foreground / contour extraction
        │
        ├──► player candidates
        │
        └──► ball candidates
        │
        ▼
Temporal association / filtering
        │
        ▼
Coordinate transformation
        │
        ▼
Top-down court visualisation
```

## Techniques demonstrated

- Video processing with OpenCV
- Color-based court analysis
- Thresholding and foreground segmentation
- Contour extraction
- Contour grouping
- Area and aspect-ratio classification
- Morphological processing
- Perspective transformation / court rectification
- Player and ball candidate filtering
- Spatial-distance calculations
- Temporal position history with `deque`
- Mapping image-space positions to a court diagram
- OpenCV and Matplotlib visualisation
- Video export

## Repository structure

```text
tennisGameProcessing/
├── project.py           # Complete processing pipeline
├── tennis_court.png     # Top-down court visualisation asset
├── requirements.txt     # Python dependencies
├── ReadMe.txt           # Original project notes
└── README.md            # Project documentation
```

The source video is expected in the repository root as:

```text
tennis.mp4
```

The video itself is not included in the repository.

## Installation

Python 3 is required.

```bash
python -m venv .venv
```

Activate the environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

## Running

Place `tennis.mp4` in the project directory and run:

```bash
python project.py
```

The script opens its configured visualisation windows and writes processed video output to:

```text
output.avi
```

## Configuration

Most detection behaviour is controlled through the `params` dictionary near the beginning of `project.py`.

It contains thresholds and switches for:

- player minimum / maximum area;
- player aspect ratio;
- ball minimum / maximum area;
- ball aspect ratio;
- contour grouping distance;
- morphology / dilation;
- court color ratio;
- court size and aspect-ratio limits;
- perspective expansion;
- player and ball search radii;
- ball circularity checking;
- tracking-history length;
- visualisation windows and movement paths.

Keeping these parameters together makes the assumptions of the handcrafted detector explicit and allows the behaviour to be tuned for different footage.

## Court representation

The project uses the standard court dimensions encoded in the source:

```text
Length: 78 ft
Doubles width: 36 ft
```

The supplied `tennis_court.png` is used as a top-down representation. Detected player and ball positions can be transformed from the camera / rectified image into this court coordinate system and rendered as markers.

This is the most interesting part of the project from a geometry perspective: detections are not only drawn on the original video, but interpreted relative to the physical playing surface.

## Player and ball candidates

Candidate moving regions are represented by contours and bounding boxes. The code uses configurable area and width/height constraints to distinguish likely player-sized regions from ball-sized regions.

Nearby contours can be grouped before classification, which is useful because foreground segmentation may split a player into several disconnected regions.

The player grouping logic also uses the player's location relative to the two halves of the court.

Ball candidates use tighter size/aspect-ratio constraints, optional circularity checks, search-radius restrictions and temporal history to reduce false positives.

## Visualisation

Depending on the configured flags, the project can show:

- the original video;
- the rectified tennis-court view;
- player bounding boxes;
- ball detections and attributes;
- movement histories;
- player/ball markers on the top-down court image.

This makes the project useful for inspecting each stage of the image-processing pipeline rather than exposing only a final result.

## Limitations

The detector is intentionally heuristic and footage-specific.

Important limitations include:

- fixed thresholds depend strongly on resolution, lighting and camera position;
- color-based court detection assumes a visually distinctive court;
- contour-based player/ball classification can be confused by shadows and background motion;
- a tennis ball occupies very few pixels and is difficult to track reliably with simple segmentation;
- camera cuts, zooms or significant camera motion can invalidate geometric assumptions;
- occlusion can disrupt player or ball association;
- many parameters were tuned experimentally for the original project video;
- the current implementation is a large single-file research/assignment prototype rather than a packaged library.

## Possible extensions

A modern version could combine this geometry pipeline with learned player/ball detectors, Kalman filtering, court-keypoint models, trajectory smoothing, bounce detection, shot classification and quantitative tracking metrics.

The existing implementation remains useful as a **classical-computer-vision baseline** because the detection and geometry decisions are visible and inspectable rather than hidden inside a neural model.

## Why this project is useful

This repository demonstrates considerably more than basic object detection. It combines scene understanding, handcrafted segmentation, object classification, perspective geometry, temporal tracking and coordinate mapping into an end-to-end sports-video analysis pipeline.
