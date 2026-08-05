import cv2
import numpy as np

from tennis_vision.config import VisionConfig
from tennis_vision.demo import generate_synthetic_frames
from tennis_vision.detection import CourtDetector, MotionObjectDetector


def test_synthetic_court_is_observed_with_four_corners():
    frame = next(generate_synthetic_frames(8))

    estimate, mask = CourtDetector(VisionConfig()).detect(frame)

    assert estimate.visible is True
    assert estimate.source == "observed"
    assert len(estimate.corners) == 4
    assert estimate.evidence["decision_rule"] == "largest_hsv_surface"
    assert cv2.countNonZero(mask) > 0


def test_candidate_rules_emit_explanations():
    config = VisionConfig()
    detector = MotionObjectDetector(config)
    frame = np.zeros((config.canonical_height, config.canonical_width, 3), dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (60, 60), (105, 180), 255, -1)
    cv2.circle(mask, (420, 180), 7, 255, -1)

    candidates = detector.extract_candidates(frame, mask)

    labels = {candidate.label for candidate in candidates}
    assert labels == {"player", "ball"}
    assert all("decision_rule" in candidate.evidence for candidate in candidates)
    assert all("thresholds" in candidate.evidence for candidate in candidates)
