"""Lightweight identity assignment and trajectory smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from tennis_vision.config import VisionConfig
from tennis_vision.models import Detection, Point


@dataclass(slots=True)
class _TrackState:
    position: Point
    missing_frames: int = 0


class TrackManager:
    """Assign semantic identities without hiding how an assignment was made."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._states: dict[str, _TrackState] = {}

    def update(self, candidates: list[Detection]) -> list[Detection]:
        assignments: list[tuple[str, Detection, str]] = []
        players = [item for item in candidates if item.label == "player"]
        balls = [item for item in candidates if item.label == "ball"]

        if len(players) == 1:
            track_id = "player_far" if players[0].court_position.y < 0.5 else "player_near"
            assignments.append((track_id, players[0], "court_half"))
        elif len(players) >= 2:
            ordered = sorted(players, key=lambda item: item.court_position.y)
            assignments.append(("player_far", ordered[0], "minimum_court_y"))
            assignments.append(("player_near", ordered[-1], "maximum_court_y"))

        if balls:
            previous_ball = self._states.get("ball_1")
            if previous_ball is None:
                selected_ball = max(balls, key=lambda item: item.confidence)
                rule = "highest_confidence"
            else:
                selected_ball = min(
                    balls,
                    key=lambda item: self._distance(item.court_position, previous_ball.position),
                )
                rule = "nearest_previous_position"
            assignments.append(("ball_1", selected_ball, rule))

        observed_ids = {track_id for track_id, _, _ in assignments}
        for track_id in list(self._states):
            if track_id not in observed_ids:
                self._states[track_id].missing_frames += 1
                if self._states[track_id].missing_frames > self.config.track_max_missing:
                    del self._states[track_id]

        tracked: list[Detection] = []
        for track_id, detection, assignment_rule in assignments:
            state = self._states.get(track_id)
            raw_position = detection.court_position
            tracking_status = "initialized"
            if state is not None:
                jump = self._distance(raw_position, state.position)
                if jump <= self.config.track_max_jump:
                    alpha = self.config.track_smoothing
                    position = Point(
                        alpha * raw_position.x + (1 - alpha) * state.position.x,
                        alpha * raw_position.y + (1 - alpha) * state.position.y,
                    )
                    tracking_status = "smoothed"
                else:
                    position = raw_position
                    tracking_status = "reset_after_large_jump"
            else:
                position = raw_position

            self._states[track_id] = _TrackState(position=position)
            evidence = {
                **detection.evidence,
                "identity_assignment": assignment_rule,
                "tracking_status": tracking_status,
                "raw_court_position": raw_position.to_dict(),
            }
            tracked.append(
                replace(
                    detection,
                    court_position=position,
                    track_id=track_id,
                    evidence=evidence,
                )
            )
        return tracked

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)
