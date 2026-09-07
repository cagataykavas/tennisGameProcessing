from tennis_vision.config import VisionConfig
from tennis_vision.models import BoundingBox, Detection, Point
from tennis_vision.tracking import TrackManager


def _player(y: float) -> Detection:
    return Detection(
        label="player",
        confidence=0.8,
        canonical_bbox=BoundingBox(10, 10, 20, 80),
        court_position=Point(0.5, y),
    )


def test_players_receive_court_relative_identities():
    tracked = TrackManager(VisionConfig()).update([_player(0.78), _player(0.24)])

    assert {item.track_id for item in tracked} == {"player_far", "player_near"}
    assert all("identity_assignment" in item.evidence for item in tracked)
