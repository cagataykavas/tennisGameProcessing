import json

import pytest

from tennis_vision.config import VisionConfig


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(VisionConfig().to_dict()), encoding="utf-8")

    loaded = VisionConfig.from_json(path)

    assert loaded == VisionConfig()


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown configuration keys: typo_threshold"):
        VisionConfig.from_mapping({"typo_threshold": 123})


def test_even_morphology_kernel_is_rejected():
    with pytest.raises(ValueError, match="positive odd"):
        VisionConfig.from_mapping({"morphology_kernel": 4})
