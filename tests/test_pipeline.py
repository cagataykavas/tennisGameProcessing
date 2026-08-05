import json

from tennis_vision.demo import generate_synthetic_frames
from tennis_vision.pipeline import analyze_frames


def test_demo_writes_machine_readable_artifacts(tmp_path):
    output = tmp_path / "demo"

    summary = analyze_frames(
        generate_synthetic_frames(18),
        fps=30.0,
        output_dir=output,
        write_video=False,
        input_metadata={"kind": "synthetic-test"},
    )

    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    disk_summary = json.loads((output / "summary.json").read_text())

    assert summary == disk_summary
    assert summary["frames_processed"] == 18
    assert summary["observed_court_frames"] == 18
    assert summary["artifacts"]["video"] is None
    assert (output / "preview.jpg").stat().st_size > 0
    assert len(events) == 18
    assert events[0]["schema_version"] == "1.0"
    assert events[0]["court"]["source"] == "observed"
    assert "metric_note" in summary
