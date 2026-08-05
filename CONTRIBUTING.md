# Contributing

Thank you for improving Tennis Vision Lab.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

Run the integration demo before opening a pull request:

```bash
tennis-vision demo --output artifacts/smoke --frames 30 --no-video
```

## Change expectations

- Keep imports free of video I/O and GUI side effects.
- Add a focused test for bug fixes and decision-rule changes.
- Preserve the JSON contract or document a deliberate schema-version change.
- Put thresholds in `VisionConfig`; do not hide them in processing functions.
- Document whether reported metrics use synthetic, public, or user-provided data.
- Do not commit match footage, model weights, credentials, or personal data.

## Pull request checklist

- [ ] `pytest` passes.
- [ ] `ruff check .` passes.
- [ ] The synthetic demo completes.
- [ ] README or guides reflect user-facing behavior.
- [ ] New confidence values are described as heuristic unless calibrated.
- [ ] Dataset and license provenance are documented for any new data.
