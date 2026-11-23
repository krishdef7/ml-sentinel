# Contributing

Issues and pull requests are welcome.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[dev,experiments]"
pytest tests/
```

## Before opening a PR

- `black src/ tests/ experiments/` (line length 100) and `flake8 src/ tests/` should pass; CI runs both
  plus the tests on Python 3.10-3.12.
- Add or update tests for library changes.
- Any number quoted in the README must come from a committed results file written by a script in
  `experiments/`. If a change alters results, rerun the benchmark and commit the new CSV and summary
  together with the code.

## License

Contributions are licensed under the MIT License.
