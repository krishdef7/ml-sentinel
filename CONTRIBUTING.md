# Contributing to ML Sentinel

Thank you for your interest in contributing to ML Sentinel! This document provides guidelines for contributing.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ml-sentinel.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Install dev dependencies: `pip install -e ".[dev]"`
5. Make your changes
6. Run tests: `pytest tests/`
7. Submit a pull request

## 📋 Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,experiments]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Format code
black src/ tests/ experiments/

# Check linting
flake8 src/ tests/
```

## 🧪 Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=ml_sentinel --cov-report=html

# Specific test file
pytest tests/test_metrics.py -v
```

## 📝 Code Style

We use:
- **Black** for code formatting (line length 100)
- **Flake8** for linting
- **Type hints** for all public functions
- **Docstrings** in NumPy format

Example:

```python
def compute_correlation(
    scores: np.ndarray,
    degradation: np.ndarray,
    method: str = 'pearson'
) -> Tuple[float, float]:
    """
    Compute correlation between scores and degradation.
    
    Parameters
    ----------
    scores : np.ndarray
        Detection scores
    degradation : np.ndarray
        Actual accuracy degradation
    method : str
        'pearson' or 'spearman'
        
    Returns
    -------
    r : float
        Correlation coefficient
    p : float
        p-value
    """
```

## 🐛 Reporting Bugs

1. Check existing issues first
2. Use the bug report template
3. Include:
   - Python version
   - Package versions (`pip freeze`)
   - Minimal reproducible example
   - Expected vs actual behavior

## ✨ Feature Requests

1. Check if it's already requested
2. Use the feature request template
3. Explain the use case and benefit

## 📊 Adding New Experiments

When adding new benchmarks or experiments:

1. Follow the existing structure in `experiments/`
2. Use consistent column naming
3. Add documentation
4. Include expected results

## 📚 Documentation

- Update docstrings for any API changes
- Update README.md if needed
- Add examples for new features

## 🔄 Pull Request Process

1. Update documentation
2. Add/update tests
3. Run the full test suite
4. Update CHANGELOG.md
5. Request review

## 📜 Code of Conduct

Be respectful and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Questions? Open an issue or reach out to the maintainers.
