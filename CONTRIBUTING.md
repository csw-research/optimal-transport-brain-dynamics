# Contributing to Optimal Transport for Dynamic Brain Connectivity

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## Development Setup

### Prerequisites

- Python 3.10+
- Git

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/csw-research/optimal-transport-brain-dynamics.git
cd optimal-transport-brain-dynamics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"
```

## Code Style

We follow standard Python conventions:

- **PEP 8** for code style
- **Type hints** for all function signatures
- **Docstrings** in NumPy format for all public functions
- **Black** for code formatting (line length 100)

### Formatting and Linting

```bash
# Format code
black src/ tests/ examples/

# Check types
mypy src/

# Lint
ruff check src/ tests/
```

## Testing

We use pytest for testing. All new features must include tests.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ot_brain_dynamics --cov-report=html

# Run specific test file
pytest tests/test_spd_geometry.py -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Aim for >80% code coverage

Example:

```python
import pytest
from ot_brain_dynamics.optimal_transport import wasserstein_spd

def test_wasserstein_distance_positive():
    """Test Wasserstein distance is positive for different matrices."""
    C1 = ...
    C2 = ...
    dist = wasserstein_spd(C1, C2)
    assert dist > 0
```

## Documentation

### Docstring Format

Use NumPy-style docstrings:

```python
def my_function(param1: np.ndarray, param2: float = 1.0) -> np.ndarray:
    """
    Brief description of function.

    Longer description providing more details about what the function
    does and how it works.

    Parameters
    ----------
    param1 : np.ndarray, shape (n, m)
        Description of param1
    param2 : float, default=1.0
        Description of param2

    Returns
    -------
    result : np.ndarray, shape (n, m)
        Description of return value

    Raises
    ------
    ValueError
        When invalid input is provided

    Notes
    -----
    Additional mathematical or implementation notes.

    Examples
    --------
    >>> result = my_function(data, param2=2.0)
    >>> print(result.shape)
    (10, 20)

    References
    ----------
    Author, A. (2020). Paper title. Journal.
    """
    ...
```

### Building Documentation

```bash
cd docs/
make html
# Open docs/_build/html/index.html
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bugfix-name
```

### 2. Make Changes

- Write clear, focused commits
- Include tests for new functionality
- Update documentation as needed
- Follow code style guidelines

### 3. Run Quality Checks

```bash
# Format code
black src/ tests/

# Run tests
pytest

# Check types
mypy src/
```

### 4. Commit Changes

Write clear commit messages:

```
Add geodesic interpolation for SPD matrices

- Implement log-Euclidean geodesic computation
- Add tests for endpoint and symmetry properties
- Include examples in docstring
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Reference to related issues (if any)
- Checklist showing you've completed required steps

### PR Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Type hints included
- [ ] No breaking changes (or clearly documented)

## Issue Reporting

### Bug Reports

Include:
- Python version
- Package version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages/tracebacks

### Feature Requests

Include:
- Clear description of proposed feature
- Use case / motivation
- Proposed API (if applicable)
- References to relevant papers (if applicable)

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Accept criticism gracefully
- Prioritize community benefit

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks
- Publishing private information
- Other unethical or unprofessional conduct

## Mathematical Contributions

This project involves rigorous mathematics. When contributing mathematical methods:

1. **Provide clear mathematical description** in docstrings or documentation
2. **Include references** to papers/textbooks
3. **Explain intuition** not just formulas
4. **Verify correctness** with theoretical properties (e.g., symmetry, positive definiteness)
5. **Compare to baselines** when possible

## Performance Considerations

When optimizing code:

- Profile before optimizing
- Document complexity in docstrings
- Consider GPU acceleration for large-scale operations
- Test on realistic data sizes
- Maintain numerical stability

## Questions?

Feel free to:
- Open an issue for discussion
- Ask questions in pull requests
- Contact maintainers

Thank you for contributing! 🧠✨
