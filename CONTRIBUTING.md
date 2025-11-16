# Contributing to Fractal Glyph Tape

Thank you for your interest in contributing to Fractal Glyph Tape! This project aims to advance semantic compression and cross-lingual understanding through innovative phrase memory systems.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Adding Experiments](#adding-experiments)
- [Documentation](#documentation)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community and project
- Show empathy towards other community members

### Unacceptable Behavior

- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- CUDA-capable GPU (recommended, but CPU fallback available)
- 16GB+ RAM

### Setting Up Development Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/fractal-glyph-tape.git
   cd fractal-glyph-tape
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

4. **Install development dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

5. **Run tests to verify setup**

   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names following this pattern:

- `feature/description` - For new features
- `fix/description` - For bug fixes
- `docs/description` - For documentation updates
- `experiment/description` - For new experiments or research

Examples:
- `feature/hybrid-tokenizer`
- `fix/clustering-memory-leak`
- `docs/api-reference`
- `experiment/multilingual-clustering`

### Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")
- Reference issues: Fixes #123, See #456
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use type hints for function signatures
- Maximum line length: 100 characters (we're flexible on this)
- Use docstrings for all public modules, functions, classes, and methods

### Formatting

We use `black` for code formatting:

```bash
black src/ scripts/ tests/
```

Before committing, run:

```bash
# Format code
black src/ scripts/ tests/

# Check for issues
flake8 src/ scripts/ tests/

# Type checking (optional but recommended)
mypy src/
```

### Example Code Style

```python
"""Module docstring explaining purpose."""

from typing import List, Optional
import numpy as np


class PhraseClusterer:
    """Clusters phrases into semantic families.

    Args:
        n_clusters: Number of clusters to create
        batch_size: Batch size for MiniBatchKMeans
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        n_clusters: int,
        batch_size: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, embeddings: np.ndarray) -> "PhraseClusterer":
        """Fit the clusterer to embeddings.

        Args:
            embeddings: Array of shape (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        # Implementation here
        return self
```

## Testing

### Writing Tests

- Place tests in `tests/` directory mirroring `src/` structure
- Use `pytest` for all tests
- Aim for >80% code coverage on new code

### Test Structure

```python
import pytest
from fgt.glyph import GlyphManager


class TestGlyphManager:
    """Test suite for GlyphManager."""

    @pytest.fixture
    def glyph_manager(self):
        """Create a GlyphManager instance for testing."""
        alphabet = ["谷", "阜", "堯", "奚"]
        return GlyphManager(alphabet)

    def test_encode_decode_roundtrip(self, glyph_manager):
        """Test that encoding and decoding are inverses."""
        for i in range(100):
            encoded = glyph_manager.encode_glyph_id(i)
            decoded = glyph_manager.decode_glyph_string(encoded)
            assert decoded == i

    def test_glyph_uniqueness(self, glyph_manager):
        """Test that different IDs produce different glyphs."""
        glyphs = [glyph_manager.encode_glyph_id(i) for i in range(1000)]
        assert len(glyphs) == len(set(glyphs))
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_glyph_manager.py -v

# Run specific test
pytest tests/test_glyph_manager.py::TestGlyphManager::test_encode_decode_roundtrip -v
```

## Pull Request Process

1. **Create a new branch** from `main`

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following code style guidelines

3. **Add tests** for new functionality

4. **Run tests locally** to ensure they pass

   ```bash
   pytest tests/ -v
   black src/ scripts/ tests/
   flake8 src/ scripts/ tests/
   ```

5. **Update documentation** if needed
   - Update docstrings
   - Update relevant `.md` files in `docs/`
   - Update README.md if adding major features

6. **Commit your changes** with clear messages

7. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request** on GitHub
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what the PR does and why
   - Include screenshots/plots if relevant

9. **Respond to review feedback**
   - Address comments promptly
   - Make requested changes
   - Mark conversations as resolved when done

10. **Wait for approval** from maintainers

### PR Title Format

- `[Feature] Add hybrid tokenizer wrapper`
- `[Fix] Resolve memory leak in clustering`
- `[Docs] Update installation instructions`
- `[Experiment] Add multilingual clustering tests`

## Adding Experiments

Experiments are a core part of FGT research. To add a new experiment:

1. **Create experiment config** in `configs/experiments/`

   ```yaml
   # configs/experiments/my_experiment.yaml
   experiment_name: "my_experiment_v1"
   description: "Testing hypothesis X"

   # ... configuration parameters
   ```

2. **Write experiment script** in `scripts/experiments/`

   ```python
   # scripts/experiments/run_my_experiment.py
   """Experiment to test hypothesis X."""

   def run_experiment(config):
       # Implementation
       pass
   ```

3. **Document the experiment** in `docs/6x-*.md` following existing format

4. **Add results** to experiment report

5. **Open PR** with experiment code, config, and documentation

## Documentation

### Documentation Standards

- Keep documentation in sync with code
- Use clear, concise language
- Include examples where helpful
- Update docs in the same PR as code changes

### Documentation Structure

- `README.md` - Project overview and quick start
- `docs/0x-*.md` - Vision and conceptual docs
- `docs/1x-*.md` - Theoretical specifications
- `docs/2x-*.md` - Mathematical and formal specs
- `docs/3x-*.md` - Architecture documentation
- `docs/4x-*.md` - Implementation guides
- `docs/5x-*.md` - Training and pipelines
- `docs/6x-*.md` - Evaluation and experiments
- `docs/7x-*.md` - Demos and integration
- `docs/9x-*.md` - Research and roadmap

### Adding New Documentation

1. Determine the appropriate category (0x, 1x, 2x, etc.)
2. Create file with sequential numbering
3. Follow existing document structure
4. Update `docs/README.md` index
5. Link from other relevant docs

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/funwae/fractal-glyph-tape/discussions)
- **Bugs**: Open a [GitHub Issue](https://github.com/funwae/fractal-glyph-tape/issues)
- **Security**: Email contact@glyphd.com directly

## Recognition

Contributors will be:
- Listed in project acknowledgments
- Credited in any academic publications using their contributions
- Invited to co-author papers for significant research contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Fractal Glyph Tape! Your work helps advance semantic compression and cross-lingual understanding for the entire NLP community.
