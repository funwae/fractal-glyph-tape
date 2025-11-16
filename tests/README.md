# Tests

This directory contains the test suite for Fractal Glyph Tape.

## Running Tests

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

## Test Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_*.py` - Test modules for each component
- `README.md` - This file

## Writing Tests

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on writing tests.

### Test Organization

Each test module should:
- Test a single component or module
- Use descriptive test names
- Include docstrings
- Use fixtures from `conftest.py`
- Aim for >80% coverage

### Example Test

```python
import pytest
from fgt.glyph import GlyphManager


class TestGlyphManager:
    """Test suite for GlyphManager."""

    @pytest.fixture
    def glyph_manager(self, sample_alphabet):
        """Create a GlyphManager instance."""
        return GlyphManager(sample_alphabet)

    def test_encode_decode_roundtrip(self, glyph_manager):
        """Test encoding and decoding are inverses."""
        for i in range(100):
            encoded = glyph_manager.encode_glyph_id(i)
            decoded = glyph_manager.decode_glyph_string(encoded)
            assert decoded == i
```

## Current Status

**Phase 0**: Test structure created, placeholder tests added.
Implementation tests will be added as modules are developed.
