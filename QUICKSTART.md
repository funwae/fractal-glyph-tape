# Quick Start Guide

Get Fractal Glyph Tape up and running in minutes.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/funwae/fractal-glyph-tape.git
cd fractal-glyph-tape
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FGT in editable mode
pip install -e .
```

### 3. Verify Installation

```bash
# Check if CLI is available
fgt version

# Should output: Fractal Glyph Tape v0.1.0
```

## First Build (Phase 0)

### Prepare Sample Data

Create a simple text file with sample phrases:

```bash
mkdir -p data/raw
cat > data/raw/sample.txt << 'EOF'
Can you send me that file?
Could you email me the document?
Please share the file with me.
Would you mind sending that over?
Hello, how are you today?
Hi there, how have you been?
Good morning! How are things?
I hope you're doing well.
Thank you so much for your help.
Thanks a lot, really appreciate it.
I'm grateful for your assistance.
EOF
```

### Run the Build Pipeline

```bash
# Run full build (once implemented)
fgt build --config configs/demo.yaml
```

**Note**: Core implementation is not yet complete. See [claude.md](claude.md) for detailed build instructions to implement the system.

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Code Formatting

```bash
# Format code
black src/ tests/ scripts/

# Check linting
flake8 src/ tests/ scripts/
```

### Adding a New Module

1. Create module file in appropriate `src/` subdirectory
2. Add imports to `src/<module>/__init__.py`
3. Write tests in `tests/test_<module>.py`
4. Update documentation if needed

## Next Steps

### For Users

- Read [Plain English Summary](docs/01-plain-english-summary.md) to understand FGT
- Explore the [documentation](docs/) for detailed concepts
- Try out experiments once Phase 0 is complete

### For Developers

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Check [claude.md](claude.md) for complete implementation instructions
- Start with Phase 0 implementation:
  1. Data ingestion ([docs/41-data-ingestion-implementation.md](docs/41-data-ingestion-implementation.md))
  2. Embedding ([docs/42-embedding-and-clustering-impl.md](docs/42-embedding-and-clustering-impl.md))
  3. Clustering
  4. Glyph assignment
  5. Fractal tape builder

### For Researchers

- Review [experimental design](docs/60-eval-metrics-overview.md)
- Propose new experiments via GitHub issues
- Check [roadmap](docs/92-roadmap-and-phases.md) for upcoming features

## Common Issues

### GPU Not Detected

If CUDA is not available, edit `configs/demo.yaml`:

```yaml
embed:
  device: "cpu"  # Change from "cuda"
```

### Import Errors

Make sure you installed the package:

```bash
pip install -e .
```

### Missing Dependencies

Reinstall requirements:

```bash
pip install -r requirements.txt --upgrade
```

## Getting Help

- **Documentation**: Browse [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/funwae/fractal-glyph-tape/issues)
- **Discussions**: [GitHub Discussions](https://github.com/funwae/fractal-glyph-tape/discussions)
- **Email**: contact@glyphd.com

## What's Next?

Once Phase 0 is implemented, you'll be able to:

1. **Build a tape**: Process 100k phrases into a fractal glyph tape
2. **Encode text**: Convert phrases to glyph-coded representations
3. **Decode glyphs**: Reconstruct text from glyph codes
4. **Visualize**: Explore the fractal map of phrase space
5. **Measure compression**: See 2-5x compression ratios

**Current Status**: Documentation complete, implementation starting.

Follow the [claude.md](claude.md) guide to build the system end-to-end!
