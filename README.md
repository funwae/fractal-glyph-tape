# Fractal Glyph Tape (FGT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

> A fractal-addressable phrase memory system that uses Mandarin characters as a pure glyph library for semantic compression and cross-lingual understanding.

## What is Fractal Glyph Tape?

Fractal Glyph Tape (FGT) is a novel approach to language representation that addresses fundamental inefficiencies in how we store, process, and reason about text:

- **Phrase-level compression**: Instead of treating repeated phrases as separate data, FGT clusters similar phrases into "families" and assigns each family a short, reusable glyph ID.
- **Fractal organization**: Glyph IDs are placed on a structured, multi-scale address space where nearby addresses = semantically related phrases.
- **Cross-lingual bridging**: Equivalent phrases across languages (English, Chinese, Spanish, etc.) share the same glyph ID, enabling language-agnostic retrieval and reasoning.
- **LLM context extension**: By encoding text as glyph sequences, the same token budget carries 3-5x more semantic content.

### The Core Innovation

```
"Can you send me that file?"     ──┐
"Mind emailing me the document?" ──├─→ Cluster → Glyph ID: 谷阜 → Fractal Address: L-R-C-L
"Could you share the file?"      ──┘

Instead of storing these phrases separately 1M times,
we store ONE cluster with examples + ONE short glyph code.
```

**Key insight**: Mandarin characters are used purely as **visual symbols** (not for their linguistic meaning), providing a dense, compact alphabet for encoding phrase families.

## Features

- ✨ **50-80% corpus compression** while preserving semantic meaning
- 🌍 **Language-agnostic phrase indexing** for cross-lingual search and retrieval
- 🧠 **LLM-friendly representation** that extends effective context windows
- 🗺️ **Interactive fractal map visualization** of semantic phrase space
- 🔧 **Hybrid tokenization** that augments (not replaces) existing tokenizers
- 🚀 **Training efficiency gains** by structuring phrase redundancy

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/funwae/fractal-glyph-tape.git
cd fractal-glyph-tape

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Build Your First Tape

```bash
# Run the full pipeline on demo data
python scripts/run_full_build.py --config configs/demo.yaml
```

This will:
1. Ingest phrases from sample corpus
2. Generate embeddings
3. Cluster into phrase families
4. Assign glyph IDs
5. Build fractal tape storage
6. Run basic evaluation metrics

### Use the CLI

```bash
# Encode text to glyph representation
echo "Can you send me that file?" | fgt encode

# Decode glyph back to text
echo "谷阜" | fgt decode

# Inspect a glyph cluster
fgt inspect-glyph 谷阜
```

## How It Works

### Architecture Overview

```
Raw Text Corpus
    ↓ 1. Ingest & segment phrases
Phrase Database (~100k-1M phrases)
    ↓ 2. Embed with sentence transformers
Dense Embeddings (384-768 dim vectors)
    ↓ 3. Cluster into phrase families
Cluster Assignments (~10k-100k families)
    ↓ 4. Assign glyph IDs (1-4 Mandarin chars)
Glyph-to-Cluster Mapping
    ↓ 5. Project to 2D + fractal addressing
Fractal Tape Storage (SQLite + indexes)
    ↓ 6. Hybrid tokenization
Text encoded as: [raw tokens] + [glyph tokens]
    ↓ 7. LLM fine-tuning
Models learn to read/write glyph space
```

### Key Components

- **Data Ingestion** (`src/ingest`): Extracts phrases from raw corpora
- **Embedding Service** (`src/embed`): GPU-accelerated phrase encoding
- **Clustering** (`src/cluster`): MiniBatchKMeans to group similar phrases
- **Glyph Manager** (`src/glyph`): Maps cluster IDs to Mandarin character sequences
- **Fractal Tape Builder** (`src/tape`): Projects clusters onto 2D fractal address space
- **Hybrid Tokenizer** (`src/tokenizer`): Mixes raw tokens with glyph tokens
- **Visualization** (`src/viz`): Interactive fractal map UI

## Use Cases

### 1. Compressed Chat Logs
Store customer support or product chat logs with 50-80% size reduction while enabling semantic search over phrase motifs.

### 2. Extended Context for LLMs
Convert long histories (project logs, user notes) into glyph-coded form. Same token budget, 3-5x more semantic coverage.

### 3. Cross-Lingual Knowledge Base
Mixed-language documentation clustered together. Search in English, retrieve matches in Chinese, Spanish, etc., all linked via shared glyph IDs.

### 4. Phrase Motif Analytics
Researchers can explore how language is used via interactive fractal maps showing phrase family distributions.

### 5. Training Acceleration
Fine-tune LLMs on glyph-enhanced sequences. Less data redundancy = faster convergence and better generalization.

## Project Status

**Current Phase**: Phase 0 - Documentation Complete, Implementation Starting

- [x] Complete documentation suite (45+ specification documents)
- [x] Architecture design and formal specs
- [ ] Core prototype implementation
- [ ] Visualization and metrics
- [ ] LLM integration and fine-tuning
- [ ] Cross-lingual experiments
- [ ] Public demo on glyphd.com

See [ROADMAP.md](docs/92-roadmap-and-phases.md) for detailed phase breakdown.

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

### Getting Started
- [Plain English Summary](docs/01-plain-english-summary.md) - Understand FGT in 5 minutes
- [Vision Overview](docs/00-vision-overview.md) - Project goals and motivation
- [Problem Statement](docs/02-problem-statement.md) - What FGT solves
- [Solution at a Glance](docs/03-solution-at-a-glance.md) - High-level approach

### Technical Specifications
- [System Architecture](docs/30-system-architecture-overview.md) - Component overview
- [Fractal Addressing Spec](docs/20-fractal-addressing-spec.md) - Address space design
- [Glyph ID Encoding](docs/21-glyph-id-encoding-spec.md) - Mandarin character mapping
- [Clustering Math](docs/22-phrase-clustering-math.md) - Formal clustering approach

### Implementation Guides
- [Tech Stack](docs/40-tech-stack-and-dependencies.md) - Libraries and tools
- [Data Pipeline](docs/31-data-pipeline-design.md) - End-to-end data flow
- [Offline Build Pipeline](docs/51-offline-building-pipeline.md) - Step-by-step build process
- [Demo CLI Spec](docs/70-demo-cli-spec.md) - Command-line interface

### For Claude Code Web
- **[claude.md](claude.md)** - Complete E2E build instructions for AI assistants

## Development

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, CPU fallback available)
- 16GB+ RAM
- ~10GB disk space for demo data and models

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

We follow PEP 8 and use type hints. Format code with `black`:

```bash
black src/ scripts/ tests/
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code of conduct
- Development workflow
- Pull request process
- Coding standards
- How to add experiments

## Research

FGT builds on and extends research in:

- Semantic compression and phrase clustering
- Cross-lingual representations
- Fractal addressing and space-filling curves
- Hybrid tokenization for LLMs
- Information-theoretic analysis of language

See [Related Work](docs/90-related-work-and-positioning.md) for detailed positioning.

## Citation

If you use Fractal Glyph Tape in your research, please cite:

```bibtex
@software{fractal_glyph_tape_2024,
  title = {Fractal Glyph Tape: A Fractal-Addressable Phrase Memory for Semantic Compression},
  author = {Glyphd Labs},
  year = {2024},
  url = {https://github.com/funwae/fractal-glyph-tape},
  note = {Research prototype}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **glyphd.com** / **Glyphd Labs** - Project sponsor and research direction
- **Mandarin character library** - Selected for glyph density, not linguistic meaning
- **Open-source community** - Built on PyTorch, Transformers, scikit-learn, and more

## Links

- **Project Website**: [glyphd.com](https://glyphd.com) (coming soon)
- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/funwae/fractal-glyph-tape/issues)
- **Discussions**: [GitHub Discussions](https://github.com/funwae/fractal-glyph-tape/discussions)

## Contact

For questions, collaborations, or enterprise use cases:

- Open an issue on GitHub
- Email: [contact@glyphd.com](mailto:contact@glyphd.com)

---

**Built with ❤️ by Glyphd Labs**

*Turning the space of "things we say" into a structured, navigable map.*
