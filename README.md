# Fractal Glyph Tape (FGT)

> A fractal-addressable phrase memory that makes language **denser**, **more cross-lingual**, and **more explorable** for LLMs.

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

Fractal Glyph Tape (FGT) is a research prototype from **Glyphd Labs** that adds a new layer beneath large language models:

- It clusters similar sentences and phrases into **phrase families**.
- It assigns each family a compact **glyph ID** built from Mandarin characters used purely as **glyphs** (no native semantics).
- It places these glyph IDs onto a **fractal tape**—a recursive triangular address space that preserves semantic neighborhoods.

Text can then be represented as a **hybrid** of:

- normal tokens, and
- **glyph tokens** that act as pointers into a shared phrase memory.

LLMs can be trained or adapted to **read and write this inner code**, enabling:

- 📦 **Semantic compression** – smaller corpora and logs with reconstructable meaning
- 🧠 **Effective context extension** – more usable signal per token under fixed context windows
- 🌍 **Cross-lingual bridging** – shared glyph IDs for phrase families spanning multiple languages
- 🔍 **Explorable phrase space** – an interactive fractal map of "things we say"

## What's in this repo?

- ✅ **Ingestion pipeline** – from raw text to structured phrases
- ✅ **Multilingual embeddings & clustering** – phrase families with rich metadata
- ✅ **Glyph encoding system** – integer glyph IDs → Mandarin glyph strings
- ✅ **Fractal tape builder** – 2D projection + recursive triangular addressing
- ✅ **Hybrid tokenizer** – wraps a base tokenizer with glyph-aware spans
- ✅ **LLM adapter** – training & inference helpers for glyph-aware models
- ✅ **Visualization API** – backend for a fractal phrase-map web UI
- ✅ **Experiment scripts** – compression, context, and cross-lingual evaluations
- ✅ **Phase 4: Agent Memory OS** – memory service with fractal-addressable storage
  - FastAPI backend with write/read/regions/addresses endpoints
  - Memory Console UI for interactive agent memory exploration
  - Policy-based foveation for efficient context retrieval
  - SQLite persistence with fractal addressing

See [`docs/README.md`](docs/README.md) for the full technical spec.

## Features

- ✨ **50-80% corpus compression** while preserving semantic meaning
- 🌍 **Language-agnostic phrase indexing** for cross-lingual search and retrieval
- 🧠 **LLM-friendly representation** that extends effective context windows
- 🗺️ **Interactive fractal map visualization** of semantic phrase space
- 🔧 **Hybrid tokenization** that augments (not replaces) existing tokenizers
- 🚀 **Training efficiency gains** by structuring phrase redundancy

## Quickstart

### Phase 4: Memory Service Quick Start

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Start the Memory Service
python scripts/run_memory_server.py --host 0.0.0.0 --port 8001

# 3) Start the Web UI (in another terminal)
cd web
npm install
npm run dev

# 4) Open the Memory Console
# Navigate to: http://localhost:3000/memory-console
```

### Classic FGT Pipeline

```bash
# Build a demo tape
python scripts/run_full_build.py --config configs/demo.yaml

# Try the CLI
echo "Can you send me that file?" | fgt encode
echo "谷阜" | fgt decode

# Launch the visualizer
uvicorn fgt.viz.app:app --reload
```

---

FGT is **research software**: we invite feedback, experiments, and extensions.
If you're working on tokenization, compression, or cross-lingual LLMs, this is for you.

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

**Current Phase**: Phase 4 - Agent Memory OS Complete

- [x] Complete documentation suite (45+ specification documents)
- [x] Architecture design and formal specs
- [x] Core prototype implementation (Phases 1-3)
- [x] Phase 4: Agent Memory OS & API
  - Memory service with fractal addressing
  - FastAPI backend with full CRUD operations
  - Memory Console UI with chat, context, timeline, glyphs, and address inspector
  - Unit tests for models, store, and policy
  - SQLite and in-memory storage implementations
- [ ] Phase 5: Advanced LLM integration and fine-tuning
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

### Phase 4: Agent Memory OS
- [Phase 4 Overview](docs/210-phase-4-agent-memory-and-api.md) - Complete FGMS architecture and API
- [Evaluation & Testing](docs/211-agent-memory-eval-and-test-plan.md) - Test plans and benchmarks
- [Memory Console UI](docs/213-agent-memory-console-ui.md) - Web interface documentation

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

This project is licensed under a Proprietary License - see the [LICENSE](LICENSE) file for details. **Non-commercial use only. Commercial use requires explicit written permission.**

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
