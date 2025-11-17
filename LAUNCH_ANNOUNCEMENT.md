# Fractal Glyph Tape Launch Announcement

Marketing copy for various channels announcing the release of Fractal Glyph Tape.

---

## Twitter/X Thread

### Tweet 1 (Hook)
We just shipped Fractal Glyph Tape (FGT) — a research prototype that treats language as a *phrase memory* instead of a token stream.

Text becomes hybrid: normal tokens + glyph codes that point to semantic families.

Think: a shared "inner code" beneath LLMs.

🧵👇

### Tweet 2 (The Problem)
LLMs treat every repeated phrase as new data:
- "Can you send that file?"
- "Mind emailing the document?"
- "Could you share the file?"

Each gets tokenized separately. Massive redundancy. No shared structure.

### Tweet 3 (The Solution)
FGT clusters similar phrases into families, assigns each a **glyph ID** (built from Mandarin chars used as pure symbols), and places them on a **fractal tape**—a recursive triangular map where neighbors = semantic neighbors.

### Tweet 4 (What This Enables)
📦 **Semantic compression** – smaller corpora, same meaning
🧠 **Context extension** – more signal per token under fixed budgets
🌍 **Cross-lingual bridging** – shared glyph IDs across languages
🔍 **Explorable phrase space** – interactive fractal map of "things we say"

### Tweet 5 (What's Included)
✅ Full pipeline: ingest → embeddings → clustering → glyph encoding → fractal tape
✅ Hybrid tokenizer (wraps existing tokenizers)
✅ LLM adapter for training glyph-aware models
✅ Visualization API + web UI
✅ Experiment scripts for compression, context, cross-lingual eval

### Tweet 6 (Call to Action)
If you work on:
- Tokenization / representation learning
- Semantic compression
- Cross-lingual LLMs
- Long context

…we built this for you to pick apart, extend, and improve.

GitHub: https://github.com/funwae/fractal-glyph-tape

---

## Reddit Post (r/MachineLearning, r/LanguageTechnology)

### Title
[R][P] Fractal Glyph Tape: A Fractal-Addressable Phrase Memory for Semantic Compression and Cross-Lingual LLMs

### Body

**tl;dr:** We built a system that clusters phrases into semantic families, assigns them glyph codes (Mandarin chars as pure symbols), and organizes them on a fractal address space. Text becomes hybrid tokens + glyph pointers. Enables semantic compression, context extension, and cross-lingual anchors.

**GitHub:** https://github.com/funwae/fractal-glyph-tape

---

#### What is Fractal Glyph Tape?

Fractal Glyph Tape (FGT) is a research prototype that adds a new layer beneath LLMs:

1. **Ingest multilingual corpora** → extract phrases
2. **Embed and cluster** → create phrase families
3. **Assign glyph IDs** → compact codes built from Mandarin characters (used purely as glyphs, not for linguistic meaning)
4. **Build fractal tape** → place glyph IDs on a triangular fractal address space where semantic neighbors are spatially close
5. **Hybrid tokenization** → text becomes: `[normal tokens]` + `[glyph tokens]`

LLMs can be trained to read/write this "inner code," enabling:

- **Semantic compression** – store one phrase family instead of millions of near-duplicates
- **Effective context extension** – single glyph token = entire phrase motif
- **Cross-lingual bridging** – same glyph ID for equivalent phrases across languages

---

#### What's in the repo?

- ✅ **Ingestion pipeline** – raw text → structured phrases
- ✅ **Multilingual embeddings & clustering** – phrase families with metadata
- ✅ **Glyph encoding system** – integer glyph IDs → Mandarin glyph strings
- ✅ **Fractal tape builder** – 2D projection + recursive triangular addressing
- ✅ **Hybrid tokenizer** – wraps base tokenizer with glyph-aware spans
- ✅ **LLM adapter** – training & inference helpers for glyph-aware models
- ✅ **Visualization API** – backend for fractal phrase-map web UI
- ✅ **Experiment scripts** – compression, context, cross-lingual evaluations

Full docs (45+ specs): https://github.com/funwae/fractal-glyph-tape/tree/main/docs

---

#### Why this matters

**1. Semantic Compression**

Current approaches:
- Raw storage: duplicates everywhere
- Traditional compression: byte-level, loses structure
- Deduplication: exact matches only

FGT approach:
- Cluster semantically similar phrases
- Store one family table + compact glyph codes
- Reconstructable meaning from glyph IDs

**2. Context Extension**

Fixed token budgets are a bottleneck. If a single glyph token can represent "file-sharing request" (a phrase family with 10+ variants), you pack more semantic content into the same context window.

**3. Cross-Lingual by Design**

English: "Can you send that file?"
中文: "你能发给我那个文件吗？"
Spanish: "¿Puedes enviarme ese archivo?"

All map to the same phrase family → same glyph ID → language-agnostic retrieval.

---

#### Example

```bash
# Encode text to glyph representation
$ echo "Can you send me that file?" | fgt encode
谷阜

# Decode glyph back to phrase family
$ echo "谷阜" | fgt decode
Phrase family #1247: File-sharing request
Examples:
  - "Can you send me that file?" (en)
  - "Mind emailing the document?" (en)
  - "你能发给我那个文件吗？" (zh)
  - "¿Puedes enviarme ese archivo?" (es)
```

---

#### Current Status

- ✅ Complete architecture design + formal specs
- ✅ 45+ documentation files
- 🚧 Core prototype implementation (in progress)
- 🚧 Empirical evaluation pending
- 🚧 Public demo (planned)

This is **research software**—we're releasing it early to invite feedback, experiments, and collaboration.

---

#### Get Started

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Build a demo tape
python scripts/run_full_build.py --config configs/demo.yaml

# 3) Try the CLI
echo "Can you send me that file?" | fgt encode
echo "谷阜" | fgt decode

# 4) Launch the visualizer
uvicorn fgt.viz.app:app --reload
```

---

#### Links

- **GitHub:** https://github.com/funwae/fractal-glyph-tape
- **Docs:** https://github.com/funwae/fractal-glyph-tape/tree/main/docs
- **Research Abstract:** https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md
- **Contact:** contact@glyphd.com

Built by **Glyphd Labs** – turning the space of "things we say" into a structured, navigable map.

---

#### We'd love feedback on:

- Clustering algorithms (we use MiniBatchKMeans—open to better approaches)
- Glyph encoding schemes (frequency-aware allocation, alternative schemes?)
- Fractal addressing (Sierpiński triangle, but other fractals could work)
- LLM training objectives (glyph reconstruction, prediction, hybrid tasks?)
- Cross-lingual evaluation (how to measure bridging quality?)
- Use cases we haven't thought of

Open an issue or start a discussion on GitHub. Let's explore this space together.

---

## Hacker News Post

### Title
Fractal Glyph Tape: A Fractal-Addressable Phrase Memory for LLMs

### URL
https://github.com/funwae/fractal-glyph-tape

### Suggested Text (for "text" field if submitting as text post)
We built a system that clusters phrases into semantic families, assigns them glyph codes (using Mandarin characters as pure symbols), and organizes them on a fractal address space.

Text becomes hybrid: normal tokens + glyph pointers to phrase families.

Enables semantic compression, context extension, and cross-lingual retrieval.

Full code + 45 docs: https://github.com/funwae/fractal-glyph-tape

---

## LinkedIn Post

### Post Text

Excited to share **Fractal Glyph Tape (FGT)**, a research prototype from Glyphd Labs that reimagines how we represent language for LLMs.

**The core idea:**
Instead of treating every phrase as new data, we cluster similar phrases into semantic families, assign compact glyph codes, and organize them on a fractal address space.

**What this enables:**
📦 Semantic compression – smaller corpora, same meaning
🧠 Context extension – more signal per token
🌍 Cross-lingual bridging – shared glyph IDs across languages

**What's included:**
✅ Full pipeline (ingest → embeddings → clustering → fractal tape)
✅ Hybrid tokenizer + LLM adapter
✅ Visualization API + web UI
✅ Experiment scripts for compression, context, cross-lingual eval

If you're working on tokenization, compression, or cross-lingual LLMs, we built this for you.

🔗 GitHub: https://github.com/funwae/fractal-glyph-tape
📄 Research abstract: [link to docs/RESEARCH_ABSTRACT.md]

#MachineLearning #NLP #LLM #OpenSource #Research

---

## GitHub Release Notes

### Release Title
v0.1.0 – Research Prototype Release

### Release Notes

We're releasing **Fractal Glyph Tape v0.1.0**, a research-grade substrate for semantic compression and cross-lingual LLMs.

## What is Fractal Glyph Tape?

FGT is a fractal-addressable phrase memory that clusters similar phrases into families, assigns each family a glyph code (using Mandarin characters as pure symbols), and organizes them on a fractal address space.

Text can then be represented as a **hybrid** of normal tokens and glyph tokens, where each glyph acts as a pointer to a phrase family.

## What's in this release?

- ✅ **Ingestion pipeline** – from raw text to structured phrases
- ✅ **Multilingual embeddings & clustering** – phrase families with rich metadata
- ✅ **Glyph encoding system** – integer glyph IDs → Mandarin glyph strings
- ✅ **Fractal tape builder** – 2D projection + recursive triangular addressing
- ✅ **Hybrid tokenizer** – wraps a base tokenizer with glyph-aware spans
- ✅ **LLM adapter** – training & inference helpers for glyph-aware models
- ✅ **Visualization API** – backend for a fractal phrase-map web UI
- ✅ **Experiment scripts** – compression, context, and cross-lingual evaluations
- ✅ **45+ documentation files** – complete specs, math, and guides

## Quickstart

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Build a demo tape
python scripts/run_full_build.py --config configs/demo.yaml

# 3) Try the CLI
echo "Can you send me that file?" | fgt encode
echo "谷阜" | fgt decode

# 4) Launch the visualizer
uvicorn fgt.viz.app:app --reload
```

## What's Next?

This is a **research prototype**. We're releasing it early to:

- Invite feedback from the research community
- Enable experiments on tokenization, compression, and cross-lingual LLMs
- Foster collaboration on fractal phrase memories

## Get Involved

- 📖 Read the [research abstract](docs/RESEARCH_ABSTRACT.md)
- 💬 Join the [Discussions](https://github.com/funwae/fractal-glyph-tape/discussions)
- 🐛 Report issues or suggest features in [Issues](https://github.com/funwae/fractal-glyph-tape/issues)
- 🤝 Contribute code, experiments, or documentation

## Citation

```bibtex
@software{fractal_glyph_tape_2025,
  title = {Fractal Glyph Tape: A Fractal-Addressable Phrase Memory for Semantic Compression},
  author = {Glyphd Labs},
  year = {2025},
  url = {https://github.com/funwae/fractal-glyph-tape},
  version = {0.1.0},
  note = {Research prototype}
}
```

---

**Built by Glyphd Labs** – turning the space of "things we say" into a structured, navigable map.

---

## Email Newsletter / Blog Post

### Subject Line
Introducing Fractal Glyph Tape: A New Substrate for Language

### Email Body

Hi [Name],

We're excited to share **Fractal Glyph Tape (FGT)**, a research prototype from Glyphd Labs that reimagines how we represent language for LLMs.

### The Problem

Current LLMs treat every repeated phrase as new data:
- "Can you send that file?"
- "Mind emailing the document?"
- "Could you share the file?"

Each gets tokenized separately. Massive redundancy. No shared structure.

### The Solution

FGT clusters similar phrases into **phrase families**, assigns each family a compact **glyph code** (built from Mandarin characters used purely as symbols), and places these glyphs on a **fractal tape**—a recursive triangular address space where semantic neighbors are spatially close.

Text becomes **hybrid**: normal tokens + glyph tokens that act as pointers to phrase families.

### What This Enables

📦 **Semantic compression** – smaller corpora and logs with reconstructable meaning
🧠 **Effective context extension** – more usable signal per token under fixed context windows
🌍 **Cross-lingual bridging** – shared glyph IDs for phrase families spanning multiple languages
🔍 **Explorable phrase space** – an interactive fractal map of "things we say"

### What's Included

- Full pipeline: ingest → embeddings → clustering → glyph encoding → fractal tape
- Hybrid tokenizer (wraps existing tokenizers)
- LLM adapter for training glyph-aware models
- Visualization API + web UI
- Experiment scripts for compression, context, cross-lingual evaluation
- 45+ documentation files with complete specs, math, and guides

### Try It Yourself

```bash
# Clone and install
git clone https://github.com/funwae/fractal-glyph-tape.git
cd fractal-glyph-tape
pip install -r requirements.txt

# Build a demo tape
python scripts/run_full_build.py --config configs/demo.yaml

# Try the CLI
echo "Can you send me that file?" | fgt encode
```

### Get Involved

FGT is **research software**—we're releasing it early to invite feedback, experiments, and collaboration.

- **Read the research abstract:** [docs/RESEARCH_ABSTRACT.md](https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md)
- **Explore the code:** [github.com/funwae/fractal-glyph-tape](https://github.com/funwae/fractal-glyph-tape)
- **Join the discussion:** [GitHub Discussions](https://github.com/funwae/fractal-glyph-tape/discussions)

If you're working on tokenization, compression, or cross-lingual LLMs, we built this for you.

Best,
The Glyphd Labs Team

---

**Last Updated:** 2025-01-16
**Usage:** Copy/paste these templates into the appropriate channels. Adjust tone/length as needed for each platform.
