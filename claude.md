# Glyphd Fractal Glyph Tape - Claude Code Build Guide

## Project Overview

**Fractal Glyph Tape (FGT)** is a revolutionary semantic compression and phrase memory system that uses Mandarin characters as a pure glyph library to create a fractal-addressable phrase map for language models.

### What Problem Does This Solve?

Modern language systems waste enormous resources:
- **Redundant storage**: The same phrase patterns are stored millions of times across corpora
- **Context limitations**: LLMs are bottlenecked by token-based context windows, not semantic capacity
- **Cross-lingual fragmentation**: No stable naming scheme connects equivalent phrases across languages
- **Training inefficiency**: Models re-learn identical patterns as separate examples

### The Core Innovation

FGT creates a **structured, compressed phrase memory** where:

1. **Phrase families** are clustered from large corpora (e.g., "Can you send me that file?" / "Mind emailing me the document?" / "Could you share the file?")
2. Each family gets a **short glyph ID** built from Mandarin characters (used as symbols, not Chinese text)
3. IDs are placed on a **fractal address space** where nearby addresses = semantically related phrases
4. LLMs can **read/write in glyph space** for massive compression and context extension

### Key Capabilities

- **Corpus compression**: 50-80% reduction while preserving semantic meaning
- **Context expansion**: Same token budget carries 3-5x more semantic content
- **Cross-lingual bridging**: Single glyph IDs connect English, Chinese, Spanish equivalents
- **Training acceleration**: Models learn phrase families as structured units, not redundant examples
- **Visual exploration**: Fractal map UI for navigating semantic phrase space

### How It Works (High-Level)

```
Raw Text Corpus
    ↓ Extract & segment phrases
Phrase Database (~100k-1M phrases)
    ↓ Embed with sentence transformers
Dense Embeddings (768-dim vectors)
    ↓ Cluster into phrase families
Cluster Assignments (~10k-100k families)
    ↓ Assign glyph IDs (1-4 Mandarin chars)
Glyph-to-Cluster Mapping
    ↓ Project to 2D + fractal addressing
Fractal Tape Storage
    ↓ Hybrid tokenization
Text encoded as: [raw tokens] + [glyph tokens]
    ↓ LLM fine-tuning
Models learn to read/write glyph space
```

---

## Budget and Resource Allocation

**Available Credits**: $250 for Claude Code web usage

**Recommended Allocation**:
- Phase 0 (Prototype): ~$100 - Core implementation and small demo
- Phase 1 (Metrics): ~$75 - Visualization and compression experiments
- Phase 2 (LLM Integration): ~$50 - Tokenizer and basic fine-tuning
- Phase 3 (Polish): ~$25 - Cross-lingual experiments and documentation

**Cost Optimization Tips**:
- Use Haiku model for file operations and routine tasks
- Use Sonnet for architecture design and complex implementation
- Batch related tasks together to minimize context switches
- Read documentation thoroughly before starting implementation

---

## Complete End-to-End Build Instructions

### Phase 0: Core System Prototype

**Goal**: Build working prototype with ~100k phrases, demonstrable compression, basic CLI

#### Step 0.1: Project Structure Setup

Create the following directory structure:

```
glyphd fractal glyph tape/
├── src/
│   ├── __init__.py
│   ├── ingest/          # Data ingestion
│   ├── embed/           # Phrase embedding
│   ├── cluster/         # Clustering service
│   ├── glyph/           # Glyph ID management
│   ├── tape/            # Fractal tape builder
│   ├── tokenizer/       # Hybrid tokenizer
│   ├── llm_adapter/     # LLM integration
│   ├── eval/            # Evaluation metrics
│   └── viz/             # Visualization backend
├── scripts/
│   ├── ingest_phrases.py
│   ├── embed_phrases.py
│   ├── cluster_embeddings.py
│   ├── cluster_metadata.py
│   ├── assign_glyphs.py
│   ├── build_addresses.py
│   ├── build_tape.py
│   ├── eval_basic.py
│   └── run_full_build.py
├── configs/
│   └── demo.yaml
├── data/
│   └── raw/
├── embeddings/
├── clusters/
├── tape/
├── tests/
├── requirements.txt
├── setup.py
├── README.md
└── docs/              # (already exists)
```

#### Step 0.2: Dependencies Setup

Create `requirements.txt`:

```text
# Core ML/NLP
torch>=2.0.0
sentence-transformers>=2.2.0
transformers>=4.30.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Dimensionality reduction
umap-learn>=0.5.3

# Storage and serialization
msgpack>=1.0.5
sqlalchemy>=2.0.0

# API and web
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
typer>=0.9.0
loguru>=0.7.0

# Optional: Advanced indexing
# faiss-cpu>=1.7.4
# annoy>=1.17.3
```

Create `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="fractal-glyph-tape",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        # ... other dependencies
    ],
    entry_points={
        "console_scripts": [
            "fgt=fgt.cli:main",
        ],
    },
)
```

#### Step 0.3: Configuration System

Create `configs/demo.yaml`:

```yaml
# Demo configuration for Phase 0 prototype
project_name: "fgt_demo_v1"
random_seed: 42

# Data ingestion
ingest:
  input_path: "data/raw"
  output_path: "data/phrases.jsonl"
  min_tokens: 3
  max_tokens: 50
  languages: ["en"]  # Start with English only
  target_phrase_count: 100000

# Embedding
embed:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384-dim
  batch_size: 128
  device: "cuda"  # or "cpu"
  output_path: "embeddings/demo_v1"

# Clustering
cluster:
  method: "minibatch_kmeans"
  n_clusters: 10000  # ~100k phrases / 10 phrases per cluster
  batch_size: 1000
  max_iter: 100
  embeddings_path: "embeddings/demo_v1"
  output_path: "clusters/demo_v1"

# Glyph assignment
glyph:
  alphabet_path: "src/glyph/mandarin_alphabet.txt"  # Curated glyph set
  max_glyph_length: 4  # 1-4 character glyph IDs
  output_path: "clusters/demo_v1/glyph_mapping.msgpack"

# Fractal tape
tape:
  projection_method: "umap"  # or "pca"
  projection_dims: 2
  fractal_type: "triangular"  # Sierpiński-like subdivision
  max_depth: 10
  output_path: "tape/v1"

# Evaluation
eval:
  test_phrases_path: "data/test_phrases.jsonl"
  metrics: ["compression_ratio", "reconstruction_quality", "cluster_coherence"]
```

#### Step 0.4: Data Ingestion Module

**File**: `src/ingest/__init__.py`, `src/ingest/reader.py`, `src/ingest/segmenter.py`, `src/ingest/phrases.py`

**Key responsibilities**:
- Read raw text files (`.txt`, `.jsonl`)
- Sentence segmentation (using regex or spaCy)
- Language detection (optional, can default to English)
- Output: `phrases.jsonl` with format:
  ```json
  {"phrase_id": "0001", "text": "Can you send me that file?", "lang": "en", "doc_id": "d123", "metadata": {}}
  ```

**Implementation notes**:
- Use `nltk.sent_tokenize()` or spaCy for sentence splitting
- Filter extremely short (<3 tokens) or long (>50 tokens) phrases
- Log statistics: documents processed, sentences extracted, phrases produced

#### Step 0.5: Embedding Module

**File**: `src/embed/__init__.py`, `src/embed/embedder.py`

**Key responsibilities**:
- Load SentenceTransformer model
- Batch encode phrases to dense vectors
- Save embeddings as `.npy` shards
- Track phrase_id → embedding_index mapping

**Implementation notes**:
- Use GPU if available: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Process in batches (128-256 phrases)
- Save periodically to avoid memory issues
- Output format: `embeddings/demo_v1/shard_0000.npy` + `phrase_index.json`

#### Step 0.6: Clustering Module

**File**: `src/cluster/__init__.py`, `src/cluster/clusterer.py`

**Key responsibilities**:
- Load embedding shards
- Run MiniBatchKMeans clustering
- Compute cluster centroids
- Assign phrases to clusters
- Extract cluster metadata (size, coherence, example phrases)

**Implementation notes**:
```python
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# Load embeddings
embeddings = np.load('embeddings/demo_v1/all_embeddings.npy')

# Cluster
kmeans = MiniBatchKMeans(
    n_clusters=10000,
    batch_size=1000,
    max_iter=100,
    random_state=42
)
labels = kmeans.fit_predict(embeddings)
centroids = kmeans.cluster_centers_

# Save
np.save('clusters/demo_v1/labels.npy', labels)
np.save('clusters/demo_v1/centroids.npy', centroids)
```

Output:
- `labels.npy`: Array mapping phrase_index → cluster_id
- `centroids.npy`: Array of cluster centroid vectors
- `metadata.json`: Per-cluster stats (size, top examples, coherence score)

#### Step 0.7: Glyph ID Manager

**File**: `src/glyph/__init__.py`, `src/glyph/manager.py`, `src/glyph/mandarin_alphabet.txt`

**Key responsibilities**:
- Load Mandarin character alphabet (~3000-6000 high-frequency chars)
- Map cluster_id → glyph_id (integer)
- Encode glyph_id → glyph_string (1-4 Mandarin characters)
- Decode glyph_string → glyph_id

**Character selection criteria**:
- High visual distinctiveness
- Common in Unicode (good font support)
- Balanced distribution across radicals
- NOT selected for linguistic meaning

**Implementation notes**:
```python
# Base-N encoding with Mandarin chars
def encode_glyph_id(glyph_id: int, alphabet: list[str]) -> str:
    """Encode integer ID as 1-4 Mandarin character string."""
    base = len(alphabet)
    if glyph_id == 0:
        return alphabet[0]

    chars = []
    while glyph_id > 0:
        chars.append(alphabet[glyph_id % base])
        glyph_id //= base
    return ''.join(reversed(chars))

def decode_glyph_string(glyph_str: str, alphabet: list[str]) -> int:
    """Decode Mandarin character string to integer ID."""
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    base = len(alphabet)
    glyph_id = 0
    for char in glyph_str:
        glyph_id = glyph_id * base + char_to_idx[char]
    return glyph_id
```

Output:
- `glyph_mapping.msgpack`: Bidirectional map (cluster_id ↔ glyph_id ↔ glyph_string)

#### Step 0.8: Fractal Tape Builder

**File**: `src/tape/__init__.py`, `src/tape/builder.py`, `src/tape/fractal.py`

**Key responsibilities**:
- Project cluster centroids from high-dim to 2D (using UMAP or PCA)
- Normalize coordinates into triangle domain [0,1] × [0,1]
- Recursively subdivide triangle (Sierpiński-like)
- Assign each cluster a fractal address (e.g., "L-R-L-R-C")
- Build storage index: glyph_id ↔ fractal_address ↔ cluster metadata

**Fractal addressing scheme**:
```
Top-level triangle divided into:
  L (left sub-triangle)
  R (right sub-triangle)
  C (center sub-triangle)

Address "L-R-C" means:
  - Enter left triangle
  - Enter its right sub-triangle
  - Enter its center sub-triangle
```

**Implementation notes**:
```python
import umap

# 1. Dimensionality reduction
reducer = umap.UMAP(n_components=2, random_state=42)
coords_2d = reducer.fit_transform(centroids)

# 2. Normalize to [0, 1]
coords_norm = (coords_2d - coords_2d.min(axis=0)) / (coords_2d.max(axis=0) - coords_2d.min(axis=0))

# 3. Assign to fractal cells (recursive subdivision)
def assign_fractal_address(x, y, depth=10):
    """Assign point (x,y) in [0,1]×[0,1] to fractal cell."""
    address = []
    # Recursive triangular subdivision logic
    # ... (see docs/20-fractal-addressing-spec.md)
    return '-'.join(address)

addresses = [assign_fractal_address(x, y) for x, y in coords_norm]
```

Output:
- `tape/v1/tape_index.db`: SQLite database with tables:
  - `glyphs`: (glyph_id, glyph_string, cluster_id)
  - `addresses`: (cluster_id, fractal_address, x_coord, y_coord)
  - `clusters`: (cluster_id, centroid_vector, size, examples)

#### Step 0.9: CLI Tool

**File**: `src/fgt/cli.py`

**Commands**:

1. **`fgt build`**: Run full pipeline
   ```bash
   fgt build --config configs/demo.yaml
   ```

2. **`fgt encode`**: Text → glyph-coded representation
   ```bash
   echo "Can you send me that file?" | fgt encode
   # Output: 谷阜 + metadata
   ```

3. **`fgt decode`**: Glyph-coded → text
   ```bash
   echo "谷阜" | fgt decode
   # Output: Representative phrase from cluster
   ```

4. **`fgt inspect-glyph`**: Show cluster details
   ```bash
   fgt inspect-glyph 谷阜
   # Output: cluster_id, example phrases, frequency, fractal address
   ```

**Implementation framework**: Use `typer` for clean CLI:
```python
import typer
app = typer.Typer()

@app.command()
def build(config: str):
    """Run full FGT build pipeline."""
    # Load config, orchestrate all steps
    ...

@app.command()
def encode(text: str):
    """Encode text to glyph representation."""
    # Load tape, find matching clusters, output glyphs
    ...
```

#### Step 0.10: Basic Evaluation

**File**: `src/eval/__init__.py`, `src/eval/metrics.py`, `scripts/eval_basic.py`

**Key metrics**:

1. **Compression ratio**:
   ```python
   original_size = len(original_text.encode('utf-8'))
   glyph_size = len(glyph_encoded_text.encode('utf-8'))
   ratio = original_size / glyph_size
   ```

2. **Cluster coherence** (average cosine similarity within clusters):
   ```python
   from sklearn.metrics import silhouette_score
   coherence = silhouette_score(embeddings, labels)
   ```

3. **Reconstruction quality** (BLEU or embedding similarity):
   ```python
   from sentence_transformers import util
   original_emb = model.encode(original_phrases)
   reconstructed_emb = model.encode(reconstructed_phrases)
   similarity = util.cos_sim(original_emb, reconstructed_emb).diagonal().mean()
   ```

Output:
- `results/demo_v1/metrics.json` with all measurements
- Log key metrics to console

---

### Phase 1: Visualization and Metrics

**Goal**: Add web visualizer, run compression experiments, generate publication-quality plots

#### Step 1.1: Visualization Backend

**File**: `src/viz/__init__.py`, `src/viz/api.py`

**Tech stack**: FastAPI serving tape data

```python
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3

app = FastAPI()

@app.get("/tape/overview")
def get_tape_overview():
    """Return tape statistics."""
    # Connect to tape DB, return cluster count, glyph count, etc.
    ...

@app.get("/tape/clusters/{cluster_id}")
def get_cluster_details(cluster_id: int):
    """Return cluster metadata and example phrases."""
    ...

@app.get("/tape/map")
def get_fractal_map():
    """Return 2D coordinates and glyph IDs for visualization."""
    ...
```

Run server:
```bash
uvicorn fgt.viz.api:app --reload
```

#### Step 1.2: Frontend Visualizer

**File**: `viz-frontend/` (Next.js or simple HTML + D3.js)

**Features**:
- 2D scatter plot of fractal tape
- Hover over points to see glyph ID and example phrases
- Click to zoom into fractal regions
- Color by language, frequency, or cluster size
- Search bar to highlight specific glyphs

**Simple implementation**: Use D3.js or Plotly.js to render scatter plot from `/tape/map` API endpoint

#### Step 1.3: Compression Experiments

**File**: `scripts/compression_experiments.py`

**Test scenarios**:
1. Encode 10k test phrases
2. Measure compression ratio vs baseline (raw text, gzip, BPE)
3. Vary cluster count (1k, 5k, 10k, 20k) and measure trade-offs
4. Generate plots: compression ratio vs cluster count, reconstruction quality vs compression

**Output**: `results/compression_report.md` with plots and tables

---

### Phase 2: LLM Integration

**Goal**: Hybrid tokenizer, minimal fine-tuning, context efficiency experiments

#### Step 2.1: Hybrid Tokenizer Wrapper

**File**: `src/tokenizer/__init__.py`, `src/tokenizer/hybrid.py`

**Responsibilities**:
- Wrap base tokenizer (e.g., GPT-2 tokenizer)
- For input text:
  1. Embed phrase
  2. Find nearest cluster
  3. If similarity > threshold, replace with glyph token
  4. Otherwise, use raw tokens
- Output: Mix of `[regular_token_ids, GLYPH_MARKER, glyph_id, ...]`

**Special tokens**:
```python
tokenizer.add_special_tokens({
    'additional_special_tokens': ['<GLYPH>', '</GLYPH>']
})
```

**Encoding logic**:
```python
def encode_hybrid(text: str, tape_index):
    """Encode text as mix of regular tokens and glyph tokens."""
    phrases = segment_into_phrases(text)
    token_ids = []

    for phrase in phrases:
        emb = embedder.encode(phrase)
        cluster_id, similarity = find_nearest_cluster(emb, tape_index)

        if similarity > 0.8:  # High confidence match
            glyph_id = tape_index.get_glyph_id(cluster_id)
            token_ids.extend([
                tokenizer.convert_tokens_to_ids('<GLYPH>'),
                glyph_id,
                tokenizer.convert_tokens_to_ids('</GLYPH>')
            ])
        else:
            # Fall back to raw tokens
            token_ids.extend(tokenizer.encode(phrase))

    return token_ids
```

#### Step 2.2: LLM Fine-Tuning (Minimal)

**Goal**: Teach small LLM to understand glyph tokens

**Approach**:
1. Take small pre-trained model (GPT-2 small, ~124M params)
2. Add glyph special tokens to vocabulary
3. Create training data:
   - Input: Glyph-encoded text
   - Target: Original text OR paraphrased text from same cluster
4. Fine-tune for 1-2 epochs on small dataset (~10k examples)

**Training script**: `scripts/finetune_glyph_model.py`

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'additional_special_tokens': ['<GLYPH>', '</GLYPH>']})
model.resize_token_embeddings(len(tokenizer))

# Prepare dataset
# ... (glyph-encoded inputs)

# Training
training_args = TrainingArguments(
    output_dir='./models/fgt_gpt2',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

#### Step 2.3: Context Efficiency Experiments

**File**: `scripts/context_experiments.py`

**Test**:
1. Prepare prompts with long context (e.g., 2000 tokens)
2. Encode as:
   - Baseline: Raw tokens
   - FGT: Glyph-encoded (should be ~500-800 tokens)
3. Measure:
   - Token count reduction
   - Semantic preservation (embedding similarity)
   - Task performance (Q&A, summarization)

**Output**: `results/context_efficiency_report.md`

---

### Phase 3: Cross-Lingual and Polish

**Goal**: Multilingual embeddings, cross-lingual experiments, public demo

#### Step 3.1: Multilingual Data Ingestion

- Add Chinese, Spanish corpora to `data/raw`
- Use multilingual sentence encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Re-run clustering with mixed-language phrases
- Verify that equivalent phrases across languages cluster together

#### Step 3.2: Cross-Lingual Retrieval Experiments

**Test**:
1. Query in English: "send me that file"
2. Find matching glyph
3. Retrieve all phrases in that cluster
4. Show matches in Chinese, Spanish, etc.

**Metric**: Cross-lingual retrieval precision@k

#### Step 3.3: Documentation and Demo Page

- Polish README.md with results, plots, examples
- Create demo video showing:
  1. Encoding text to glyphs
  2. Visualizing fractal map
  3. Cross-lingual search
  4. Compression statistics
- Prepare for glyphd.com integration

---

## Pipeline Orchestration

**Master script**: `scripts/run_full_build.py`

```python
#!/usr/bin/env python3
"""
Full FGT pipeline orchestrator.
Usage: python scripts/run_full_build.py --config configs/demo.yaml
"""
import subprocess
import yaml
import sys
from pathlib import Path

def run_step(script_name, config_path):
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, f"scripts/{script_name}", "--config", config_path]
    result = subprocess.run(cmd, check=True)

    if result.returncode != 0:
        print(f"ERROR: {script_name} failed!")
        sys.exit(1)

def main(config_path):
    # Load config to check parameters
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Starting FGT build: {config['project_name']}")

    # Run pipeline steps in order
    steps = [
        "ingest_phrases.py",
        "embed_phrases.py",
        "cluster_embeddings.py",
        "cluster_metadata.py",
        "assign_glyphs.py",
        "build_addresses.py",
        "build_tape.py",
        "eval_basic.py",
    ]

    for step in steps:
        run_step(step, config_path)

    print(f"\n{'='*60}")
    print("FGT build complete!")
    print(f"Tape version: {config['tape']['output_path']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
```

---

## Testing Strategy

**File**: `tests/test_*.py`

**Key test modules**:

1. **`test_glyph_manager.py`**: Encoding/decoding correctness
   ```python
   def test_encode_decode_roundtrip():
       manager = GlyphManager(alphabet)
       for i in range(100000):
           encoded = manager.encode_glyph_id(i)
           decoded = manager.decode_glyph_string(encoded)
           assert decoded == i
   ```

2. **`test_tape_storage.py`**: Database integrity
   ```python
   def test_tape_query():
       tape = TapeIndex('tape/v1')
       glyph_str = '谷阜'
       cluster = tape.get_cluster_by_glyph(glyph_str)
       assert cluster is not None
       assert len(cluster['examples']) > 0
   ```

3. **`test_hybrid_tokenizer.py`**: Tokenization correctness
   ```python
   def test_encode_decode():
       text = "Can you send me that file?"
       encoded = tokenizer.encode_hybrid(text)
       decoded = tokenizer.decode_hybrid(encoded)
       # Check semantic similarity
       assert embedding_similarity(text, decoded) > 0.9
   ```

Run tests:
```bash
pytest tests/ -v
```

---

## Quality Assurance Checklist

Before considering each phase complete:

### Phase 0 Checklist
- [ ] All directory structure created
- [ ] Dependencies installed and tested
- [ ] Config system working
- [ ] Ingestion produces valid `phrases.jsonl`
- [ ] Embeddings generated successfully
- [ ] Clustering completes without errors
- [ ] Glyph IDs assigned to all clusters
- [ ] Fractal addresses computed
- [ ] Tape database created and queryable
- [ ] CLI commands (`build`, `encode`, `decode`, `inspect-glyph`) working
- [ ] Basic metrics computed and logged
- [ ] Compression ratio > 1.5x achieved
- [ ] Unit tests passing

### Phase 1 Checklist
- [ ] Visualization API serving data
- [ ] Frontend showing fractal map
- [ ] Can hover/click to inspect glyphs
- [ ] Compression experiments completed
- [ ] Results documented with plots
- [ ] Comparison with baseline methods (gzip, BPE)

### Phase 2 Checklist
- [ ] Hybrid tokenizer encoding/decoding working
- [ ] Special tokens added to vocabulary
- [ ] Fine-tuning script functional
- [ ] Model checkpoint saved
- [ ] Context efficiency experiments run
- [ ] Token count reduction measured (target: 50-70%)
- [ ] Semantic preservation verified

### Phase 3 Checklist
- [ ] Multilingual data ingested
- [ ] Cross-lingual clustering verified
- [ ] Retrieval experiments showing cross-lingual matches
- [ ] Documentation polished
- [ ] Demo materials prepared
- [ ] All code cleaned and commented

---

## Common Issues and Solutions

### GPU Memory Issues
**Problem**: OOM errors during embedding or clustering
**Solution**:
- Reduce batch size in config
- Process embeddings in shards
- Use CPU for clustering if needed: `device='cpu'`

### Clustering Quality Poor
**Problem**: Clusters too large/small or incoherent
**Solution**:
- Tune `n_clusters` parameter
- Try different embedding models (larger models = better separation)
- Filter low-quality phrases before clustering
- Experiment with hierarchical clustering

### Glyph Encoding Collisions
**Problem**: Running out of glyph space
**Solution**:
- Increase alphabet size (use more Mandarin characters)
- Allow longer glyph sequences (increase `max_glyph_length`)
- Current scheme: 3000^4 ≈ 81 trillion possible IDs (more than sufficient)

### Fractal Address Assignment Slow
**Problem**: UMAP taking too long on large centroid sets
**Solution**:
- Use PCA instead for faster projection
- Subsample centroids for projection, then assign remainder by nearest neighbor

---

## Success Criteria

**Phase 0 Success** = Working prototype that can:
- Ingest 100k phrases
- Cluster into ~10k families
- Assign glyph IDs
- Build fractal tape
- Encode/decode text via CLI
- Show compression ratio > 2x

**Phase 1 Success** = Visualization that:
- Renders 2D fractal map
- Allows interactive exploration
- Documents clear compression gains vs baselines

**Phase 2 Success** = LLM integration that:
- Reduces token count by 50%+ while preserving semantics
- Fine-tuned model can decode glyph tokens
- Context efficiency experiments show measurable gains

**Phase 3 Success** = Cross-lingual system that:
- Clusters multilingual phrases together
- Enables cross-lingual search via glyph IDs
- Ready for public demo on glyphd.com

---

## Next Steps After Build

1. **Scale up**: Process 1M+ phrases, 50k+ clusters
2. **Production hardening**: Add error handling, monitoring, logging
3. **Advanced experiments**: Training efficiency, few-shot learning with glyphs
4. **Integration**: Connect to glyphd.com platform, EarthCloud products
5. **Research publication**: Write paper on compression results, cross-lingual capabilities
6. **Open source release**: Clean code, comprehensive docs, tutorial notebooks

---

## Additional Resources

- **Full Documentation**: See [docs/](docs/) folder for detailed specs
- **Key Concepts**:
  - [Vision Overview](docs/00-vision-overview.md)
  - [Plain English Summary](docs/01-plain-english-summary.md)
  - [System Architecture](docs/30-system-architecture-overview.md)
- **Implementation Details**:
  - [Data Pipeline](docs/31-data-pipeline-design.md)
  - [Clustering Math](docs/22-phrase-clustering-math.md)
  - [Fractal Addressing](docs/20-fractal-addressing-spec.md)
- **Experiments**:
  - [Compression Experiments](docs/61-corpus-compression-experiments.md)
  - [Context Efficiency](docs/62-context-window-efficiency-experiments.md)
  - [Multilingual Bridge](docs/63-multilingual-bridge-experiments.md)

---

## For Claude Code Web Assistant

**When implementing this project**:

1. **Start with Phase 0** - Build working prototype first
2. **Read docs thoroughly** before implementing each module
3. **Follow the exact directory structure** specified
4. **Test incrementally** - Don't build everything before testing
5. **Log verbosely** - Use loguru or Python logging for detailed progress
6. **Handle errors gracefully** - Add try/except with informative messages
7. **Optimize for cost** - Use Haiku for routine tasks, Sonnet for complex logic
8. **Ask for clarification** if specs are ambiguous
9. **Document as you go** - Add docstrings and comments
10. **Checkpoint frequently** - Save intermediate results to avoid re-computation

**Budget-conscious development**:
- Batch file operations together
- Read documentation before asking questions
- Use existing libraries instead of reinventing (scikit-learn, sentence-transformers, etc.)
- Test on small data first (1k phrases) before scaling to 100k
- Cache embeddings and intermediate results

**Expected timeline** (for Claude Code web):
- Phase 0: 2-3 hours of focused implementation
- Phase 1: 1-2 hours for viz + experiments
- Phase 2: 1-2 hours for tokenizer + basic fine-tuning
- Phase 3: 1 hour for multilingual + polish

**Total: ~5-8 hours of development time across $250 budget**

Good luck building the Fractal Glyph Tape! This is a genuinely novel approach to semantic compression and could be a significant contribution to NLP/LLM research.
