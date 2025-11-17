# Phase 1: Visualization and Metrics

This document describes the Phase 1 implementation of the Fractal Glyph Tape project, which adds visualization tools and comprehensive compression experiments.

## Overview

Phase 1 adds:
- **Interactive web visualization** of the fractal semantic space
- **FastAPI backend** for serving tape data
- **Compression experiments** comparing FGT with baselines
- **Publication-quality plots** for analysis and reporting

## Components

### 1. Visualization Backend (`src/viz/api.py`)

A FastAPI-based REST API for accessing tape data:

**Endpoints:**
- `GET /` - API overview and documentation
- `GET /api/tape/overview` - Tape statistics
- `GET /api/tape/map` - All cluster coordinates for mapping
- `GET /api/tape/clusters/{cluster_id}` - Detailed cluster information
- `GET /api/tape/search?query={text}` - Search glyphs and addresses
- `GET /viz` - Interactive fractal map visualization
- `GET /docs` - Auto-generated API documentation

**Start the server:**
```bash
python scripts/run_viz_server.py --tape tape/v1/tape_index.db --port 8000
```

Then visit:
- http://localhost:8000 - API overview
- http://localhost:8000/viz - Interactive map
- http://localhost:8000/docs - API docs

### 2. Interactive Fractal Map

The `/viz` endpoint provides an interactive visualization built with Plotly.js:

**Features:**
- Scatter plot of semantic phrase space
- Color-coded by cluster ID
- Size proportional to cluster size
- Hover to see glyph, size, and fractal address
- Click to view detailed cluster information
- Zoom and pan for exploration

**Data Format:**
Each point on the map represents a cluster/glyph with:
- X, Y coordinates (UMAP/PCA projection)
- Glyph string (Mandarin characters)
- Cluster size (number of phrases)
- Fractal address (hierarchical location)
- Example phrases from the cluster

### 3. Compression Experiments (`scripts/compression_experiments.py`)

Comprehensive comparison of compression methods:

**Methods Tested:**
1. **Raw Text** - Baseline (no compression)
2. **Gzip** - General-purpose compression
3. **BPE (GPT-2)** - Subword tokenization
4. **FGT Glyphs** - Semantic glyph encoding

**Metrics:**
- Compression ratio (higher is better)
- Bytes per phrase (lower is better)
- Encoding/decoding time
- Semantic preservation (embedding similarity)

**Run experiments:**
```bash
python scripts/compression_experiments.py \
    --phrases data/phrases.jsonl \
    --tape tape/v1/tape_index.db \
    --output results/compression_experiments \
    --sample-size 1000
```

**Output:**
- `compression_results.json` - Raw results data
- `compression_comparison.png` - Comparison plots
- `compression_report.md` - Detailed markdown report

### 4. Analysis Plots (`scripts/generate_plots.py`)

Generate publication-quality visualizations:

**Plots Generated:**
1. **Summary Dashboard** - Key metrics overview
2. **Cluster Size Distribution** - Histogram and box plot
3. **Fractal Map 2D** - Semantic space visualization
4. **Address Depth Distribution** - Fractal addressing stats
5. **Glyph Length Distribution** - Glyph encoding efficiency

**Run plot generation:**
```bash
python scripts/generate_plots.py \
    --tape tape/v1/tape_index.db \
    --output results/plots
```

**Output:**
- `summary_dashboard.png` - Overview metrics
- `cluster_size_distribution.png` - Size analysis
- `fractal_map_2d.png` - 2D projection
- `address_depth_distribution.png` - Addressing stats
- `glyph_length_distribution.png` - Encoding efficiency
- `statistics.json` - Summary statistics

## Usage Examples

### Example 1: Explore the Fractal Map

1. Start the visualization server:
```bash
python scripts/run_viz_server.py
```

2. Open browser to http://localhost:8000/viz

3. Interact with the map:
   - Hover over points to see cluster info
   - Click to view detailed examples
   - Zoom in to explore dense regions
   - Use search to find specific glyphs

### Example 2: Compare Compression Methods

Run comprehensive experiments:
```bash
# Run on 1000 sample phrases
python scripts/compression_experiments.py \
    --sample-size 1000 \
    --output results/exp_1k

# Run on 10000 phrases for more robust stats
python scripts/compression_experiments.py \
    --sample-size 10000 \
    --output results/exp_10k
```

View results in `results/exp_*/compression_report.md`

### Example 3: Generate Analysis Report

1. Generate all plots:
```bash
python scripts/generate_plots.py --output results/analysis
```

2. Create final report combining plots and experiments:
```bash
# Plots are saved to results/analysis/
# Include them in your documentation or presentations
```

## API Usage

### Get Cluster Information
```python
import requests

# Get cluster details
response = requests.get("http://localhost:8000/api/tape/clusters/42")
cluster = response.json()

print(f"Glyph: {cluster['glyph_string']}")
print(f"Size: {cluster['size']} phrases")
print(f"Address: {cluster['fractal_address']}")
print(f"Examples: {cluster['examples'][:3]}")
```

### Search for Glyphs
```python
# Search by glyph or address
response = requests.get(
    "http://localhost:8000/api/tape/search",
    params={"query": "L-R-C", "limit": 5}
)
results = response.json()

for r in results:
    print(f"{r['glyph_string']}: {r['fractal_address']}")
```

### Get All Map Points
```python
# Get coordinates for visualization
response = requests.get("http://localhost:8000/api/tape/map")
points = response.json()

# Plot with matplotlib
import matplotlib.pyplot as plt
x = [p['x'] for p in points]
y = [p['y'] for p in points]
sizes = [p['size'] for p in points]

plt.scatter(x, y, s=sizes, alpha=0.5)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Semantic Phrase Space')
plt.show()
```

## Performance Considerations

### Visualization Server
- **Database queries** are optimized with indexes
- **Map data** can be limited with `?limit=N` parameter
- **CORS** enabled for frontend integration
- **Connection pooling** for concurrent requests

### Compression Experiments
- **Sample size** controls experiment duration
- **Embedding** is the bottleneck for FGT encoding
- **Batch processing** improves throughput
- **GPU acceleration** recommended for large samples

### Plot Generation
- **High DPI** (300) for publication quality
- **Vector formats** (PDF) available if needed
- **Style customization** via seaborn/matplotlib
- **Memory efficient** for large datasets

## Expected Results

### Compression Ratios (Typical)
- **Gzip**: 2.0-3.0x compression
- **BPE**: 0.7-1.2x (larger due to vocab overhead)
- **FGT Glyphs**: 1.5-2.5x (with semantic preservation)

### Key Advantages of FGT
1. **Semantic clustering** - Similar phrases map to same glyph
2. **Cross-lingual** - Equivalent phrases in different languages share glyphs
3. **Searchable** - Find semantically related content via glyph space
4. **Structured** - Fractal addressing enables hierarchical navigation
5. **Composable** - Can combine with other compression methods

## Troubleshooting

### Visualization server won't start
- Check that tape database exists: `ls -l tape/v1/tape_index.db`
- Run build pipeline first: `fgt build --config configs/demo.yaml`
- Verify port is available: `lsof -i :8000`

### Compression experiments fail
- Ensure transformers library is installed for BPE
- Check that phrases file exists and is valid JSONL
- Reduce sample size if memory issues occur
- Use CPU mode if GPU not available

### Plots look incorrect
- Verify tape database has data
- Check that matplotlib/seaborn are installed
- Try different style settings
- Ensure output directory is writable

## Next Steps

After Phase 1:
- **Phase 2**: LLM integration (hybrid tokenizer, fine-tuning)
- **Phase 3**: Cross-lingual experiments, public demo
- **Production**: Scaling to millions of phrases
- **Research**: Publish compression and context efficiency results

## References

- FastAPI documentation: https://fastapi.tiangolo.com/
- Plotly.js: https://plotly.com/javascript/
- Compression benchmarks: See `results/compression_report.md`
- System architecture: `docs/30-system-architecture-overview.md`
