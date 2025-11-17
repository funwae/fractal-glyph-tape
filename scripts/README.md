# Scripts

This directory contains scripts for running experiments and managing the Fractal Glyph Tape system.

## Available Scripts

### Compression Experiments

#### `run_compression_experiment.py`

Run compression and reconstruction experiments to evaluate FGT performance.

**Usage:**

```bash
python scripts/run_compression_experiment.py \
  <original_file> \
  <reconstructed_file> \
  --fgt-sequences-bytes <bytes> \
  --fgt-tables-bytes <bytes> \
  [--output-dir <dir>] \
  [--use-bertscore]
```

**Arguments:**

- `original_file`: Path to original corpus file
- `reconstructed_file`: Path to reconstructed corpus file
- `--fgt-sequences-bytes`: Bytes used for FGT sequences (required)
- `--fgt-tables-bytes`: Bytes used for lookup tables (required)
- `--output-dir`: Directory to save results (default: `results/experiments`)
- `--use-bertscore`: Compute BERTScore (slow but more accurate)

**Example:**

```bash
python scripts/run_compression_experiment.py \
  data/eval/wiki_sample.txt \
  data/eval/wiki_sample_reconstructed.txt \
  --fgt-sequences-bytes 1200000 \
  --fgt-tables-bytes 500000 \
  --output-dir results/experiments
```

**Output:**

The script generates a JSON file with compression and reconstruction metrics:

```json
{
  "dataset_name": "wiki_sample",
  "sentence_count": 10000,
  "compression": {
    "raw_bytes": 10000000,
    "fgt_bytes_total": 1700000,
    "compression_ratio": 5.88,
    "compression_percentage": 83.0
  },
  "reconstruction": {
    "bleu": 0.8234,
    "rouge1_f1": 0.8567,
    "rouge2_f1": 0.7123,
    "rougeL_f1": 0.8345,
    "bertscore_f1": 0.9123
  }
}
```

### Visualization

#### `generate_plots.py`

Generate plots and tables from experiment results.

**Usage:**

```bash
python scripts/generate_plots.py \
  [--results-dir <dir>] \
  [--output-dir <dir>]
```

**Arguments:**

- `--results-dir`: Directory containing result JSON files (default: `results/experiments`)
- `--output-dir`: Directory to save outputs (default: `results/plots`)

**Example:**

```bash
python scripts/generate_plots.py \
  --results-dir results/experiments \
  --output-dir results/plots
```

**Output:**

The script generates:

1. **CSV tables:**
   - `compression_metrics.csv`: Compression ratios and sizes
   - `reconstruction_metrics.csv`: BLEU, ROUGE, BERTScore

2. **PNG plots:**
   - `compression_ratios.png`: Bar chart of compression ratios
   - `reconstruction_quality.png`: Grouped bar chart of quality metrics
   - `compression_vs_quality.png`: Scatter plot showing trade-offs

#### `start_visualizer.py`

Start the FastAPI backend server for the interactive visualizer.

**Usage:**

```bash
python scripts/start_visualizer.py \
  [--tape-dir <dir>] \
  [--host <host>] \
  [--port <port>]
```

**Arguments:**

- `--tape-dir`: Path to tape directory (default: `tape/v1`)
- `--host`: Host to bind to (default: `127.0.0.1`)
- `--port`: Port to bind to (default: `8000`)

**Example:**

```bash
# Start server on default port
python scripts/start_visualizer.py

# Start on specific port
python scripts/start_visualizer.py --port 8080

# Use custom tape directory
python scripts/start_visualizer.py --tape-dir tape/v2
```

The server will start at `http://localhost:8000` and provide the following endpoints:

- `GET /`: API information
- `GET /clusters`: List all clusters
- `GET /cluster/{id}`: Get cluster details
- `GET /glyph/{glyph}`: Lookup cluster by glyph
- `GET /layout`: Get 2D layout coordinates

## Workflow Example

### Phase 1: Run Experiments

1. **Generate tape data** (using main pipeline):
   ```bash
   python -m src.fgt.cli build --config configs/demo.yaml
   ```

2. **Compute layout**:
   ```bash
   python -m src.viz.layout tape/v1/clusters
   ```

3. **Run compression experiment**:
   ```bash
   python scripts/run_compression_experiment.py \
     data/eval/dataset.txt \
     data/eval/dataset_reconstructed.txt \
     --fgt-sequences-bytes 1000000 \
     --fgt-tables-bytes 400000
   ```

4. **Generate plots**:
   ```bash
   python scripts/generate_plots.py
   ```

### Phase 2: Interactive Visualization

1. **Start API server**:
   ```bash
   python scripts/start_visualizer.py
   ```

2. **Start web frontend** (in separate terminal):
   ```bash
   cd web
   npm install
   npm run dev
   ```

3. **Open visualizer**:
   Navigate to `http://localhost:3000/explore`

## Requirements

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

For the web frontend:

```bash
cd web
npm install
```

## Tips

- Use `--use-bertscore` only when needed, as it's computationally expensive
- Run multiple experiments with different configurations to compare results
- The visualizer requires computed layout data (`clusters/layout.npy`)
- Check the API server is running before opening the web visualizer
