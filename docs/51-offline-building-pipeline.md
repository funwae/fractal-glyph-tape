# Offline Building Pipeline

This document describes the end-to-end offline build of a tape.

## 1. Steps

1. **Ingest phrases**

   - `python scripts/ingest_phrases.py --config configs/demo.yaml`

2. **Embed phrases**

   - `python scripts/embed_phrases.py --config configs/demo.yaml`

3. **Cluster embeddings**

   - `python scripts/cluster_embeddings.py --config configs/demo.yaml`

4. **Compute cluster metadata**

   - `python scripts/cluster_metadata.py --config configs/demo.yaml`

5. **Assign glyph IDs**

   - `python scripts/assign_glyphs.py --config configs/demo.yaml`

6. **Build fractal addresses**

   - `python scripts/build_addresses.py --config configs/demo.yaml`

7. **Build tape storage**

   - `python scripts/build_tape.py --config configs/demo.yaml`

8. **Run basic eval**

   - `python scripts/eval_basic.py --config configs/demo.yaml`

## 2. Artifacts

At the end:

- `clusters/` filled.

- `tape/vX/` created.

- Basic metrics logged.

## 3. Automation

Use a master script:

```bash
python scripts/run_full_build.py --config configs/demo.yaml
```

with clear logging and checkpoints.

