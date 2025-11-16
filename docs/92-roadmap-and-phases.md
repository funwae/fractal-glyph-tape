# Roadmap and Phases

## Phase 0 – Prototype Tape (Demo Scale)

- Scope:

  - ~100k phrases (English, optionally some Chinese).

- Deliverables:

  - Full offline pipeline.

  - Working tape (`tape/v1`).

  - CLI demo for encode/decode.

- Success criteria:

  - Demonstrable compression.

  - Visual map with meaningful clusters.

## Phase 1 – Visualization + Metrics

- Add:

  - Web visualizer.

  - Compression and reconstruction experiments.

- Deliverables:

  - Plots and tables for `61` experiments.

- Success criteria:

  - Show clear compression with acceptable reconstruction.

## Phase 2 – LLM Integration

- Add:

  - Hybrid tokenizer wrapper.

  - Minimal fine-tuning to use glyph tokens.

- Deliverables:

  - Context efficiency experiments.

- Success criteria:

  - Evidence of effective context multiplier.

## Phase 3 – Multilingual and glyphd.com

- Add:

  - Multilingual embeddings.

  - Cross-lingual experiments.

  - Public demo on glyphd.com.

- Success criteria:

  - Cross-lingual retrieval gains.

  - Compelling public story.

## Phase 4 – Productionization (optional)

- Harden pipelines.

- Move toward integration with real Glyphd/EarthCloud products.

