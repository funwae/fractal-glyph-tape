# Fractal Glyph Tape Docs

This `docs/` folder specifies the **Glyphd Fractal Glyph Tape (FGT)** project end-to-end:

- Concept & theory
- Math & specs
- System architecture
- Implementation plans
- Training & evaluation
- Demos & glyphd.com integration

FGT = **fractal-addressable phrase memory** that uses **Mandarin characters as a pure glyph library** (no natural-language semantics) to:

- Cluster and store **phrase families** (similar phrases/templates).
- Assign each family a compact **glyph ID**.
- Place glyph IDs on a **fractal tape** (multi-scale address space).
- Let LLMs **read/write in glyph space** for compression, context extension, and cross-lingual bridging.

## File Index

### Vision / narrative:

- `00-vision-overview.md`
- `01-plain-english-summary.md`
- `02-problem-statement.md`
- `03-solution-at-a-glance.md`
- `04-use-cases-and-demos.md`
- `05-terminology-glossary-en.md`

### Concept / theory:

- `10-fractal-tape-concept.md`
- `11-glyph-library-design.md`
- `12-clustered-phrase-memory.md`
- `13-hybrid-tokenizer-theory.md`
- `14-cross-lingual-anchor-theory.md`

### Math / formal specs:

- `20-fractal-addressing-spec.md`
- `21-glyph-id-encoding-spec.md`
- `22-phrase-clustering-math.md`
- `23-probabilistic-modeling.md`
- `24-information-theoretic-analysis.md`

### Architecture:

- `30-system-architecture-overview.md`
- `31-data-pipeline-design.md`
- `32-storage-layout-and-indexes.md`
- `33-tokenizer-integration-architecture.md`
- `34-llm-integration-patterns.md`

### Implementation:

- `40-tech-stack-and-dependencies.md`
- `41-data-ingestion-implementation.md`
- `42-embedding-and-clustering-impl.md`
- `43-glyph-id-manager-impl.md`
- `44-fractal-tape-storage-impl.md`
- `45-tokenizer-wrapper-impl.md`
- `46-llm-adapter-impl.md`

### Training / pipelines:

- `50-training-objectives-and-losses.md`
- `51-offline-building-pipeline.md`
- `52-online-update-strategy.md`
- `53-resource-and-time-estimates.md`

### Evaluation:

- `60-eval-metrics-overview.md`
- `61-corpus-compression-experiments.md`
- `62-context-window-efficiency-experiments.md`
- `63-multilingual-bridge-experiments.md`
- `64-training-efficiency-experiments.md`
- `65-human-eval-protocols.md`

### Demos / integration:

- `70-demo-cli-spec.md`
- `71-web-visualizer-spec.md`
- `72-glyphd-com-integration.md`
- `73-api-design-and-endpoints.md`

### Mandarin internal docs:

- `80-工程概览-zh.md`
- `81-训练步骤说明-zh.md`
- `82-评估流程-zh.md`
- `83-数据格式与约定-zh.md`

### Research / roadmap:

- `90-related-work-and-positioning.md`
- `91-open-questions-and-risks.md`
- `92-roadmap-and-phases.md`
- `93-contribution-guide.md`

## Recommended Build Phases

- **Phase 0 – Small demo (single GPU)**
  - Implement ingestion → embeddings → clustering → glyph assignment → fractal addressing → tape storage.
  - Provide a CLI that compresses/expands text using glyphs.

- **Phase 1 – Visualization + metrics**
  - Add web visualizer.
  - Run compression + reconstruction experiments.

- **Phase 2 – LLM integration**
  - Hybrid tokenizer wrapper.
  - Minimal fine-tune / instruction prompting to use glyph tokens.

- **Phase 3 – Cross-lingual + glyphd.com**
  - Multilingual embeddings.
  - Cross-lingual experiments.
  - Public research/demo page on glyphd.com.

