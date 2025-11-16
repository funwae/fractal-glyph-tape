# Tokenizer Integration Architecture

This describes how the hybrid tokenizer integrates with FGT components.

## 1. Components

1. **Base tokenizer**

   - Off-the-shelf tokenizer (HuggingFace or similar).

2. **Phrase matcher**

   - Detects spans in text that correspond to phrase families.

3. **Glyph codec**

   - Converts `cluster_id` ↔ `glyph_id` ↔ glyph string.

4. **Wrapper**

   - Combines base tokenizer + glyph insertion logic.

## 2. Offline vs online

- **Offline preprocessing**

  - For corpora/logs:

    - Run full phrase matching.

    - Emit glyph-coded versions for storage/training.

- **Online runtime**

  - For interactive prompts:

    - Optionally run a lighter phrase matching or rely on glyph-coded history.

## 3. Phrase matching strategies

Options:

1. **Exact / normalized text match**

   - Hash normalized phrases → `cluster_id`.

2. **Approximate match**

   - Use a separate embedding model + nearest neighbor search.

3. **Hybrid**

   - Use exact match for frequent patterns, approximate for tail cases.

The chosen strategy is configured per deployment.

## 4. Wrapper flow

1. Receive raw text.

2. Run phrase matcher → list of spans with `cluster_id`.

3. Build segmented view: raw segments + glyph spans.

4. Emit token IDs using:

   - Base tokenizer for raw segments.

   - Glyph codec for glyph spans (as glyph characters).

## 5. Extensibility

We design the wrapper as a library:

- Python API.

- Possible Rust/Go bindings for low-level systems.

It should be pluggable into:

- Training dataloaders.

- Online inference servers.

- CLI tools.

