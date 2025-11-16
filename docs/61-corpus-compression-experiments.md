# Corpus Compression Experiments

This document describes how we measure and report compression achieved by Fractal Glyph Tape (FGT).

---

## 1. Objective

Quantify how much storage space we save when representing corpora using FGT compared to raw text, while preserving reconstructable meaning.

We compare:

- **Raw text representation** (UTF-8).

- **FGT representation**, which includes:

  - Glyph-coded phrase sequences.

  - Necessary metadata for reconstruction.

---

## 2. Datasets

Use at least two datasets:

1. **General-domain corpus**

   - Example: Wikipedia subset, news articles, or technical documentation.

2. **Conversational corpus**

   - Example: chat logs, support dialogs, etc.

For each dataset:

- Document the source and size.

- Split into:

  - Train (used for clustering and glyph assignment).

  - Eval (used only for compression measurement and reconstruction).

---

## 3. Representations

### 3.1 Raw baseline

Store eval corpus as:

- UTF-8 encoded text files.

- Minimal metadata (e.g., document boundaries) if needed.

Compute:

- `B_raw_total` — total bytes.

- `B_raw_per_sentence` — average bytes per sentence.

- `B_raw_per_token` — if a tokenizer is chosen.

### 3.2 FGT representation

For the same eval corpus:

1. Convert text to FGT representation:

   - Detect phrases.

   - Map phrase spans to cluster IDs.

   - Replace spans with glyph IDs in a compact format.

2. Store:

   - Glyph-coded sequences.

   - Any residual raw text needed where glyphs are not applied.

   - Necessary lookup tables:

     - `cluster_id -> glyph_id`

     - `glyph_id -> representative phrases` (for reconstruction).

Compute:

- `B_fgt_sequences` — bytes of glyph-coded sequences.

- `B_fgt_tables` — bytes of lookup tables.

- `B_fgt_total = B_fgt_sequences + B_fgt_tables`.

---

## 4. Metrics

Primary metric:

\[
\text{CompressionRatio} = \frac{B_\text{raw_total}}{B_\text{fgt_total}}
\]

We also report:

- `CompressionRatio_sequences = B_raw_total / B_fgt_sequences`

  (ignoring shared tables, which are amortized over large corpora).

Per-unit metrics:

- `Bytes_per_sentence_raw`

- `Bytes_per_sentence_fgt`

- `Bytes_per_token_raw`

- `Bytes_per_token_fgt` (if tokenization is defined).

---

## 5. Reconstruction procedure

To validate that compression is meaningful:

1. For each FGT-coded document in the eval set:

   - Decode glyph IDs back into text:

     - Replace glyphs with representative phrases or model-generated paraphrases.

2. Compare reconstructed text to the original.

Metrics:

- **BLEU**, **ROUGE**, **BERTScore** between original and reconstructed text.

- Optional human evaluation on a sample (see `65-human-eval-protocols.md`).

We expect:

- High semantic similarity.

- Acceptable fluency.

---

## 6. Experimental setup

Document:

- Tape configuration:

  - Number of clusters.

  - Embedding model.

  - Clustering algorithm and hyperparameters.

- Glyph alphabet:

  - Range of Unicode characters used.

  - Glyph length distribution.

- Phrase detection strategy:

  - N-gram sizes.

  - Confidence thresholds.

Also record:

- Hardware (GPU, CPU).

- Runtime for:

  - Building FGT representation.

  - Reconstruction.

---

## 7. Result reporting

Produce tables like:

```text
Table 1: Compression and reconstruction metrics (General-domain corpus)

Representation        Bytes total   Bytes/sentence   BLEU   BERTScore (F1)
-------------------   -----------   --------------   ----   --------------
Raw text              10.0 GB       512              1.00   1.000
FGT (incl. tables)     2.5 GB       128              0.82   0.945
FGT (seq only)         1.2 GB        61              0.82   0.945
```

And similar for the conversational corpus.

Plots:

* Bar plots for `Bytes/sentence` by representation.

* Scatter plot:

  * Compression ratio vs reconstruction quality for multiple configurations.

---

## 8. Interpretation

We discuss:

* Trade-offs between compression ratio and reconstruction quality.

* How much of the storage savings comes from:

  * Shared phrase families.

  * Short glyph sequences.

* How results change with:

  * Number of clusters.

  * Phrase detection aggressiveness.

This sets the stage for:

* Choosing default configurations.

* Designing real-world storage systems based on FGT.

