# Evaluation Metrics Overview

This document defines the metrics used to evaluate Fractal Glyph Tape (FGT).

We group metrics into four categories:

1. **Compression metrics**

2. **Context efficiency metrics**

3. **Model performance metrics**

4. **Human and qualitative metrics**

Each experiment file (`61-*.md`–`65-*.md`) references these definitions.

---

## 1. Compression metrics

### 1.1 Byte compression ratio

Measures how much storage we save by using FGT.

Given:

- `B_raw` = size in bytes of original corpus (UTF-8 text).

- `B_fgt` = size in bytes of FGT representation (glyph tape + necessary metadata).

Define:

\[
\text{CompressionRatio} = \frac{B_\text{raw}}{B_\text{fgt}}
\]

Higher is better.

We also report:

- `B_raw_per_sentence`

- `B_fgt_per_sentence`

for human scale.

### 1.2 Information density

Average semantic content per byte.

Approximation:

- Use a proxy, such as:

  - Average embedding entropy.

  - Or mutual information estimates.

For first prototype, we can informally compare:

- How many distinct phrase families can be represented per MB of storage.

---

## 2. Context efficiency metrics

We want to show that FGT effectively **expands context**.

### 2.1 Effective context multiplier

Define:

- `N_tokens_raw` = number of tokens in a raw-context prompt.

- `N_tokens_fgt` = number of tokens in FGT-encoded prompt that carries equivalent or better information.

- `TaskScore_raw` = downstream performance (e.g., QA accuracy) with raw context.

- `TaskScore_fgt` = performance with FGT context, under same token budget.

We define:

\[
\text{EffectiveContextMultiplier} = \frac{\text{EquivalentInformationRawTokens}}{N_\text{tokens_fgt}}
\]

Operationally:

- For a fixed performance level (e.g., 80% QA accuracy),

- Compare the token budgets required with and without FGT.

### 2.2 Context retention curves

For a context length `L`:

1. Truncate context at various positions.

2. Evaluate task performance for both:

   - Raw text.

   - FGT representation.

Plot:

- Task performance vs token budget.

- Show FGT curve dominating raw curve if successful.

---

## 3. Model performance metrics

We evaluate model behavior with and without FGT.

### 3.1 Language modeling metrics

Standard metrics:

- **Perplexity (PPL)** on held-out corpora.

- **Negative log-likelihood (NLL)**.

Comparison:

- Model trained/fine-tuned on raw data only.

- Model trained/fine-tuned with FGT-augmented data or glyph tokens.

### 3.2 Task metrics

Select downstream tasks (depending on corpus):

- Question answering (accuracy, F1).

- Summarization (ROUGE, BERTScore).

- Classification (accuracy, macro-F1).

Measure:

- Performance under:

  - Fixed token budgets.

  - Fixed training FLOP budgets.

We're looking for:

- Equal or better performance when FGT is used.

- Especially under constrained token or compute budgets.

---

## 4. Cross-lingual metrics

If FGT uses multilingual embeddings:

### 4.1 Cross-lingual retrieval

Given:

- Query in language A.

- Corpus documents in language B.

We can:

1. Encode documents into FGT representation (with glyph IDs).

2. Run retrieval based on:

   - Glyph IDs.

   - Or combination of glyph IDs and embeddings.

Metrics:

- Recall@k

- MRR (Mean Reciprocal Rank)

- nDCG@k

Compare to:

- Baseline embedding-only retrieval.

### 4.2 Alignment purity

For each cluster (glyph family):

- Compute **language purity**:

  - Fraction of phrases from the dominant language.

- For cross-lingual clusters:

  - Compute how evenly languages are represented.

We report:

- Distribution of cluster language purity.

- Number of clusters with cross-lingual membership above a threshold.

---

## 5. Human and qualitative metrics

Some evaluations require human judgment.

### 5.1 Reconstruction quality

Given original text and FGT-based reconstruction:

- Human raters judge:

  - **Meaning preservation**: "Same meaning?" (Yes/No/Close).

  - **Fluency**: 1–5 scale.

  - **Acceptability**: "Could this be used in an application?" (Yes/No).

We report:

- Percent of reconstructions rated as:

  - Meaning preserved.

  - Fluent (>= 4 / 5).

  - Acceptable or better.

### 5.2 Interpretability of glyph families

Show raters:

- A glyph code.

- A sample of its phrases.

Ask:

- "Do these phrases feel like a coherent family?" (1–5).

- "How would you label this family in one short phrase?"

This measures:

- Whether clusters align with human intuitions.

- Whether glyph families can be named and reasoned about.

---

## 6. Visualization quality (for demos)

Visualization is partly qualitative but can be guided by metrics:

- **Neighborhood coherence**:

  - For a point on the fractal map, neighbors should have similar embeddings.

- **Cluster separation**:

  - Overlap of different high-level categories in visual space.

We can quantify using:

- Silhouette score in projected space.

- Local density vs semantic variance.

These metrics justify the visual story shown on glyphd.com.

---

## 7. Reporting standards

For each experiment, we should:

1. Specify:

   - Datasets.

   - Models.

   - Hyperparameters.

2. Report:

   - Mean, standard deviation, and number of runs.

3. Provide:

   - Tables and plots (stored under `docs/figures/` or similar).

4. State:

   - Limitations.

   - Caveats.

This allows external researchers to:

- Reproduce the results.

- Compare with their own approaches.

