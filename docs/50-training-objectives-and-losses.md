# Training Objectives and Losses

We define how to train models to understand and use FGT.

## 1. Reconstruction objective

Task:

- Given glyph-coded input + context, generate text close to original.

Loss:

\[
\mathcal{L}_{\text{recon}} = -\sum_t \log p(y_t \mid y_{<t}, \text{glyph-coded input})
\]

## 2. Glyph prediction objective

Task:

- Given raw text, predict glyph tokens corresponding to phrase families.

Approach:

- Tag training data with gold glyph spans.

- Add auxiliary head for glyph classification.

Loss:

\[
\mathcal{L}_{\text{glyph}} = - \sum_{i} \log p(k_i \mid x)
\]

## 3. Combined loss

Total loss:

\[
\mathcal{L} = \lambda_1 \mathcal{L}_{\text{recon}} + \lambda_2 \mathcal{L}_{\text{glyph}} + \lambda_3 \mathcal{L}_{\text{task}}
\]

Where:

- `L_task` is any downstream task loss.

## 4. Curriculum

Stages:

1. Pre-train on reconstruction + glyph prediction.

2. Fine-tune on downstream tasks using FGT inputs.

## 5. Evaluation

Use metrics in `60-eval-metrics-overview.md` to compare:

- Baseline models vs FGT-augmented models.

