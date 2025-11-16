# Training Efficiency Experiments

We test whether FGT helps training/fine-tuning.

## 1. Setup

- Baseline model:

  - Trained/fine-tuned on raw data.

- FGT model:

  - Trained/fine-tuned on FGT-augmented data under same compute budget.

## 2. Metrics

- Perplexity on held-out corpora.

- Task metrics (QA, summarization, classification).

- Training curves (loss vs steps).

## 3. Hypothesis

FGT-augmented models:

- Converge faster.

- Achieve equal or better performance at same FLOPs.

## 4. Reporting

- Curves comparing loss vs step.

- Tables comparing final metrics.

