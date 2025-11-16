# Context Window Efficiency Experiments

We test whether FGT acts as a context multiplier.

## 1. Setup

- Choose tasks:

  - QA over long documents.

  - Dialogue summarization.

- Compare:

  - Raw text inputs.

  - FGT-coded inputs (glyphs replacing common motifs).

## 2. Protocol

For various token budgets `L`:

1. Construct prompts with raw context truncated at `L`.

2. Construct prompts with FGT-coded context truncated at `L`.

3. Evaluate task performance.

## 3. Metrics

- Task accuracy / F1.

- Plot performance vs `L`.

## 4. Expected outcome

- For the same `L`, FGT-coded prompts retain more relevant context.

- Curves show FGT dominating raw baselines, approximating a larger context window.

