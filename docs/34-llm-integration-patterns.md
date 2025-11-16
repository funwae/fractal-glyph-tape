# LLM Integration Patterns

FGT can interact with LLMs at multiple layers.

## 1. Preprocessing-only

Simplest pattern:

- Only **preprocess inputs** to include glyph-coded histories.

- LLM itself is unaware of glyph semantics but learns patterns empirically.

Useful as:

- Baseline for compression and context experiments.

## 2. Fine-tuned glyph-aware model

We fine-tune an LLM on:

- Data where glyph tokens appear in input and/or output.

- Tasks:

  - Reconstruct text from glyph-coded inputs.

  - Predict glyphs given text.

Benefits:

- Model explicitly learns to use glyph tokens as pointers.

## 3. Retrieval-augmented FGT

Pattern:

- During inference:

  - Parse prompt.

  - Detect or insert glyph tokens.

  - Use glyph IDs to fetch phrase-family examples or summaries.

  - Compose augmented context.

LLM sees:

- Glyphs + retrieved evidence.

## 4. Internal chain-of-thought (CoT) with glyphs (conceptual)

In a research setting:

- Let the model generate internal reasoning using glyph tokens as shorthand.

- Then decode into natural language.

This is conceptual but informs how we design training tasks.

## 5. Distillation

We can:

- Train a powerful teacher using full FGT stack.

- Distill behavior into a smaller model that still understands glyph tokens.

Integration details are fleshed out in `46-llm-adapter-impl.md` and `50-training-objectives-and-losses.md`.

