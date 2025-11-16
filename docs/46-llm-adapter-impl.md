# LLM Adapter Implementation

This module provides glue code for integrating FGT with LLM workflows.

## 1. Training-time hooks

- Dataset wrappers:

  - Convert raw text to FGT representation on the fly or from precomputed files.

- Collators:

  - Pad sequences.

  - Attach glyph metadata (for auxiliary losses).

## 2. Loss components

- Reconstruction loss:

  - Predict original text given glyph-coded input.

- Glyph prediction loss:

  - Predict glyph tokens from raw text.

- Task-specific losses:

  - QA, summarization, etc.

Implementation references `50-training-objectives-and-losses.md`.

## 3. Inference-time hooks

Helpers for:

- Taking raw text + FGT model config.

- Running hybrid tokenization.

- Applying retrieval via tape if desired.

- Calling model.

- Decoding outputs.

## 4. API sketch

```python
class FGTLLMAdapter:
    def __init__(self, model, tokenizer, tape_store, glyph_manager):
        ...

    def encode_input(self, text: str) -> dict:
        ...

    def generate(self, prompt: str, **gen_kwargs) -> str:
        ...
```

Backed by actual model APIs (transformers, etc.).

