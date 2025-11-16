# Hybrid Tokenizer Theory

FGT's tokenizer is **hybrid**:

- It combines a standard subword tokenizer with glyph-based phrase codes.

## 1. Baseline: pure subword tokenization

Subword tokenizers:

- Break text into frequent sub-string units.

- Are agnostic to semantics; they optimize compression of character sequences.

Limitations:

- No notion of phrase families.

- Redundant representation of common patterns.

- No direct tie-in to a semantic index.

## 2. Hybrid approach

Hybrid tokenizer output:

- **Raw tokens** for unique or delicate content.

- **Glyph tokens** for spans that match known phrase families.

Encoding rule of thumb:

- If a span maps confidently to a phrase family:

  - Replace span with glyph.

- Otherwise:

  - Leave as raw text.

## 3. Advantages

- **Semantic compression**:

  - Fewer tokens for frequent patterns.

- **Consistent naming**:

  - All variations of a phrase family share a code.

- **Link to external memory**:

  - Glyph tokens directly index phrase-family entries in FGT.

## 4. Train-time behavior

During training:

- Model sees glyph tokens along with raw tokens.

- It learns:

  - That glyph tokens are "handles" for distributions of phrases.

  - To expand glyphs into appropriate text given context.

## 5. Inference-time behavior

At inference:

- Input can already be glyphified:

  - E.g., logs pre-compressed with FGT.

- Model:

  - Uses glyphs to retrieve and condense knowledge.

  - Generates responses with or without glyphs depending on configuration.

## 6. Design parameters

Key knobs:

- Aggressiveness of glyph insertion.

- Confidence thresholds for phrase-family matching.

- Whether glyph tokens are:

  - Specially marked.

  - Or just raw characters in context.

These affect:

- Compression vs fidelity.

- How non-FGT-aware models behave (degradation mode).

