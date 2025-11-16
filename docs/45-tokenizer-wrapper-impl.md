# Tokenizer Wrapper Implementation

This document specifies the implementation of the **hybrid tokenizer** that combines:

- A standard subword tokenizer (e.g., BPE).

- The Fractal Glyph Tape (FGT) glyph ID layer.

---

## 1. Goals

The tokenizer wrapper must:

1. Accept raw text input and produce:

   - A sequence of tokens, where some tokens are **glyph tokens**.

2. Support decoding back to:

   - Text (approximate / paraphrased).

   - Or a mixed representation (text + glyph metadata).

3. Integrate with existing LLM stacks:

   - Token IDs map into the model's vocabulary.

4. Provide knobs for:

   - How aggressively to "glyphify" phrases.

   - Whether glyph tokens are visible to the model or hidden behind virtual IDs.

---

## 2. Token types

We define three token types:

1. **Raw tokens** — from the base tokenizer.

2. **Glyph tokens (surface)** — literal Mandarin characters in the tokenizer vocab.

3. **Glyph meta tokens (virtual)** — internal IDs not exposed to the model directly, but used for bookkeeping.

For the first prototype:

- Use **surface glyph tokens**:

  - Treat glyph characters as normal Unicode.

  - Ensure the base tokenizer's vocab includes them (or extend it).

---

## 3. Base tokenizer

Use a standard tokenizer, e.g.:

- HuggingFace BPE-based tokenizer.

Requirements:

- It must handle Mandarin characters without splitting them into sub-codepoints (or we provide special handling).

- It must be extendable by adding new tokens if needed.

---

## 4. Phrase → glyph detection

To insert glyph tokens, we must detect spans of text that correspond to known phrase families.

We use:

- A **phrase index** built from the clustering stage:

  - Contains canonical string patterns per cluster.

  - Possibly n-gram patterns or templates.

Simplest first prototype:

- Use fuzzy string matching / n-gram matching to detect phrases that:

  - Have high similarity to cluster representatives.

- This can be refined later.

### 4.1 Matching algorithm (prototype)

1. Split input text into sentences.

2. For each sentence:

   - Tokenize into words.

   - Generate n-grams up to length `N_max` (e.g. 6).

3. For each n-gram:

   - Look up candidate clusters via:

     - Hash of normalized text.

     - Or approximate match index.

4. If a high-confidence match is found:

   - Replace the span with a glyph token representing the cluster.

Implementation details are documented in `33-tokenizer-integration-architecture.md`.

This file focuses on how to **wrap** the base tokenizer once glyph spans are determined.

---

## 5. Encoding pipeline

Given:

- Raw text `s`.

- A list of glyph spans `(start_char, end_char, cluster_id)`.

Encoding steps:

1. **Segment text**:

   - Break into a sequence of segments:

     - Raw text segments.

     - Glyph spans.

2. **Convert glyph spans to characters**:

   - For each glyph span:

     - Get `glyph_id = glyph_id_for_cluster(cluster_id)`.

     - Convert `glyph_id` to Unicode characters (see `21-glyph-id-encoding-spec.md`).

     - Insert these characters as a single segment.

3. **Tokenize segments** in order:

   - For raw text segments:

     - Use base tokenizer.

   - For glyph segments:

     - Either:

       - Use base tokenizer, assuming glyph characters have their own tokens.

       - Or bypass and manually emit glyph token IDs.

4. **Optionally tag glyph tokens**:

   - E.g., wrap glyphs inside special markers like `<GLYPH>谷阜</GLYPH>`.

   - This allows the model to distinguish them explicitly.

### 5.1 Pseudo-code

```python
def encode_with_glyphs(text: str, glyph_spans: list[GlyphSpan], tokenizer, glyph_codec):
    # GlyphSpan: start_char, end_char, cluster_id
    segments = segment_text_with_spans(text, glyph_spans)
    token_ids = []
    token_meta = []

    for seg in segments:
        if seg.type == "raw":
            ids = tokenizer.encode(seg.text, add_special_tokens=False)
            token_ids.extend(ids)
            token_meta.extend([{"type": "raw"} for _ in ids])
        elif seg.type == "glyph":
            glyph_id = glyph_codec.cluster_to_glyph(seg.cluster_id)
            glyph_chars = glyph_codec.glyph_id_to_unicode(glyph_id)
            ids = tokenizer.encode(glyph_chars, add_special_tokens=False)
            # Assume this yields 1–4 tokens.
            token_ids.extend(ids)
            token_meta.extend([{"type": "glyph", "cluster_id": seg.cluster_id} for _ in ids])

    return token_ids, token_meta
```

---

## 6. Decoding pipeline

Decoding depends on the target:

* **To text only**:

  * Map glyph tokens back to a *representative phrase*.

* **To mixed representation**:

  * Show glyphs as special markers, possibly with tooltips.

### 6.1 Glyph-aware decode to text

1. Decode token IDs to a raw string `s`.

2. Scan `s` for contiguous glyph-character sequences.

3. For each glyph sequence:

   * Convert to `glyph_id`.

   * Look up `cluster_id`.

   * Pick a prototype phrase (or sample) from the cluster.

4. Replace glyph sequence with that phrase.

This may produce paraphrases different from the original training text; this is a feature, not a bug.

### 6.2 Mixed decode

Alternatively:

* Keep glyph sequences visible as e.g. `⟨谷阜⟩` or `[GLYPH:12345]`.

* Use them as clickable elements in UI to reveal underlying phrases.

---

## 7. Integration with LLMs

### 7.1 Vocabulary considerations

To ensure glyph characters are handled well:

1. Pre-extend the tokenizer vocab with the chosen glyph alphabet.

2. Ensure each glyph character is treated as a standalone token:

   * No splitting into byte-level or surrogate tokens.

3. Optionally add special tokens:

   * `<GLYPH_START>` and `<GLYPH_END>`.

### 7.2 Training / fine-tuning

When fine-tuning LLMs to use FGT:

* Provide training examples where:

  * Input or internal sequences contain glyph tokens.

  * Targets require the model to:

    * Expand glyph tokens into text.

    * Use glyph tokens as pointers for reasoning.

Loss functions (see `50-training-objectives-and-losses.md`) can include:

* Cross-entropy on text reconstruction.

* Auxiliary tasks:

  * Predict cluster IDs from text.

  * Predict text from glyph IDs.

---

## 8. Configuration knobs

Config file example:

```yaml
tokenizer_wrapper:
  base_tokenizer: "gpt2"
  glyph_markers:
    enabled: true
    start_token: "<GLYPH>"
    end_token: "</GLYPH>"
  glyph_insertion:
    min_confidence: 0.9
    max_span_length: 6
    allow_overlaps: false
  decoding:
    mode: "paraphrase"  # or "literal", "mixed"
```

These knobs allow experiments with:

* How aggressively to use glyphs.

* How visible glyphs are to the model and user.

---

## 9. Testing

Key tests:

1. **Round-trip text → tokens → text** with glyphs disabled:

   * Should match base tokenizer behavior.

2. **Glyph insertion correctness**:

   * For known phrases, ensure glyph spans are detected.

   * Ensure no invalid overlaps.

3. **Decode robustness**:

   * For random mixed sequences:

     * Decoding should not crash.

     * Should produce reasonable text.

4. **Performance**:

   * Measure overhead vs base tokenizer.

   * Ensure acceptable latency for typical context sizes.

These tests ensure the tokenizer wrapper is stable enough for:

* Offline experiments.

* Interactive demos.

* Integration with glyphd.com.

