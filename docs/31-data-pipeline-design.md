# Data Pipeline Design

This describes the steps from raw corpus to phrase list.

## 1. Inputs

- Raw text files:

  - `.txt`, `.jsonl`, etc.

- Each document may include:

  - Language tag.

  - Source metadata.

## 2. Steps

1. **Document reading**

   - Stream input files.

   - Normalize encoding (UTF-8).

2. **Sentence segmentation**

   - Use language-aware sentence splitter.

   - Output sentences as base units.

3. **Phrase extraction**

   - Depending on config:

     - Use full sentences.

     - Or extract sub-sentence chunks (e.g., via punctuation or dependency heuristics).

   - Optional: generate n-grams (2–6 tokens).

4. **Metadata tagging**

   - For each phrase:

     - `phrase_id` (monotonic int).

     - `doc_id`.

     - `sentence_index`.

     - `language`.

     - `source` (corpus name).

5. **Output format**

Write:

- `data/phrases.jsonl`

Each line:

```json
{
  "phrase_id": 12345,
  "text": "Can you send me that file?",
  "lang": "en",
  "doc_id": "doc_0001",
  "sent_idx": 12,
  "source": "support_chat"
}
```

## 3. Configurability

Config options:

* Languages to include.

* Min/max phrase length.

* Whether to include punctuation-heavy phrases.

* Filters (e.g., remove boilerplate, extremely short phrases).

## 4. Quality checks

Metrics:

* Distribution of phrase lengths.

* Language distribution.

* Sample inspection logs.

These checks ensure the embedding and clustering stages receive clean, meaningful phrases.

