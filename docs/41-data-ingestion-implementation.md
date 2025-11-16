# Data Ingestion Implementation

This document describes the code structure for ingestion.

## 1. Module layout

- `src/ingest/__init__.py`

- `src/ingest/reader.py`

- `src/ingest/segmenter.py`

- `src/ingest/phrases.py`

- `src/ingest/cli.py`

## 2. Reader

Responsibilities:

- Iterate over raw files.

- Handle formats:

  - `.txt`, `.jsonl`, simple custom formats.

- Provide stream of documents: `{doc_id, text, meta}`.

## 3. Segmenter

Responsibilities:

- Sentence segmentation using library (e.g., spaCy, nltk, or regex-based).

- Language detection (if not given) using lightweight detector.

Outputs:

- `Sentence` objects with `doc_id`, `sent_idx`, `text`, `lang`.

## 4. Phrase extractor

Responsibilities:

- Decide what constitutes a phrase:

  - Full sentence.

  - Sub-sentence fragments.

  - n-grams.

- Filter:

  - Empty / extremely short phrases.

  - Highly boilerplate content (if known).

Writes:

- `data/phrases.jsonl` (see `31-data-pipeline-design.md`).

## 5. CLI

Example usage:

```bash
python -m fgt.ingest.cli \
  --input data/raw_corpus \
  --output data/phrases.jsonl \
  --min-tokens 3 \
  --max-tokens 50
```

Configurable by YAML.

## 6. Logging

* Log counts:

  * Documents processed.

  * Sentences extracted.

  * Phrases produced.

* Sample phrases printed to logs for inspection.

