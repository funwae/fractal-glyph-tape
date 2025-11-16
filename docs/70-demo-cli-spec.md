# Demo CLI Specification

We provide a CLI for basic FGT operations.

## 1. Commands

### 1.1 `fgt build`

Run full pipeline:

```bash
fgt build --config configs/demo.yaml
```

### 1.2 `fgt encode`

Encode text to glyph-coded representation:

```bash
echo "Can you send me that file?" | fgt encode
```

Output:

* Glyph-coded string.

* Metadata about clusters used.

### 1.3 `fgt decode`

Decode glyph-coded string back to text:

```bash
echo "谷阜" | fgt decode
```

Outputs:

* Representative phrase or paraphrase.

### 1.4 `fgt inspect-glyph`

Inspect glyph / cluster:

```bash
fgt inspect-glyph 谷阜
```

Shows:

* Cluster ID.

* Example phrases.

* Frequency.

* Address on tape.

## 2. Configuration

Global config:

* Path to tape version.

* Model and tokenizer selection.

## 3. Implementation

* Python package `fgt.cli`.

* Uses `argparse` or `typer`.

