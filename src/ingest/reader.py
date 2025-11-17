"""Corpus reader for loading raw text data."""

import json
from pathlib import Path
from typing import Iterator, Dict, Any
from loguru import logger


class CorpusReader:
    """Read raw text files from various formats."""

    def __init__(self, input_path: str):
        """
        Initialize corpus reader.

        Args:
            input_path: Path to directory containing raw text files
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

    def read_documents(self) -> Iterator[Dict[str, Any]]:
        """
        Read documents from input directory.

        Yields:
            Dictionary containing document text and metadata
        """
        # Support .txt, .jsonl formats
        txt_files = list(self.input_path.glob("**/*.txt"))
        jsonl_files = list(self.input_path.glob("**/*.jsonl"))

        logger.info(f"Found {len(txt_files)} .txt files and {len(jsonl_files)} .jsonl files")

        # Read .txt files
        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()
                    yield {
                        "doc_id": txt_file.stem,
                        "text": text,
                        "source": str(txt_file),
                        "metadata": {},
                    }
            except Exception as e:
                logger.warning(f"Failed to read {txt_file}: {e}")
                continue

        # Read .jsonl files
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            # Extract text field (support various key names)
                            text = data.get("text") or data.get("content") or data.get("body")
                            if not text:
                                logger.warning(f"No text field in {jsonl_file}:{line_num}")
                                continue

                            yield {
                                "doc_id": data.get("id", f"{jsonl_file.stem}_{line_num}"),
                                "text": text,
                                "source": str(jsonl_file),
                                "metadata": {k: v for k, v in data.items() if k not in ["text", "content", "body"]},
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON in {jsonl_file}:{line_num}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Failed to read {jsonl_file}: {e}")
                continue

    def count_documents(self) -> int:
        """Count total number of documents."""
        return sum(1 for _ in self.read_documents())
