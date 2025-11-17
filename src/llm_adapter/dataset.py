"""Dataset wrappers for training with FGT."""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..tokenizer.hybrid import HybridTokenizer


class FGTTextDataset(Dataset):
    """
    Dataset for training with FGT-encoded text.

    Converts raw text to FGT representation on-the-fly or loads
    precomputed encodings.
    """

    def __init__(
        self,
        texts: List[str],
        hybrid_tokenizer: HybridTokenizer,
        max_length: int = 512,
        return_metadata: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text samples
            hybrid_tokenizer: Hybrid tokenizer for encoding
            max_length: Maximum sequence length
            return_metadata: Whether to return token metadata
        """
        self.texts = texts
        self.tokenizer = hybrid_tokenizer
        self.max_length = max_length
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get encoded sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, labels, etc.
        """
        text = self.texts[idx]

        # Encode
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            return_metadata=self.return_metadata,
        )

        # Truncate
        input_ids = encoded["input_ids"][:self.max_length]
        attention_mask = encoded["attention_mask"][:self.max_length]

        # For language modeling, labels = input_ids
        labels = input_ids.copy()

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "glyph_count": encoded["glyph_count"],
        }

        if self.return_metadata:
            result["metadata"] = encoded["metadata"]

        return result

    @classmethod
    def from_file(
        cls,
        file_path: Path,
        hybrid_tokenizer: HybridTokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ) -> "FGTTextDataset":
        """
        Create dataset from text file.

        Args:
            file_path: Path to text file (one sample per line)
            hybrid_tokenizer: Hybrid tokenizer
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load

        Returns:
            Initialized dataset
        """
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        if max_samples:
            texts = texts[:max_samples]

        return cls(
            texts=texts,
            hybrid_tokenizer=hybrid_tokenizer,
            max_length=max_length,
        )


class FGTReconstructionDataset(Dataset):
    """
    Dataset for training reconstruction tasks.

    Provides:
    - Input: Glyph-coded text
    - Target: Original text

    This trains the model to expand glyphs correctly.
    """

    def __init__(
        self,
        texts: List[str],
        hybrid_tokenizer: HybridTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize reconstruction dataset.

        Args:
            texts: List of text samples
            hybrid_tokenizer: Hybrid tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = hybrid_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get reconstruction sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with glyph input and text target
        """
        text = self.texts[idx]

        # Encode with glyphs
        glyph_encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
        )

        # Encode original (baseline tokenizer)
        text_encoded = self.tokenizer.base_tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )

        # Truncate glyph input
        glyph_input_ids = glyph_encoded["input_ids"][:self.max_length]
        glyph_attention_mask = glyph_encoded["attention_mask"][:self.max_length]

        return {
            "input_ids": torch.tensor(glyph_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(glyph_attention_mask, dtype=torch.long),
            "labels": torch.tensor(text_encoded, dtype=torch.long),
            "glyph_count": glyph_encoded["glyph_count"],
        }


def fgt_collate_fn(
    batch: List[Dict],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for FGT datasets.

    Pads sequences to the same length within a batch.

    Args:
        batch: List of samples from dataset
        pad_token_id: Padding token ID

    Returns:
        Batched tensors
    """
    # Find max length in batch
    max_length = max(len(sample["input_ids"]) for sample in batch)

    # Pad all sequences
    padded_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "glyph_count": [],
    }

    for sample in batch:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]

        # Padding length
        padding_length = max_length - len(input_ids)

        # Pad
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            labels = torch.cat([
                labels,
                torch.full((padding_length,), -100, dtype=torch.long)  # Ignore in loss
            ])

        padded_batch["input_ids"].append(input_ids)
        padded_batch["attention_mask"].append(attention_mask)
        padded_batch["labels"].append(labels)
        padded_batch["glyph_count"].append(sample["glyph_count"])

    # Stack into tensors
    return {
        "input_ids": torch.stack(padded_batch["input_ids"]),
        "attention_mask": torch.stack(padded_batch["attention_mask"]),
        "labels": torch.stack(padded_batch["labels"]),
        "glyph_count": torch.tensor(padded_batch["glyph_count"]),
    }


if __name__ == "__main__":
    # Example usage
    print("FGT Dataset Example")
    print("=" * 60)
    print("\nDataset classes:")
    print("  - FGTTextDataset: For language modeling with FGT")
    print("  - FGTReconstructionDataset: For training glyph expansion")
    print("\nUsage:")
    print("  dataset = FGTTextDataset.from_file(")
    print("      'data/train.txt',")
    print("      hybrid_tokenizer=tokenizer,")
    print("      max_length=512")
    print("  )")
