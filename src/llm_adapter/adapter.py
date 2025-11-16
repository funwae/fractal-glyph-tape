"""LLM adapter for glyph-aware models."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from tokenizer import HybridTokenizer


class LLMAdapter:
    """Adapter for integrating FGT glyphs with language models."""

    def __init__(
        self,
        model_name: str = "gpt2",
        tape_db_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize LLM adapter.

        Args:
            model_name: Base model name or path
            tape_db_path: Path to tape database
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        logger.info(f"Initializing LLMAdapter with model: {model_name}")

        self.model_name = model_name
        self.tape_db_path = tape_db_path

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize components (lazy-loaded)
        self.model = None
        self.tokenizer = None
        self.hybrid_tokenizer = None

    def load_model(self):
        """Load the base model and resize embeddings for special tokens."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add special tokens for glyphs
            special_tokens = {
                "additional_special_tokens": ["<GLYPH>", "</GLYPH>"],
                "pad_token": "<PAD>"
            }
            num_added = self.tokenizer.add_special_tokens(special_tokens)

            if num_added > 0:
                # Resize model embeddings
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Resized embeddings for {num_added} new tokens")

            # Move to device
            self.model.to(self.device)

            logger.info(f"Model loaded with vocab size: {len(self.tokenizer)}")

        return self.model, self.tokenizer

    def load_hybrid_tokenizer(self) -> HybridTokenizer:
        """Load hybrid tokenizer for glyph encoding."""
        if self.hybrid_tokenizer is None:
            self.hybrid_tokenizer = HybridTokenizer(
                base_tokenizer=self.model_name,
                tape_db_path=self.tape_db_path,
                similarity_threshold=0.75
            )
            logger.info("Loaded hybrid tokenizer")

        return self.hybrid_tokenizer

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        use_hybrid: bool = False,
        **generation_kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            use_hybrid: Whether to use hybrid tokenization
            **generation_kwargs: Additional arguments for generation

        Returns:
            Generated text
        """
        model, tokenizer = self.load_model()

        if use_hybrid and self.tape_db_path:
            # Use hybrid tokenizer
            hybrid_tok = self.load_hybrid_tokenizer()
            input_ids = hybrid_tok.encode_hybrid(prompt)
            input_ids = torch.tensor([input_ids]).to(self.device)
        else:
            # Use regular tokenizer
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                **generation_kwargs
            )

        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text

    def prepare_training_data(
        self,
        texts: List[str],
        use_hybrid: bool = True,
        max_length: int = 512
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Prepare training data with hybrid tokenization.

        Args:
            texts: List of training texts
            use_hybrid: Whether to use hybrid encoding
            max_length: Maximum sequence length

        Returns:
            List of tokenized examples
        """
        _, tokenizer = self.load_model()

        if use_hybrid and self.tape_db_path:
            hybrid_tok = self.load_hybrid_tokenizer()

        examples = []
        for text in texts:
            if use_hybrid and self.tape_db_path:
                # Hybrid encoding
                token_ids = hybrid_tok.encode_hybrid(text)
            else:
                # Regular encoding
                token_ids = tokenizer.encode(text)

            # Truncate if needed
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            examples.append({
                "input_ids": torch.tensor(token_ids),
                "labels": torch.tensor(token_ids)
            })

        return examples

    def save_model(self, output_dir: str):
        """Save fine-tuned model and tokenizer."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(output_dir)
            logger.info(f"Saved model to {output_dir}")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved tokenizer to {output_dir}")

    def load_from_checkpoint(self, checkpoint_dir: str):
        """Load model from checkpoint."""
        logger.info(f"Loading from checkpoint: {checkpoint_dir}")
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        logger.info("Checkpoint loaded successfully")


class GlyphDataset(torch.utils.data.Dataset):
    """Dataset for training with glyph-encoded data."""

    def __init__(self, examples: List[Dict[str, torch.Tensor]]):
        """
        Initialize dataset.

        Args:
            examples: List of tokenized examples
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
