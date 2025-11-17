"""LLM adapter for integrating FGT with language models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..tokenizer.hybrid import HybridTokenizer


# Placeholder for legacy LLMAdapter
class LLMAdapter:
    """Legacy adapter - use FGTLLMAdapter instead."""

    def __init__(self):
        raise NotImplementedError("Use FGTLLMAdapter instead")


class FGTLLMAdapter:
    """
    Adapter for using FGT with LLMs.

    Provides helpers for:
    - Encoding inputs with glyph tokens
    - Running generation with hybrid tokenization
    - Decoding outputs with glyph expansion
    """

    def __init__(
        self,
        model: PreTrainedModel,
        hybrid_tokenizer: HybridTokenizer,
    ):
        """
        Initialize LLM adapter.

        Args:
            model: Pre-trained language model
            hybrid_tokenizer: Hybrid tokenizer with FGT support
        """
        self.model = model
        self.tokenizer = hybrid_tokenizer

        # Check device
        self.device = next(model.parameters()).device

    def encode_input(
        self,
        text: str,
        max_length: Optional[int] = None,
        return_tensors: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode input text with glyph insertion.

        Args:
            text: Input text
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors

        Returns:
            Dictionary with input_ids, attention_mask, and metadata
        """
        # Encode with hybrid tokenizer
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            return_metadata=True,
        )

        # Truncate if needed
        if max_length:
            encoded["input_ids"] = encoded["input_ids"][:max_length]
            encoded["attention_mask"] = encoded["attention_mask"][:max_length]
            if "metadata" in encoded:
                encoded["metadata"] = encoded["metadata"][:max_length]

        # Convert to tensors
        if return_tensors:
            encoded["input_ids"] = torch.tensor([encoded["input_ids"]], device=self.device)
            encoded["attention_mask"] = torch.tensor([encoded["attention_mask"]], device=self.device)

        return encoded

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        max_new_tokens: Optional[int] = None,
        expand_glyphs: bool = True,
        glyph_expansion_mode: str = "representative",
        **generation_kwargs
    ) -> str:
        """
        Generate text from prompt using FGT-enhanced context.

        Args:
            prompt: Input prompt
            max_length: Maximum total sequence length
            max_new_tokens: Maximum new tokens to generate
            expand_glyphs: Whether to expand glyphs in output
            glyph_expansion_mode: How to expand glyphs
            **generation_kwargs: Additional arguments for model.generate()

        Returns:
            Generated text
        """
        # Encode prompt
        inputs = self.encode_input(prompt, max_length=max_length)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length if max_new_tokens is None else None,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )

        # Decode
        output_text = self.tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True,
            expand_glyphs=expand_glyphs,
            glyph_expansion_mode=glyph_expansion_mode,
        )

        return output_text

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode batch of texts.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            return_tensors: Whether to return PyTorch tensors

        Returns:
            Batched encodings
        """
        # Batch encode
        encoded = self.tokenizer.batch_encode(
            texts,
            padding=padding,
            max_length=max_length,
            return_metadata=True,
        )

        # Convert to tensors
        if return_tensors:
            encoded["input_ids"] = torch.tensor(encoded["input_ids"], device=self.device)
            encoded["attention_mask"] = torch.tensor(encoded["attention_mask"], device=self.device)

        return encoded

    def compute_context_compression(
        self,
        text: str,
        baseline_tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dict[str, float]:
        """
        Compute compression ratio compared to baseline tokenization.

        Args:
            text: Input text
            baseline_tokenizer: Baseline tokenizer (uses base tokenizer if None)

        Returns:
            Dictionary with compression metrics
        """
        if baseline_tokenizer is None:
            baseline_tokenizer = self.tokenizer.base_tokenizer

        # Encode with FGT
        fgt_encoded = self.tokenizer.encode(text, return_metadata=True)
        fgt_length = len(fgt_encoded["input_ids"])
        glyph_count = fgt_encoded["glyph_count"]

        # Encode with baseline
        baseline_encoded = baseline_tokenizer.encode(text)
        baseline_length = len(baseline_encoded)

        # Compute metrics
        compression_ratio = baseline_length / fgt_length if fgt_length > 0 else 1.0
        tokens_saved = baseline_length - fgt_length
        percent_saved = (tokens_saved / baseline_length * 100) if baseline_length > 0 else 0.0

        return {
            "baseline_tokens": baseline_length,
            "fgt_tokens": fgt_length,
            "glyph_count": glyph_count,
            "tokens_saved": tokens_saved,
            "compression_ratio": compression_ratio,
            "percent_saved": percent_saved,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tape_dir: Path,
        base_tokenizer: Optional[str] = None,
        device: str = "cpu",
        **model_kwargs
    ) -> "FGTLLMAdapter":
        """
        Create adapter from pretrained model and tape.

        Args:
            model_name_or_path: Model name or path
            tape_dir: Path to tape directory
            base_tokenizer: Base tokenizer name (defaults to model tokenizer)
            device: Device to load model on
            **model_kwargs: Additional arguments for model loading

        Returns:
            Initialized FGTLLMAdapter
        """
        from transformers import AutoModelForCausalLM

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        model = model.to(device)
        model.eval()

        # Create hybrid tokenizer
        if base_tokenizer is None:
            base_tokenizer = model_name_or_path

        hybrid_tokenizer = HybridTokenizer.from_tape(
            tape_dir,
            base_tokenizer=base_tokenizer,
        )

        return cls(model=model, hybrid_tokenizer=hybrid_tokenizer)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FGTLLMAdapter("
            f"model={type(self.model).__name__}, "
            f"tokenizer={self.tokenizer})"
        )


if __name__ == "__main__":
    # Example usage
    print("FGT LLM Adapter Example")
    print("=" * 60)
    print("\nTo use this adapter:")
    print("1. Load a pre-trained model")
    print("2. Create a tape with cluster data")
    print("3. Use FGTLLMAdapter.from_pretrained()")
    print("\nExample:")
    print("  adapter = FGTLLMAdapter.from_pretrained(")
    print("      'gpt2',")
    print("      tape_dir='tape/v1',")
    print("      device='cuda'")
    print("  )")
    print("  output = adapter.generate('Your prompt here')")
