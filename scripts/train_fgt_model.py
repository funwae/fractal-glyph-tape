#!/usr/bin/env python3
"""Training script for fine-tuning models with FGT."""

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from src.llm_adapter import FGTTextDataset, fgt_collate_fn
from src.llm_adapter.adapter import FGTLLMAdapter
from src.tokenizer.hybrid import HybridTokenizer


def train_fgt_model(
    model_name: str,
    tape_dir: Path,
    train_file: Path,
    output_dir: Path,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    device: str = "cpu",
):
    """
    Fine-tune a model with FGT-encoded data.

    Args:
        model_name: Pre-trained model name
        tape_dir: Path to tape directory
        train_file: Path to training data file
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        device: Device to train on
    """
    print(f"\n{'=' * 60}")
    print(f"FGT Model Training")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Tape: {tape_dir}")
    print(f"Training file: {train_file}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max length: {max_length}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load adapter
    print("Loading model and tokenizer...")
    adapter = FGTLLMAdapter.from_pretrained(
        model_name,
        tape_dir=tape_dir,
        device=device,
    )

    # Create dataset
    print(f"Loading training data from {train_file}...")
    dataset = FGTTextDataset.from_file(
        train_file,
        hybrid_tokenizer=adapter.tokenizer,
        max_length=max_length,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: fgt_collate_fn(
            batch,
            pad_token_id=adapter.tokenizer.base_tokenizer.pad_token_id or 0,
        ),
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(adapter.model.parameters(), lr=learning_rate)

    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Training loop
    print(f"\nStarting training...")
    adapter.model.train()

    global_step = 0
    total_glyph_tokens = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        epoch_loss = 0.0
        epoch_glyph_count = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            glyph_count = batch["glyph_count"]

            # Forward pass
            outputs = adapter.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update metrics
            epoch_loss += loss.item()
            epoch_glyph_count += glyph_count.sum().item()
            total_glyph_tokens += glyph_count.sum().item()
            global_step += 1

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} - "
                      f"Loss: {loss.item():.4f} - "
                      f"Avg Loss: {avg_loss:.4f} - "
                      f"Glyphs: {glyph_count.sum().item()}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average loss: {avg_epoch_loss:.4f}")
        print(f"  Total glyph tokens: {epoch_glyph_count}")

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Total steps: {global_step}")
    print(f"Total glyph tokens processed: {total_glyph_tokens}")
    print(f"{'=' * 60}\n")

    # Save model
    print(f"Saving model to {output_dir}...")
    adapter.model.save_pretrained(output_dir)
    print("Model saved successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a model with FGT glyph tokens"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Pre-trained model name (default: gpt2)"
    )

    parser.add_argument(
        "--tape-dir",
        type=Path,
        default=Path("tape/v1"),
        help="Path to tape directory (default: tape/v1)"
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Path to training data file (one sample per line)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/fgt_finetuned"),
        help="Directory to save model (default: models/fgt_finetuned)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available, else cpu)"
    )

    args = parser.parse_args()

    # Train model
    train_fgt_model(
        model_name=args.model,
        tape_dir=args.tape_dir,
        train_file=args.train_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
