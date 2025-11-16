#!/usr/bin/env python3
"""
Fine-tune a language model to understand FGT glyph tokens.

This script:
1. Loads a pre-trained LLM (e.g., GPT-2)
2. Adds glyph special tokens to vocabulary
3. Creates training data with hybrid tokenization
4. Fine-tunes the model to read/write glyph tokens
5. Evaluates on held-out test set
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from loguru import logger
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from llm_adapter import LLMAdapter
from llm_adapter.adapter import GlyphDataset


def load_training_texts(phrases_file: str, max_samples: int = 10000) -> List[str]:
    """
    Load training texts from phrases file.

    Args:
        phrases_file: Path to phrases JSONL file
        max_samples: Maximum number of samples to load

    Returns:
        List of text strings
    """
    logger.info(f"Loading training data from {phrases_file}...")

    texts = []
    with open(phrases_file, "r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line)
            texts.append(data["text"])

    logger.info(f"Loaded {len(texts)} training examples")
    return texts


def create_training_examples(
    texts: List[str],
    adapter: LLMAdapter,
    use_hybrid: bool = True,
    max_length: int = 128
) -> GlyphDataset:
    """
    Create training dataset.

    Args:
        texts: List of training texts
        adapter: LLM adapter
        use_hybrid: Whether to use hybrid encoding
        max_length: Maximum sequence length

    Returns:
        GlyphDataset ready for training
    """
    logger.info("Creating training examples...")

    examples = adapter.prepare_training_data(
        texts,
        use_hybrid=use_hybrid,
        max_length=max_length
    )

    logger.info(f"Created {len(examples)} training examples")
    return GlyphDataset(examples)


def run_finetuning(
    model_name: str = "gpt2",
    phrases_file: str = "data/phrases.jsonl",
    tape_db_path: str = "tape/v1/tape_index.db",
    output_dir: str = "models/fgt_gpt2",
    num_train_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_samples: int = 10000,
    use_hybrid: bool = True,
    save_steps: int = 500,
):
    """
    Run the fine-tuning process.

    Args:
        model_name: Base model to fine-tune
        phrases_file: Path to training phrases
        tape_db_path: Path to tape database
        output_dir: Directory to save fine-tuned model
        num_train_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_samples: Maximum training samples
        use_hybrid: Whether to use hybrid tokenization
        save_steps: Save checkpoint every N steps
    """
    logger.info("=" * 80)
    logger.info("GLYPH-AWARE LLM FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Base model: {model_name}")
    logger.info(f"Hybrid encoding: {use_hybrid}")
    logger.info(f"Training samples: {max_samples}")
    logger.info(f"Epochs: {num_train_epochs}")

    # Initialize adapter
    adapter = LLMAdapter(
        model_name=model_name,
        tape_db_path=tape_db_path if use_hybrid else None
    )

    # Load model and tokenizer
    model, tokenizer = adapter.load_model()

    # Load training data
    texts = load_training_texts(phrases_file, max_samples)

    # Split into train/val (90/10)
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    logger.info(f"Train set: {len(train_texts)} examples")
    logger.info(f"Val set: {len(val_texts)} examples")

    # Create datasets
    train_dataset = create_training_examples(train_texts, adapter, use_hybrid)
    val_dataset = create_training_examples(val_texts, adapter, use_hybrid)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to=["tensorboard"],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    train_result = trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    logger.info(f"Training loss: {train_result.training_loss:.4f}")

    # Evaluate
    logger.info("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()

    logger.info("Evaluation results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save model
    logger.info(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info = {
        "model_name": model_name,
        "use_hybrid": use_hybrid,
        "num_train_epochs": num_train_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "train_loss": train_result.training_loss,
        "eval_results": eval_results,
    }

    info_path = Path(output_dir) / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved training info to {info_path}")

    logger.info("\n" + "=" * 80)
    logger.info("FINE-TUNING COMPLETE!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 80)


def test_model(model_path: str, test_prompts: List[str]):
    """
    Test the fine-tuned model with sample prompts.

    Args:
        model_path: Path to fine-tuned model
        test_prompts: List of test prompts
    """
    logger.info("=" * 80)
    logger.info("TESTING FINE-TUNED MODEL")
    logger.info("=" * 80)

    adapter = LLMAdapter(model_name=model_path)
    adapter.load_model()

    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\nTest {i}:")
        logger.info(f"Prompt: {prompt}")

        output = adapter.generate(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9
        )

        logger.info(f"Output: {output}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for FGT glyphs")
    parser.add_argument("--model", default="gpt2", help="Base model name")
    parser.add_argument("--phrases", default="data/phrases.jsonl", help="Training phrases file")
    parser.add_argument("--tape", default="tape/v1/tape_index.db", help="Tape database")
    parser.add_argument("--output", default="models/fgt_gpt2", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max training samples")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid tokenization")
    parser.add_argument("--test-only", action="store_true", help="Only test existing model")
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(f"{args.output}/training.log", level="DEBUG")

    if args.test_only:
        # Test mode
        test_prompts = [
            "The fractal glyph tape is a",
            "Can you send me",
            "This is a demonstration of"
        ]
        test_model(args.output, test_prompts)
    else:
        # Training mode
        run_finetuning(
            model_name=args.model,
            phrases_file=args.phrases,
            tape_db_path=args.tape,
            output_dir=args.output,
            num_train_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            use_hybrid=not args.no_hybrid
        )


if __name__ == "__main__":
    main()
