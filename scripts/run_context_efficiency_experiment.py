#!/usr/bin/env python3
"""Run context window efficiency experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_adapter.adapter import FGTLLMAdapter
from src.tokenizer.hybrid import HybridTokenizer


def run_context_experiment(
    model_name: str,
    tape_dir: Path,
    test_texts: List[str],
    context_budgets: List[int],
    output_dir: Path,
    device: str = "cpu",
):
    """
    Run context window efficiency experiment.

    Compares how much context can be preserved with FGT vs baseline tokenization
    at various token budgets.

    Args:
        model_name: Pre-trained model name
        tape_dir: Path to tape directory
        test_texts: List of test texts
        context_budgets: List of token budgets to test
        output_dir: Directory to save results
        device: Device to run on
    """
    print(f"\n{'=' * 60}")
    print(f"Context Efficiency Experiment")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Tape: {tape_dir}")
    print(f"Test texts: {len(test_texts)}")
    print(f"Context budgets: {context_budgets}")
    print(f"{'=' * 60}\n")

    # Create adapter
    print("Loading model and tokenizer...")
    adapter = FGTLLMAdapter.from_pretrained(
        model_name,
        tape_dir=tape_dir,
        device=device,
    )

    # Results storage
    results = {
        "model": model_name,
        "tape_dir": str(tape_dir),
        "timestamp": datetime.now().isoformat(),
        "experiments": [],
    }

    # Run experiments for each text and budget
    for text_idx, text in enumerate(test_texts):
        print(f"\nProcessing text {text_idx + 1}/{len(test_texts)}...")
        print(f"Text length: {len(text)} chars")

        text_results = {
            "text_idx": text_idx,
            "text_length_chars": len(text),
            "budgets": [],
        }

        # Compute compression metrics
        compression_metrics = adapter.compute_context_compression(text)

        print(f"\nCompression metrics:")
        print(f"  Baseline tokens: {compression_metrics['baseline_tokens']}")
        print(f"  FGT tokens: {compression_metrics['fgt_tokens']}")
        print(f"  Glyph count: {compression_metrics['glyph_count']}")
        print(f"  Compression ratio: {compression_metrics['compression_ratio']:.3f}x")
        print(f"  Tokens saved: {compression_metrics['tokens_saved']}")
        print(f"  Percent saved: {compression_metrics['percent_saved']:.1f}%")

        text_results["compression"] = compression_metrics

        # Test different context budgets
        for budget in context_budgets:
            print(f"\n  Testing budget: {budget} tokens")

            # Encode with FGT (truncated)
            fgt_encoded = adapter.encode_input(text, max_length=budget, return_tensors=False)

            # Encode with baseline (truncated)
            baseline_encoded = adapter.tokenizer.base_tokenizer.encode(
                text,
                max_length=budget,
                truncation=True,
            )

            # Decode both to see what was preserved
            fgt_preserved_text = adapter.tokenizer.decode(
                fgt_encoded["input_ids"],
                expand_glyphs=True,
            )

            baseline_preserved_text = adapter.tokenizer.base_tokenizer.decode(
                baseline_encoded,
                skip_special_tokens=True,
            )

            # Measure how much was preserved
            fgt_chars_preserved = len(fgt_preserved_text)
            baseline_chars_preserved = len(baseline_preserved_text)

            fgt_preservation_ratio = fgt_chars_preserved / len(text) if len(text) > 0 else 0
            baseline_preservation_ratio = baseline_chars_preserved / len(text) if len(text) > 0 else 0

            budget_result = {
                "budget": budget,
                "fgt_tokens_used": len(fgt_encoded["input_ids"]),
                "fgt_glyph_count": fgt_encoded["glyph_count"],
                "fgt_chars_preserved": fgt_chars_preserved,
                "fgt_preservation_ratio": fgt_preservation_ratio,
                "baseline_tokens_used": len(baseline_encoded),
                "baseline_chars_preserved": baseline_chars_preserved,
                "baseline_preservation_ratio": baseline_preservation_ratio,
                "relative_improvement": (fgt_preservation_ratio - baseline_preservation_ratio)
                    / baseline_preservation_ratio if baseline_preservation_ratio > 0 else 0,
            }

            text_results["budgets"].append(budget_result)

            print(f"    FGT: {fgt_chars_preserved}/{len(text)} chars "
                  f"({fgt_preservation_ratio * 100:.1f}%)")
            print(f"    Baseline: {baseline_chars_preserved}/{len(text)} chars "
                  f"({baseline_preservation_ratio * 100:.1f}%)")
            print(f"    Improvement: {budget_result['relative_improvement'] * 100:+.1f}%")

        results["experiments"].append(text_results)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"context_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}\n")

    # Print summary
    print("\nSummary:")
    for exp in results["experiments"]:
        print(f"\nText {exp['text_idx'] + 1}:")
        print(f"  Compression ratio: {exp['compression']['compression_ratio']:.3f}x")
        for budget_result in exp["budgets"]:
            print(f"  Budget {budget_result['budget']}: "
                  f"{budget_result['relative_improvement'] * 100:+.1f}% improvement")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run context window efficiency experiments"
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
        "--test-file",
        type=Path,
        help="Path to file with test texts (one per line)"
    )

    parser.add_argument(
        "--test-texts",
        type=str,
        nargs="+",
        help="Test texts as command-line arguments"
    )

    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Token budgets to test (default: 128 256 512 1024)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/context_efficiency"),
        help="Directory to save results (default: results/context_efficiency)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu)"
    )

    args = parser.parse_args()

    # Load test texts
    if args.test_file:
        with open(args.test_file, "r", encoding="utf-8") as f:
            test_texts = [line.strip() for line in f if line.strip()]
    elif args.test_texts:
        test_texts = args.test_texts
    else:
        print("Error: Either --test-file or --test-texts must be provided", file=sys.stderr)
        sys.exit(1)

    # Run experiment
    run_context_experiment(
        model_name=args.model,
        tape_dir=args.tape_dir,
        test_texts=test_texts,
        context_budgets=args.budgets,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
