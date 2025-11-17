#!/usr/bin/env python3
"""
Phase 5 Standalone Context Efficiency Benchmark

Self-contained benchmark that doesn't require FGMS API.
Compares RAW-TRUNCATE vs simulated FGT-CONTEXT strategies.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load episodes from JSONL file."""
    episodes = []
    with open(dataset_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1.3 tokens per word)."""
    return int(len(text.split()) * 1.3)


def raw_truncate_strategy(
    episode: Dict[str, Any],
    token_budget: int
) -> Tuple[List[Dict[str, str]], int]:
    """
    RAW-TRUNCATE: Take last N tokens from conversation.

    This is the baseline - just grab the most recent conversation
    until we hit the token budget.
    """
    turns = episode["turns"]
    selected_turns = []
    tokens_used = 0

    # Work backwards from most recent
    for turn in reversed(turns):
        turn_tokens = estimate_tokens(turn["text"])

        if tokens_used + turn_tokens <= token_budget:
            selected_turns.insert(0, turn)
            tokens_used += turn_tokens
        else:
            break

    return selected_turns, tokens_used


def fgt_foveated_strategy(
    episode: Dict[str, Any],
    token_budget: int,
    question: str
) -> Tuple[List[Dict[str, str]], int]:
    """
    FGT-FOVEATED: Improved foveation strategy.

    Strategy:
    - 30% budget for very early context (first few turns - often contain key info)
    - 30% budget for semantically relevant context (keyword matching)
    - 40% budget for recent context (recency)

    This simulates a multi-scale foveation approach.
    """
    turns = episode["turns"]

    # Allocate budgets
    early_budget = int(token_budget * 0.3)
    semantic_budget = int(token_budget * 0.3)
    recent_budget = token_budget - early_budget - semantic_budget

    selected_turns = []
    used_indices = set()

    # 1. Get VERY early context (first few turns often have critical setup info)
    early_tokens = 0
    for i, turn in enumerate(turns[:5]):  # First 5 turns max
        turn_tokens = estimate_tokens(turn["text"])
        if early_tokens + turn_tokens <= early_budget:
            selected_turns.append((i, turn, "early"))
            used_indices.add(i)
            early_tokens += turn_tokens
        else:
            break

    # 2. Get semantically relevant context (keyword-based)
    question_terms = set(question.lower().split())
    semantic_candidates = []

    for i, turn in enumerate(turns):
        if i in used_indices:
            continue

        turn_text = turn["text"].lower()
        turn_terms = set(turn_text.split())
        overlap = question_terms & turn_terms

        if len(overlap) > 0:
            # Relevance score: keyword overlap + recency bonus
            recency_bonus = 1.0 / (len(turns) - i + 1)  # Slight preference for more recent
            relevance_score = len(overlap) + recency_bonus
            semantic_candidates.append((i, turn, relevance_score))

    # Sort by relevance and take top ones within budget
    semantic_candidates.sort(key=lambda x: -x[2])
    semantic_tokens = 0

    for i, turn, score in semantic_candidates:
        turn_tokens = estimate_tokens(turn["text"])
        if semantic_tokens + turn_tokens <= semantic_budget:
            selected_turns.append((i, turn, "semantic"))
            used_indices.add(i)
            semantic_tokens += turn_tokens

    # 3. Get recent context
    recent_turns = []
    recent_tokens = 0

    for i, turn in enumerate(reversed(turns)):
        real_idx = len(turns) - 1 - i
        if real_idx in used_indices:
            continue

        turn_tokens = estimate_tokens(turn["text"])
        if recent_tokens + turn_tokens <= recent_budget:
            recent_turns.insert(0, (real_idx, turn, "recent"))
            used_indices.add(real_idx)
            recent_tokens += turn_tokens
        else:
            break

    selected_turns.extend(recent_turns)

    # Sort by original position to maintain temporal order
    selected_turns.sort(key=lambda x: x[0])

    # Extract just the turns
    final_turns = [t[1] for t in selected_turns]
    total_tokens = early_tokens + semantic_tokens + recent_tokens

    return final_turns, total_tokens


def evaluate_context(
    context_turns: List[Dict[str, str]],
    question: str,
    answer: str,
    episode: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate whether context contains the information needed.

    Checks:
    1. Answer keywords present in context
    2. Turn where answer was mentioned is included
    """
    if not context_turns:
        return {
            "answer_keywords_present": False,
            "relevant_turn_included": False,
            "context_completeness": 0.0,
            "success": False
        }

    # Build context text
    context_text = " ".join([turn["text"] for turn in context_turns]).lower()

    # Check 1: Answer keywords
    answer_keywords = set(answer.lower().split())
    context_keywords = set(context_text.split())
    keywords_present = len(answer_keywords & context_keywords) > 0

    # Check 2: Relevant turn included
    turn_mentioned = episode.get("turn_mentioned", 0)
    if turn_mentioned < len(episode["turns"]):
        relevant_turn_text = episode["turns"][turn_mentioned]["text"].lower()
        turn_included = relevant_turn_text in context_text
    else:
        turn_included = False

    # Context completeness
    context_completeness = len(context_turns) / len(episode["turns"])

    # Success if both checks pass
    success = keywords_present and turn_included

    return {
        "answer_keywords_present": keywords_present,
        "relevant_turn_included": turn_included,
        "context_completeness": context_completeness,
        "success": success
    }


def run_benchmark(
    dataset_path: Path,
    token_budgets: List[int],
    output_path: Path,
    num_episodes: int = None
):
    """Run the standalone benchmark."""
    print("=" * 70)
    print("Phase 5 Standalone Context Efficiency Benchmark")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    all_episodes = load_dataset(dataset_path)

    if num_episodes:
        episodes = all_episodes[:num_episodes]
        print(f"Using first {num_episodes} episodes (out of {len(all_episodes)})")
    else:
        episodes = all_episodes
        print(f"Using all {len(episodes)} episodes")

    all_results = []

    for budget in token_budgets:
        print(f"\n{'=' * 70}")
        print(f"Token Budget: {budget}")
        print('=' * 70)

        budget_results = {
            "token_budget": budget,
            "num_episodes": len(episodes),
            "raw_truncate": {
                "correct": 0,
                "total_tokens": 0,
                "avg_completeness": 0.0
            },
            "fgt_foveated": {
                "correct": 0,
                "total_tokens": 0,
                "avg_completeness": 0.0
            },
            "episodes": []
        }

        for i, episode in enumerate(episodes):
            episode_id = episode["episode_id"]

            # Handle both single question and multi-question formats
            if "question" in episode:
                # Single question format
                questions_to_test = [(episode["question"], episode["answer"])]
            elif "questions" in episode:
                # Multi-question format
                questions_to_test = [(q["question"], q["answer"]) for q in episode["questions"]]
            else:
                # Skip episodes without questions
                continue

            # Test each question (for multi-question episodes, we test all)
            for question, answer in questions_to_test:
                # RAW-TRUNCATE
                raw_turns, raw_tokens = raw_truncate_strategy(episode, budget)
                raw_eval = evaluate_context(raw_turns, question, answer, episode)

                # FGT-FOVEATED
                fgt_turns, fgt_tokens = fgt_foveated_strategy(episode, budget, question)
                fgt_eval = evaluate_context(fgt_turns, question, answer, episode)

                # Record results
                episode_result = {
                    "episode_id": episode_id,
                    "question": question,
                    "answer": answer,
                    "num_turns": len(episode["turns"]),
                    "raw_truncate": {
                        "tokens": raw_tokens,
                        "turns_retrieved": len(raw_turns),
                        "success": raw_eval["success"],
                        "answer_keywords_present": raw_eval["answer_keywords_present"],
                        "relevant_turn_included": raw_eval["relevant_turn_included"],
                        "completeness": raw_eval["context_completeness"]
                    },
                    "fgt_foveated": {
                        "tokens": fgt_tokens,
                        "turns_retrieved": len(fgt_turns),
                        "success": fgt_eval["success"],
                        "answer_keywords_present": fgt_eval["answer_keywords_present"],
                        "relevant_turn_included": fgt_eval["relevant_turn_included"],
                        "completeness": fgt_eval["context_completeness"]
                    }
                }

                if raw_eval["success"]:
                    budget_results["raw_truncate"]["correct"] += 1
                budget_results["raw_truncate"]["total_tokens"] += raw_tokens
                budget_results["raw_truncate"]["avg_completeness"] += raw_eval["context_completeness"]

                if fgt_eval["success"]:
                    budget_results["fgt_foveated"]["correct"] += 1
                budget_results["fgt_foveated"]["total_tokens"] += fgt_tokens
                budget_results["fgt_foveated"]["avg_completeness"] += fgt_eval["context_completeness"]

                budget_results["episodes"].append(episode_result)

            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(episodes)} episodes...")

        # Calculate averages
        num_tests = len(budget_results["episodes"])  # Number of actual question tests

        if num_tests == 0:
            print("Warning: No test questions found in dataset!")
            continue

        budget_results["raw_truncate"]["avg_tokens"] = budget_results["raw_truncate"]["total_tokens"] / num_tests
        budget_results["raw_truncate"]["success_rate"] = budget_results["raw_truncate"]["correct"] / num_tests
        budget_results["raw_truncate"]["avg_completeness"] /= num_tests

        budget_results["fgt_foveated"]["avg_tokens"] = budget_results["fgt_foveated"]["total_tokens"] / num_tests
        budget_results["fgt_foveated"]["success_rate"] = budget_results["fgt_foveated"]["correct"] / num_tests
        budget_results["fgt_foveated"]["avg_completeness"] /= num_tests

        # Print summary
        print(f"\nResults ({num_tests} test questions):")
        print(f"  RAW-TRUNCATE:")
        print(f"    Success rate: {budget_results['raw_truncate']['success_rate']:.1%} "
              f"({budget_results['raw_truncate']['correct']}/{num_tests})")
        print(f"    Avg tokens:   {budget_results['raw_truncate']['avg_tokens']:.0f}")
        print(f"    Avg complete: {budget_results['raw_truncate']['avg_completeness']:.1%}")

        print(f"  FGT-FOVEATED:")
        print(f"    Success rate: {budget_results['fgt_foveated']['success_rate']:.1%} "
              f"({budget_results['fgt_foveated']['correct']}/{num_tests})")
        print(f"    Avg tokens:   {budget_results['fgt_foveated']['avg_tokens']:.0f}")
        print(f"    Avg complete: {budget_results['fgt_foveated']['avg_completeness']:.1%}")

        improvement = (budget_results['fgt_foveated']['success_rate'] -
                      budget_results['raw_truncate']['success_rate']) * 100
        print(f"    Improvement:  {improvement:+.1f} percentage points")

        all_results.append(budget_results)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "num_episodes": len(episodes),
        "token_budgets": token_budgets,
        "fgms_available": False,
        "note": "Standalone benchmark with simulated foveation strategy",
        "results": all_results
    }

    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to {output_path}")
    print('=' * 70)

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Standalone Benchmark")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/phase5/dialog_test.jsonl"),
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "--budget",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Token budgets to test"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/phase5/standalone_bench.json"),
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to use (default: all)"
    )

    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        token_budgets=args.budget,
        output_path=args.out,
        num_episodes=args.num_episodes
    )


if __name__ == "__main__":
    main()
