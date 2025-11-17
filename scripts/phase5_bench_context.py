#!/usr/bin/env python3
"""
Phase 5 Context Efficiency Benchmark

Compares RAW-TRUNCATE vs FGT-CONTEXT under fixed token budgets.

Strategies:
- RAW-TRUNCATE: Take last N tokens of conversation history
- FGT-CONTEXT: Use FGMS with foveation to select relevant memories

Evaluation:
- Checks if context preserves information needed to answer questions about early conversation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
import requests
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict


# FGMS API configuration
FGMS_BASE_URL = "http://localhost:8000"


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
    RAW-TRUNCATE strategy: Take last N tokens from conversation.

    Returns:
        (context_turns, tokens_used)
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


def fgt_context_strategy(
    episode: Dict[str, Any],
    token_budget: int,
    question: str,
    actor_id: str = "bench_actor"
) -> Tuple[Optional[List[Dict[str, str]]], int, Dict[str, Any]]:
    """
    FGT-CONTEXT strategy: Use FGMS to retrieve relevant context.

    Steps:
    1. Write all episode turns to FGMS as memories
    2. Query FGMS with the question and token budget
    3. Return retrieved context

    Returns:
        (context_turns, tokens_used, metadata)
    """
    try:
        # Clear any existing memories for this actor (fresh start for each episode)
        # In production, we'd use unique actor IDs per episode

        # Write all turns to FGMS
        for i, turn in enumerate(episode["turns"]):
            role_tag = turn["role"]
            write_response = requests.post(
                f"{FGMS_BASE_URL}/api/memory/write",
                json={
                    "actor_id": f"{actor_id}_{episode['episode_id']}",
                    "text": turn["text"],
                    "tags": [role_tag, f"turn_{i}"],
                    "source": role_tag
                },
                timeout=5
            )

            if write_response.status_code != 200:
                print(f"Warning: Failed to write turn {i}: {write_response.text}")
                return None, 0, {"error": "write_failed"}

        # Read with foveation
        read_response = requests.post(
            f"{FGMS_BASE_URL}/api/memory/read",
            json={
                "actor_id": f"{actor_id}_{episode['episode_id']}",
                "query": question,
                "token_budget": token_budget,
                "mode": "mixed"  # Blend of recent and relevant
            },
            timeout=5
        )

        if read_response.status_code != 200:
            print(f"Warning: Failed to read memories: {read_response.text}")
            return None, 0, {"error": "read_failed"}

        read_data = read_response.json()

        # Convert memories back to turns format
        context_turns = []
        for memory in read_data.get("memories", []):
            # Determine role from tags
            tags = memory.get("tags", [])
            role = "user" if "user" in tags else "assistant"

            context_turns.append({
                "role": role,
                "text": memory["text"]
            })

        metadata = {
            "memories_selected": read_data.get("memories_selected", 0),
            "candidates_considered": read_data.get("candidates_considered", 0),
            "policy": read_data.get("policy", "unknown"),
            "addresses": read_data.get("addresses", [])
        }

        return context_turns, read_data.get("token_estimate", 0), metadata

    except requests.exceptions.RequestException as e:
        print(f"Warning: FGMS API request failed: {e}")
        return None, 0, {"error": str(e)}


def evaluate_context(
    context_turns: List[Dict[str, str]],
    question: str,
    answer: str,
    episode: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate if context contains information needed to answer the question.

    Checks:
    1. Answer keywords present in context
    2. Turn where answer was mentioned is included
    3. Sufficient context completeness
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

    # Check 3: Context completeness (what fraction of episode is in context)
    context_completeness = len(context_turns) / len(episode["turns"])

    # Success if both answer keywords and relevant turn are present
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
    num_episodes: Optional[int] = None,
    use_fgms: bool = True
):
    """Run the context efficiency benchmark."""
    print("=" * 70)
    print("Phase 5 Context Efficiency Benchmark")
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

    # Check FGMS availability
    fgms_available = False
    if use_fgms:
        try:
            health_check = requests.get(f"{FGMS_BASE_URL}/", timeout=2)
            fgms_available = health_check.status_code == 200
            print(f"✓ FGMS API available at {FGMS_BASE_URL}")
        except requests.exceptions.RequestException:
            print(f"✗ FGMS API not available at {FGMS_BASE_URL}")
            print("  FGT-CONTEXT strategy will be skipped")
            fgms_available = False

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
            "fgt_context": {
                "correct": 0,
                "total_tokens": 0,
                "avg_completeness": 0.0,
                "errors": 0
            },
            "episodes": []
        }

        for i, episode in enumerate(episodes):
            episode_id = episode["episode_id"]
            question = episode["question"]
            answer = episode["answer"]

            # RAW-TRUNCATE
            raw_turns, raw_tokens = raw_truncate_strategy(episode, budget)
            raw_eval = evaluate_context(raw_turns, question, answer, episode)

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
                }
            }

            if raw_eval["success"]:
                budget_results["raw_truncate"]["correct"] += 1
            budget_results["raw_truncate"]["total_tokens"] += raw_tokens
            budget_results["raw_truncate"]["avg_completeness"] += raw_eval["context_completeness"]

            # FGT-CONTEXT (if available)
            if fgms_available:
                fgt_turns, fgt_tokens, fgt_metadata = fgt_context_strategy(
                    episode, budget, question
                )

                if fgt_turns is not None:
                    fgt_eval = evaluate_context(fgt_turns, question, answer, episode)

                    episode_result["fgt_context"] = {
                        "tokens": fgt_tokens,
                        "turns_retrieved": len(fgt_turns),
                        "success": fgt_eval["success"],
                        "answer_keywords_present": fgt_eval["answer_keywords_present"],
                        "relevant_turn_included": fgt_eval["relevant_turn_included"],
                        "completeness": fgt_eval["context_completeness"],
                        "memories_selected": fgt_metadata.get("memories_selected", 0),
                        "candidates_considered": fgt_metadata.get("candidates_considered", 0)
                    }

                    if fgt_eval["success"]:
                        budget_results["fgt_context"]["correct"] += 1
                    budget_results["fgt_context"]["total_tokens"] += fgt_tokens
                    budget_results["fgt_context"]["avg_completeness"] += fgt_eval["context_completeness"]
                else:
                    episode_result["fgt_context"] = {"error": fgt_metadata.get("error", "unknown")}
                    budget_results["fgt_context"]["errors"] += 1

            budget_results["episodes"].append(episode_result)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(episodes)} episodes...")

        # Calculate averages
        num_episodes_val = len(episodes)
        budget_results["raw_truncate"]["avg_tokens"] = (
            budget_results["raw_truncate"]["total_tokens"] / num_episodes_val
        )
        budget_results["raw_truncate"]["success_rate"] = (
            budget_results["raw_truncate"]["correct"] / num_episodes_val
        )
        budget_results["raw_truncate"]["avg_completeness"] = (
            budget_results["raw_truncate"]["avg_completeness"] / num_episodes_val
        )

        if fgms_available:
            valid_fgt = num_episodes_val - budget_results["fgt_context"]["errors"]
            if valid_fgt > 0:
                budget_results["fgt_context"]["avg_tokens"] = (
                    budget_results["fgt_context"]["total_tokens"] / valid_fgt
                )
                budget_results["fgt_context"]["success_rate"] = (
                    budget_results["fgt_context"]["correct"] / num_episodes_val
                )
                budget_results["fgt_context"]["avg_completeness"] = (
                    budget_results["fgt_context"]["avg_completeness"] / valid_fgt
                )

        # Print summary
        print(f"\nResults:")
        print(f"  RAW-TRUNCATE:")
        print(f"    Success rate: {budget_results['raw_truncate']['success_rate']:.1%} "
              f"({budget_results['raw_truncate']['correct']}/{num_episodes_val})")
        print(f"    Avg tokens:   {budget_results['raw_truncate']['avg_tokens']:.0f}")
        print(f"    Avg complete: {budget_results['raw_truncate']['avg_completeness']:.1%}")

        if fgms_available and valid_fgt > 0:
            print(f"  FGT-CONTEXT:")
            print(f"    Success rate: {budget_results['fgt_context']['success_rate']:.1%} "
                  f"({budget_results['fgt_context']['correct']}/{num_episodes_val})")
            print(f"    Avg tokens:   {budget_results['fgt_context']['avg_tokens']:.0f}")
            print(f"    Avg complete: {budget_results['fgt_context']['avg_completeness']:.1%}")

            improvement = (
                (budget_results['fgt_context']['success_rate'] -
                 budget_results['raw_truncate']['success_rate']) * 100
            )
            print(f"    Improvement:  {improvement:+.1f} percentage points")

        all_results.append(budget_results)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "num_episodes": len(episodes),
        "token_budgets": token_budgets,
        "fgms_available": fgms_available,
        "results": all_results
    }

    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to {output_path}")
    print('=' * 70)

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Context Efficiency Benchmark")
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
        default=[512, 1024, 2048],
        help="Token budgets to test"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/phase5/context_bench.json"),
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to use (default: all)"
    )
    parser.add_argument(
        "--no-fgms",
        action="store_true",
        help="Skip FGT-CONTEXT strategy (only run RAW-TRUNCATE)"
    )

    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        token_budgets=args.budget,
        output_path=args.out,
        num_episodes=args.num_episodes,
        use_fgms=not args.no_fgms
    )


if __name__ == "__main__":
    main()
