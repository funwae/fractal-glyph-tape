#!/usr/bin/env python3
"""
Context efficiency benchmark for FGMS

Compares:
- Baseline: Raw truncated context (last N tokens)
- FGT: Glyph-coded history with foveation

Tests whether FGT preserves more relevant early information
within the same token budget.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import random


def create_episode_data() -> List[Dict[str, Any]]:
    """
    Create sample multi-turn episodes with questions.

    Each episode has:
    - Multiple turns of conversation
    - A decision or fact mentioned early on
    - A question later that requires that early info
    """
    episodes = [
        {
            "id": "ep1",
            "turns": [
                {"user": "I prefer using PostgreSQL for our database.", "assistant": "Great choice! PostgreSQL is robust."},
                {"user": "Can you help me design the schema?", "assistant": "Sure, what tables do you need?"},
                {"user": "We need users, posts, and comments.", "assistant": "I'll create a normalized schema."},
                {"user": "Also add tags and categories.", "assistant": "Will do."},
                {"user": "Make sure to add indexes.", "assistant": "I'll add appropriate indexes."},
            ],
            "question": "What database did I choose earlier?",
            "answer": "PostgreSQL",
            "turn_mentioned": 0
        },
        {
            "id": "ep2",
            "turns": [
                {"user": "We're deploying to AWS Lambda.", "assistant": "Lambda is great for serverless."},
                {"user": "What's the best runtime?", "assistant": "For Python, use Python 3.11."},
                {"user": "How about memory settings?", "assistant": "Start with 512MB and adjust."},
                {"user": "What about timeout?", "assistant": "30 seconds is a good default."},
                {"user": "Should I use layers?", "assistant": "Yes, for shared dependencies."},
            ],
            "question": "Where are we deploying this?",
            "answer": "AWS Lambda",
            "turn_mentioned": 0
        },
        {
            "id": "ep3",
            "turns": [
                {"user": "I want to use React for the frontend.", "assistant": "React is excellent."},
                {"user": "Should I use TypeScript?", "assistant": "Highly recommended!"},
                {"user": "What about state management?", "assistant": "Try Redux Toolkit or Zustand."},
                {"user": "I'll go with Zustand.", "assistant": "Good choice, simpler than Redux."},
                {"user": "How do I handle forms?", "assistant": "React Hook Form is great."},
            ],
            "question": "Which state management library did I choose?",
            "answer": "Zustand",
            "turn_mentioned": 3
        },
        {
            "id": "ep4",
            "turns": [
                {"user": "Set the API timeout to 5000ms.", "assistant": "Noted, 5 second timeout."},
                {"user": "Also enable retry logic.", "assistant": "I'll add exponential backoff."},
                {"user": "Max retries should be 3.", "assistant": "Setting max_retries=3."},
                {"user": "What about rate limiting?", "assistant": "I'll add a rate limiter."},
                {"user": "Log all requests.", "assistant": "Will add comprehensive logging."},
            ],
            "question": "What timeout did I set for the API?",
            "answer": "5000ms or 5 seconds",
            "turn_mentioned": 0
        },
        {
            "id": "ep5",
            "turns": [
                {"user": "Use JWT tokens for authentication.", "assistant": "JWT is a solid choice."},
                {"user": "Set expiry to 24 hours.", "assistant": "24 hour expiry configured."},
                {"user": "Store refresh tokens in Redis.", "assistant": "Good idea for scalability."},
                {"user": "Enable 2FA for admin users.", "assistant": "I'll add TOTP support."},
                {"user": "How should we handle password resets?", "assistant": "Email-based reset links."},
            ],
            "question": "Where should we store refresh tokens?",
            "answer": "Redis",
            "turn_mentioned": 2
        }
    ]

    return episodes


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1.3 tokens per word)."""
    return int(len(text.split()) * 1.3)


def baseline_truncated_context(
    episode: Dict[str, Any],
    token_budget: int
) -> Tuple[str, int]:
    """
    Baseline approach: Truncate to last N tokens.

    Returns:
        (context_text, tokens_used)
    """
    turns = episode["turns"]
    context_parts = []
    tokens_used = 0

    # Start from most recent and work backwards
    for turn in reversed(turns):
        turn_text = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        turn_tokens = estimate_tokens(turn_text)

        if tokens_used + turn_tokens <= token_budget:
            context_parts.insert(0, turn_text)
            tokens_used += turn_tokens
        else:
            break

    return "\n\n".join(context_parts), tokens_used


def fgt_foveated_context(
    episode: Dict[str, Any],
    token_budget: int,
    question: str
) -> Tuple[str, int]:
    """
    FGT approach: Use foveation to include relevant early memories.

    Strategy:
    - Reserve 50% budget for recent context
    - Reserve 50% budget for relevant early context

    Returns:
        (context_text, tokens_used)
    """
    turns = episode["turns"]
    recent_budget = token_budget // 2
    relevant_budget = token_budget - recent_budget

    # Get recent context (last few turns)
    recent_parts = []
    recent_tokens = 0
    for turn in reversed(turns):
        turn_text = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        turn_tokens = estimate_tokens(turn_text)

        if recent_tokens + turn_tokens <= recent_budget:
            recent_parts.insert(0, turn_text)
            recent_tokens += turn_tokens
        else:
            break

    # Get relevant early context (keyword matching for demo)
    question_terms = set(question.lower().split())
    relevant_parts = []
    relevant_tokens = 0

    for i, turn in enumerate(turns):
        # Skip if already in recent context
        if i >= len(turns) - len(recent_parts):
            continue

        turn_text = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        turn_terms = set((turn['user'] + " " + turn['assistant']).lower().split())

        # Check relevance (simple keyword overlap)
        overlap = question_terms & turn_terms
        if len(overlap) > 0:
            turn_tokens = estimate_tokens(turn_text)
            if relevant_tokens + turn_tokens <= relevant_budget:
                relevant_parts.append((i, turn_text, turn_tokens))
                relevant_tokens += turn_tokens

    # Combine: [relevant early] ... [recent]
    context_parts = []
    if relevant_parts:
        context_parts.extend([text for _, text, _ in relevant_parts])
        if len(relevant_parts) < len(turns) - len(recent_parts):
            context_parts.append("[... earlier conversation ...]")

    context_parts.extend(recent_parts)

    return "\n\n".join(context_parts), recent_tokens + relevant_tokens


def evaluate_context_quality(
    context: str,
    episode: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate if the context contains the answer.

    Checks:
    - Does context include the turn where answer was mentioned?
    - Keyword presence
    """
    answer = episode["answer"].lower()
    context_lower = context.lower()

    # Check if answer keywords are in context
    answer_keywords = set(answer.split())
    context_keywords = set(context_lower.split())
    keyword_match = len(answer_keywords & context_keywords) > 0

    # Check if the specific turn is included
    turn_mentioned = episode["turn_mentioned"]
    mentioned_turn = episode["turns"][turn_mentioned]
    turn_in_context = (mentioned_turn["user"].lower() in context_lower or
                      mentioned_turn["assistant"].lower() in context_lower)

    return {
        "keyword_match": keyword_match,
        "turn_in_context": turn_in_context,
        "answer_preserved": keyword_match and turn_in_context
    }


def run_benchmark():
    """Run context efficiency benchmark."""
    print("=" * 70)
    print("FGMS Context Efficiency Benchmark")
    print("=" * 70)

    episodes = create_episode_data()
    token_budgets = [512, 1024, 2048]

    print(f"\nEvaluating {len(episodes)} episodes with budgets: {token_budgets}")

    all_results = []

    for budget in token_budgets:
        print(f"\n{'=' * 70}")
        print(f"Token Budget: {budget}")
        print('=' * 70)

        baseline_correct = 0
        fgt_correct = 0

        episode_results = []

        for episode in episodes:
            # Baseline
            baseline_context, baseline_tokens = baseline_truncated_context(episode, budget)
            baseline_eval = evaluate_context_quality(baseline_context, episode)

            # FGT
            fgt_context, fgt_tokens = fgt_foveated_context(episode, budget, episode["question"])
            fgt_eval = evaluate_context_quality(fgt_context, episode)

            if baseline_eval["answer_preserved"]:
                baseline_correct += 1
            if fgt_eval["answer_preserved"]:
                fgt_correct += 1

            episode_results.append({
                "episode_id": episode["id"],
                "question": episode["question"],
                "answer": episode["answer"],
                "baseline": {
                    "tokens": baseline_tokens,
                    "answer_preserved": baseline_eval["answer_preserved"],
                    "turn_in_context": baseline_eval["turn_in_context"]
                },
                "fgt": {
                    "tokens": fgt_tokens,
                    "answer_preserved": fgt_eval["answer_preserved"],
                    "turn_in_context": fgt_eval["turn_in_context"]
                }
            })

        print(f"\nBaseline (truncated): {baseline_correct}/{len(episodes)} correct")
        print(f"FGT (foveated):       {fgt_correct}/{len(episodes)} correct")

        improvement = ((fgt_correct - baseline_correct) / len(episodes)) * 100 if len(episodes) > 0 else 0
        print(f"Improvement:          {improvement:+.1f}%")

        all_results.append({
            "token_budget": budget,
            "baseline_correct": baseline_correct,
            "fgt_correct": fgt_correct,
            "total_episodes": len(episodes),
            "improvement_pct": improvement,
            "episodes": episode_results
        })

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"\n{'Budget':<10} {'Baseline':<15} {'FGT':<15} {'Improvement':<15}")
    print("-" * 70)

    for result in all_results:
        budget = result["token_budget"]
        baseline = f"{result['baseline_correct']}/{result['total_episodes']}"
        fgt = f"{result['fgt_correct']}/{result['total_episodes']}"
        improvement = f"{result['improvement_pct']:+.1f}%"

        print(f"{budget:<10} {baseline:<15} {fgt:<15} {improvement:<15}")

    # Save report
    report_path = Path("reports/context_bench.json")
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results
        }, f, indent=2)

    print(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    run_benchmark()
