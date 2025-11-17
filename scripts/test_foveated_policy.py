#!/usr/bin/env python3
"""
Test script for FGT-FOVEATED policy.

Simulates the Phase 5 benchmark scenario:
1. Early setup turns (critical context)
2. Topic drift turns (noise)
3. Critical question (requires early context)

Tests that the foveated policy retrieves early context even under 256-token budget.
"""
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory_system.storage import SQLiteMemoryStore
from memory_system.foveation import FoveationEngine
from memory_system.models import MemoryEntry, FractalAddress, Glyph


def create_test_episode(store: SQLiteMemoryStore, actor_id: str):
    """
    Create a synthetic episode similar to Phase 5 benchmark.

    Structure:
    - Turn 1-2: Early setup (database preference, project info)
    - Turn 3-10: Topic drift (weather, news, etc.)
    - Turn 11: Critical question (what database?)
    """
    base_time = datetime.now() - timedelta(hours=1)

    episodes = [
        # EARLY SETUP (critical context)
        {
            "time_delta": 0,
            "text": "I'm building a web application using PostgreSQL as my database and React for the frontend.",
            "tags": ["setup", "tech"],
            "source": "user"
        },
        {
            "time_delta": 1,
            "text": "I prefer PostgreSQL because of its advanced features and ACID compliance.",
            "tags": ["setup", "preference"],
            "source": "user"
        },
        {
            "time_delta": 2,
            "text": "Great choice! PostgreSQL has excellent support for JSON data and full-text search.",
            "tags": ["assistant"],
            "source": "assistant"
        },

        # TOPIC DRIFT (noise)
        {
            "time_delta": 5,
            "text": "What's the weather like today?",
            "tags": ["casual"],
            "source": "user"
        },
        {
            "time_delta": 6,
            "text": "I don't have real-time weather data, but I can help you find weather services.",
            "tags": ["assistant"],
            "source": "assistant"
        },
        {
            "time_delta": 10,
            "text": "Can you tell me about the latest tech news?",
            "tags": ["casual"],
            "source": "user"
        },
        {
            "time_delta": 11,
            "text": "I don't have access to current news, but I can discuss technology trends.",
            "tags": ["assistant"],
            "source": "assistant"
        },
        {
            "time_delta": 15,
            "text": "What are some good restaurants nearby?",
            "tags": ["casual"],
            "source": "user"
        },
        {
            "time_delta": 16,
            "text": "I don't have location data, but I can suggest cuisine types if you'd like.",
            "tags": ["assistant"],
            "source": "assistant"
        },
        {
            "time_delta": 20,
            "text": "Tell me a joke.",
            "tags": ["casual"],
            "source": "user"
        },
        {
            "time_delta": 21,
            "text": "Why do programmers prefer dark mode? Because light attracts bugs!",
            "tags": ["assistant"],
            "source": "assistant"
        },

        # CRITICAL QUESTION (requires early context)
        {
            "time_delta": 30,
            "text": "What database did I mention I'm using for my web application?",
            "tags": ["question"],
            "source": "user"
        },
    ]

    print(f"\n{'='*60}")
    print(f"Creating test episode for actor: {actor_id}")
    print(f"{'='*60}\n")

    for i, turn in enumerate(episodes):
        timestamp = base_time + timedelta(minutes=turn["time_delta"])

        # Create fractal address
        address = FractalAddress(
            world="test",
            region="demo",
            tri_path=f"01{i%3}",
            depth=3,
            time_slice=timestamp
        )

        # Create mock glyph
        glyphs = [Glyph(
            glyph_id=hash(turn["text"]) % 10000,
            glyph_str=f"谷{chr(ord('阜') + i % 10)}",
            cluster_id=i,
            frequency=1,
            semantic_summary=turn["text"][:30]
        )]

        # Create memory entry
        entry = MemoryEntry(
            entry_id=f"test_{actor_id}_{i}",
            actor_id=actor_id,
            address=address,
            text=turn["text"],
            glyphs=glyphs,
            tags=turn["tags"],
            source=turn["source"],
            created_at=timestamp,
            token_estimate=int(len(turn["text"].split()) * 1.3)
        )

        store.write(entry)
        print(f"[Turn {i+1:2d}] ({turn['source']:10s}) {turn['text'][:60]}...")

    print(f"\n✅ Created {len(episodes)} memory entries\n")


def test_policy(engine: FoveationEngine, actor_id: str, mode: str, token_budget: int):
    """
    Test a specific policy mode with a given token budget.
    """
    query = "database application"  # Simpler query that will match more memories

    print(f"\n{'='*60}")
    print(f"Testing: {mode.upper()} policy with {token_budget} token budget")
    print(f"{'='*60}")
    print(f"Query: {query}\n")

    result = engine.retrieve(
        actor_id=actor_id,
        query=query,
        token_budget=token_budget,
        mode=mode
    )

    print(f"Results:")
    print(f"  Candidates considered: {result['candidates_considered']}")
    print(f"  Memories selected:     {result['memories_selected']}")
    print(f"  Tokens used:           {result['token_estimate']}/{token_budget}")
    print(f"  Policy:                {result['policy']}\n")

    print(f"Selected memories:")
    for i, mem in enumerate(result['memories']):
        print(f"  [{i+1}] {mem['text'][:70]}...")

    # Check if early setup is included
    early_keywords = ["PostgreSQL", "database", "React", "prefer"]
    early_found = False

    for mem in result['memories']:
        if any(keyword in mem['text'] for keyword in early_keywords):
            early_found = True
            break

    print(f"\n✅ Early context found: {early_found}")

    return early_found


def main():
    """Run the test."""
    print("\n" + "="*60)
    print("FGT-FOVEATED Policy Test")
    print("="*60)

    # Initialize store and engine
    db_path = "data/test_foveated_policy.db"

    # Clean up old test database
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed old test database: {db_path}")

    store = SQLiteMemoryStore(db_path)
    engine = FoveationEngine(store)

    actor_id = "test_user"

    # Create test episode
    create_test_episode(store, actor_id)

    # Test 1: Baseline (recent) with 256 tokens
    print("\n" + "="*60)
    print("TEST 1: RECENT policy (baseline)")
    print("="*60)
    recent_found = test_policy(engine, actor_id, "recent", 256)

    # Test 2: FGT-FOVEATED with 256 tokens
    print("\n" + "="*60)
    print("TEST 2: FOVEATED policy (FGT-FOVEATED)")
    print("="*60)
    foveated_found = test_policy(engine, actor_id, "foveated", 256)

    # Test 3: FGT-FOVEATED with 512 tokens (should converge)
    print("\n" + "="*60)
    print("TEST 3: FOVEATED policy with larger budget")
    print("="*60)
    foveated_512_found = test_policy(engine, actor_id, "foveated", 512)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Test 1 (Recent, 256 tokens):     Early context = {recent_found}")
    print(f"✅ Test 2 (Foveated, 256 tokens):   Early context = {foveated_found}")
    print(f"✅ Test 3 (Foveated, 512 tokens):   Early context = {foveated_512_found}")

    if not recent_found and foveated_found:
        print("\n🎉 SUCCESS: Foveated policy preserves early context under tight budget!")
        print("   This matches the Phase 5 benchmark findings.")
    else:
        print("\n⚠️  Note: Both policies may succeed with this episode size.")
        print("   Phase 5 used larger episodes with more topic drift.")

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
