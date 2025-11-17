#!/usr/bin/env python3
"""
Smoke test script for FGMS API

Tests:
1. Write memory
2. Read memory
3. Stats endpoint
4. Agent chat
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests
import json
from datetime import datetime


API_URL = "http://localhost:8001"
ACTOR_ID = "test_user"


def test_write_memory():
    """Test writing a memory."""
    print("\n=== Test 1: Write Memory ===")

    data = {
        "actor_id": ACTOR_ID,
        "text": "Today I started using a fractal glyph memory service for my AI projects.",
        "tags": ["devlog", "ai"],
        "source": "user"
    }

    response = requests.post(f"{API_URL}/api/memory/write", json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Memory written successfully")
        print(f"  Entry ID: {result['entry_id']}")
        print(f"  Address: {result['address']}")
        print(f"  Token estimate: {result['token_estimate']}")
        return result
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return None


def test_read_memory():
    """Test reading memories."""
    print("\n=== Test 2: Read Memory ===")

    data = {
        "actor_id": ACTOR_ID,
        "query": "fractal glyph AI projects",
        "token_budget": 2048,
        "mode": "mixed"
    }

    response = requests.post(f"{API_URL}/api/memory/read", json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Memory read successfully")
        print(f"  Memories selected: {result['memories_selected']}")
        print(f"  Candidates considered: {result['candidates_considered']}")
        print(f"  Token estimate: {result['token_estimate']}")
        print(f"  Policy: {result['policy']}")
        print(f"  Addresses: {len(result['addresses'])}")
        print(f"  Glyphs: {len(result['glyphs'])}")

        if result['memories']:
            print(f"\n  Sample memory:")
            print(f"    Text: {result['memories'][0]['text'][:80]}...")
        return result
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return None


def test_stats():
    """Test stats endpoint."""
    print("\n=== Test 3: Stats ===")

    response = requests.get(f"{API_URL}/api/memory/stats?actor_id={ACTOR_ID}")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Stats retrieved successfully")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return result
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return None


def test_agent_chat():
    """Test agent chat endpoint."""
    print("\n=== Test 4: Agent Chat ===")

    data = {
        "actor_id": ACTOR_ID,
        "messages": [
            {"role": "user", "content": "What did I say earlier about my projects?"}
        ],
        "token_budget": 2048,
        "mode": "mixed",
        "llm_provider": "mock"
    }

    response = requests.post(f"{API_URL}/api/agent/chat", json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Agent chat successful")
        print(f"  Response: {result['response'][:100]}...")
        print(f"  Memories used: {result['memories_used']}")
        print(f"  Tokens used: {result['tokens_used']}")
        return result
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return None


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("FGMS API Smoke Tests")
    print(f"API URL: {API_URL}")
    print(f"Actor ID: {ACTOR_ID}")
    print("=" * 60)

    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code != 200:
            print("\n✗ API is not responding correctly")
            print("  Make sure the API is running: python scripts/start_memory_api.py")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to API")
        print("  Make sure the API is running: python scripts/start_memory_api.py")
        sys.exit(1)

    # Run tests
    results = {
        "write": test_write_memory(),
        "read": test_read_memory(),
        "stats": test_stats(),
        "chat": test_agent_chat()
    }

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r is not None)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
