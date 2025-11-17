#!/usr/bin/env python3
"""
Storage compression benchmark for FGMS

Compares:
- Raw text + zstd
- Deduped text + zstd
- FGT format (glyph + metadata)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import zlib
import time
from collections import defaultdict
from typing import List, Dict, Any
from memory_system.models import MemoryEntry, FractalAddress, Glyph
from datetime import datetime
import hashlib


def create_sample_conversations(num_conversations: int = 50) -> List[str]:
    """Create sample conversation data with realistic patterns."""
    templates = [
        "Can you help me with {task}?",
        "I need to {action} for my {project}.",
        "How do I {question}?",
        "Please {request}.",
        "What's the best way to {goal}?",
        "I'm working on {topic} and need help with {subtopic}.",
        "Could you explain {concept}?",
        "I'd like to learn more about {subject}.",
        "Can you recommend {recommendation}?",
        "What are your thoughts on {opinion_topic}?",
    ]

    tasks = ["deployment", "testing", "debugging", "optimization", "documentation"]
    actions = ["implement", "refactor", "design", "analyze", "review"]
    projects = ["the backend", "the frontend", "the API", "the database", "the UI"]
    questions = ["set up Docker", "write tests", "optimize queries", "handle errors"]
    requests = ["review this code", "check my logic", "suggest improvements"]
    goals = ["scale the application", "improve performance", "reduce latency"]
    topics = ["authentication", "caching", "monitoring", "deployment"]
    subtopics = ["JWT tokens", "Redis", "logging", "CI/CD"]
    concepts = ["microservices", "event sourcing", "CQRS", "DDD"]
    subjects = ["Kubernetes", "GraphQL", "WebSockets", "gRPC"]
    recommendations = ["tools", "libraries", "patterns", "best practices"]
    opinion_topics = ["architecture choices", "tech stack", "design patterns"]

    substitutions = {
        "task": tasks,
        "action": actions,
        "project": projects,
        "question": questions,
        "request": requests,
        "goal": goals,
        "topic": topics,
        "subtopic": subtopics,
        "concept": concepts,
        "subject": subjects,
        "recommendation": recommendations,
        "opinion_topic": opinion_topics,
    }

    conversations = []
    for i in range(num_conversations):
        template = templates[i % len(templates)]
        text = template

        for key, values in substitutions.items():
            if f"{{{key}}}" in text:
                text = text.replace(f"{{{key}}}", values[i % len(values)])

        conversations.append(text)

    return conversations


def calculate_compression_ratio(original: bytes, compressed: bytes) -> float:
    """Calculate compression ratio."""
    return len(original) / len(compressed) if compressed else float('inf')


def benchmark_raw_zstd(texts: List[str]) -> Dict[str, Any]:
    """Benchmark: Raw text + zstd compression."""
    start = time.time()

    # Concatenate all text
    raw_text = "\n".join(texts)
    raw_bytes = raw_text.encode('utf-8')

    # Compress with zlib (similar to zstd)
    compressed_bytes = zlib.compress(raw_bytes, level=9)

    elapsed = time.time() - start

    return {
        "method": "Raw + zstd",
        "raw_bytes": len(raw_bytes),
        "compressed_bytes": len(compressed_bytes),
        "compression_ratio": calculate_compression_ratio(raw_bytes, compressed_bytes),
        "elapsed_sec": elapsed
    }


def benchmark_dedup_zstd(texts: List[str]) -> Dict[str, Any]:
    """Benchmark: Deduplicated text + zstd compression."""
    start = time.time()

    # Simple deduplication by hash
    unique_texts = {}
    text_refs = []

    for text in texts:
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        if text_hash not in unique_texts:
            unique_texts[text_hash] = text
        text_refs.append(text_hash)

    # Store unique texts + references
    unique_text = "\n".join(unique_texts.values())
    refs_text = ",".join(text_refs)

    combined = f"{unique_text}\n---REFS---\n{refs_text}"
    raw_bytes = combined.encode('utf-8')
    compressed_bytes = zlib.compress(raw_bytes, level=9)

    elapsed = time.time() - start

    return {
        "method": "Dedup + zstd",
        "raw_bytes": len(raw_bytes),
        "compressed_bytes": len(compressed_bytes),
        "compression_ratio": calculate_compression_ratio(raw_bytes, compressed_bytes),
        "unique_texts": len(unique_texts),
        "total_texts": len(texts),
        "dedup_ratio": len(unique_texts) / len(texts),
        "elapsed_sec": elapsed
    }


def benchmark_fgt_format(texts: List[str]) -> Dict[str, Any]:
    """Benchmark: FGT glyph format."""
    start = time.time()

    # Create mock glyph mappings (in reality, these come from clustering)
    # For demo: group texts by first 3 words
    glyph_map = {}
    glyph_sequences = []

    for text in texts:
        # Simple clustering by first few words
        words = text.split()
        key = " ".join(words[:3]) if len(words) >= 3 else text

        if key not in glyph_map:
            glyph_id = len(glyph_map)
            glyph_str = f"谷{chr(ord('阜') + glyph_id % 100)}"
            glyph_map[key] = {
                "glyph_id": glyph_id,
                "glyph_str": glyph_str,
                "texts": []
            }

        glyph_map[key]["texts"].append(text)
        glyph_sequences.append(glyph_map[key]["glyph_id"])

    # Serialize glyph table + sequences
    glyph_table = {
        k: {"glyph_id": v["glyph_id"], "glyph_str": v["glyph_str"], "count": len(v["texts"])}
        for k, v in glyph_map.items()
    }

    fgt_data = {
        "glyph_table": glyph_table,
        "sequences": glyph_sequences,
        "metadata": {
            "num_texts": len(texts),
            "num_glyphs": len(glyph_map),
            "timestamp": datetime.now().isoformat()
        }
    }

    # Serialize and compress
    json_bytes = json.dumps(fgt_data).encode('utf-8')
    compressed_bytes = zlib.compress(json_bytes, level=9)

    elapsed = time.time() - start

    # Calculate raw bytes equivalent
    raw_bytes = sum(len(t.encode('utf-8')) for t in texts)

    return {
        "method": "FGT (glyph + meta)",
        "raw_bytes": raw_bytes,
        "json_bytes": len(json_bytes),
        "compressed_bytes": len(compressed_bytes),
        "compression_ratio": calculate_compression_ratio(bytes(str(raw_bytes), 'utf-8'), compressed_bytes),
        "num_glyphs": len(glyph_map),
        "num_texts": len(texts),
        "glyph_coverage": len(glyph_map) / len(texts),
        "elapsed_sec": elapsed
    }


def run_benchmark(num_conversations: int = 100):
    """Run storage compression benchmark."""
    print("=" * 70)
    print("FGMS Storage Compression Benchmark")
    print("=" * 70)

    print(f"\nGenerating {num_conversations} sample conversations...")
    texts = create_sample_conversations(num_conversations)

    print(f"Generated {len(texts)} conversations")
    print(f"Average length: {sum(len(t) for t in texts) / len(texts):.1f} chars")
    print(f"Total size: {sum(len(t) for t in texts):,} bytes")

    print("\n" + "-" * 70)
    print("Running benchmarks...")
    print("-" * 70)

    # Run benchmarks
    results = []

    print("\n1. Raw + zstd...")
    results.append(benchmark_raw_zstd(texts))

    print("2. Dedup + zstd...")
    results.append(benchmark_dedup_zstd(texts))

    print("3. FGT format...")
    results.append(benchmark_fgt_format(texts))

    # Display results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    baseline_size = results[0]["compressed_bytes"]

    print(f"\n{'Method':<20} {'Bytes':>12} {'Relative':>10} {'Ratio':>8} {'Time':>8}")
    print("-" * 70)

    for result in results:
        size = result["compressed_bytes"]
        relative = size / baseline_size
        ratio = result["compression_ratio"]
        time_sec = result["elapsed_sec"]

        print(f"{result['method']:<20} {size:>12,} {relative:>10.2f}x {ratio:>8.2f} {time_sec:>7.3f}s")

    # Additional details
    print("\n" + "-" * 70)
    print("Details")
    print("-" * 70)

    for result in results:
        print(f"\n{result['method']}:")
        for key, value in result.items():
            if key != "method":
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

    # Save report
    report_path = Path("reports/storage_bench.json")
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_conversations": num_conversations,
            "results": results
        }, f, indent=2)

    print(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run storage compression benchmark")
    parser.add_argument("--num", type=int, default=100, help="Number of conversations")
    args = parser.parse_args()

    run_benchmark(args.num)
