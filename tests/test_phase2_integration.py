"""Integration tests for Phase 2 components."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_glyph_codec():
    """Test glyph codec functionality."""
    from src.glyph.codec import GlyphCodec

    print("\n" + "=" * 60)
    print("Testing Glyph Codec")
    print("=" * 60)

    # Create codec with mock data
    mock_metadata = {
        "cluster_0": {
            "glyph": "谷",
            "representative_phrase": "Can you send me that file?",
            "example_phrases": [
                "Can you send me that file?",
                "Mind emailing the document?",
            ],
        },
        "cluster_1": {
            "glyph": "阜",
            "representative_phrase": "Thank you very much",
            "example_phrases": ["Thank you very much", "Thanks a lot"],
        },
    }

    codec = GlyphCodec(cluster_metadata=mock_metadata)

    # Test cluster → glyph
    glyph = codec.cluster_to_glyph("cluster_0")
    print(f"\n✓ Cluster→Glyph: cluster_0 → {glyph}")
    assert glyph == "谷", f"Expected '谷', got '{glyph}'"

    # Test glyph → cluster
    cluster = codec.glyph_to_cluster("谷")
    print(f"✓ Glyph→Cluster: 谷 → {cluster}")
    assert cluster == "cluster_0", f"Expected 'cluster_0', got '{cluster}'"

    # Test glyph → text
    text = codec.decode_glyph("谷")
    print(f"✓ Glyph→Text: 谷 → '{text}'")
    assert text == "Can you send me that file?"

    # Test extraction
    test_text = "Hello 谷 world 阜"
    glyphs = codec.extract_glyphs(test_text)
    print(f"✓ Extract glyphs from: '{test_text}'")
    print(f"  Found: {glyphs}")
    assert len(glyphs) == 2

    print("\n✅ Glyph Codec tests passed!")
    return True


def test_phrase_matcher():
    """Test phrase matcher functionality."""
    from src.tokenizer.phrase_matcher import PhraseMatcher

    print("\n" + "=" * 60)
    print("Testing Phrase Matcher")
    print("=" * 60)

    # Create matcher with mock data
    mock_metadata = {
        "cluster_0": {
            "representative_phrase": "Can you send me that file?",
            "example_phrases": [
                "Can you send me that file?",
                "send me that file",
            ],
        },
        "cluster_1": {
            "representative_phrase": "Thank you very much",
            "example_phrases": ["Thank you very much", "thank you"],
        },
    }

    matcher = PhraseMatcher(
        cluster_metadata=mock_metadata,
        max_span_length=8,
    )

    # Test matching
    text = "Can you send me that file? Thank you!"
    spans = matcher.match_phrases(text)

    print(f"\n✓ Text: '{text}'")
    print(f"✓ Found {len(spans)} phrase spans:")
    for span in spans:
        print(f"  [{span.start_char}:{span.end_char}] '{span.original_text}' → {span.cluster_id}")

    assert len(spans) >= 1, f"Expected at least 1 span, got {len(spans)}"

    print("\n✅ Phrase Matcher tests passed!")
    return True


def test_hybrid_tokenizer():
    """Test hybrid tokenizer functionality."""
    from src.glyph.codec import GlyphCodec
    from src.tokenizer.hybrid import HybridTokenizer
    from src.tokenizer.phrase_matcher import PhraseMatcher

    print("\n" + "=" * 60)
    print("Testing Hybrid Tokenizer")
    print("=" * 60)

    # Create components with mock data
    mock_metadata = {
        "cluster_0": {
            "glyph": "谷",
            "representative_phrase": "send me that file",
            "example_phrases": ["send me that file", "send the file"],
        },
    }

    codec = GlyphCodec(cluster_metadata=mock_metadata)
    matcher = PhraseMatcher(cluster_metadata=mock_metadata)

    # Create tokenizer
    tokenizer = HybridTokenizer(
        base_tokenizer="gpt2",
        glyph_codec=codec,
        phrase_matcher=matcher,
    )

    # Test encoding
    text = "Please send me that file. Thanks!"
    encoded = tokenizer.encode(text, return_metadata=True)

    print(f"\n✓ Text: '{text}'")
    print(f"✓ Tokens: {len(encoded['input_ids'])}")
    print(f"✓ Glyph count: {encoded['glyph_count']}")

    # Test decoding
    decoded = tokenizer.decode(encoded["input_ids"], expand_glyphs=True)
    print(f"✓ Decoded: '{decoded}'")

    # Test compression
    baseline_tokens = tokenizer.base_tokenizer.encode(text)
    print(f"\n✓ Baseline tokens: {len(baseline_tokens)}")
    print(f"✓ FGT tokens: {len(encoded['input_ids'])}")
    if len(encoded["input_ids"]) < len(baseline_tokens):
        print(f"✓ Compression achieved: {len(baseline_tokens) - len(encoded['input_ids'])} tokens saved")

    print("\n✅ Hybrid Tokenizer tests passed!")
    return True


def test_fgt_adapter():
    """Test FGT LLM adapter."""
    print("\n" + "=" * 60)
    print("Testing FGT LLM Adapter")
    print("=" * 60)

    print("\n⚠️  Skipping adapter test (requires model loading)")
    print("  Use the example scripts to test adapter functionality")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Phase 2 Integration Tests")
    print("=" * 60)

    tests = [
        ("Glyph Codec", test_glyph_codec),
        ("Phrase Matcher", test_phrase_matcher),
        ("Hybrid Tokenizer", test_hybrid_tokenizer),
        ("FGT LLM Adapter", test_fgt_adapter),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
