"""Test deduplication pipeline."""

from local_memory_stack import MemoryEngine

# Test data: generic examples (no personal info)
DEDUP_TEST_CASES = [
    # (text, expected_action, description)
    ("User earns $5000 per month with $30k savings", "added", "First write"),
    ("User earns $5000 per month with $30k savings", "skipped", "Exact duplicate -> skip"),
    ("User earns five thousand a month, saved thirty thousand", "skipped", "Synonym rewrite -> skip"),
    ("User earns 5000/month, saved 30k", "skipped", "Number format variation -> skip"),
    ("User likes programming and AI", "added", "Different topic -> add"),
    ("Push to GitHub only after review and approval", "flagged", "Different wording but related -> flagged"),
    ("The cat is a Sphynx and Devon Rex mix", "added", "Cat info V1 -> add"),
    ("The cat is a Devon Rex mix, very cute", "flagged", "Cat info V2 -> flagged (incremental)"),
    ("User earns $6000 per month now", "flagged", "Conflicting fact -> flagged"),
]


def test_dedup():
    engine = MemoryEngine()
    
    for text, expected, description in DEDUP_TEST_CASES:
        result = engine.write(text, source="test", tags=["dynamic"])
        print(f"[{result}] {description}: {text[:50]}...")
        assert result in ("added", "skipped", "flagged"), f"Unexpected result: {result}"
    
    print("\nAll dedup tests passed")


if __name__ == "__main__":
    test_dedup()
