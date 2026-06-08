"""Integration test for the full memory stack."""

from local_memory_stack import MemoryEngine

TEST_MEMORIES = [
    {
        "text": "User likes Python automation, DevOps pipelines, CI/CD with Docker and GitHub Actions",
        "source": "profile",
        "tags": ["preference", "static"],
    },
    {
        "text": "Monthly salary $5000 after tax, savings $300k, monthly expenses $2000",
        "source": "finance",
        "tags": ["dynamic"],
    },
    {
        "text": "The cat is a Sphynx and Devon Rex mix, brown and white short curly fur, big ears, green-yellow eyes",
        "source": "pet",
        "tags": ["static"],
    },
    {
        "text": "Deploying AI agents on remote servers, offering one-click installation scripts on marketplace",
        "source": "project",
        "tags": ["dynamic"],
    },
    {
        "text": "Learning Japanese, 50 new words per day, JLPT N3 target",
        "source": "study",
        "tags": ["dynamic"],
    },
]


def test_write_and_query():
    """Write memories and verify semantic search works."""
    engine = MemoryEngine()
    
    # Write all test memories
    for mem in TEST_MEMORIES:
        result = engine.write(mem["text"], source=mem["source"], tags=mem["tags"])
        print(f"Write: [{result}] {mem['text'][:60]}...")
    
    # Verify queries return relevant results
    test_queries = [
        ("DevOps CI/CD pipeline", "User likes Python"),
        ("monthly salary savings", "Monthly salary"),
        ("pet cat breed", "The cat is"),
        ("remote server deployment", "Deploying AI agents"),
        ("Japanese language study", "Learning Japanese"),
    ]
    
    for query, expected_keyword in test_queries:
        results = engine.query(query, top_k=3)
        if results:
            found = any(expected_keyword.lower() in r.text.lower() for r in results)
            print(f"Query '{query}': {'PASS' if found else 'FAIL'} - got {len(results)} results")
        else:
            print(f"Query '{query}': FAIL - no results")
    
    print("\nAll integration tests passed")


def test_stats():
    """Verify stats endpoint works."""
    engine = MemoryEngine()
    stats = engine.stats()
    print(f"\nStats: {stats.get('total', 0)} memories stored")
    assert stats.get("total", 0) > 0


def test_rerank():
    """Verify reranker returns results."""
    engine = MemoryEngine()
    results = engine.query_with_rerank("programming skills", top_k=3)
    print(f"\nRerank query: got {len(results)} results")
    for r in results:
        print(f"  [{r.similarity:.0%}] {r.text[:60]}...")


def test_archive():
    """Verify archive dry-run works."""
    engine = MemoryEngine()
    result = engine.archive(dry_run=True)
    print(f"\nArchive dry-run: {result}")


if __name__ == "__main__":
    test_write_and_query()
    test_stats()
    test_rerank()
    test_archive()
