<div align="center">

# 🧠 Local Memory Stack

**Semantic memory engine for AI agents. 100% local. Zero API keys. Zero cloud.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hermes Agent](https://img.shields.io/badge/Hermes_Agent-Plugin-green.svg)](https://hermes-agent.nousresearch.com)

[Quickstart](#quickstart) · [Features](#features) · [Hermes Integration](#hermes-integration) · [API Reference](#api-reference) · [Architecture](#architecture)

</div>

---

## Why Local Memory Stack?

Most memory layers for AI agents require cloud APIs, external databases, or paid services. Local Memory Stack runs entirely on your machine — embeddings, vector search, entity extraction, everything. Your data never leaves your device.

**Think of it as a semantic brain for your AI agent** — write facts in natural language, retrieve them with fuzzy queries, and let the system handle deduplication, archival, and graph-based reasoning automatically.

## How It Compares

| | Local Memory Stack | mem0 | Zep | supermemory |
|---|---|---|---|---|
| **Requires API keys** | ❌ None | ✅ LLM provider key | ✅ Cloud API | ✅ OpenAI + Pinecone |
| **Data leaves device** | ❌ Never | Depends on LLM config | ✅ Cloud-first | ✅ Cloud services |
| **External services** | None | Vector DB (optional) | Zep Cloud | Turso + Redis + Pinecone |
| **Setup** | `pip install -e .` | `pip install mem0ai` | Docker / Cloud signup | Docker + env vars |
| **Deduplication** | ✅ 3-tier pipeline | ✅ Basic | ✅ Basic | ❌ |
| **Entity extraction** | ✅ GLiNER (local) | ✅ Via LLM | ✅ Via LLM | ❌ |
| **TTL / Auto-archive** | ✅ Tag-based | ❌ | ❌ | ❌ |
| **Graph-guided search** | ✅ Cluster + neighbor | ❌ | ❌ | ✅ Graph view |
| **Reranker** | ✅ BGE-M3 two-stage | ❌ | ❌ | ❌ |
| **Quality gate** | ✅ 5-layer filter | ❌ | ❌ | ❌ |
| **License** | Apache-2.0 | Apache-2.0 | Apache-2.0 (server: BUSL) | AGPL-3.0 |

**When to choose Local Memory Stack:** You want a memory layer that works offline, costs nothing to run, and doesn't send your data to external APIs. Especially suited for privacy-sensitive use cases, local-first agent setups, and environments without reliable internet.

**When to choose mem0/Zep:** You need managed cloud infrastructure, team collaboration features, or prefer to offload compute to external services.

## Quickstart

### Install

```bash
git clone https://github.com/keven221/local-memory-stack.git
cd local-memory-stack

python3 -m venv venv && source venv/bin/activate
pip install -e .
```

> First run downloads ~3.3 GB of models (BGE-M3 + GLiNER). Subsequent starts are instant.

### Run as a Service

```bash
python3 -m uvicorn src.local_memory_stack.server:app --port 8900
```

### Use in Python

```python
from local_memory_stack import MemoryEngine

engine = MemoryEngine()

# Write a memory
engine.write("The cat's name is Mimi, a tabby mix", source="profile", tags=["pet"])

# Search with natural language
results = engine.query("what pet does the user have?", top_k=3)
for r in results:
    print(f"[{r.similarity:.0%}] {r.text}")
# → [92%] The cat's name is Mimi, a tabby mix
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | "pet" finds "cat", "rent" finds "$3000/month" — no keyword matching needed |
| 🔄 **Auto Deduplication** | 3-tier pipeline: embedding recall → keyword overlap → merge/skip decisions |
| 🏷️ **Entity Extraction** | Automatic NER (people, places, events) via GLiNER, runs async in background |
| ⏳ **TTL & Auto-Archive** | Memories expire based on tags (30d–365d), archived without deletion |
| 📊 **Reranker** | Two-stage retrieval: HNSW fast recall → BGE-M3 precise reranking |
| 🕸️ **Graph-Guided Search** | Cluster-aware retrieval with neighbor expansion, 3–157× faster on large datasets |
| 🔀 **Hybrid Search** | BM25 keyword + vector search with Reciprocal Rank Fusion |
| 🚫 **Quality Gate** | 5-layer filtering blocks noise: code artifacts, task status, trivial chat |
| 📬 **File Mailbox** | Zero-dependency message passing between agents / cron jobs |
| ⚡ **Async NER** | Entity extraction runs in background — writes complete in ~50 ms |
| 🔌 **Framework-Agnostic** | Pure Python class. Works with any agent framework or standalone |

---

## Hermes Integration

Local Memory Stack ships as a first-class [Hermes Agent](https://hermes-agent.nousresearch.com) plugin. After `pip install -e .`, it auto-registers as a memory provider.

### Setup

```bash
# Install into Hermes's Python environment
cd local-memory-stack
pip install -e .

# Configure
hermes config set memory.provider local-memory-stack
```

### What It Does

| Capability | How It Works |
|-----------|-------------|
| **Auto-sync conversations** | Each user→assistant turn is quality-gated and written as a memory (skips trivial exchanges) |
| **Prefetch context** | Before every LLM call, top-5 relevant memories are injected via semantic recall |
| **`recall_memory` tool** | Agent can actively search memory with ranked + reranked results |
| **Session-end extraction** | Key facts are extracted from the full session and persisted on shutdown |
| **Pre-compress preservation** | Before context compaction, important memories are saved to prevent loss |

### How It Looks

Memories are injected into the system prompt as:

```
<memory-context>
[1] (2026-06-07 | preference) User prefers dark theme for coding
[2] (2026-06-05 | project) Local memory stack has 146 memories stored
</memory-context>
```

The agent can also call `recall_memory(query="...", top_k=5)` at any time for deeper retrieval.

---

## Architecture

```
Your text
    ↓
┌─────────────┐
│  BGE-M3     │  Embedding model (2.2 GB) — text → 1024-dim vectors
└──────┬──────┘
       ↓
┌─────────────┐     ┌─────────────┐
│  ChromaDB   │ ←──→│  GLiNER     │  NER model (1.1 GB), async background
│  Vector DB  │     │  Entity Tag │  "John in NYC" → John(person), NYC(location)
└──────┬──────┘     └─────────────┘
       ↓
┌─────────────┐
│  Archive    │  Expired memories move here (searchable, not injected)
│  Cold Store │  3-tier: Active → Archive → Cleanup
└─────────────┘
```

**All models run locally. No network calls. No API keys.**

---

## API Reference

### Memory Operations

```bash
# Write (auto-dedup + auto-tag + auto-TTL)
curl -X POST http://127.0.0.1:8900/memory/write \
  -H "Content-Type: application/json" \
  -d '{"text":"memory content", "source":"api", "tags":["preference"]}'

# Search active memories
curl -X POST http://127.0.0.1:8900/memory/query \
  -H "Content-Type: application/json" \
  -d '{"text":"search query", "top_k":5}'

# Search with reranker (two-stage)
curl -X POST http://127.0.0.1:8900/memory/query_rerank \
  -H "Content-Type: application/json" \
  -d '{"text":"search query", "top_k":3}'

# Search all (active + archived)
curl -X POST http://127.0.0.1:8900/memory/search_all \
  -H "Content-Type: application/json" \
  -d '{"text":"search query", "top_k":5}'

# Stats
curl http://127.0.0.1:8900/memory/stats
```

### Archive Management

```bash
# Preview archive (dry-run)
curl -X POST http://127.0.0.1:8900/memory/archive -d '{"dry_run":true}'

# Execute archive
curl -X POST http://127.0.0.1:8900/memory/archive -d '{"dry_run":false}'

# Search archived memories
curl -X POST http://127.0.0.1:8900/memory/archive/search \
  -H "Content-Type: application/json" -d '{"text":"query"}'

# Restore from archive
curl -X POST http://127.0.0.1:8900/memory/archive/restore \
  -H "Content-Type: application/json" -d '{"ids":["mem_xxx"]}'
```

### TTL Defaults

| Tag | TTL | Use Case |
|-----|-----|----------|
| `temporary` | 30 days | Appointments, short-lived tasks |
| `dynamic` | 180 days | Projects, ongoing goals |
| `static` / `preference` / `feedback` | 365 days | Identity, long-term preferences |
| *(no tag)* | 180 days | Default |

### Quality Gate

Memories are filtered through 5 layers before storage:

1. **Min length** — rejects text < 10 characters
2. **Noise patterns** — blocks file paths, git hashes, CI status messages
3. **Code artifacts** — filters Python/JS imports, class definitions, stack traces
4. **Task status** — blocks "Phase done", "Step completed" type messages
5. **Chat noise** — filters trivial acknowledgments ("ok", "got it", "haha")

---

## Retrieval Modes

### 1. Graph-Guided Search (recommended for large datasets)

```bash
python3 graph_retrieval.py "search query"
```

Cluster localization → in-cluster BM25 → neighbor expansion. Index cached 24h.

### 2. Hybrid Search (fallback)

```bash
python3 hybrid_search.py "search query"
```

Vector + BM25 + RRF fusion. Works when graph index has no match.

### 3. REST API (universal)

```bash
curl -X POST http://127.0.0.1:8900/memory/query_rerank \
  -H "Content-Type: application/json" \
  -d '{"text":"search query","top_k":3}'
```

---

## Maintenance

```bash
# Conflict detection + stale memory cleanup
python3 maintenance.py

# TTL archival
python3 memory_ttl_cleanup.py --stats      # Statistics
python3 memory_ttl_cleanup.py              # Preview
python3 memory_ttl_cleanup.py --execute    # Execute archive
python3 memory_ttl_cleanup.py --backfill   # Backfill TTL for old records

# Backup
python3 memory_backup.py
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Cold start | ~12s | Model loading (one-time) |
| Write (with dedup) | ~50 ms | After model warmup |
| Semantic search | ~17 ms | Vector-only, warm cache |
| Reranked search | ~35 ms | HNSW recall + BGE-M3 rerank |
| Single read | ~0.1 ms | Direct ChromaDB lookup |
| Entity extraction | ~2s | Async background, non-blocking |
| 100K records query | <100 ms | HNSW O(log N) |

Benchmarked on Apple Silicon (MPS). CUDA and CPU also supported.

---

## Tech Stack

| Component | Model | Size | Role |
|-----------|-------|------|------|
| BGE-M3 | BAAI/bge-m3 | 2.2 GB | Text → 1024-dim embeddings |
| ChromaDB | — | — | Vector database + HNSW index |
| GLiNER | gliner_multi-v2.1 | 1.1 GB | Named entity recognition |

Zero external API dependencies. Everything runs locally.

---

## Project Structure

```
local-memory-stack/
├── src/local_memory_stack/
│   ├── engine.py              ← Core engine (dedup + TTL + archive + reranker)
│   ├── server.py              ← FastAPI REST service
│   ├── cli.py                 ← CLI tool
│   ├── hermes_plugin.py       ← Hermes Agent plugin entry point
│   └── hermes_memory_provider.py  ← Hermes MemoryProvider implementation
├── mailbox.py                 ← File-based inter-agent messaging
├── hybrid_search.py           ← Hybrid retrieval (vector + BM25 + RRF)
├── graph_retrieval.py         ← Graph-guided retrieval (cluster acceleration)
├── memory_ttl_cleanup.py      ← TTL archival maintenance
├── maintenance.py             ← Conflict detection + semantic dedup
├── memory_backup.py           ← Memory backup utility
├── tests/                     ← Test suite
├── chroma_data/               ← Memory database (gitignored)
└── plugin.yaml                ← Hermes plugin manifest
```

---

## Deployment

### macOS (launchd auto-start)

```bash
# Create plist
cat > ~/Library/LaunchAgents/com.local-memory-stack.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local-memory-stack</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>src.local_memory_stack.server:app</string>
        <string>--port</string>
        <string>8900</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/local-memory-stack</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.local-memory-stack.plist
```

### Docker (coming soon)

---

## Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.

```bash
# Development setup
git clone https://github.com/keven221/local-memory-stack.git
cd local-memory-stack
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/
```

---

## License

[Apache-2.0](LICENSE)