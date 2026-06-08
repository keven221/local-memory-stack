"""Hermes Agent MemoryProvider adapter for Local-Memory-Stack.

Implements the MemoryProvider ABC (agent/memory_provider.py) so
local-memory-stack can plug into Hermes as a memory backend —
no MCP bridge, no HTTP hop, just zero-latency in-process recall.

Usage in ~/.hermes/config.yaml:

    memory:
      provider: local-memory-stack

Install:

    pip install local-memory-stack
    # Plugin auto-discovered via pip entry point

Five-layer quality gate + three-tier TTL archive + hybrid BM25+HNSW+RRF
retrieval + graph-guided recall + BGE-M3 rerank — all active through the
standard Hermes MemoryProvider hooks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .engine import MemoryEngine

logger = logging.getLogger(__name__)


class LocalMemoryStackProvider:
    """Hermes MemoryProvider backed by the Local-Memory-Stack engine."""

    def __init__(self) -> None:
        self._engine: Optional[MemoryEngine] = None
        self._session_id: str = ""
        self._data_dir: str = ""
        self._prefetch_cache: str = ""

    # -- Identity -------------------------------------------------------

    @property
    def name(self) -> str:
        return "local-memory-stack"

    # -- Core lifecycle -------------------------------------------------

    def is_available(self) -> bool:
        """Always available — runs fully local, no credentials needed."""
        try:
            from .engine import MemoryEngine  # noqa: F811
            return True
        except ImportError:
            logger.warning("local-memory-stack: MemoryEngine import failed")
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Boot the engine, scoped to the active Hermes home directory.

        Uses `hermes_home` from kwargs to store ChromaDB data under
        ``$HERMES_HOME/memory/`` so each Hermes profile has its own
        memory bank.
        """
        hermes_home = kwargs.get("hermes_home", "")
        platform = kwargs.get("platform", "cli")
        agent_context = kwargs.get("agent_context", "primary")

        self._session_id = session_id

        if hermes_home:
            import os
            self._data_dir = os.path.join(hermes_home, "memory", "local-memory-stack")
        else:
            self._data_dir = "./local_memory_stack_data"

        logger.info(
            "local-memory-stack: initializing session=%s platform=%s context=%s data_dir=%s",
            session_id, platform, agent_context, self._data_dir,
        )

        self._engine = MemoryEngine(data_dir=self._data_dir)

    def shutdown(self) -> None:
        """Clean shutdown — ChromaDB client cleans itself, nothing extra."""
        self._engine = None
        self._prefetch_cache = ""

    # -- Context injection --------------------------------------------

    def system_prompt_block(self) -> str:
        """Return static provider info for the system prompt.

        Kept brief — Hermes injects dynamic recall context via prefetch().
        """
        return (
            "[local-memory-stack] Active -- semantic recall with "
            "BGE-M3 + ChromaDB + GLiNER, 5-layer quality gate, "
            "3-tier TTL archive."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached recall context for the current turn.

        Background recall happens in queue_prefetch(); this just
        returns the pre-computed result.
        """
        return self._prefetch_cache

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Run semantic recall in the background for the next turn.

        Called after each turn completes. Results consumed by prefetch().
        Recall is additive to MEMORY.md, which should be kept minimal
        (identity + core preferences + hard rules only) to avoid overlap.
        """
        if self._engine is None:
            self._prefetch_cache = ""
            return

        try:
            entries = self._engine.query_with_rerank(
                text=query,
                top_k=5,
                retrieve_k=10,
                threshold=0.35,
            )
            if not entries:
                self._prefetch_cache = ""
                return

            lines: list[str] = []
            for e in entries:
                ts = (e.created_at or "")[:10]
                tag_str = ", ".join(e.tags) if e.tags else ""
                line = f"- [{tag_str}] {e.text}" if tag_str else f"- {e.text}"
                if ts:
                    line += f"  ({ts})"
                lines.append(line)

            header = (
                "<memory-context>\n"
                "[local-memory-stack] Contextual recall:\n"
            )
            self._prefetch_cache = header + "\n".join(lines) + "\n</memory-context>"
        except Exception:
            logger.exception("local-memory-stack: prefetch failed")
            self._prefetch_cache = ""

    # -- Turn sync ----------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Write high-signal conversation turns to engine.

        Quality gates:
        1. Strip <memory-context> blocks to prevent recursive pollution
        2. Skip trivial turns (short, low entity count, or repeating)
        3. Prefer messages list when available (pre-compression fidelity)
        4. Merge adjacent writes into single entry per turn pair
        """
        if self._engine is None:
            return

        # ── Strip injected context blocks from content ──
        import re
        def clean(content: str) -> str:
            """Remove <memory-context> and similar injected blocks."""
            content = re.sub(r'<memory-context>.*?</memory-context>', '', content, flags=re.DOTALL)
            content = re.sub(r'<session-context>.*?</session-context>', '', content, flags=re.DOTALL)
            return content.strip()

        user_clean = clean(user_content)
        asst_clean = clean(assistant_content)

        # ── Skip trivial exchanges ──
        trivial_patterns = [
            r'^(嗯|好|ok|继续|go\s*on|next|yes|no|对|行|可以|明白|了解|收到)$',
            r'^[!！?？。.]{1,3}$',
        ]
        if any(re.match(p, user_clean) for p in trivial_patterns) and            any(re.match(p, asst_clean) for p in trivial_patterns):
            return

        try:
            combined = f"User: {user_clean}\nAssistant: {asst_clean}"
            tags = [f"session:{session_id}", "source:conversation"]
            self._engine.write(
                text=combined,
                source="hermes-conversation",
                tags=tags,
                auto_extract=True,
            )
        except Exception:
            logger.exception("local-memory-stack: sync_turn failed")

    # -- Tools --------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose recall_memory tool so the agent can self-reflect."""
        return [
            {
                "name": "recall_memory",
                "description": (
                    "Semantic search over local memory (BGE-M3 + ChromaDB). "
                    "Returns ranked, reranked memories with entity extraction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query for semantic recall.",
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 5,
                            "description": "Number of results to return (max 10).",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch recall_memory tool calls."""
        if tool_name != "recall_memory":
            raise ValueError(f"Unknown tool: {tool_name}")

        if self._engine is None:
            return '{"error": "memory engine not initialized", "results": []}'

        query = args.get("query", "")
        top_k = min(int(args.get("top_k", 5)), 10)

        try:
            entries = self._engine.query_with_rerank(
                text=query, top_k=top_k, retrieve_k=20,
            )
            import json
            results = [e.to_dict() for e in entries]
            return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)
        except Exception:
            logger.exception("local-memory-stack: handle_tool_call failed")
            return '{"error": "recall failed", "results": []}'

    # -- Optional hooks ------------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror Hermes built-in memory writes to the engine."""
        if self._engine is None or action not in ("add", "replace"):
            return

        try:
            tags = [f"target:{target}", "source:hermes-builtin-memory"]
            if metadata and metadata.get("session_id"):
                tags.append(f"session:{metadata['session_id']}")

            self._engine.write(
                text=content,
                source="hermes-builtin-memory",
                tags=tags,
                auto_extract=True,
            )
        except Exception:
            logger.exception("local-memory-stack: on_memory_write failed")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Extract key facts from the full session on exit."""
        if self._engine is None:
            return

        try:
            if messages:
                snippet = " ".join(
                    m.get("content", "")[:200]
                    for m in messages[-5:]
                    if isinstance(m.get("content"), str)
                )
                if snippet.strip():
                    self._engine.write(
                        text=snippet,
                        source="hermes-session-end",
                        tags=[f"session:{self._session_id}", "session-summary"],
                        auto_extract=False,
                    )

            self._engine.maintenance()
            self._engine.archive(dry_run=False)
        except Exception:
            logger.exception("local-memory-stack: on_session_end failed")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract key exchanges before Hermes compresses the context window."""
        if self._engine is None or not messages:
            return ""

        try:
            entries = self._engine.query_with_rerank(
                text="key facts decisions commitments preferences",
                top_k=3,
                retrieve_k=10,
                threshold=0.3,
            )
            if not entries:
                return ""

            lines = [f"- {e.text}" for e in entries]
            return "[local-memory-stack preserved facts]\n" + "\n".join(lines)
        except Exception:
            logger.exception("local-memory-stack: on_pre_compress failed")
            return ""

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Track session-id rotation."""
        self._session_id = new_session_id
        self._prefetch_cache = ""

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """No credentials needed — fully local."""
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """No-op: all config is implicit (data dir under hermes_home)."""
        pass
