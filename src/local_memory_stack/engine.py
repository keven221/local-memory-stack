"""MemoryEngine — 本地语义记忆核心引擎。

零外部依赖（除 ML 模型），任何 Agent 框架都可直接使用。

用法:
    engine = MemoryEngine()
    engine.write("ThisIsMyCat", source="profile")
    results = engine.query("User的猫是什么", top_k=3)
    engine.maintenance()  # 定期清理
"""

from __future__ import annotations

import os
import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("memstack")

# 离线模式 — 只读本地缓存，不联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DEDUP_THRESHOLD = 0.85
DEFAULT_ENTITY_LABELS = ["person", "location", "organization", "money", "technology", "date", "event"]


@dataclass
class MemoryEntry:
    id: str
    text: str
    source: str = ""
    tags: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    similarity: float = 0.0
    merged_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryEngine:
    """本地语义记忆引擎，封装 BGE-M3 + ChromaDB + GLiNER。"""

    def __init__(self, data_dir: str = "./chroma_data", device: str = "auto"):
        """初始化引擎（首次会下载模型，后续秒加载）。

        Args:
            data_dir: ChromaDB 持久化目录
            device: 'auto' 自动选择 MPS/CUDA/CPU，或手动指定
        """
        self._data_dir = data_dir
        self._encoder = None
        self._db = None
        self._collection = None
        self._ner = None
        self._ner_lock = threading.Lock()
        self._bg_executor = None

        if device == "auto":
            import torch
            self._device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._load_models()

    def _load_models(self):
        """加载三大组件（只执行一次）。"""
        import torch
        from sentence_transformers import SentenceTransformer

        t0 = time.time()
        logger.info(f"加载 BGE-M3 → {self._device}")
        self._encoder = SentenceTransformer("BAAI/bge-m3", device=self._device)
        logger.info(f"BGE-M3 就绪 ({time.time()-t0:.1f}s)")

        import chromadb
        self._db = chromadb.PersistentClient(path=self._data_dir)
        self._collection = self._db.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB 就绪 ({self._collection.count()} 条记忆)")

        from gliner import GLiNER
        logger.info("加载 GLiNER...")
        self._ner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        logger.info(f"GLiNER 就绪 ({time.time()-t0:.1f}s 总计)")

    # ── 写入 ────────────────────────────────

    def write(self, text: str, source: str = "api", tags: Optional[List[str]] = None,
              auto_extract: bool = True) -> Dict[str, Any]:
        """写入一条记忆，自动去重合并。

        Args:
            text: 记忆内容
            source: 来源标记（api/chat/profile/...）
            tags: 标签列表
            auto_extract: 是否自动提取实体（后台异步）

        Returns:
            {"id": ..., "action": "added"|"merged", "text": ..., "similarity": ...}
        """
        now = datetime.now(timezone.utc).isoformat()
        vector = self._encoder.encode([text], normalize_embeddings=True)[0]

        # 去重检查
        if self._collection.count() > 0:
            results = self._collection.query(query_embeddings=[vector.tolist()], n_results=3)
            if results["documents"] and results["documents"][0]:
                best_sim, best_i = 0, 0
                for i, dist in enumerate(results["distances"][0]):
                    sim = 1 - dist
                    if sim > best_sim:
                        best_sim, best_i = sim, i

                if best_sim >= DEDUP_THRESHOLD:
                    return self._merge(
                        existing_id=results["ids"][0][best_i],
                        existing_text=results["documents"][0][best_i],
                        existing_meta=results["metadatas"][0][best_i] or {},
                        new_text=text,
                        new_vector=vector,
                        source=source,
                        tags=tags or [],
                        similarity=best_sim,
                        timestamp=now,
                        auto_extract=auto_extract,
                    )

        # 新增
        return self._add_new(text=text, vector=vector, source=source, tags=tags or [],
                           timestamp=now, auto_extract=auto_extract)

    def _add_new(self, text, vector, source, tags, timestamp, auto_extract) -> dict:
        mem_id = f"mem_{uuid.uuid4().hex[:12]}"
        meta = {"source": source, "created_at": timestamp, "merged_count": "0"}
        if tags:
            meta["tags"] = ",".join(tags)

        self._collection.add(ids=[mem_id], documents=[text], embeddings=[vector.tolist()], metadatas=[meta])

        if auto_extract:
            self._extract_async(mem_id, text)

        return {"id": mem_id, "text": text[:200], "action": "added", "metadata": meta}

    def _merge(self, existing_id, existing_text, existing_meta, new_text, new_vector,
               source, tags, similarity, timestamp, auto_extract) -> dict:
        # 保留更长的文本
        merged_text = new_text if len(new_text) >= len(existing_text) else existing_text
        merged_vector = new_vector if len(new_text) >= len(existing_text) else self._encoder.encode([existing_text], normalize_embeddings=True)[0]

        old_tags = existing_meta.get("tags", "").split(",") if existing_meta.get("tags") else []
        merged_tags = list(set(old_tags + tags))

        meta = {
            "source": f"{existing_meta.get('source', '')},{source}".strip(","),
            "created_at": existing_meta.get("created_at", timestamp),
            "updated_at": timestamp,
            "merged_count": str(int(existing_meta.get("merged_count", "0")) + 1),
        }
        if merged_tags:
            meta["tags"] = ",".join(merged_tags)

        self._collection.update(ids=[existing_id], documents=[merged_text],
                               embeddings=[merged_vector.tolist()], metadatas=[meta])

        if auto_extract:
            self._extract_async(existing_id, merged_text)

        return {"id": existing_id, "text": merged_text[:200], "action": "merged",
                "similarity": round(similarity, 4), "metadata": meta}

    # ── 查询 ────────────────────────────────

    def query(self, text: str, top_k: int = 5, threshold: float = 0.3,
              source: Optional[str] = None) -> List[MemoryEntry]:
        """语义查询记忆。

        Args:
            text: 查询文本
            top_k: 返回条数
            threshold: 最小相似度（1-cosine距离，越高越相似）
            source: 按来源过滤（None=全局）

        Returns:
            MemoryEntry 列表，按相似度降序
        """
        query_vec = self._encoder.encode([text], normalize_embeddings=True)[0]
        kwargs: dict = {"query_embeddings": [query_vec.tolist()], "n_results": top_k}
        if source:
            kwargs["where"] = {"source": source}

        results = self._collection.query(**kwargs)

        entries = []
        for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
            sim = 1 - dist
            if sim >= threshold:
                tags = (meta.get("tags", "").split(",") if meta.get("tags") else []) if meta else []
                entities_raw = (meta.get("entities", "") if meta else "")
                entities = []
                if entities_raw:
                    for e in entities_raw.split("|"):
                        parts = e.rsplit(":", 1)
                        entities.append({"text": parts[0], "label": parts[1] if len(parts) > 1 else "unknown"})

                entries.append(MemoryEntry(
                    id=meta.get("id", ""),
                    text=doc,
                    source=meta.get("source", "") if meta else "",
                    tags=tags,
                    entities=entities,
                    merged_count=int(meta.get("merged_count", 0)) if meta else 0,
                    created_at=meta.get("created_at", "") if meta else "",
                    updated_at=meta.get("updated_at", "") if meta else "",
                    similarity=round(sim, 4),
                ))
        return entries

    # ── 实体提取 ────────────────────────────

    def extract_entities(self, text: str, labels: Optional[List[str]] = None,
                         threshold: float = 0.3) -> List[Dict[str, Any]]:
        """提取实体（同步，用于调试）。"""
        with self._ner_lock:
            return self._ner.predict_entities(text, labels or DEFAULT_ENTITY_LABELS, threshold=threshold)

    def _extract_async(self, mem_id: str, text: str):
        """后台异步提取实体并更新 metadata。"""
        def _run():
            try:
                with self._ner_lock:
                    entities = self._ner.predict_entities(text, DEFAULT_ENTITY_LABELS, threshold=0.3)
                if entities:
                    entity_str = "|".join(f"{e['text']}:{e['label']}" for e in entities)
                    result = self._collection.get(ids=[mem_id], include=["metadatas"])
                    if result["metadatas"]:
                        meta = result["metadatas"][0] or {}
                        meta["entities"] = entity_str
                        self._collection.update(ids=[mem_id], metadatas=[meta])
            except Exception as e:
                logger.debug(f"异步实体提取失败: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── 维护 ────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """统计信息。"""
        return {
            "total_memories": self._collection.count(),
            "device": self._device,
            "data_dir": self._data_dir,
        }

    def cleanup(self) -> Dict[str, int]:
        """清理：删除对话摘要和过时信息。"""
        if self._collection.count() == 0:
            return {"removed": 0, "remaining": 0}

        all_data = self._collection.get(include=["documents"])
        to_delete = []
        for i, doc in enumerate(all_data["documents"]):
            if doc.startswith("[对话]"):
                to_delete.append(all_data["ids"][i])

        if to_delete:
            self._collection.delete(ids=to_delete)

        return {"removed": len(to_delete), "remaining": self._collection.count()}

    def maintenance(self) -> Dict[str, Any]:
        """日常维护：检测数值冲突 + 标记低频记忆。"""
        if self._collection.count() == 0:
            return {"conflicts": 0, "stale": 0}

        all_data = self._collection.get(include=["metadatas", "documents"])

        import re
        conflicts = 0
        seen_facts: Dict[str, str] = {}

        for i, (doc, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
            # GLiNER 提取实体
            try:
                with self._ner_lock:
                    entities = self._ner.predict_entities(doc, DEFAULT_ENTITY_LABELS, threshold=0.3)
            except Exception:
                entities = []

            persons = [e["text"] for e in entities if e["label"] == "person"]
            moneys = re.findall(r'[\d,.]+[万亿千百]', doc) + \
                     [e["text"] for e in entities if e["label"] == "money"]
            projects = [e["text"] for e in entities if e["label"] in ("organization", "technology")]

            subject = ",".join(persons[:1]) if persons else "unknown"
            for proj in projects[:2]:
                key = f"{subject}+{proj}"
                if key in seen_facts:
                    conflicts += 1
                else:
                    seen_facts[key] = doc

            for money in moneys[:2]:
                key = f"{subject}+money"
                if key in seen_facts:
                    conflicts += 1
                else:
                    seen_facts[key] = doc

        # 检查过期
        now = datetime.now(timezone.utc)
        stale = 0
        for meta in all_data["metadatas"]:
            updated = meta.get("updated_at") or meta.get("created_at", "")
            if updated:
                try:
                    dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    if (now - dt).days > 30:
                        stale += 1
                except Exception:
                    pass

        return {"conflicts": conflicts, "stale": stale}
