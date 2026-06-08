"""MemoryEngine — 本地语义记忆核心引擎。

零外部依赖（除 ML 模型），任何 Agent 框架都可直接使用。

用法:
    engine = MemoryEngine()
    engine.write("ThisIsMyCat", source="profile")
    results = engine.query("User的猫是什么", top_k=3)
    engine.maintenance()  # 定期清理
"""

from __future__ import annotations

import json
import os
import re
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

# ── 去重配置（三档策略 + 二次判定）──────────────
DEDUP_THRESHOLD_RECALL = 0.55   # 第一轮：embedding 召回候选（粗筛）
DEDUP_KEYWORD_OVERLAP = 0.25    # 第二轮：关键词重叠率阈值（低于此认为不同主题）
DEDUP_THRESHOLD_SKIP = 0.84     # 高度重复：跳过不存
DEDUP_THRESHOLD_MERGE = 0.72    # 灰色地带：打标等审核
# 决策流程：
# 1. embedding 召回 top-3，找最佳候选（sim >= 0.65）
# 2. 对候选做关键词重叠判定：
#    - 关键词重叠率 >= 0.35 → 同主题，用 embedding 相似度分三档
#    - 关键词重叠率 < 0.35 → 不同主题，视为全新

DEFAULT_ENTITY_LABELS = ["person", "location", "organization", "money", "technology", "date", "event"]

# ── TTL / 归档配置 ─────────────────────────────────
TAG_TTL_DAYS = {
    "static": 365,       # 用户身份/偏好 → 一年
    "dynamic": 180,      # 项目/计划 → 半年
    "temporary": 30,     # 临时事项 → 一个月
    "preference": 365,   # 偏好 → 一年
    "feedback": 365,     # 反馈 → 一年
    "reference": 365,    # 参考链接 → 一年
}
DEFAULT_TTL_DAYS = 180  # 无标签时的默认 TTL


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
    """本地语义记忆引擎，封装 BGE-M3 + ChromaDB + GLiNER。

    三档去重策略（写入时实时判断）：
        - similarity > 0.88: 高度重复 → 跳过
        - 0.78~0.88: 灰色地带 → 两条都保留，打 needs_review 标记
        - < 0.78: 全新 → 直接存
    """

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
        self._write_lock = threading.Lock()  # 写入锁，防止连续写入时序窗口

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
        self._archive = self._db.get_or_create_collection(
            name="memories_archive",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB 就绪 (活跃: {self._collection.count()}, 归档: {self._archive.count()})")

        from gliner import GLiNER
        logger.info("加载 GLiNER...")
        self._ner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        logger.info(f"GLiNER 就绪 ({time.time()-t0:.1f}s 总计)")

    @staticmethod
    def _extract_keywords(text: str) -> list:
        """从文本提取关键词（jieba 分词 + 去停用词 + 保留数字/单位）。"""
        import jieba
        
        # 用 jieba 精确模式分词
        tokens = list(jieba.cut(text))
        
        stopwords = {'包括但不限于','对于','以及','关于','所有','这些','一个','一种','这个','那个',
                     '已经','可以','需要','必须','不能','不是','没有','不会','应该','可能',
                     '但是','因为','所以','如果','虽然','而且','或者','然后','那么','这样',
                     '的','是','在','和','了','有','不','也','都','就','但','而','与','或',
                     '了','着','过','把','被','让','给','对','从','到','上','下','中','里',
                     '很','非常','比较','更加','最','太','还','再','又','也','都',
                     '我','你','他','她','它','我们','你们','他们',
                     'the','and','for','with','this','that','from','has','was',
                     '，','。','！','？','：','；','、','（','）','【','】','…',' ', '\n', '\t'}
        
        keywords = []
        for t in tokens:
            t = t.strip()
            if not t or t in stopwords:
                continue
            # 保留：数字(含单位)、2字以上中文、3字以上英文
            if re.match(r'[\d,.]+[万亿千百%]?', t):
                keywords.append(t)
            elif re.match(r'[\u4e00-\u9fff]{2,}', t):
                keywords.append(t)
            elif re.match(r'[a-zA-Z]{3,}', t):
                keywords.append(t.lower())
        
        # 去重保序
        seen = set()
        result = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return result[:25]

    # ── 写入（三档去重）────────────────────────

    def write(self, text: str, source: str = "api", tags: Optional[List[str]] = None,
              auto_extract: bool = True) -> Dict[str, Any]:
        """写入一条记忆，三档去重策略。

        Returns:
            {"id": ..., "action": "added"|"skipped"|"flagged", ...}
        """
        with self._write_lock:
            # ── 质量门控：低价值内容直接拒绝 ──
            if not self._is_worth_storing(text):
                logger.info(f"质量门控拒绝: {text[:80]}")
                return {"id": "", "text": text[:200], "action": "rejected",
                        "reason": "质量门控：低价值内容（临时状态/代码产物/任务进度）"}

            now = datetime.now(timezone.utc).isoformat()
            vector = self._encoder.encode([text], normalize_embeddings=True)[0]

            # 去重检查：两轮判定
            if self._collection.count() > 0:
                results = self._collection.query(query_embeddings=[vector.tolist()], n_results=3)
                if results["documents"] and results["documents"][0]:
                    best_sim, best_i = 0, 0
                    for i, dist in enumerate(results["distances"][0]):
                        sim = 1 - dist
                        if sim > best_sim:
                            best_sim, best_i = sim, i

                    logger.info(f"去重候选: best_sim={best_sim:.3f}, recall_threshold={DEDUP_THRESHOLD_RECALL}, total_candidates={len(results['distances'][0])}")

                    existing_id = results["ids"][0][best_i]
                    existing_text = results["documents"][0][best_i]
                    existing_meta = results["metadatas"][0][best_i] if results["metadatas"][0] else {}

                    if best_sim >= DEDUP_THRESHOLD_RECALL:
                        # 第二轮：关键词重叠判定
                        new_kw = set(self._extract_keywords(text))
                        old_kw = set(self._extract_keywords(existing_text))
                        overlap = len(new_kw & old_kw) / max(len(new_kw | old_kw), 1) if (new_kw or old_kw) else 0

                        logger.info(f"去重二判: sim={best_sim:.3f}, kw_overlap={overlap:.3f}, new_kw={new_kw}, old_kw={old_kw}")
                        if overlap < DEDUP_KEYWORD_OVERLAP:
                            # 关键词差异大 → 不同主题，视为全新（即使 embedding 相似）
                            logger.debug(f"去重跳过: embedding={best_sim:.2f} 但关键词重叠={overlap:.2f}")
                        else:
                            # 同主题，走三档策略
                            if best_sim >= DEDUP_THRESHOLD_SKIP:
                                return {
                                    "id": existing_id,
                                    "text": existing_text[:200],
                                    "action": "skipped",
                                    "similarity": round(best_sim, 4),
                                    "reason": "高度重复，核心语义一致",
                                }

                            if best_sim >= DEDUP_THRESHOLD_MERGE:
                                return self._flag_for_review(
                                    existing_id=existing_id,
                                    existing_text=existing_text,
                                    existing_meta=existing_meta,
                                    new_text=text,
                                    new_vector=vector,
                                    source=source,
                                    tags=tags or [],
                                    similarity=best_sim,
                                    timestamp=now,
                                    auto_extract=auto_extract,
                                )

            # 新增
            return self._add_new(text=text, vector=vector, source=source,
                                tags=tags or [], timestamp=now, auto_extract=auto_extract)

    def _is_worth_storing(self, text: str) -> bool:
        """质量门控：过滤低价值内容，避免噪音入库。

        返回 False = 跳过存储（明显是临时状态/代码产物/任务进度）。
        边界情况宁可放过——dedup 层会处理近似重复。
        """
        t = text.strip()

        # 太短 = 信息量不足
        if len(t) < 10:
            return False

        low = t.lower()

        # ── 硬排除：文件路径 / Git / 临时状态 ──
        noise_patterns = [
            r'^/[\w/]+\.\w+$',                      # 纯文件路径
            r'^[a-f0-9]{7,40}$',                    # Git commit hash
            r'(?:fixed?|resolved?|completed?|done)\s+(?:bug|issue|error)',
            r'^PR\s*#\d+',                           # PR 引用
            r'^commit\s+[a-f0-9]',                   # git commit 消息
            r'temporary|temp fix|workaround',        # 临时方案
            r'^Running\s+tests?',                    # 测试运行
            r'^Build\s+(?:passed|failed)',           # CI 状态
        ]
        if any(re.search(p, low) for p in noise_patterns):
            return False

        # ── 代码产物（可从仓库读到）──
        code_noise = [
            r'(?:class|def|function|import|from)\s+\w+',
            r'(?:\.py|\.js|\.ts|\.yaml|\.json|\.toml)\b.*(?:modified|updated|changed)',
            r'(?:error|traceback|exception):.*(?:line|file|module)',
            r'File\s+"[^"]+",\s+line\s+\d+',
        ]
        if any(re.search(p, t) for p in code_noise):
            return False

        # ── 临时任务状态 ──
        task_noise = [
            r'^(?:Phase|Step|阶段)\s*\d+\s*(?:完成|done|completed)',
            r'已(?:完成|修复|解决|处理)\s*[：:]',
            r'^\d+\s*(?:个|条)\s*(?:文件|commit|PR)',
        ]
        if any(re.search(p, t) for p in task_noise):
            return False

        return True

    def _add_new(self, text, vector, source, tags, timestamp, auto_extract) -> dict:
        mem_id = f"mem_{uuid.uuid4().hex[:12]}"
        # 自动计算 TTL：取 tags 中最具体的 TTL（有标签用标签值，没有才用默认值）
        ttl = None
        for t in tags:
            if t in TAG_TTL_DAYS:
                if ttl is None or TAG_TTL_DAYS[t] < ttl:
                    ttl = TAG_TTL_DAYS[t]  # 取最短的（最具体的标签优先）
        if ttl is None:
            ttl = DEFAULT_TTL_DAYS
        meta = {"source": source, "created_at": timestamp, "merged_count": "0",
                "ttl_days": str(ttl), "last_accessed_at": timestamp}
        if tags:
            meta["tags"] = json.dumps(tags, ensure_ascii=False)

        self._collection.add(
            ids=[mem_id], documents=[text],
            embeddings=[vector.tolist()], metadatas=[meta]
        )

        # 强制 flush，防止连续写入时下一条查不到这一条
        self._collection.get(ids=[mem_id])

        if auto_extract:
            self._extract_async(mem_id, text)

        return {"id": mem_id, "text": text[:200], "action": "added", "metadata": meta}

    def _flag_for_review(self, existing_id, existing_text, existing_meta,
                         new_text, new_vector, source, tags, similarity,
                         timestamp, auto_extract) -> dict:
        """灰色地带：两条都保留，但打 needs_review 标记。"""
        mem_id = f"mem_{uuid.uuid4().hex[:12]}"
        meta = {
            "source": source,
            "created_at": timestamp,
            "merged_count": "0",
            "needs_review": "true",
            "review_similar_to": existing_id,
            "review_similarity": str(round(similarity, 4)),
        }
        if tags:
            meta["tags"] = json.dumps(tags, ensure_ascii=False)

        self._collection.add(
            ids=[mem_id], documents=[new_text],
            embeddings=[new_vector.tolist()], metadatas=[meta]
        )

        # 给旧的那条也打标
        existing_meta["needs_review"] = "true"
        existing_meta["review_similar_to"] = mem_id
        existing_meta["review_similarity"] = str(round(similarity, 4))
        self._collection.update(ids=[existing_id], metadatas=[existing_meta])

        if auto_extract:
            self._extract_async(mem_id, new_text)

        return {
            "id": mem_id,
            "text": new_text[:200],
            "action": "flagged",
            "similarity": round(similarity, 4),
            "reason": "灰色地带，已打标等待审核",
            "similar_to": existing_text[:200],
            "metadata": meta,
        }

    # ── 查询 ────────────────────────────────

    def query(self, text: str, top_k: int = 5, threshold: float = 0.3,
              source: Optional[str] = None) -> List[MemoryEntry]:
        """语义查询记忆。"""
        query_vec = self._encoder.encode([text], normalize_embeddings=True)[0]
        kwargs: dict = {"query_embeddings": [query_vec.tolist()], "n_results": top_k}
        if source:
            kwargs["where"] = {"source": source}

        results = self._collection.query(**kwargs)

        entries = []
        if not results["documents"] or not results["documents"][0]:
            return entries

        for doc, dist, meta, mid in zip(
            results["documents"][0], results["distances"][0],
            results["metadatas"][0], results["ids"][0]
        ):
            meta = meta or {}
            sim = 1 - dist
            if sim >= threshold:
                tags = self._parse_tags(meta.get("tags", ""))
                entities = self._parse_entities(meta.get("entities", ""))

                entries.append(MemoryEntry(
                    id=mid,  # ← 从 ChromaDB ids 回填，不是从 metadata
                    text=doc,
                    source=meta.get("source", ""),
                    tags=tags,
                    entities=entities,
                    merged_count=int(meta.get("merged_count", 0)),
                    created_at=meta.get("created_at", ""),
                    updated_at=meta.get("updated_at", ""),
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

    # ── 列表 / 更新 / 删除 ──────────────────────

    def list_memories(self, limit: int = 100, offset: int = 0,
                      needs_review: Optional[bool] = None) -> Dict[str, Any]:
        """列出所有记忆（支持分页和 needs_review 过滤）。"""
        count = self._collection.count()
        include = ["documents", "metadatas"]

        if needs_review is True:
            all_data = self._collection.get(include=include)
            filtered_ids, filtered_docs, filtered_metas = [], [], []
            for i, (mid, doc, meta) in enumerate(zip(all_data["ids"], all_data["documents"], all_data["metadatas"])):
                if meta and meta.get("needs_review") == "true":
                    filtered_ids.append(mid)
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
            items = [
                {"id": mid, "text": doc, "metadata": meta}
                for mid, doc, meta in zip(filtered_ids, filtered_docs, filtered_metas)
            ]
            return {"total": len(items), "items": items[offset:offset+limit]}
        else:
            all_data = self._collection.get(include=include)
            items = [
                {"id": mid, "text": doc, "metadata": meta}
                for mid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
            ]
            return {"total": len(items), "items": items[offset:offset+limit]}

    def update_metadata(self, mem_id: str, metadata: Dict[str, Any],
                        delete_keys: Optional[List[str]] = None,
                        replace: bool = False) -> Dict[str, Any]:
        """更新指定记忆的 metadata（不改变文本和向量）。

        默认合并 metadata；传 delete_keys 可删除指定字段；replace=True 时整体替换。
        """
        result = self._collection.get(ids=[mem_id], include=["metadatas"])
        if not result["metadatas"]:
            return {"error": "not found"}

        existing_meta = result["metadatas"][0] or {}
        next_meta = dict(metadata) if replace else {**existing_meta, **metadata}
        for key in delete_keys or []:
            next_meta.pop(key, None)

        self._collection.update(ids=[mem_id], metadatas=[next_meta])
        return {"id": mem_id, "action": "updated", "metadata": next_meta}

    def delete(self, ids: List[str]) -> Dict[str, int]:
        """按 ID 删除记忆（自动去重防止 ChromaDB 批量失败）。"""
        if not ids:
            return {"deleted": 0}
        unique_ids = list(set(ids))
        try:
            self._collection.delete(ids=unique_ids)
            return {"deleted": len(unique_ids)}
        except Exception as e:
            logger.warning(f"删除失败: {e}")
            return {"deleted": 0}

    # ── 维护 ────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """统计信息。"""
        return {
            "total_memories": self._collection.count(),
            "archived_memories": self._archive.count(),
            "device": self._device,
            "data_dir": self._data_dir,
        }

    def cleanup(self) -> Dict[str, int]:
        """清理：删除对话摘要。"""
        if self._collection.count() == 0:
            return {"removed": 0, "remaining": 0}

        all_data = self._collection.get(include=["documents"])
        to_delete = [all_data["ids"][i] for i, doc in enumerate(all_data["documents"])
                     if doc.startswith("[对话]")]

        if to_delete:
            self.delete(to_delete)

        return {"removed": len(to_delete), "remaining": self._collection.count()}

    def maintenance(self) -> Dict[str, Any]:
        """日常维护：检测数值冲突 + 标记低频记忆。"""
        if self._collection.count() == 0:
            return {"conflicts": 0, "stale": 0}

        all_data = self._collection.get(include=["metadatas", "documents"])

        conflicts = 0
        seen_facts: Dict[str, str] = {}

        for i, (doc, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
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

    # ── 归档系统 ────────────────────────────────

    def archive(self, dry_run: bool = False, now: Optional[datetime] = None) -> Dict[str, Any]:
        """扫描活跃记忆，将超龄的移到归档集合。

        判定逻辑：created_at + ttl_days < now → 归档
        对于没有 ttl_days 的旧记忆，用 tag 推断或 DEFAULT_TTL_DAYS。

        Args:
            dry_run: True 时只报告不执行
            now: 测试用时间，默认当前 UTC
        """
        if now is None:
            now = datetime.now(timezone.utc)

        all_data = self._collection.get(include=["documents", "metadatas", "embeddings"])
        if not all_data["ids"]:
            return {"archived": 0, "remaining": 0}

        to_archive_ids = []
        to_archive_docs = []
        to_archive_embeds = []
        to_archive_metas = []

        for i, (mid, doc, meta, embed) in enumerate(zip(
            all_data["ids"], all_data["documents"],
            all_data["metadatas"], all_data["embeddings"]
        )):
            meta = meta or {}
            created_str = meta.get("created_at", "")
            if not created_str:
                continue

            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except Exception:
                continue

            # TTL：优先用 metadata 里的，否则按 tag 推断
            ttl_str = meta.get("ttl_days", "")
            if ttl_str:
                ttl = int(ttl_str)
            else:
                tags = self._parse_tags(meta.get("tags", ""))
                ttl = None
                for t in tags:
                    if t in TAG_TTL_DAYS:
                        if ttl is None or TAG_TTL_DAYS[t] < ttl:
                            ttl = TAG_TTL_DAYS[t]
                if ttl is None:
                    ttl = DEFAULT_TTL_DAYS

            age_days = (now - created).days
            if age_days > ttl:
                # 更新 last_accessed_at（如果有）
                meta["archived_at"] = now.isoformat()
                to_archive_ids.append(mid)
                to_archive_docs.append(doc)
                to_archive_embeds.append(embed)
                to_archive_metas.append(meta)

        if dry_run:
            return {"would_archive": len(to_archive_ids),
                    "examples": [{"id": mid, "text": doc[:80]}
                                 for mid, doc in zip(to_archive_ids[:5], to_archive_docs[:5])]}

        if to_archive_ids:
            # 写入归档集合
            self._archive.add(
                ids=to_archive_ids,
                documents=to_archive_docs,
                embeddings=to_archive_embeds,
                metadatas=to_archive_metas,
            )
            # 从活跃集合删除
            self._collection.delete(ids=to_archive_ids)

        return {"archived": len(to_archive_ids),
                "remaining": self._collection.count(),
                "archive_total": self._archive.count()}

    def restore(self, mem_ids: List[str]) -> Dict[str, Any]:
        """从归档恢复到活跃集合。"""
        if not mem_ids:
            return {"restored": 0}
        data = self._archive.get(ids=mem_ids, include=["documents", "metadatas", "embeddings"])
        if not data["ids"]:
            return {"restored": 0, "reason": "not found in archive"}

        # 清除 archived_at 标记
        metas = []
        for m in data["metadatas"]:
            m = m or {}
            m.pop("archived_at", None)
            metas.append(m)

        self._collection.add(
            ids=data["ids"], documents=data["documents"],
            embeddings=data["embeddings"], metadatas=metas,
        )
        self._archive.delete(ids=mem_ids)
        return {"restored": len(data["ids"]), "remaining_archive": self._archive.count()}

    def search_archived(self, text: str, top_k: int = 5,
                        threshold: float = 0.3) -> List[MemoryEntry]:
        """只搜归档记忆。"""
        query_vec = self._encoder.encode([text], normalize_embeddings=True)[0]
        results = self._archive.query(
            query_embeddings=[query_vec.tolist()], n_results=top_k
        )
        entries = []
        if not results["documents"] or not results["documents"][0]:
            return entries
        for doc, dist, meta, mid in zip(
            results["documents"][0], results["distances"][0],
            results["metadatas"][0], results["ids"][0]
        ):
            meta = meta or {}
            sim = 1 - dist
            if sim >= threshold:
                entries.append(MemoryEntry(
                    id=mid, text=doc, source=meta.get("source", ""),
                    tags=self._parse_tags(meta.get("tags", "")),
                    entities=self._parse_entities(meta.get("entities", "")),
                    merged_count=int(meta.get("merged_count", 0)),
                    created_at=meta.get("created_at", ""),
                    updated_at=meta.get("updated_at", ""),
                    similarity=round(sim, 4),
                ))
        return entries

    def search_all(self, text: str, top_k: int = 5,
                   threshold: float = 0.3) -> Dict[str, List[MemoryEntry]]:
        """同时搜活跃 + 归档，分组返回。"""
        active = self.query(text, top_k=top_k, threshold=threshold)
        archived = self.search_archived(text, top_k=top_k, threshold=threshold)
        return {"active": active, "archived": archived,
                "total": len(active) + len(archived)}

    def backfill_ttl(self) -> Dict[str, int]:
        """为旧记忆补填 TTL 元数据（无 ttl_days 字段的）。"""
        all_data = self._collection.get(include=["metadatas"])
        updated = 0
        for mid, meta in zip(all_data["ids"], all_data["metadatas"]):
            meta = meta or {}
            if "ttl_days" in meta:
                continue  # 已有 TTL
            tags = self._parse_tags(meta.get("tags", ""))
            ttl = None
            for t in tags:
                if t in TAG_TTL_DAYS:
                    if ttl is None or TAG_TTL_DAYS[t] < ttl:
                        ttl = TAG_TTL_DAYS[t]
            if ttl is None:
                ttl = DEFAULT_TTL_DAYS
            meta["ttl_days"] = str(ttl)
            if "last_accessed_at" not in meta:
                meta["last_accessed_at"] = meta.get("created_at", "")
            self._collection.update(ids=[mid], metadatas=[meta])
            updated += 1
        return {"backfilled": updated, "total": self._collection.count()}

    # ── 内部工具 ──────────────────────────────

    @staticmethod
    def _parse_tags(raw: str) -> List[str]:
        """解析 tags 字段（兼容旧逗号格式和新 JSON 格式）。"""
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return [t.strip() for t in raw.split(",") if t.strip()]

    @staticmethod
    def _parse_entities(raw: str) -> List[Dict[str, Any]]:
        """解析 entities 字段。"""
        if not raw:
            return []
        entities = []
        for e in raw.split("|"):
            parts = e.rsplit(":", 1)
            entities.append({"text": parts[0], "label": parts[1] if len(parts) > 1 else "unknown"})
        return entities
