"""
本地记忆栈 API 服务 — BGE-M3 + ChromaDB + GLiNER
启动: python3 server.py
接口:
  POST /memory/write   — 写入记忆（自动提取实体）
  POST /memory/query   — 语义查询
  POST /memory/entities — 实体提取
  GET  /memory/stats   — 统计信息
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time, uuid, datetime, torch, threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# ── 全局线程池（异步实体提取） ─────────────────
_bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory")

# ── 全局模型（启动时加载一次，常驻内存） ──────────
models = {}

def load_models():
    """加载三大组件，只执行一次"""
    print("🚀 加载模型中...")
    t0 = time.time()

    # 1. BGE-M3 向量编码器
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   📦 BGE-M3 → {device}")
    from sentence_transformers import SentenceTransformer
    models["encoder"] = SentenceTransformer("BAAI/bge-m3", device=device)
    print(f"   ✅ BGE-M3 就绪")

    # 2. ChromaDB 持久存储
    import chromadb
    models["db"] = chromadb.PersistentClient(path="./chroma_data")
    models["collection"] = models["db"].get_or_create_collection(
        name="memories",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"   ✅ ChromaDB 就绪 ({models['collection'].count()} 条记忆)")

    # 3. GLiNER 实体提取
    print(f"   📦 GLiNER 加载中...")
    from gliner import GLiNER
    models["ner"] = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    print(f"   ✅ GLiNER 就绪")

    print(f"⚡ 全部加载完成，耗时 {time.time()-t0:.1f}s")

@asynccontextmanager
async def lifespan(app):
    load_models()
    yield
    print("👋 服务关闭")

app = FastAPI(title="本地记忆栈", lifespan=lifespan)

# ── 请求/响应模型 ──────────────────────────────
class WriteRequest(BaseModel):
    text: str
    source: Optional[str] = "api"      # 来源标记（api/chat/cron...）
    tags: Optional[list[str]] = None
    auto_extract: bool = True           # 自动提取实体

class QueryRequest(BaseModel):
    text: str
    top_k: int = 5
    threshold: float = 0.5
    source: Optional[str] = None  # 按来源分区查询（finance/profile/project...）

class EntityRequest(BaseModel):
    text: str
    labels: list[str] = ["person", "location", "organization", "money", "technology", "date", "event"]
    threshold: float = 0.3

class DeleteRequest(BaseModel):
    ids: list[str]

# ── 去重配置 ────────────────────────────────────
DEDUP_THRESHOLD = 0.85  # 语义相似度超过此值视为重复，合并而非新增

# ── 接口 ────────────────────────────────────────
@app.post("/memory/write")
async def write_memory(req: WriteRequest):
    """写入一条记忆，自动去重合并。实体提取异步执行，不阻塞返回。"""
    timestamp = datetime.datetime.now().isoformat()

    # 编码
    vector = models["encoder"].encode([req.text], normalize_embeddings=True)[0]

    # 去重：查询是否有语义相似的旧记忆
    if models["collection"].count() > 0:
        results = models["collection"].query(
            query_embeddings=[vector.tolist()],
            n_results=3
        )
        if results["documents"] and results["documents"][0]:
            # 找最相似的一条
            best_sim = 0
            best_i = 0
            for i, dist in enumerate(results["distances"][0]):
                sim = 1 - dist
                if sim > best_sim:
                    best_sim = sim
                    best_i = i

            if best_sim >= DEDUP_THRESHOLD:
                existing_text = results["documents"][0][best_i]
                existing_id = results["ids"][0][best_i]
                existing_meta = results["metadatas"][0][best_i] if results["metadatas"][0] else {}

                # 合并文本
                new_text = req.text if len(req.text) >= len(existing_text) else existing_text
                old_tags = existing_meta.get("tags", "").split(",") if existing_meta.get("tags") else []
                new_tags = list(set(old_tags + (req.tags or [])))
                new_vector = vector if len(req.text) >= len(existing_text) else models["encoder"].encode([existing_text], normalize_embeddings=True)[0]

                metadata = {
                    "source": f"{existing_meta.get('source', '')},{req.source}".strip(","),
                    "created_at": existing_meta.get("created_at", timestamp),
                    "updated_at": timestamp,
                    "merged_count": str(int(existing_meta.get("merged_count", "0")) + 1),
                }
                if new_tags:
                    metadata["tags"] = ",".join(new_tags)

                models["collection"].update(
                    ids=[existing_id],
                    documents=[new_text],
                    embeddings=[new_vector.tolist()],
                    metadatas=[metadata]
                )

                # 异步实体提取（后台跑，不阻塞返回）
                if req.auto_extract:
                    _bg_executor.submit(_extract_and_update, existing_id, new_text)

                return {
                    "id": existing_id,
                    "text": new_text[:200],
                    "action": "merged",
                    "similarity": round(best_sim, 4),
                    "metadata": metadata
                }

    # 新增
    mem_id = f"mem_{uuid.uuid4().hex[:12]}"
    metadata = {
        "source": req.source,
        "created_at": timestamp,
        "merged_count": "0",
    }
    if req.tags:
        metadata["tags"] = ",".join(req.tags)

    models["collection"].add(
        ids=[mem_id],
        documents=[req.text],
        embeddings=[vector.tolist()],
        metadatas=[metadata]
    )

    # 异步实体提取
    if req.auto_extract:
        _bg_executor.submit(_extract_and_update, mem_id, req.text)

    return {
        "id": mem_id,
        "text": req.text[:200],
        "action": "added",
        "metadata": metadata
    }


def _extract_and_update(mem_id: str, text: str):
    """后台线程：提取实体并更新 metadata"""
    try:
        entities = models["ner"].predict_entities(text,
            ["person","location","organization","money","technology","date","event"],
            threshold=0.3)
        if entities:
            entity_str = "|".join(f"{e['text']}:{e['label']}" for e in entities)
            result = models["collection"].get(ids=[mem_id], include=["metadatas"])
            if result["metadatas"]:
                meta = result["metadatas"][0] or {}
                meta["entities"] = entity_str
                models["collection"].update(ids=[mem_id], metadatas=[meta])
    except Exception as e:
        print(f"⚠️ 异步实体提取失败: {e}")

@app.post("/memory/query")
async def query_memory(req: QueryRequest):
    """语义查询记忆，支持按 source 分区"""
    query_vec = models["encoder"].encode([req.text], normalize_embeddings=True)[0]
    
    kwargs = {
        "query_embeddings": [query_vec.tolist()],
        "n_results": req.top_k,
    }
    if req.source:
        kwargs["where"] = {"source": req.source}
    
    results = models["collection"].query(**kwargs)

    memories = []
    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        similarity = 1 - dist  # cosine distance → similarity
        if similarity >= req.threshold:
            memories.append({
                "text": doc,
                "similarity": round(similarity, 4),
                "metadata": meta
            })

    return {"query": req.text, "results": memories, "count": len(memories)}

@app.post("/memory/entities")
async def extract_entities(req: EntityRequest):
    """纯实体提取（不写入）"""
    entities = models["ner"].predict_entities(req.text, req.labels, threshold=req.threshold)
    return {
        "text": req.text,
        "entities": [{"text": e["text"], "label": e["label"], "score": round(e["score"], 2)} for e in entities]
    }

@app.get("/memory/stats")
async def stats():
    """记忆库统计"""
    count = models["collection"].count()
    return {"total_memories": count, "device": str(models["encoder"].device)}

@app.post("/memory/delete")
async def delete_memories(req: DeleteRequest):
    """按ID列表批量删除记忆"""
    try:
        models["collection"].delete(ids=req.ids)
        return {"deleted": len(req.ids)}
    except Exception as e:
        return {"error": str(e), "deleted": 0}

@app.post("/memory/cleanup")
async def cleanup():
    """清理：删除所有对话摘要 + 过时猫品种信息"""
    count = models["collection"].count()
    if count == 0:
        return {"removed": 0, "remaining": 0}
    
    all_data = models["collection"].get(include=["documents"])
    to_delete = []
    for i, doc_id in enumerate(all_data["ids"]):
        doc = all_data["documents"][i]
        # 删除所有 "[对话]" 开头的
        if doc.startswith("[对话]"):
            to_delete.append(doc_id)
    
    if to_delete:
        models["collection"].delete(ids=to_delete)
    
    remaining = models["collection"].count()
    return {"removed": len(to_delete), "remaining": remaining}

@app.get("/health")
async def health():
    return {"status": "ok"}

# ── 启动 ────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8900)
