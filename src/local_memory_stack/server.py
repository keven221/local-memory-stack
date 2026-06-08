"""FastAPI 服务 — 唯一入口，薄封装 MemoryEngine。

所有 API 调用都走 MemoryEngine，不再有独立实现。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_memory_stack import MemoryEngine
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

engine = MemoryEngine()


class WriteRequest(BaseModel):
    text: str
    source: str = "api"
    tags: Optional[list[str]] = None
    auto_extract: bool = True


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5
    threshold: float = 0.3
    source: Optional[str] = None


class EntityRequest(BaseModel):
    text: str
    labels: list[str] = ["person", "location", "organization", "money", "technology", "date", "event"]
    threshold: float = 0.3


class UpdateRequest(BaseModel):
    id: str = ""
    metadata: dict = {}
    delete_keys: Optional[list[str]] = None
    replace: bool = False


class ListRequest(BaseModel):
    limit: int = 100
    offset: int = 0
    needs_review: Optional[bool] = None


class DeleteRequest(BaseModel):
    ids: list[str]


class ArchiveRequest(BaseModel):
    dry_run: bool = False


class RestoreRequest(BaseModel):
    ids: list[str]


class SearchAllRequest(BaseModel):
    text: str
    top_k: int = 5
    threshold: float = 0.3


@asynccontextmanager
async def lifespan(app):
    yield

app = FastAPI(title="Local Memory Stack", lifespan=lifespan)


@app.post("/memory/write")
async def write_memory(req: WriteRequest):
    return engine.write(req.text, source=req.source, tags=req.tags, auto_extract=req.auto_extract)


@app.post("/memory/query")
async def query_memory(req: QueryRequest):
    entries = engine.query(req.text, top_k=req.top_k, threshold=req.threshold, source=req.source)
    return {
        "query": req.text,
        "results": [
            {
                "id": e.id,
                "text": e.text,
                "similarity": e.similarity,
                "metadata": {
                    "source": e.source,
                    "tags": e.tags,
                    "entities": e.entities,
                },
            }
            for e in entries
        ],
        "count": len(entries),
    }


@app.post("/memory/entities")
async def extract_entities(req: EntityRequest):
    entities = engine.extract_entities(req.text, req.labels, req.threshold)
    return {"text": req.text, "entities": entities}


@app.get("/memory/stats")
async def stats():
    return engine.stats()


@app.post("/memory/cleanup")
async def cleanup():
    return engine.cleanup()


@app.post("/memory/delete")
async def delete_memories(req: DeleteRequest):
    return engine.delete(req.ids)


@app.post("/memory/list")
async def list_memories(req: ListRequest = None):
    """列出所有记忆（支持 needs_review 过滤）。"""
    if req is None:
        req = ListRequest()
    return engine.list_memories(limit=req.limit, offset=req.offset, needs_review=req.needs_review)


@app.patch("/memory/update")
@app.post("/memory/update")
async def update_metadata(mem_id: str = "", req: UpdateRequest = None):
    """更新指定记忆的 metadata。兼容两种调用：
    - PATCH/POST body={"id": "xxx", "metadata": {...}}
    - PATCH/POST body={"id": "xxx", "delete_keys": ["needs_review"]}
    - POST ?mem_id=xxx body={"metadata": {...}}
    """
    target_id = mem_id or (req.id if req else "")
    target_meta = req.metadata if req else {}
    if not target_id:
        return {"error": "no id provided"}
    return engine.update_metadata(
        target_id,
        target_meta,
        delete_keys=req.delete_keys if req else None,
        replace=req.replace if req else False,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 归档系统 API ──────────────────────────────

@app.post("/memory/archive")
async def archive_memories(req: ArchiveRequest = ArchiveRequest()):
    """归档超龄记忆。dry_run=True 只报告不执行。"""
    return engine.archive(dry_run=req.dry_run)


@app.post("/memory/archive/search")
async def search_archived(req: QueryRequest):
    """搜索归档记忆。"""
    entries = engine.search_archived(req.text, top_k=req.top_k, threshold=req.threshold)
    return {
        "query": req.text,
        "results": [
            {"id": e.id, "text": e.text, "similarity": e.similarity,
             "metadata": {"source": e.source, "tags": e.tags, "entities": e.entities}}
            for e in entries
        ],
        "count": len(entries),
    }


@app.post("/memory/archive/restore")
async def restore_memories(req: RestoreRequest):
    """从归档恢复到活跃。"""
    return engine.restore(req.ids)


@app.post("/memory/search_all")
async def search_all_memories(req: SearchAllRequest):
    """同时搜活跃 + 归档，分组返回。"""
    result = engine.search_all(req.text, top_k=req.top_k, threshold=req.threshold)
    return {
        "query": req.text,
        "active": [
            {"id": e.id, "text": e.text, "similarity": e.similarity}
            for e in result["active"]
        ],
        "archived": [
            {"id": e.id, "text": e.text, "similarity": e.similarity}
            for e in result["archived"]
        ],
        "total": result["total"],
    }


@app.post("/memory/backfill_ttl")
async def backfill_ttl():
    """为旧记忆补填 TTL 元数据。"""
    return engine.backfill_ttl()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8900)
