"""FastAPI 服务 — 薄封装 MemoryEngine，提供 REST API。"""
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

class DeleteRequest(BaseModel):
    ids: list[str]

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
        "results": [{"text": e.text, "similarity": e.similarity, "metadata": {"id": e.id, "source": e.source, "tags": ",".join(e.tags), "entities": "|".join(f"{x['text']}:{x['label']}" for x in e.entities)}} for e in entries],
        "count": len(entries)
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
    return {"error": "Use /memory/cleanup for bulk cleanup"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8900)
