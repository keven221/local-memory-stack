"""Benchmark local-memory-stack without leaving test memories behind."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import uuid
from typing import Any


API = "http://127.0.0.1:8900"


def request(method: str, path: str, payload: dict[str, Any] | None = None,
            timeout: int = 30) -> tuple[float, dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, json.loads(body) if body else {}


def post(path: str, payload: dict[str, Any], timeout: int = 30) -> tuple[float, dict[str, Any]]:
    return request("POST", path, payload, timeout)


def summarize(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} avg={statistics.mean(values):.1f}ms "
        f"p50={statistics.median(values):.1f}ms max={max(values):.1f}ms"
    )


def list_perf_records(run_id: str) -> list[dict[str, Any]]:
    _, payload = post("/memory/list", {"limit": 500, "offset": 0})
    items = payload.get("items", [])
    return [
        item for item in items
        if run_id in item.get("text", "") or (item.get("metadata") or {}).get("run_id") == run_id
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark memory read/write performance safely.")
    parser.add_argument("--writes", type=int, default=10, help="Number of temporary unique writes.")
    parser.add_argument("--queries", type=int, default=10, help="Number of repeated query rounds.")
    args = parser.parse_args()

    run_id = f"perf_{uuid.uuid4().hex}"
    created_ids: list[str] = []
    query_times: list[float] = []
    write_times: list[float] = []
    duplicate_times: list[float] = []
    list_times: list[float] = []

    print(f"run_id={run_id}")
    try:
        ms, stats = request("GET", "/memory/stats")
        print(f"stats: {ms:.1f}ms {stats}")

        for _ in range(3):
            ms, payload = post("/memory/list", {"limit": 100, "offset": 0})
            list_times.append(ms)
        print(f"list: {summarize(list_times)}")

        queries = ["Kevin", "探索 Agent", "闲鱼收入", "语言 去重 中文 英文", "项目 记忆 栈"]
        for _ in range(args.queries):
            for text in queries:
                ms, _ = post("/memory/query", {"text": text, "top_k": 5, "threshold": 0})
                query_times.append(ms)
        print(f"query: {summarize(query_times)}")

        for index in range(args.writes):
            text = f"{run_id} 临时压测记忆 #{index}: 本条用于测量写入速度，脚本结束必须删除。"
            ms, payload = post(
                "/memory/write",
                {
                    "text": text,
                    "source": "perf_test",
                    "tags": ["perf", run_id],
                    "auto_extract": False,
                },
            )
            write_times.append(ms)
            mem_id = payload.get("id")
            if mem_id and payload.get("action") != "skipped":
                created_ids.append(mem_id)
        print(f"write_unique: {summarize(write_times)}")

        duplicate_text = f"{run_id} 重复压测记忆: 本条用于测量重复写入去重速度，脚本结束必须删除。"
        _, first = post(
            "/memory/write",
            {"text": duplicate_text, "source": "perf_test", "tags": ["perf", run_id], "auto_extract": False},
        )
        if first.get("id"):
            created_ids.append(first["id"])
        for _ in range(max(3, min(args.writes, 10))):
            ms, _ = post(
                "/memory/write",
                {"text": duplicate_text, "source": "perf_test", "tags": ["perf", run_id], "auto_extract": False},
            )
            duplicate_times.append(ms)
        print(f"write_duplicate: {summarize(duplicate_times)}")
    finally:
        leftovers = list_perf_records(run_id)
        cleanup_ids = sorted({item.get("id") for item in leftovers if item.get("id")} | set(created_ids))
        if cleanup_ids:
            ms, payload = post("/memory/delete", {"ids": cleanup_ids})
            print(f"cleanup: {ms:.1f}ms deleted={payload.get('deleted')} ids={len(cleanup_ids)}")

        remaining = list_perf_records(run_id)
        if remaining:
            print(f"cleanup_verify: FAILED remaining={len(remaining)}")
            for item in remaining[:10]:
                print(f"  remaining {item.get('id')}: {item.get('text', '')[:80]}")
            raise SystemExit(1)
        print("cleanup_verify: OK no benchmark memories remain")


if __name__ == "__main__":
    main()
