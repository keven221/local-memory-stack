"""Backfill missing metadata for existing memories.

This script only updates metadata on existing records. It does not create test
memories or delete user memories.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request
from typing import Any


API = "http://127.0.0.1:8900"


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(
        API + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode())


def list_all(limit: int = 100) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = post("/memory/list", {"limit": limit, "offset": offset})
        items = payload.get("items", [])
        memories.extend(items)
        if not items or offset + limit >= payload.get("total", len(memories)):
            return memories
        offset += limit


def build_backfill(mem: dict[str, Any], timestamp: str) -> dict[str, Any]:
    meta = mem.get("metadata") or {}
    patch: dict[str, Any] = {}
    if not meta.get("source"):
        patch["source"] = "legacy"
    if not meta.get("created_at"):
        patch["created_at"] = timestamp
    if not meta.get("merged_count"):
        patch["merged_count"] = "0"
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing memory metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned updates.")
    args = parser.parse_args()

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    memories = list_all()
    planned = [(mem, build_backfill(mem, timestamp)) for mem in memories]
    planned = [(mem, patch) for mem, patch in planned if patch]

    print(f"memories={len(memories)} missing_metadata={len(planned)} dry_run={args.dry_run}")
    for mem, patch in planned:
        print(f"  {mem.get('id')}: {patch} | {mem.get('text', '')[:60]}")
        if not args.dry_run:
            post("/memory/update", {"id": mem.get("id"), "metadata": patch})

    if not args.dry_run:
        print(f"updated={len(planned)}")


if __name__ == "__main__":
    main()
