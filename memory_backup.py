"""Export and restore local-memory-stack memories as JSON.

Export is exact for text and metadata. Restore uses the public API, so Chroma IDs
may change; metadata is re-applied after each write. Use this for migration and
disaster recovery, not for merging two active databases blindly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


API = "http://127.0.0.1:8900"


def request(method: str, path: str, payload: dict[str, Any] | None = None,
            timeout: int = 30) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode()
    return json.loads(body) if body else {}


def list_all(limit: int = 100) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = request("POST", "/memory/list", {"limit": limit, "offset": offset})
        items = payload.get("items", [])
        memories.extend(items)
        if not items or offset + limit >= payload.get("total", len(memories)):
            return memories
        offset += limit


def export_memories(output: Path | None) -> Path:
    memories = list_all()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "exported_at": now,
        "count": len(memories),
        "items": memories,
    }

    if output is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path("backups") / f"local-memory-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported={len(memories)} path={output}")
    return output


def restore_memories(path: Path, dry_run: bool, source_suffix: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    print(f"restore_plan items={len(items)} dry_run={dry_run}")

    restored = 0
    skipped = 0
    for item in items:
        text = item.get("text", "")
        metadata = dict(item.get("metadata") or {})
        if not text:
            skipped += 1
            continue

        source = metadata.get("source") or "restore"
        if source_suffix:
            source = f"{source}{source_suffix}"
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else None

        if dry_run:
            print(f"  would_restore old_id={item.get('id')} source={source} text={text[:60]}")
            restored += 1
            continue

        write_payload = {
            "text": text,
            "source": source,
            "tags": tags,
            "auto_extract": False,
        }
        result = request("POST", "/memory/write", write_payload)
        new_id = result.get("id")
        if not new_id:
            skipped += 1
            continue

        metadata["restored_from_id"] = item.get("id", "")
        metadata["restored_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        metadata["source"] = source
        request("POST", "/memory/update", {"id": new_id, "metadata": metadata})
        restored += 1

    print(f"restored={restored} skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or restore local memory JSON backups.")
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="Export all memories to JSON.")
    export_parser.add_argument("-o", "--output", type=Path, default=None)

    restore_parser = sub.add_parser("restore", help="Restore memories from JSON through the API.")
    restore_parser.add_argument("path", type=Path)
    restore_parser.add_argument("--dry-run", action="store_true")
    restore_parser.add_argument(
        "--source-suffix",
        default="",
        help="Optional suffix appended to restored source values, e.g. ':restored'.",
    )

    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if args.command == "export":
        export_memories(args.output)
    elif args.command == "restore":
        restore_memories(args.path, args.dry_run, args.source_suffix)


if __name__ == "__main__":
    main()
