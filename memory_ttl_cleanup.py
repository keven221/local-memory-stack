#!/usr/bin/env python3
"""记忆 TTL 清理 + 归档脚本。

功能：
  1. 为旧记忆补填 TTL 元数据（backfill）
  2. dry-run 预览将被归档的记忆
  3. 执行归档（超龄记忆移到 memories_archive 集合）
  4. 统计报告（活跃/归档/各 TTL 分布）

用法：
  python3 memory_ttl_cleanup.py              # dry-run 预览
  python3 memory_ttl_cleanup.py --execute    # 真正执行归档
  python3 memory_ttl_cleanup.py --stats      # 只看统计
  python3 memory_ttl_cleanup.py --backfill   # 只补填 TTL 元数据
  python3 memory_ttl_cleanup.py --restore ID # 从归档恢复

适合 cron 调用：python3 memory_ttl_cleanup.py --execute --quiet
"""

import argparse
import sys
import requests
import json
from collections import Counter
from datetime import datetime, timezone

API = "http://127.0.0.1:8900"


def api_get(endpoint, timeout=10):
    """调用记忆栈 GET API。"""
    try:
        r = requests.get(f"{API}{endpoint}", timeout=timeout)
        return r.json()
    except requests.ConnectionError:
        print(f"❌ 记忆栈未启动（{API}）。请先启动：")
        print(f"   cd ~/projects/local-memory-stack && venv/bin/uvicorn src.local_memory_stack.server:app --port 8900")
        sys.exit(1)
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return None


def api_post(endpoint, data=None, timeout=30):
    """调用记忆栈 API。"""
    try:
        r = requests.post(f"{API}{endpoint}", json=data or {}, timeout=timeout)
        return r.json()
    except requests.ConnectionError:
        print(f"❌ 记忆栈未启动（{API}）。请先启动：")
        print(f"   cd ~/projects/local-memory-stack && venv/bin/uvicorn src.local_memory_stack.server:app --port 8900")
        sys.exit(1)
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return None


def fetch_all_memories():
    """获取所有活跃记忆。"""
    memories = []
    offset = 0
    limit = 200
    while True:
        r = api_post("/memory/list", {"limit": limit, "offset": offset})
        if not r:
            break
        items = r.get("items", [])
        memories.extend(items)
        if offset + limit >= r.get("total", len(memories)) or not items:
            break
        offset += limit
    return memories


def show_stats(quiet=False):
    """显示统计信息。"""
    stats = api_get("/memory/stats")
    if not stats:
        return

    if quiet:
        print(f"active={stats.get('total_memories', 0)} archived={stats.get('archived_memories', 0)}")
        return

    print(f"\n📊 记忆栈统计")
    print(f"   活跃: {stats.get('total_memories', 0)} 条")
    print(f"   归档: {stats.get('archived_memories', 0)} 条")
    print(f"   设备: {stats.get('device', '?')}")
    print(f"   存储: {stats.get('data_dir', '?')}")

    # TTL 分布
    memories = fetch_all_memories()
    if not memories:
        return

    ttl_dist = Counter()
    no_ttl = 0
    for mem in memories:
        meta = mem.get("metadata", {}) or {}
        ttl = meta.get("ttl_days", "")
        if ttl:
            ttl_dist[int(ttl)] += 1
        else:
            no_ttl += 1

    print(f"\n   TTL 分布:")
    for days, count in sorted(ttl_dist.items()):
        label = {30: "temporary", 180: "dynamic/default", 365: "static/preference"}.get(days, "")
        print(f"     {days}天 ({label}): {count} 条")
    if no_ttl:
        print(f"     无TTL（旧记忆）: {no_ttl} 条 → 需要 backfill")


def do_backfill(quiet=False):
    """为旧记忆补填 TTL。"""
    result = api_post("/memory/backfill_ttl")
    if not result:
        return
    n = result.get("backfilled", 0)
    total = result.get("total", 0)
    if quiet:
        print(f"backfill: {n}/{total}")
    else:
        print(f"✅ 补填 TTL: {n} 条（共 {total} 条）")


def do_preview(quiet=False):
    """Dry-run 预览归档。"""
    result = api_post("/memory/archive", {"dry_run": True})
    if not result:
        return

    n = result.get("would_archive", 0)
    examples = result.get("examples", [])

    if quiet:
        print(f"would_archive={n}")
        return

    if n == 0:
        print("✅ 没有需要归档的记忆（全部在 TTL 内）")
        return

    print(f"\n⚠️ 将归档 {n} 条记忆:")
    for ex in examples:
        print(f"   [{ex['id']}] {ex['text']}")
    if n > 5:
        print(f"   ... 还有 {n - 5} 条")
    print(f"\n   加 --execute 执行归档")


def do_archive(quiet=False):
    """执行归档。"""
    if not quiet:
        print("🔄 正在归档...")
    result = api_post("/memory/archive", {"dry_run": False})
    if not result:
        return

    archived = result.get("archived", 0)
    remaining = result.get("remaining", 0)
    archive_total = result.get("archive_total", 0)

    if quiet:
        print(f"archived={archived} remaining={remaining} archive_total={archive_total}")
    else:
        if archived == 0:
            print("✅ 没有需要归档的记忆")
        else:
            print(f"✅ 归档完成: {archived} 条 → 归档库")
            print(f"   活跃: {remaining} 条 | 归档: {archive_total} 条")


def do_restore(mem_id):
    """从归档恢复。"""
    result = api_post("/memory/archive/restore", {"ids": [mem_id]})
    if not result:
        return
    n = result.get("restored", 0)
    if n:
        print(f"✅ 已恢复 {mem_id}，归档剩余: {result.get('remaining_archive', '?')}")
    else:
        print(f"❌ 未找到 {mem_id}（可能不在归档中）")


def main():
    parser = argparse.ArgumentParser(description="记忆 TTL 清理 + 归档")
    parser.add_argument("--execute", action="store_true", help="真正执行归档（默认 dry-run）")
    parser.add_argument("--stats", action="store_true", help="只看统计")
    parser.add_argument("--backfill", action="store_true", help="只补填 TTL 元数据")
    parser.add_argument("--restore", type=str, help="从归档恢复指定 ID")
    parser.add_argument("--quiet", action="store_true", help="安静模式（适合 cron）")
    args = parser.parse_args()

    if not args.quiet:
        print("🧹 记忆 TTL 清理系统")
        print("=" * 40)

    if args.stats:
        show_stats(quiet=args.quiet)
        return

    if args.backfill:
        do_backfill(quiet=args.quiet)
        return

    if args.restore:
        do_restore(args.restore)
        return

    # 默认流程：backfill → preview/execute
    do_backfill(quiet=args.quiet)

    if args.execute:
        do_archive(quiet=args.quiet)
    else:
        do_preview(quiet=args.quiet)


if __name__ == "__main__":
    main()
