"""CLI 工具 — local-memory 命令行接口。"""
import argparse
from local_memory_stack import MemoryEngine


def main():
    parser = argparse.ArgumentParser(description="Local Memory Stack CLI")
    sub = parser.add_subparsers(dest="command")

    # start
    p = sub.add_parser("start", help="启动 REST API 服务")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--host", default="127.0.0.1")

    # query
    p = sub.add_parser("query", help="查询记忆")
    p.add_argument("text")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("-s", "--source", default=None)

    # write
    p = sub.add_parser("write", help="写入记忆")
    p.add_argument("text")
    p.add_argument("-s", "--source", default="cli")
    p.add_argument("-t", "--tags", nargs="*", default=[])

    # cleanup
    sub.add_parser("cleanup", help="清理垃圾记忆")

    # stats
    sub.add_parser("stats", help="显示统计信息")

    args = parser.parse_args()

    if args.command == "start":
        import uvicorn
        from local_memory_stack.server import app
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

    elif args.command == "query":
        engine = MemoryEngine()
        entries = engine.query(args.text, top_k=args.top_k, source=args.source)
        for e in entries:
            tags = ", ".join(e.tags) if e.tags else "—"
            print(f"[{e.similarity:.0%}] [{e.source}] {e.text[:100]}")

    elif args.command == "write":
        engine = MemoryEngine()
        result = engine.write(args.text, source=args.source, tags=args.tags)
        print(f"{result['action']}: {result['id']} | {result['text'][:80]}")

    elif args.command == "cleanup":
        engine = MemoryEngine()
        result = engine.cleanup()
        print(f"清理: 删除 {result['removed']} 条, 剩余 {result['remaining']} 条")

    elif args.command == "stats":
        engine = MemoryEngine()
        s = engine.stats()
        print(f"记忆: {s['total_memories']} 条 | 设备: {s['device']} | 数据: {s['data_dir']}")


if __name__ == "__main__":
    main()
