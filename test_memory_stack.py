#!/usr/bin/env python3
"""记忆栈闭环测试 — 写入 → 检索 → 图引导 → 性能对比 → 清理"""

import json
import os
import sys
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V1_API = "http://127.0.0.1:8900/memory"
TEST_TAG = "memory_stack_test_20260606"

# ─── 测试数据 ─────────────────────────────────────
TEST_MEMORIES = [
    {"text": "Kevin喜欢用Python写自动化部署脚本，最擅长DevOps和CI/CD流水线搭建，技术栈包括Docker和GitHub Actions",
     "tags": ["user", "preference", "devops"]},
    {"text": "闲鱼技术服务店铺2026年5月27日开张，主营AI Agent远程部署服务，定价30元起，一键部署脚本支持Hermes和OpenClaw",
     "tags": ["business", "xianyu"]},
    {"text": "Kevin月薪1.7万税前，存款30万，月支出6500元，目标买200万配售型保障房，朝阳住建委电话010-64186100",
     "tags": ["finance", "salary", "housing"]},
    {"text": "毛球是斯芬克斯无毛猫和德文卷毛猫的混血串串，棕色白色相间的短卷毛，超大招风耳，绿黄色大眼睛，Kevin的AI猫咪助手",
     "tags": ["pet", "cat", "identity"]},
    {"text": "本地记忆栈使用ChromaDB做向量存储，BGE-M3模型做embedding，支持BM25关键词和向量混合检索，RRF融合排序",
     "tags": ["tech", "memory", "architecture"]},
]


def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def test_write():
    """① 写入测试记忆"""
    header("① 写入测试")
    ids = []
    for i, mem in enumerate(TEST_MEMORIES):
        t0 = time.time()
        resp = requests.post(f"{V1_API}/write", json={
            "text": mem["text"],
            "source": TEST_TAG,
            "tags": mem["tags"],
        }, timeout=30)
        elapsed = (time.time() - t0) * 1000
        data = resp.json()
        mid = data.get("id", "unknown")
        ids.append(mid)
        status = "✅" if resp.status_code == 200 else "❌"
        print(f"  {status} [{elapsed:.0f}ms] {mid}: {mem['text'][:50]}...")
    return ids


def test_list():
    """② 列出记忆，验证写入（用向量搜索验证）"""
    header("② 列出记忆验证")
    # 用向量搜索验证测试记忆是否写入成功
    test_queries = [
        ("DevOps CI/CD 流水线", "Kevin喜欢用Python"),
        ("闲鱼AI部署", "闲鱼技术服务"),
        ("月薪存款保障房", "Kevin月薪"),
        ("无毛猫德文卷毛", "斯芬克斯"),
        ("ChromaDB BGE-M3 embedding", "ChromaDB"),
    ]
    found = 0
    for query, expected in test_queries:
        resp = requests.post(f"{V1_API}/query", json={
            "text": query, "top_k": 1, "threshold": 0.3
        }, timeout=30)
        results = resp.json().get("results", [])
        if results and expected in results[0].get("text", ""):
            found += 1
            print(f"  ✅ 找到: {results[0]['text'][:50]}...")
        else:
            print(f"  ❌ 未匹配: \"{query}\" → {results[0]['text'][:40] if results else '无结果'}")
    print(f"\n  匹配: {found}/5")
    return found


def test_vector_search():
    """③ 向量搜索"""
    header("③ 向量搜索")
    queries = ["Kevin的薪资", "Python自动化", "猫咪品种"]
    for q in queries:
        t0 = time.time()
        resp = requests.post(f"{V1_API}/query", json={
            "text": q, "top_k": 3, "threshold": 0.2
        }, timeout=30)
        elapsed = (time.time() - t0) * 1000
        results = resp.json().get("results", [])
        print(f"\n  🔍 \"{q}\" ({elapsed:.0f}ms, {len(results)} 条):")
        for r in results:
            score = r.get("score", 0)
            text = r.get("text", "")[:60]
            print(f"    [{score:.4f}] {text}")


def test_bm25():
    """④ BM25 关键词搜索"""
    header("④ BM25 关键词搜索")
    from hybrid_search import bm25_search

    # 获取全部记忆
    resp = requests.post(f"{V1_API}/list", json={"limit": 1000}, timeout=10)
    all_memories = resp.json().get("items", [])

    queries = ["闲鱼部署", "薪资存款", "ChromaDB embedding"]
    for q in queries:
        t0 = time.time()
        results = bm25_search(q, all_memories, top_k=3)
        elapsed = (time.time() - t0) * 1000
        print(f"\n  🔍 \"{q}\" ({elapsed:.0f}ms, {len(results)} 条):")
        for r in results:
            score = r.get("bm25_score", 0)
            text = r.get("text", "")[:60]
            print(f"    [{score:.4f}] {text}")


def test_hybrid():
    """⑤ 混合检索（RRF 融合）"""
    header("⑤ 混合检索 (RRF)")
    from hybrid_search import hybrid_search

    queries = ["Kevin的财务状况", "AI Agent部署方案", "宠物猫品种"]
    for q in queries:
        t0 = time.time()
        results = hybrid_search(q, top_k=3)
        elapsed = (time.time() - t0) * 1000
        print(f"\n  🔍 \"{q}\" ({elapsed:.0f}ms, {len(results)} 条):")
        for r in results:
            score = r.get("score", 0)
            text = r.get("text", "")[:60]
            vrank = r.get("vector_rank", "-")
            krank = r.get("bm25_rank", "-")
            print(f"    [{score:.4f}] (v{vrank} k{krank}) {text}")


def test_graph_retrieval():
    """⑥ 图引导检索"""
    header("⑥ 图引导检索")
    from graph_retrieval import GraphRetriever

    t0 = time.time()
    r = GraphRetriever(rebuild=True)
    build_time = (time.time() - t0) * 1000
    print(f"  索引构建: {build_time:.0f}ms")

    stats = r.stats()
    print(f"  统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    queries = ["Kevin的薪资", "闲鱼部署", "猫咪品种"]
    for q in queries:
        t0 = time.time()
        results = r.search(q, top_k=3)
        elapsed = (time.time() - t0) * 1000
        print(f"\n  🔍 \"{q}\" ({elapsed:.0f}ms, {len(results)} 条):")
        for res in results:
            score = res.get("score", 0)
            text = res.get("text", "")[:60]
            method = res.get("method", "")
            extra = ""
            if res.get("expanded_from"):
                extra = " ← 邻居"
            print(f"    [{score:.4f}] ({method}) {text}{extra}")


def test_performance_comparison():
    """⑦ 性能对比"""
    header("⑦ 性能对比: 全局 vs 图引导")
    from hybrid_search import hybrid_search
    from graph_retrieval import GraphRetriever

    r = GraphRetriever()
    queries = ["Kevin的薪资", "闲鱼部署", "AI Agent", "Python脚本", "宠物猫"]

    print(f"\n  {'查询':<15} {'全局(ms)':>10} {'图引导(ms)':>10} {'提速':>8}")
    print(f"  {'-'*48}")

    for q in queries:
        # 全局
        t0 = time.time()
        g_res = hybrid_search(q, top_k=5)
        t_global = (time.time() - t0) * 1000

        # 图引导
        t0 = time.time()
        gr_res = r.search(q, top_k=5)
        t_graph = (time.time() - t0) * 1000

        speedup = t_global / max(t_graph, 1)
        print(f"  {q:<15} {t_global:>10.0f} {t_graph:>10.0f} {speedup:>7.1f}x")


def test_cleanup(ids):
    """⑧ 清理测试数据"""
    header("⑧ 清理测试数据")
    # 直接用写入返回的 ID 清理
    valid_ids = [i for i in ids if i and i != "unknown"]
    if valid_ids:
        resp = requests.post(f"{V1_API}/delete", json={"ids": valid_ids}, timeout=10)
        print(f"  🗑️ 删除 {len(valid_ids)} 条测试记忆: {resp.json()}")
    else:
        print(f"  ℹ️ 无有效 ID 可清理")

    # 清理图索引缓存
    import os
    idx_path = os.path.expanduser("~/projects/local-memory-stack/graph_index.json")
    if os.path.exists(idx_path):
        os.remove(idx_path)
        print(f"  🗑️ 删除图索引缓存")


def main():
    print("🧪 记忆栈闭环测试")
    print(f"   时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ① 写入
    ids = test_write()

    # ② 验证
    found = test_list()
    assert found >= 4, f"写入验证失败: 只找到 {found} 条"

    # ③ 向量搜索
    test_vector_search()

    # ④ BM25
    test_bm25()

    # ⑤ 混合检索
    test_hybrid()

    # ⑥ 图引导检索
    test_graph_retrieval()

    # ⑦ 性能对比
    test_performance_comparison()

    # ⑧ 清理
    test_cleanup(ids)

    header("✅ 闭环测试完成")


if __name__ == "__main__":
    main()
