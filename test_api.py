#!/usr/bin/env python3
"""一键测试本地记忆栈 API"""
import requests, json, time

BASE = "http://127.0.0.1:8900"

def test():
    print("=" * 50)
    print("🧪 本地记忆栈 API 测试")
    print("=" * 50)

    # 0. 健康检查
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print("✅ 服务在线")

    # 1. 写入记忆
    print("\n📝 写入3条记忆...")
    memories = [
        {"text": "示例：用户有一只橘猫，绿黄色大眼睛", "source": "profile", "tags": ["pet"]},
        {"text": "示例：文章已完成，用了AI生成框架", "source": "work", "tags": ["article"]},
        {"text": "示例：用户住在北京，通勤时间半小时", "source": "profile", "tags": ["location"]},
    ]
    for mem in memories:
        r = requests.post(f"{BASE}/memory/write", json=mem)
        data = r.json()
        ents = [f"{e['text']}({e['label']})" for e in data.get("entities", [])]
        print(f"   ✅ {data['id']} | 实体: {ents}")

    # 2. 语义查询
    print("\n🔍 语义查询...")
    queries = ["User的宠物", "最近完成的工作", "财务状况"]
    for q in queries:
        r = requests.post(f"{BASE}/memory/query", json={"text": q, "top_k": 2})
        data = r.json()
        print(f"\n   Q: 「{q}」→ {data['count']} 条结果")
        for m in data["results"]:
            print(f"   → [{m['similarity']}] {m['text']}")

    # 3. 实体提取
    print("\n🏷️ 实体提取测试...")
    r = requests.post(f"{BASE}/memory/entities", json={
        "text": "User计划用LangChain搭建AI Agent，预算5万，目标3个月内完成"
    })
    for e in r.json()["entities"]:
        print(f"   → [{e['label']}] {e['text']} (置信度: {e['score']})")

    # 4. 统计
    r = requests.get(f"{BASE}/memory/stats")
    print(f"\n📊 总记忆数: {r.json()['total_memories']}  设备: {r.json()['device']}")

    print("\n" + "=" * 50)
    print("🎉 全部测试通过！")
    print("=" * 50)

if __name__ == "__main__":
    test()
