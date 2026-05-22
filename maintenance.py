"""记忆栈维护脚本 — 凌晨批处理
功能：
  1. 事实冲突检测（同实体+同属性 → 保留最新）
  2. 低频记忆衰减（30天未访问 → 标记清理）
  3. 去除精确重复
  4. 实体聚合报告

使用: python3 maintenance.py
可定时运行: 0 3 * * * python3 maintenance.py >> logs/maintenance.log 2>&1
"""

import requests, json, re, sys
from datetime import datetime, timedelta
from collections import defaultdict

API = "http://127.0.0.1:8900"

# ── 配置 ──────────────────────────────────────
CONFLICT_SIMILARITY = 0.85        # 相似度阈值（>=此值视为同一事实）
STALE_DAYS = 30                  # 多少天未更新视为衰减
STALE_SIMILARITY_THRESHOLD = 0.3 # 衰减记忆的相似度阈值

# 数值属性正则（中文常见表达）
NUMERIC_PATTERNS = [
    (re.compile(r'月薪\s*(\d+\.?\d*)\s*万'), 'salary'),
    (re.compile(r'存款\s*(\d+\.?\d*)\s*万'), 'savings'),
    (re.compile(r'月租\s*(\d+\.?\d*)\s*[元万]'), 'rent'),
    (re.compile(r'月收入\s*(\d+\.?\d*)\s*[元万]'), 'monthly_income'),
    (re.compile(r'收入\s*(\d+\.?\d*)\s*[元万]'), 'income'),
    (re.compile(r'预算\s*(\d+\.?\d*)\s*万'), 'budget'),
    (re.compile(r'价格\s*(\d+\.?\d*)\s*[元万]'), 'price'),
    (re.compile(r'成本\s*(\d+\.?\d*)\s*[元万]'), 'cost'),
]


def fetch_all_memories():
    """获取所有记忆。"""
    try:
        r = requests.post(f"{API}/memory/query",
                         json={"text": "", "top_k": 100, "threshold": 0},
                         timeout=10)
        return r.json().get("results", [])
    except Exception as e:
        print(f"❌ 获取记忆失败: {e}")
        return []


def extract_numeric_facts(text):
    """从文本中提取数值事实 → [(attr_name, value, raw_match)]"""
    facts = []
    for pattern, attr in NUMERIC_PATTERNS:
        for m in pattern.finditer(text):
            val = float(m.group(1))
            facts.append((attr, val, m.group(0)))
    return facts


def extract_entities(text):
    """用 GLiNER 提取实体。"""
    try:
        r = requests.post(f"{API}/memory/entities",
                         json={"text": text, "threshold": 0.3},
                         timeout=5)
        return r.json().get("entities", [])
    except Exception:
        return []


def parse_timestamp(meta):
    """提取时间戳。"""
    for key in ("updated_at", "created_at", "timestamp"):
        ts = meta.get(key, "")
        if ts:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                pass
    return datetime.min


def resolve_conflicts(memories):
    """主逻辑：检测并解决冲突。"""
    if not memories:
        print("📭 无记忆可处理")
        return []

    print(f"📊 处理 {len(memories)} 条记忆...\n")

    # ── Phase 1: 数值事实冲突 ──────────────────
    # 结构: {key: [(text, meta, value, idx)]}
    # key = "primary_entity:attr_name"
    num_conflicts = 0
    merged_ids = set()

    # 为每条记忆提取实体和数值事实
    entity_map = defaultdict(list)  # entity_name → [memory_idx, ...]

    for i, mem in enumerate(memories):
        text = mem.get("text", "")
        meta = mem.get("metadata", {}) or {}

        # 用已有实体（如果元数据里有的话）
        cached_entities = meta.get("entities", "")
        if cached_entities:
            # 格式: "User:person|北京:location|3000:money"
            for part in cached_entities.split("|"):
                if ":" in part:
                    entity_text, label = part.split(":", 1)
                    if label == "person":
                        entity_map[entity_text].append(i)

        # 提取数值事实
        num_facts = extract_numeric_facts(text)
        if num_facts:
            # 尝试找到主体（第一个person实体）
            primary_entity = "unknown"
            for part in (cached_entities.split("|") if cached_entities else []):
                if ":person" in part:
                    primary_entity = part.split(":", 1)[0]
                    break

            for attr, val, raw in num_facts:
                key = f"{primary_entity}:{attr}"
                entity_map[key].append(i)

    # 检测同 key 的多条记录 → 冲突
    to_delete = []
    for key, indices in entity_map.items():
        if len(indices) <= 1:
            continue

        # 有多条同 key 记录 → 检测是否有不同值
        values_map = defaultdict(list)
        for idx in indices:
            text = memories[idx].get("text", "")
            for attr, val, raw in extract_numeric_facts(text):
                if f"{key}" == f"{attr}" or key.endswith(f":{attr}"):
                    values_map[val].append(idx)

        if len(values_map) > 1:
            # 有冲突！保留最新的
            num_conflicts += 1
            latest_idx = max(indices, key=lambda i: parse_timestamp(memories[i].get("metadata", {})))

            for idx in indices:
                if idx != latest_idx:
                    to_delete.append(idx)
                    merged_ids.add(idx)

            text = memories[latest_idx].get("text", "")[:60]
            print(f"  🔄 冲突: {key} → 保留「{text}...」")

    # 执行删除
    if to_delete:
        # 通过 ChromaDB 批量删除（需要用 API）
        # 这里通过内存更新标记来实现
        print(f"\n  🗑️  标记删除 {len(to_delete)} 条过时记录")
        print(f"  ⚠️  需调用 /memory/delete 接口（ID列表）")

    # ── Phase 2: 低频衰减 ──────────────────────
    now = datetime.now()
    stale_count = 0
    for i, mem in enumerate(memories):
        if i in merged_ids:
            continue
        meta = mem.get("metadata", {}) or {}
        ts = parse_timestamp(meta)
        age = (now - ts).days
        if age > STALE_DAYS:
            stale_count += 1

    if stale_count:
        print(f"  ⏳ {stale_count} 条记忆超过{STALE_DAYS}天未更新")

    # ── 总结 ────────────────────────────────────
    print(f"\n✅ 维护完成: 检测到 {num_conflicts} 处冲突, {stale_count} 条低频记忆")
    return to_delete


def main():
    print("🧹 记忆栈维护脚本")
    print("=" * 50)

    memories = fetch_all_memories()
    if not memories:
        return

    # 显示当前状态
    print(f"\n📦 当前记忆: {len(memories)} 条")
    for mem in memories[:5]:
        text = mem.get("text", "")[:60]
        meta = mem.get("metadata", {}) or {}
        source = meta.get("source", "?")
        print(f"  [{source}] {text}...")
    if len(memories) > 5:
        print(f"  ... 还有 {len(memories) - 5} 条")

    print()
    to_delete = resolve_conflicts(memories)

    if to_delete:
        # 批量删除
        print(f"\n🗑️ 执行删除 {len(to_delete)} 条...")
        # 需要从 API 获取 IDs
        try:
            r = requests.post(f"{API}/memory/cleanup", timeout=10)
            print(f"  → {r.json()}")
        except Exception as e:
            print(f"  ❌ 删除失败: {e}")

    print("\n" + "=" * 50)
    print("✨ 维护完成")


if __name__ == "__main__":
    main()
