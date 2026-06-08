"""记忆栈维护脚本 — 凌晨批处理
功能：
  1. 事实冲突检测（同实体+同属性 → 保留最新）
  2. 低频记忆衰减（30天未访问 → 标记清理）
  3. 语义去重（BGE-M3全量对比，同义不同词也会检测）
  4. 清理对话垃圾

使用: python3 maintenance.py
可定时运行: 0 3 * * * python3 maintenance.py >> logs/maintenance.log 2>&1
"""

import os
import requests, json, re, sys
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from local_memory_stack.engine import (  # noqa: E402
    DEDUP_THRESHOLD_MERGE,
    DEDUP_THRESHOLD_RECALL,
    DEDUP_THRESHOLD_SKIP,
)

API = "http://127.0.0.1:8900"

# ── 配置 ──────────────────────────────────────
CONFLICT_SIMILARITY = DEDUP_THRESHOLD_RECALL  # 语义去重查询阈值（跟 engine 粗筛阈值保持一致）
STALE_DAYS = 30                  # 多少天未更新视为衰减

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
        memories = []
        offset = 0
        limit = 100
        while True:
            r = requests.post(f"{API}/memory/list",
                              json={"limit": limit, "offset": offset},
                              timeout=10)
            payload = r.json()
            items = payload.get("items", [])
            memories.extend(items)
            if offset + limit >= payload.get("total", len(memories)) or not items:
                break
            offset += limit
        return memories
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
    """提取时间戳（返回 naive datetime）。"""
    for key in ("updated_at", "created_at", "timestamp"):
        ts = meta.get(key, "")
        if ts:
            try:
                # 移除时区信息，转为 naive
                ts_clean = ts.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts_clean)
                return dt.replace(tzinfo=None)
            except Exception:
                pass
    return datetime.min


def find_semantic_duplicates(memories, skip_ids=None):
    """Phase 3: 语义去重 — 用 BGE-M3 全量对比同义不同词。
    
    三档策略：>0.92 高度重复删一条，0.85~0.92 保留（已由server.py打标），
    <0.85 不管。
    """
    if skip_ids is None:
        skip_ids = set()
    
    to_delete = []
    processed = set()
    dedup_count = 0
    
    for i, mem in enumerate(memories):
        if i in skip_ids or i in processed:
            continue
        
        text = mem.get("text", "")
        if len(text) < 10:
            continue
        
        meta = mem.get("metadata", {}) or {}
        mem_id = mem.get("id", "")
        
        try:
            r = requests.post(f"{API}/memory/query",
                             json={"text": text, "top_k": 5, "threshold": CONFLICT_SIMILARITY},
                             timeout=10)
            results = r.json().get("results", [])
            
            for match in results[1:]:
                match_text = match.get("text", "")
                match_id = match.get("id", "")
                match_sim = match.get("similarity", 0)
                
                for j, m2 in enumerate(memories):
                    if j in skip_ids or j in processed or j == i:
                        continue
                    if m2.get("id", "") == match_id:
                        if match_sim >= DEDUP_THRESHOLD_SKIP:
                            if len(text) >= len(match_text):
                                to_delete.append(j)
                                skip_ids.add(j)
                                processed.add(j)
                                dedup_count += 1
                                print(f"  ⊛ 高度重复 → 删除: 「{match_text[:50]}...」({match_sim:.0%})")
                            else:
                                to_delete.append(i)
                                skip_ids.add(i)
                                processed.add(i)
                                dedup_count += 1
                                print(f"  ⊛ 高度重复 → 删除: 「{text[:50]}...」({match_sim:.0%})")
                        elif match_sim >= DEDUP_THRESHOLD_MERGE:
                            print(f"  ⚖ 相关但不同 → 保留: 「{match_text[:50]}...」({match_sim:.0%})")
                        break
        except Exception:
            pass

    if dedup_count:
        print(f"  📊 发现 {dedup_count} 处高度语义重复")
    
    return to_delete, skip_ids


def review_flagged_memories(memories):
    """Phase 4: LLM审核 needs_review 标记的记忆对。
    
    找所有 needs_review=true 的记忆，配对后调本地LLM判断：
    - duplicate: 核心语义一致，删短的
    - keep: 有信息增量，两条都保留，清除标记
    """
    import subprocess
    
    # 收集所有打标记忆
    flagged = []
    for i, mem in enumerate(memories):
        meta = mem.get("metadata", {}) or {}
        if meta.get("needs_review") == "true":
            flagged.append(i)
    
    if not flagged:
        print("\n📋 LLM审核: 无待审核标记")
        return 0, []
    
    print(f"\n🤖 LLM审核: 发现 {len(flagged)} 条待审核记忆")
    
    # 配对（通过 review_similar_to 互相关联）
    processed_pairs = set()
    to_delete = []
    review_count = 0
    
    for i in flagged:
        mem = memories[i]
        meta = mem.get("metadata", {}) or {}
        partner_id = meta.get("review_similar_to", "")
        similarity = meta.get("review_similarity", "?")
        
        if not partner_id:
            continue
        
        pair_key = tuple(sorted([mem.get("id", ""), partner_id]))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        # 找到伙伴记忆的文本
        partner_text = ""
        partner_idx = None
        for j, m2 in enumerate(memories):
            if m2.get("id", "") == partner_id:
                partner_text = m2.get("text", "")
                partner_idx = j
                break
        
        if not partner_text:
            continue
        
        text_a = mem.get("text", "")
        text_b = partner_text
        
        # 调本地LLM判断
        prompt = f"""判断以下两条记忆是否语义重复。

记忆A: {text_a}
记忆B: {text_b}
相似度: {similarity}

判断标准:
- duplicate = 核心事实完全一致，无信息增量（如只是换了个说法）
- keep = 有信息增量（如B比A多出具体细节、数字、新事实）

只回答一个词: duplicate 或 keep"""

        try:
            result = subprocess.run(
                ["python3", "-m", "hermes_tools"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=30
            )
            # fallback: 用简单规则判断
            decision = "keep"  # 默认保留
            llm_output = result.stdout.strip().lower()
            if "duplicate" in llm_output and "keep" not in llm_output:
                decision = "duplicate"
        except Exception:
            # LLM 不可用时，用长度差异规则：短的不到长的70%就保留
            ratio = min(len(text_a), len(text_b)) / max(len(text_a), len(text_b), 1)
            decision = "duplicate" if ratio > 0.85 else "keep"
        
        if decision == "duplicate":
            # 删短的
            if len(text_a) >= len(text_b):
                to_delete.append(partner_idx)
                review_count += 1
                print(f"  🗑 LLM判重 → 删除: 「{text_b[:50]}...」")
            else:
                to_delete.append(i)
                review_count += 1
                print(f"  🗑 LLM判重 → 删除: 「{text_a[:50]}...」")
        else:
            # 保留两条，清除标记
            print(f"  ✅ LLM判异 → 保留: 「{text_a[:30]}...」vs「{text_b[:30]}...」")
            try:
                requests.patch(f"{API}/memory/update",
                              json={
                                  "id": mem.get("id", ""),
                                  "delete_keys": ["needs_review", "review_similar_to", "review_similarity"],
                              },
                              timeout=5)
                if partner_idx is not None:
                    requests.patch(f"{API}/memory/update",
                                  json={
                                      "id": partner_id,
                                      "delete_keys": ["needs_review", "review_similar_to", "review_similarity"],
                                  },
                                  timeout=5)
            except Exception:
                pass
    
    return review_count, to_delete


def resolve_conflicts(memories):
    """主逻辑：检测并解决冲突。"""
    if not memories:
        print("📭 无记忆可处理")
        return []

    print(f"📊 处理 {len(memories)} 条记忆...\n")

    # ── Phase 1: 数值事实冲突 ──────────────────
    num_conflicts = 0
    merged_ids = set()
    entity_map = defaultdict(list)

    for i, mem in enumerate(memories):
        text = mem.get("text", "")
        meta = mem.get("metadata", {}) or {}
        cached_entities = meta.get("entities", "")
        if cached_entities:
            for part in cached_entities.split("|"):
                if ":" in part:
                    entity_text, label = part.split(":", 1)
                    if label == "person":
                        entity_map[entity_text].append(i)

        num_facts = extract_numeric_facts(text)
        if num_facts:
            primary_entity = "unknown"
            for part in (cached_entities.split("|") if cached_entities else []):
                if ":person" in part:
                    primary_entity = part.split(":", 1)[0]
                    break

            for attr, val, raw in num_facts:
                key = f"{primary_entity}:{attr}"
                entity_map[key].append(i)

    to_delete = []
    for key, indices in entity_map.items():
        if len(indices) <= 1:
            continue

        values_map = defaultdict(list)
        for idx in indices:
            text = memories[idx].get("text", "")
            for attr, val, raw in extract_numeric_facts(text):
                if f"{key}" == f"{attr}" or key.endswith(f":{attr}"):
                    values_map[val].append(idx)

        if len(values_map) > 1:
            num_conflicts += 1
            latest_idx = max(indices, key=lambda i: parse_timestamp(memories[i].get("metadata", {})))
            for idx in indices:
                if idx != latest_idx:
                    to_delete.append(idx)
                    merged_ids.add(idx)
            text = memories[latest_idx].get("text", "")[:60]
            print(f"  🔄 数值冲突: {key} → 保留「{text}...」")

    if to_delete:
        print(f"\n  🗑️  标记删除 {len(to_delete)} 条过时记录\n")

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
        print(f"  ⏳ {stale_count} 条记忆超过{STALE_DAYS}天未更新\n")

    # ── Phase 3: 语义去重 ──────────────────────
    printed_header = False
    print(f"🔍 语义去重（BGE-M3 阈值{CONFLICT_SIMILARITY}）...")
    dedup_deletes, merged_ids2 = find_semantic_duplicates(memories, skip_ids=merged_ids)
    to_delete.extend(dedup_deletes)
    merged_ids.update(merged_ids2)
    dedup_count = len(dedup_deletes)

    # ── Phase 4: LLM审核灰色地带标记 ──────────
    review_count, review_deletes = review_flagged_memories(memories)
    to_delete.extend(review_deletes)
    dedup_count += review_count

    # ── 总结 ────────────────────────────────────
    print(f"\n✅ 维护完成: {num_conflicts}处数值冲突, {stale_count}条低频, {dedup_count}处语义重复({review_count}条LLM审核)")
    return to_delete


def main():
    print("🧹 记忆栈维护脚本")
    print("=" * 50)

    memories = fetch_all_memories()
    if not memories:
        return

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
        # 收集要删除的 ID
        ids_to_delete = []
        for idx in to_delete:
            mem = memories[idx]
            mem_id = mem.get("id", "")
            if mem_id:
                ids_to_delete.append(mem_id)

        if ids_to_delete:
            ids_to_delete = list(set(ids_to_delete))  # 去重
            print(f"\n🗑️ 执行删除 {len(ids_to_delete)} 条...")
            try:
                # 分批次处理，容错
                deleted = 0
                for mem_id in ids_to_delete:
                    try:
                        r = requests.post(f"{API}/memory/delete",
                                         json={"ids": [mem_id]},
                                         timeout=5)
                        if r.json().get("deleted", 0) > 0:
                            deleted += 1
                    except Exception:
                        pass
                print(f"  → 已删除 {deleted} 条")
            except Exception as e:
                print(f"  ❌ 删除失败: {e}")

    # 清理对话垃圾
    try:
        r = requests.post(f"{API}/memory/cleanup", timeout=10)
        removed = r.json().get("removed", 0)
        if removed:
            print(f"  🧹 清理对话垃圾: {removed} 条")
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("✨ 维护完成")


if __name__ == "__main__":
    main()
