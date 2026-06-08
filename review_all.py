"""审核全部记忆：找重复、冲突、冗余"""
import requests, json, numpy as np
from sentence_transformers import SentenceTransformer

API = "http://127.0.0.1:8900"

# 拉全部记忆
print("📥 拉取全部记忆...")
r = requests.post(f"{API}/memory/query", json={"text": "全部", "top_k": 100, "threshold": 0}, timeout=15)
memories = r.json().get("results", [])
print(f"共 {len(memories)} 条\n")

# 加载 BGE-M3
print("🔧 加载 BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3", device="mps")

# 算相似度矩阵
texts = [m.get("text", "") for m in memories]
embeddings = model.encode(texts, normalize_embeddings=True)
sim_matrix = np.inner(embeddings, embeddings)

# 找相似对
print("\n" + "=" * 80)
print("📋 全量审核结果")
print("=" * 80)

# 先列所有记忆
for i, m in enumerate(memories):
    meta = m.get("metadata", {}) or {}
    src = meta.get("source", "?")
    created = meta.get("created_at", "?")[:10]
    print(f"\n{i+1:2d}. [{src}] ({created}) id={m.get('id','?')[:16]}")
    print(f"    {texts[i][:150]}")

# 找相似对
print("\n" + "=" * 80)
print("🔍 相似对分析（>0.80）")
print("=" * 80)

pairs = []
for i in range(len(memories)):
    for j in range(i+1, len(memories)):
        sim = sim_matrix[i][j]
        if sim > 0.80:
            pairs.append((sim, i, j))

pairs.sort(reverse=True)
for sim, i, j in pairs:
    print(f"\n相似度 {sim:.4f} | #{i+1} ↔ #{j+1}")
    print(f"  A: {texts[i][:100]}")
    print(f"  B: {texts[j][:100]}")
    
    if sim >= 0.92:
        verdict = "🔴 高度重复 → 建议删除短的"
    elif sim >= 0.85:
        verdict = "🟡 灰色地带 → 建议打标等审核"
    else:
        verdict = "🟢 相关但不同 → 保留"
    print(f"  → {verdict}")

print(f"\n共 {len(pairs)} 对相似度 > 0.80")
