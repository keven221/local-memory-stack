"""
本地记忆栈 Demo — BGE-M3 + ChromaDB + GLiNER
验证三大组件都能正常工作
"""
import time

print("=" * 50)
print("🧠 Local Memory Stack — 验证开始")
print("=" * 50)

# ── 1. BGE-M3 向量嵌入 ──────────────────────
print("\n📦 [1/3] 加载 BGE-M3 嵌入模型...")
t0 = time.time()

from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("BAAI/bge-m3", device="mps")  # Metal 加速

print(f"   ✅ 加载耗时: {time.time() - t0:.1f}s")

docs = [
    "User正在学习机器学习模型训练",
    "项目文档已完成，使用Markdown格式",
    "在线学习平台完成了Python课程",
    "咪咪是User养的橘猫",
]
print(f"   🔢 编码 {len(docs)} 条记忆...")
t1 = time.time()
vectors = encoder.encode(docs, normalize_embeddings=True)
print(f"   ✅ 向量维度: {vectors.shape}  耗时: {time.time() - t1:.2f}s")

# ── 2. ChromaDB 向量存储 ────────────────────
print("\n📦 [2/3] 初始化 ChromaDB...")
t0 = time.time()

import chromadb
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="memories",
    metadata={"hnsw:space": "cosine"}
)

# 写入
ids = [f"mem_{i}" for i in range(len(docs))]
collection.add(ids=ids, documents=docs, embeddings=vectors.tolist())
print(f"   ✅ 写入 {len(docs)} 条记录  耗时: {time.time() - t0:.1f}s")

# 查询
query_text = "User的猫叫什么"
query_vec = encoder.encode([query_text], normalize_embeddings=True)
results = collection.query(query_embeddings=query_vec.tolist(), n_results=2)
print(f"   🔍 查询: 「{query_text}」")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"   → {doc}  (距离: {dist:.4f})")

# ── 3. GLiNER 实体提取 ──────────────────────
print("\n📦 [3/3] 加载 GLiNER 实体提取...")
t0 = time.time()

from gliner import GLiNER
ner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
print(f"   ✅ 加载耗时: {time.time() - t0:.1f}s")

text = "用户在写Python单元测试，最近在学习FastAPI"
labels = ["person", "location", "money", "technology"]
entities = ner_model.predict_entities(text, labels, threshold=0.3)
print(f"   🔍 输入: 「{text}」")
for e in entities:
    print(f"   → [{e['label']}] {e['text']}  (置信度: {e['score']:.2f})")

# ── 完成 ─────────────────────────────────────
print("\n" + "=" * 50)
print("🎉 三大组件全部验证通过！")
print("=" * 50)
