"""混合检索 — 向量搜索 + BM25 关键词搜索，RRF 融合排序。

不依赖 V1 引擎的重型模型加载，直接调 V1 的 HTTP API。
用法:
    results = hybrid_search("Kevin的薪资", top_k=5)
"""
import math
import re
import requests
from typing import Dict, List, Optional
from collections import Counter

V1_API = "http://127.0.0.1:8900/memory"


def _tokenize(text: str) -> List[str]:
    """中英文混合分词（纯正则，不依赖 jieba）。"""
    stopwords = {'的','是','在','了','有','不','也','都','就','但','而','与','或',
                 '我','你','他','她','它','这','那','一个','可以','没有','已经',
                 'the','and','for','with','this','that','from','has','was','is'}
    # 中文按字/词切分 + 英文按空格
    tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}|[\d,.]+[万亿千百%]?', text.lower())
    return [t for t in tokens if t not in stopwords]


def bm25_search(query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
    """纯 BM25 关键词搜索（本地计算，无外部依赖）。"""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # 计算 IDF
    N = len(documents)
    doc_freq = Counter()
    doc_tokens_list = []
    for doc in documents:
        tokens = _tokenize(doc.get("text", ""))
        doc_tokens_list.append(tokens)
        unique = set(tokens)
        for t in unique:
            doc_freq[t] += 1

    # BM25 参数
    k1, b = 1.5, 0.75
    avgdl = sum(len(t) for t in doc_tokens_list) / max(N, 1)

    scores = []
    for i, doc in enumerate(documents):
        tokens = doc_tokens_list[i]
        dl = len(tokens)
        tf_counter = Counter(tokens)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf_counter:
                continue
            tf = tf_counter[qt]
            df = doc_freq.get(qt, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
            score += idf * tf_norm
        if score > 0:
            scores.append({**doc, "bm25_score": round(score, 4)})

    scores.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scores[:top_k]


def hybrid_search(query: str, top_k: int = 5, vector_weight: float = 0.6) -> List[Dict]:
    """混合检索：向量搜索 + BM25，RRF 融合。

    Args:
        query: 搜索查询
        top_k: 返回条数
        vector_weight: 向量搜索权重 (0~1)，BM25 权重 = 1 - vector_weight
    """
    # 1. 向量搜索（调 V1 API）
    try:
        resp = requests.post(f"{V1_API}/query", json={
            "text": query, "top_k": top_k * 2, "threshold": 0.2
        }, timeout=10)
        vector_results = resp.json().get("results", [])
    except Exception:
        vector_results = []

    # 2. 获取全部记忆做 BM25（V1 没有关键词搜索接口）
    try:
        resp = requests.post(f"{V1_API}/list", json={"limit": 1000}, timeout=10)
        all_memories = resp.json().get("items", [])
    except Exception:
        all_memories = []

    bm25_results = bm25_search(query, all_memories, top_k=top_k * 2)

    # 3. RRF 融合 (Reciprocal Rank Fusion)
    k = 60  # RRF 常数
    scores = {}

    for rank, item in enumerate(vector_results):
        mid = item.get("id", "")
        rrf = vector_weight / (k + rank + 1)
        scores[mid] = {"item": item, "score": rrf, "vector_rank": rank, "bm25_rank": None}

    for rank, item in enumerate(bm25_results):
        mid = item.get("id", "")
        rrf = (1 - vector_weight) / (k + rank + 1)
        if mid in scores:
            scores[mid]["score"] += rrf
            scores[mid]["bm25_rank"] = rank
        else:
            scores[mid] = {"item": item, "score": rrf, "vector_rank": None, "bm25_rank": rank}

    # 排序取 top_k
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    return [{
        "id": s["item"].get("id", ""),
        "text": s["item"].get("text", ""),
        "score": round(s["score"], 4),
        "vector_rank": s["vector_rank"],
        "bm25_rank": s["bm25_rank"],
        "metadata": s["item"].get("metadata", {}),
    } for s in ranked]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Kevin"
    results = hybrid_search(q, top_k=5)
    for r in results:
        print(f"[{r['score']:.4f}] (v{r['vector_rank']} k{r['bm25_rank']}) {r['text'][:80]}")
