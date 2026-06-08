"""去重评测集 — 验证三档去重 + 关键词二次判定的效果。

测试用例设计：
1. 完全相同 → 应 skipped
2. 措辞不同但同义（BGE-M3 < 0.85 的）→ 应 skipped 或 flagged
3. 相关但不同主题 → 应 added
4. 冲突事实（同一人不同数值）→ 应 flagged
5. 中英混合 → 正确处理
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_memory_stack import MemoryEngine
import json

# 使用测试数据库，不影响生产
TEST_DIR = "./test_chroma_data"
import shutil
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)

engine = MemoryEngine(data_dir=TEST_DIR)

# ── 测试用例 ──────────────────────────────
test_cases = [
    # (写入文本, 期望action, 描述)
    ("Kevin月薪1.7万，存款30万", "added", "首次写入"),
    ("Kevin月薪1.7万，存款30万", "skipped", "完全相同 → 应跳过"),
    ("Kevin月薪一万七，存款三十万", "skipped", "中文数字写法 → 应跳过（同义）"),
    ("Kevin 月薪 17000，存了 30 万", "skipped", "数字格式不同 → 应跳过"),
    ("Kevin 喜欢编程和 AI", "added", "不同话题 → 应新增"),
    ("⚠️ 推代码前先列清单给你审", "added", "铁律V1 → 应新增"),
    ("⚠️ 推GitHub前先列文件清单等Kevin审查确认", "flagged", "铁律V2 → 灰色地带 flagged（措辞差异大但有关联）"),
    ("⚠️ 公开推送铁律：先cp -r复制副本再改隐私数据", "added", "铁律V3 → 不同维度规则，应新增"),
    ("毛球是Kevin的猫", "added", "猫信息V1 → 应新增"),
    ("毛球是Kevin养的德文卷毛猫，很可爱", "flagged", "猫信息V2 → 应 flagged（有增量信息）"),
    ("Kevin月薪涨到2.0万了", "flagged", "冲突事实 → 应 flagged"),
]

# ── 运行 ──────────────────────────────
results = []
for text, expected, desc in test_cases:
    r = engine.write(text, source="test", tags=["eval"], auto_extract=False)
    actual = r.get("action", "unknown")
    passed = actual == expected
    results.append({
        "desc": desc,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "sim": r.get("similarity"),
        "text": text[:40],
    })
    status = "✅" if passed else "❌"
    sim_str = f" (sim={r.get('similarity', 'N/A')})" if r.get("similarity") else ""
    # 也打印 best_sim（即使是 added 的情况，可能没过 recall）
    extra = ""
    if not passed:
        # 诊断：手动查 top-1 相似度
        q_vec = engine._encoder.encode([text], normalize_embeddings=True)[0]
        if engine._collection.count() > 0:
            qr = engine._collection.query(query_embeddings=[q_vec.tolist()], n_results=1)
            if qr["distances"] and qr["distances"][0]:
                real_sim = 1 - qr["distances"][0][0]
                extra = f" [top1_sim={real_sim:.3f}]"
    print(f"  {status} {desc}: expected={expected}, actual={actual}{sim_str}{extra}")

# ── 汇总 ──────────────────────────────
total = len(results)
passed = sum(1 for r in results if r["passed"])
print(f"\n{'='*50}")
print(f"📊 通过率: {passed}/{total} ({passed/total*100:.0f}%)")

if passed < total:
    print("\n❌ 失败用例:")
    for r in results:
        if not r["passed"]:
            print(f"  - {r['desc']}: expected={r['expected']}, actual={r['actual']}, sim={r['sim']}")

# 清理
shutil.rmtree(TEST_DIR, ignore_errors=True)
