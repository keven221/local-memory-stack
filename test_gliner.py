import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from gliner import GLiNER
print("加载GLiNER模型...")
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

text = "用户在写Python单元测试，最近在学习FastAPI"
labels = ["person", "location", "money", "technology"]
entities = model.predict_entities(text, labels, threshold=0.3)

print(f"\n输入: {text}")
for e in entities:
    print(f"  [{e['label']}] {e['text']}  (置信度: {e['score']:.2f})")

print("\n✅ GLiNER 验证通过！")
