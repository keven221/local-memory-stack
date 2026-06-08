# 🧠 本地记忆栈（Local Memory Stack）

> 纯本地语义记忆系统 — 零联网、零 API Key、零外部服务。  
> 文本存进去 → 自动理解含义 → 用大白话就能搜出来。

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 它能干什么？

| 能力 | 说明 |
|------|------|
| 🔍 **语义搜索** | "宠物" 搜到 "猫"、"房租" 搜到 "月租3000" — 不用记关键词 |
| 🔄 **写入去重** | 同一个事实存两次？自动合并，不会出现 3 条重复记录 |
| 🏷️ **自动标注** | 每条记忆自动提取人名/地点/事件标签 |
| ⏳ **TTL 自动归档** | 记忆带存活时间，超龄自动归档，不污染日常上下文 |
| 📬 **文件邮箱** | Agent/Cron 间零依赖消息传递 |
| 🔎 **混合检索 + 图引导** | 向量 + BM25 + RRF 融合，图引导加速 3-157x |
| ⚡ **不卡写入** | 实体提取后台异步，写入 741ms 完成 |
| 🔌 **框架无关** | 纯 Python 类，换任何 Agent 框架都能用 |

---

## 快速开始

```bash
# 克隆
git clone https://github.com/keven221/local-memory-stack.git
cd local-memory-stack

# 安装
python3 -m venv venv && source venv/bin/activate
pip install -e .

# 启动服务
python3 -m uvicorn src.local_memory_stack.server:app --port 8900
```

首次启动会自动下载模型（约 3.3GB），之后秒级启动。

### 30 秒体验

```python
from local_memory_stack import MemoryEngine

engine = MemoryEngine()

# 存记忆
engine.write("咪咪是张三养的橘猫串串", source="profile", tags=["pet"])

# 用白话搜
results = engine.query("张三养了什么宠物？", top_k=3)
for r in results:
    print(f"[{r.similarity:.0%}] {r.text}")
# → [92%] 咪咪是张三养的橘猫串串
```

---

## 架构

```
你的一句话
    ↓
┌─────────────┐
│  BGE-M3     │  ← 嵌入模型（2.2GB），把文字变成 1024 维向量
│  翻译官      │     "张三的猫" → [0.12, -0.34, 0.56, ...]
└──────┬──────┘
       ↓
┌─────────────┐     ┌─────────────┐
│  ChromaDB   │ ←──→│  GLiNER     │  ← 实体提取（1.1GB），后台异步
│  图书馆管理员 │     │  标注员      │     "张三在杭州" → 张三(人)、杭州(地)
└──────┬──────┘     └─────────────┘
       ↓
┌─────────────┐
│  归档层      │  ← TTL 过期的记忆自动移到这里
│  冷存储      │     不注入上下文，但随时能搜回来
└─────────────┘
```

三个模型各司其职，全部跑在本地，不联网。

---

## API 接口

### 记忆操作

```bash
# 写入（自动去重 + 自动标注 + 自动计算 TTL）
curl -X POST http://127.0.0.1:8900/memory/write \
  -H "Content-Type: application/json" \
  -d '{"text":"记忆内容","source":"api","tags":["preference"]}'

# 搜索（只搜活跃记忆）
curl -X POST http://127.0.0.1:8900/memory/query \
  -H "Content-Type: application/json" \
  -d '{"text":"搜索内容","top_k":5}'

# 搜索全部（活跃 + 归档）
curl -X POST http://127.0.0.1:8900/memory/search_all \
  -H "Content-Type: application/json" \
  -d '{"text":"搜索内容","top_k":5}'

# 统计
curl http://127.0.0.1:8900/memory/stats
```

### 归档管理

```bash
# 预览归档（dry-run）
curl -X POST http://127.0.0.1:8900/memory/archive -d '{"dry_run":true}'

# 执行归档
curl -X POST http://127.0.0.1:8900/memory/archive -d '{"dry_run":false}'

# 搜索归档
curl -X POST http://127.0.0.1:8900/memory/archive/search \
  -H "Content-Type: application/json" -d '{"text":"搜索内容"}'

# 从归档恢复
curl -X POST http://127.0.0.1:8900/memory/archive/restore \
  -H "Content-Type: application/json" -d '{"ids":["mem_xxx"]}'
```

---

## TTL 机制

每条记忆写入时自动计算存活时间：

| 标签 | TTL | 适用场景 |
|------|-----|---------|
| `temporary` | 30 天 | 临时事项、约会安排 |
| `dynamic` | 180 天 | 项目进展、阶段性目标 |
| `static` / `preference` / `feedback` | 365 天 | 用户身份、长期偏好 |
| 无标签 | 180 天 | 默认 |

**三层存储：**

| 层级 | 状态 | 用途 |
|------|------|------|
| 🟢 活跃层 | TTL 内 | 自动注入上下文，日常搜索 |
| 🟡 归档层 | TTL 超期 | 不注入上下文，`search_all` 可查 |
| 🔴 清理层 | 归档 365 天未引用 | 可选永久删除 |

```bash
# 维护脚本
python3 memory_ttl_cleanup.py --stats      # 统计
python3 memory_ttl_cleanup.py              # 预览
python3 memory_ttl_cleanup.py --execute    # 执行归档
python3 memory_ttl_cleanup.py --backfill   # 为旧记忆补填 TTL
```

---

## 文件邮箱

Agent/Cron 间零依赖消息传递。每个消息是一个 JSON 文件，写入邮箱目录即投递。

```python
from mailbox import Mailbox

mb = Mailbox()

# 发送
mb.send("cron-research", subject="发现", body="详细内容", sender="agent")

# 接收
msgs = mb.list("cron-research", status="unread")
msg = mb.read("cron-research")  # 自动标记已读
mb.ack("cron-research", msg["id"])

# 统计
mb.stats()
```

```bash
# 命令行
python3 mailbox.py send --to my-box --subject "主题" --body "内容"
python3 mailbox.py list --box my-box
python3 mailbox.py read --box my-box
python3 mailbox.py stats
python3 mailbox.py cleanup
```

消息存储在 `~/.hermes/mailbox/<box>/msg_xxx.json`，默认 72 小时过期。

---

## 检索方式

### 1. 图引导检索（首选，3-157x 加速）

```bash
python3 graph_retrieval.py "搜索内容"
```

两阶段：聚类定位（毫秒）→ 簇内 BM25 → 邻居扩展。索引缓存 24h，新记忆后需 `--rebuild`。

### 2. 混合检索（兜底）

```bash
python3 hybrid_search.py "搜索内容"
```

向量 + BM25 关键词 + RRF 融合，适合图引导无结果时 fallback。

### 3. REST API（通用）

```bash
curl -X POST http://127.0.0.1:8900/memory/query \
  -H "Content-Type: application/json" \
  -d '{"text":"搜索内容","top_k":5,"threshold":0.3}'
```

---

## 维护

```bash
# 事实冲突检测 + 低频衰减 + 语义去重 + LLM 审核
python3 maintenance.py

# TTL 归档清理
python3 memory_ttl_cleanup.py --execute

# 记忆备份
python3 memory_backup.py
```

---

## 性能

| 操作 | 耗时 | 说明 |
|------|------|------|
| 首次启动 | ~12s | 加载 3.3GB 模型到内存 |
| 写入一条记忆 | ~741ms | BGE-M3 编码 + ChromaDB 写入 |
| 搜索 5 条结果 | ~50ms | BGE-M3 编码 + HNSW 检索 |
| 实体提取 | ~2s | 后台异步，不阻塞写入 |
| 10 万条数据查询 | <100ms | HNSW O(log N) |

测试环境：Apple Silicon (MPS GPU)

---

## 项目结构

```
local-memory-stack/
├── src/local_memory_stack/
│   ├── engine.py              ← ⭐ 核心引擎（去重 + TTL + 归档）
│   ├── server.py              ← FastAPI REST 服务
│   └── cli.py                 ← 命令行工具
├── mailbox.py                 ← 文件邮箱（Agent 间消息传递）
├── hybrid_search.py           ← 混合检索（向量 + BM25 + RRF）
├── graph_retrieval.py         ← 图引导检索（聚类加速）
├── memory_ttl_cleanup.py      ← TTL 归档维护脚本
├── maintenance.py             ← 事实冲突 + 语义去重 + LLM 审核
├── memory_backup.py           ← 记忆备份
├── tests/                     ← 测试套件
├── chroma_data/               ← 记忆数据库（gitignore）
└── README.md
```

---

## 部署

### macOS（launchd 开机自启）

```bash
# 创建 plist
cat > ~/Library/LaunchAgents/com.local-memory-stack.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local-memory-stack</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which python3)</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>src.local_memory_stack.server:app</string>
        <string>--port</string>
        <string>8900</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PWD</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.local-memory-stack.plist
```

### 迁移

```bash
# 1. 复制项目目录（chroma_data 必须带上）
scp -r local-memory-stack/ 新机器:~/projects/

# 2. 安装
cd ~/projects/local-memory-stack
python3 -m venv venv && source venv/bin/activate
pip install -e .

# 3. 启动（首次会下载模型）
python3 -m uvicorn src.local_memory_stack.server:app --port 8900
```

---

## 技术栈

| 组件 | 模型 | 大小 | 作用 |
|------|------|------|------|
| BGE-M3 | BAAI/bge-m3 | 2.2GB | 文本 → 1024 维向量 |
| ChromaDB | — | — | 向量数据库 + HNSW 索引 |
| GLiNER | gliner_multi-v2.1 | 1.1GB | 实体提取（人名/地点/事件） |

零外部 API 依赖，全部跑在本地。

---

## License

[Apache-2.0](LICENSE)
