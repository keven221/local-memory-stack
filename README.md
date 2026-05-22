# 🧠 本地记忆栈（Local Memory Stack）

> 一份完全跑在本地的语义记忆系统。不需要联网，不需要 API Key，不需要外部服务。  
> 文本存进去 → 自动理解含义 → 以后用大白话就能搜出来。

---

## 一、这玩意是干什么的？

打个比方：你有一个笔记本，你在上面写了"张三的猫叫咪咪，是橘猫和英短的混血串串"。

普通的 txt 文档，你以后想找"张三养了什么宠物？"，它匹配不到"宠物"这两个字——因为你写的是"猫"。

**本地记忆栈不一样：它把每句话变成 1024 个数字（叫"向量"），数字保存的不只是词，而是意思。** 以后你问"张三养了什么宠物？"，它算出来的向量和你存的向量距离很近，就给你返回"咪咪是张三养的橘猫 × 英短串串"。

**核心价值：用白话搜，不用记关键词。**

---

## 二、文件在哪？

```
~/projects/local-memory-stack/

├── pyproject.toml                    ← 打包配置（pip install 用的）
├── src/local_memory_stack/           ← 核心代码
│   ├── __init__.py                   ← 包入口，导出了 MemoryEngine
│   ├── engine.py                     ← ⭐ 核心引擎（500行，最核心的文件）
│   ├── server.py                     ← FastAPI REST 服务（薄封装 engine）
│   └── cli.py                        ← 命令行工具
├── server.py                         ← 旧的 server（可删，新server在src里）
├── demo.py                           ← 验证脚本（跑一遍看三大组件是否正常）
├── test_api.py                       ← API 测试脚本
├── chroma_data/                      ← ⚡ 记忆数据库文件（不要删！）
└── logs/                             ← 服务日志
```

**最有价值的文件就是 `src/local_memory_stack/engine.py`**，整个系统的核心都在这里。

---

## 三、里面装了三个模型，各干什么？

### 1. BGE-M3（嵌入模型，2.2GB）

| 是什么 | 干什么 | 快不快 |
|--------|--------|--------|
| 把一段中文变成 1024 个数字（向量） | "张三的猫叫咪咪" → `[0.12, -0.34, 0.56, ...]` | M5 Metal GPU 加速，0.27 秒/条 |

**大白话解释：** 它是翻译官，把"人话"翻译成"机器能比的数字"。两句话意思越近，它们对应的数字就越像。这就是为什么你问"宠物"能搜到"猫"——翻译官知道它们是近义词。

### 2. ChromaDB（向量数据库）

| 是什么 | 干什么 | 快不快 |
|--------|--------|--------|
| 存向量 + 快速搜最像的 | 把 1024 个数字存起来，以后搜最接近的 | HNSW 索引，百万级数据查询 < 100ms |

**大白话解释：** 它是图书馆管理员。你把翻译好的数字交给他，他按规律放好。下次你问一个问题，他不用一个一个翻，而是直接跳到最像的那个书架去找。这个"直接跳"的算法叫 HNSW，数据再多也不怕。

### 3. GLiNER（实体提取模型，1.1GB）

| 是什么 | 干什么 | 快不快 |
|--------|--------|--------|
| 从文本里抓出人名、地点、事件类型等 | "张三在杭州，刚完成项目部署" → 张三(人)、杭州(地点)、项目部署(事件) | 后台异步跑，不阻塞 |

**大白话解释：** 它是标注员，把你存的每条记忆贴上标签。"这个人是谁""说的是哪个地方""做了什么"。贴完标签后，你可以按标签搜，比如"所有和项目有关的记忆"。

---

## 四、数据流是怎么走的？

### 写入（存记忆）

```
你写一条记忆："用户昨天完成了项目部署"
          ↓
    BGE-M3 翻译成 1024 个数字
          ↓
    先去 ChromaDB 查：有没有相似的旧记忆？
          ↓
    找到了"用户正在部署项目"（相似度 0.88，超过 0.85 阈值）
          ↓
    合并！保留更完整的文字，标签也合并
          ↓
    GLiNER 后台标注：用户(人)、昨天(时间)、项目部署(事件)
          ↓
    完成。返回 "merged"（而不是新增一条重复的）
```

### 查询（搜记忆）

```
你问："用户最近项目进展如何？"
          ↓
    BGE-M3 翻译成 1024 个数字
          ↓
    ChromaDB HNSW 索引快速定位最近邻
          ↓
    返回："用户昨天完成了项目部署"
         "用户上周修复了登录bug"
         "用户在写单元测试"
          ↓
    按相似度从高到低排序，返回给 Agent
```

### 对话中的自动流程（Hermes 插件）

```
每轮对话开始：
    prefetch → 用户消息 → BGE-M3 翻译 → ChromaDB 查询 → 相关记忆注入对话上下文

每轮对话结束：
    sync_turn → 已禁用（避免产生垃圾）

每次 memory 工具调用：
    on_memory_write → 自动镜像到本地记忆栈
```

---

## 五、核心功能清单

| 功能 | 怎么实现的 | 效果 |
|------|-----------|------|
| 🔍 语义搜索 | BGE-M3 1024维向量 + ChromaDB | "宠物"搜到"猫" |
| 🔄 写入去重 | 新记忆和旧记忆比相似度 > 0.85 就合并 | 不会出现 3 条"咪咪" |
| 🏷️ 自动标注 | GLiNER 后台异步提取实体 | 每条记忆带人名/地点/事件标签 |
| 📂 按来源分区 | 写入时打 source 标签，查询可按 source 过滤 | "只看项目相关的记忆" |
| ⚡ 不卡写入 | 实体提取后台慢慢做，写入立即返回 | 741ms 写入完成 |
| 🧹 自动清理 | 凌晨 2:00 cron 跑检测脚本 | 删除过期记忆，报告冲突 |
| 🔌 框架无关 | 纯 Python 类 MemoryEngine，零依赖框架 | 换任何 Agent 都能用 |

---

## 六、三种使用方式

### 方式 1：Python SDK（推荐，换框架无缝迁移）

```python
from local_memory_stack import MemoryEngine

# 初始化（12 秒首次加载，之后秒级）
engine = MemoryEngine()

# 存记忆
result = engine.write("咪咪是张三养的橘猫串串", source="profile", tags=["pet"])
# → {"action": "added", "id": "mem_abc123"}

# 再次存相似的，自动合并
result = engine.write("咪咪是张三养的橘猫×英短混血", source="profile")
# → {"action": "merged", "similarity": 0.88}

# 搜记忆
entries = engine.query("张三养了什么宠物？", top_k=3)
for e in entries:
    print(f"[{e.similarity:.0%}] {e.text}")

# 看统计
stats = engine.stats()
# → {"total_memories": 15, "device": "mps:0"}
```

### 方式 2：CLI 命令行

```bash
# 安装后直接用
pip install .

# 写入
local-memory write "项目记忆栈开发完成" --source project --tags milestone

# 查询
local-memory query "最近完成了什么"

# 查看统计
local-memory stats

# 清理垃圾
local-memory cleanup

# 启动 REST 服务
local-memory start --port 8900
```

### 方式 3：REST API

```bash
# 写入
curl -X POST http://127.0.0.1:8900/memory/write \
  -H "Content-Type: application/json" \
  -d '{"text":"记忆内容","source":"api","tags":["test"]}'

# 查询
curl -X POST http://127.0.0.1:8900/memory/query \
  -H "Content-Type: application/json" \
  -d '{"text":"搜索内容","top_k":5}'

# 实体提取
curl -X POST http://127.0.0.1:8900/memory/entities \
  -H "Content-Type: application/json" \
  -d '{"text":"用户在北京租房"}'

# 统计
curl http://127.0.0.1:8900/memory/stats
```

---

## 七、如何部署？

### 本机（macOS M5）

服务已经通过 launchd 开机自动启动：

```bash
# 查看状态
launchctl list | grep local-memory

# 手动启动/停止
launchctl start com.local-memory-stack
launchctl stop com.local-memory-stack

# 查看日志
tail -f ~/projects/local-memory-stack/logs/server.log
```

### 搬到新机器

```bash
# 1. 复制整个项目目录
scp -r local-memory-stack/ 新机器:~/projects/

# 2. 安装
cd ~/projects/local-memory-stack
python3 -m venv venv
source venv/bin/activate
pip install .

# 3. 首次启动（会下载模型到 ~/.cache/huggingface/，约 3.3GB）
local-memory start

# 4. 设置开机自启（macOS）
cp local-memory-stack.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/local-memory-stack.plist
```

**注意：`chroma_data/` 目录是记忆数据库，迁移时必须带上。**

---

## 八、性能数据（M5 MacBook Air）

| 操作 | 耗时 | 瓶颈 |
|------|------|------|
| 首次启动（加载模型） | ~12 秒 | 从硬盘读 3.3GB 模型到内存 |
| 写入一条记忆 | ~741ms | BGE-M3 编码 0.27s + ChromaDB 写入 |
| 查询 5 条结果 | ~50ms | BGE-M3 编码 0.27s + HNSW 检索 |
| 实体提取（异步） | ~2s（后台） | GLiNER 模型推理 |
| 10万条数据查询 | <100ms | HNSW O(log N) 几乎不变 |

---

## 九、目前的限制和未来方向

### 现在能做的 ✅
- 语义搜索（白话搜到意思相近的内容）
- 写入自动去重（同一个事实不会存两份）
- 实体自动标注
- 无论换什么 Agent 框架都能用

### 现在做不了的 ❌（需要接入本地 LLM）
- **事实更新**：存了"用户用 Python 3.10"又存"用户升级到 Python 3.12"，不知道这是同一事实的更新
- **跨记忆推理**：不知道"用户有数据库经验 + 会 FastAPI = 适合做后端任务"
- **自动总结**：100 条关于某个项目的记忆，不会压缩成一段综述
- **时效衰减**：半年前的记忆和昨天的记忆一样重要

### 未来可以做（如果接 Ollama 跑个 Qwen3:7B）
- 写入时判断"这是新事实还是旧事实的更新"
- 定期总结同主题记忆
- 自动标记过期信息

---

---
