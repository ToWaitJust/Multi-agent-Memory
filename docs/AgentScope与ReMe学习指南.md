# AgentScope 与 ReMe 学习指南

> 面向：SRTP「多智能体记忆共享调度」课题强势入门
> 依据：AgentScope **2.0.6** + reme-ai **0.4.1.6** 本机源码（2026-08-14 核验）
> 目标：搞清楚"每个组件有什么用、被怎么用"，能独立装配一个带记忆的 Agent

---

## 1. AgentScope 是什么

一句话：**多智能体应用开发框架**。你写 Agent（谁在说话）、挂上 Model（用什么脑子）、配上 Tool（能做什么事）、插上 Middleware（在哪一步拦截），它就自己跑推理-工具循环。

```
装配一个 Agent 的样子：
Agent(
    name="assistant",
    system_prompt="...",
    model=ChatModelBase(...),
    toolkit=Toolkit(tools=[...]),
    middlewares=[TracingMiddleware(), ReMeMiddleware(...)],
)
agent.reply(Msg(...))   # 一次回复 = 一次完整推理-工具循环
```

**关键特性**（为什么适合我们课题）：
- 中间件机制：记忆读写**完全中间件化**，调度算法只需写一个 Middleware，不碰 Agent/ReMe 源码（NFR-5）
- 模型无关：换模型只改 Model，调度代码不动
- 内置 Tracing（OpenTelemetry）：Token/耗时/调用链自动采集
- 自带 ReMe 记忆中间件 + RAG 工具包

---

## 2. AgentScope 核心组件速览

### 2.1 Agent（智能体，一切围绕它）

| 项 | 内容 |
|----|------|
| **是什么** | 一个"会说话+会做事"的角色，持有自己的对话上下文 |
| **被怎么用** | `Agent(name, system_prompt, model, toolkit, middlewares, state)` |
| **核心方法** | `reply(inputs)` 返回最终 Msg；`reply_stream()` 流式；`observe()` 接收观察消息 |
| **源码** | `agentscope/agent/_agent.py`：`__init__` 114-214 行、`reply` 295 行、`_reply` 633 行（中间件链 execute_chain）、`_reasoning` 1257 行 |

**重要机制**：Agent 把传入的 `middlewares` 按 hook 类型分成 7 组（`_reply_middlewares`/`_reasoning_middlewares`/...），每组按列表顺序用 `execute_chain` 洋葱式调用——这就是为什么中间件有**挂载顺序**。

### 2.2 Model（大脑）

| 项 | 内容 |
|----|------|
| **是什么** | 所有 LLM 调用的统一接口，Agent 用它"思考" |
| **被怎么用** | 实例化后注入 Agent；`ChatModelBase.__call__(messages, tools, tool_choice)` 异步调用，返回 `ChatResponse`（含流式 delta + usage） |
| **实现** | `DeepSeekChatModel` / `DashScopeChatModel` / `OpenAIChatModel`（都在 `agentscope/model/` 下，都继承 `ChatModelBase`） |
| **源码** | `agentscope/model/_base.py`：`__call__` 158 行、`_call_api` 269 行（带重试 max_retries） |

### 2.3 Tool（能力）

| 项 | 内容 |
|----|------|
| **是什么** | Agent 能调用的"技能"，LLM 决策何时调用 |
| **被怎么用** | 子类化 `ToolBase`，实现 `name`/`description`/`input_schema`/`__call__`/`check_permissions`；通过 `Toolkit(tools=[...])` 注入 Agent |
| **源码** | `agentscope/tool/`：`ToolBase` 基类、`Toolkit` 装配器 |

示例（ReMe 的 memory_search 工具，已核验 `agentscope/middleware/_longterm_memory/_reme/_tools.py` 73 行）：
```python
class _MemorySearchTool(_ReMeMemoryToolBase):
    name: str = "memory_search"
    description: str = "Retrieve memories from past conversations..."
    input_schema: dict[str, Any] = {...}
    async def __call__(self, query: str, limit: int | None = None) -> ToolChunk: ...
```

### 2.4 Message（消息，Agent 之间/与模型的通信单位）

| 项 | 内容 |
|----|------|
| **是什么** | `Msg(role, name, content=List[ContentBlock])`，content 是**消息块列表**（2.0 核心变化） |
| **块类型** | `TextBlock`/`HintBlock`/`ThinkingBlock`/`DataBlock`/`ToolCallBlock`/`ToolResultBlock` |
| **关键块** | `HintBlock(hint=...)`——向模型注入"提示/记忆"的块，**我们调度注入记忆就用它**（ReMe 注入也用它，name="memory"） |
| **源码** | `agentscope/message/_block.py`：TextBlock 11 行、HintBlock 101 行、ToolCallBlock 138 行 |

### 2.5 Middleware（中间件，课题主战场）

| 项 | 内容 |
|----|------|
| **是什么** | 在 Agent 7 个执行点拦截的"插件"，**继承 `MiddlewareBase`** |
| **7 个 hook** | `on_reply`（整个回复）、`on_reasoning`（推理前）、`on_acting`（工具执行）、`on_check_permission`（权限）、`on_model_call`（模型调用）、`on_compress_context`（压缩）、`on_system_prompt`（系统提示） |
| **被怎么用** | 子类实现需要的 hook（`is_implemented` 自动检测），实例挂进 Agent `middlewares=[...]` |
| **源码** | `agentscope/middleware/_base.py` 13 行起，7 个 hook 的完整签名 68-279 行 |

**我们的调度中间件就是第 8 个 Middleware**，实现 `on_reply`（调度前置）+ `on_reasoning`（注入共享池）+ `on_system_prompt`（追加调度说明）。

### 2.6 RAG 工具包（知识问答）

| 项 | 内容 |
|----|------|
| **是什么** | 文档型 RAG 全流程：解析→分块→向量化→向量库→检索→Agent 注入 |
| **核心类** | `KnowledgeBase(name, description, embedding_model, vector_store, collection)`：`search()`/`insert_document()`/`delete_document()`/`list_documents()` |
| **解析器** | `PDFParser`/`WordParser`/`ExcelParser`/`PPTParser`/`TextParser`/`ImageParser` |
| **向量库** | `MilvusLiteStore`/`QdrantStore`/`MongoDBStore`/`ElasticsearchStore` |
| **源码** | `agentscope/rag/`：`__init__.py` 导出 30 行、`_knowledge.py` 44 行起 |

> ⚠️ 局限：`KnowledgeBase.search()` = **纯向量检索**，无 BM25、无 RRF、无关系展开（详见"为什么不拿它当主记忆层"）。

### 2.7 Channel（外部通道）

| 项 | 内容 |
|----|------|
| **是什么** | 把 Agent 接入外部平台（飞书/钉钉/Discord/网页） |
| **源码** | `agentscope/app/channel/`：飞书 `_feishu/`、Discord、基础 `_base.py` |

### 2.8 TracingMiddleware（全链路追踪）

| 项 | 内容 |
|----|------|
| **是什么** | 自动把 Agent 回复/LLM 调用/工具调用记录为 OpenTelemetry Span |
| **hook** | `on_reply`/`on_model_call`/`on_acting` |
| **源码** | `agentscope/middleware/_tracing/_trace.py`：`on_reply` 137 行、`on_model_call` 259 行、`on_acting` 312 行 |

---

## 3. AgentScope 典型调用路径（一次 reply 到底发生了什么）

```
agent.reply(Msg)
  → _reply(inputs)
      → execute_chain(middleware 链):
          TracingMiddleware.on_reply   (开 Agent Span)
            → MemorySchedulingMiddleware.on_reply   (调度前置：候选召回→打分→注入)
              → ReMeMiddleware.on_reply              (自动检索任务+快照pre_ids)
                → ... → _reply_impl
                    → _reasoning:
                        execute_chain(reasoning 链):
                            TracingMiddleware.on_reasoning
                            → MemorySchedulingMiddleware.on_reasoning  (注入HintBlock)
                              → ReMeMiddleware.on_reasoning            (注入原生检索HintBlock)
                                → _reasoning_impl → ChatModelBase.__call__ → LLM
                    → 工具循环(_batch_tool_calls → _execute_tool_call → on_acting 链)
                → ReMeMiddleware finally: _write_back → auto_memory 蒸馏
              → MemorySchedulingMiddleware finally: 埋点 flush
          → TracingMiddleware finally: 关 Agent Span
```

**要点**：
1. **中间件是洋葱链**，顺序 = 挂载顺序（D-2：调度挂 ReMe 之前）
2. `on_reasoning` 在**每次推理前**都触发——这是注入记忆的正确时机
3. 工具调用也有自己的中间件链（`on_acting`/`on_check_permission`）

---

## 4. ReMe 是什么

一句话：**AgentScope 生态的 local-first 文件化记忆层**，核心理念 **"Memory as File"**——把对话与资料沉淀为带 frontmatter 的 Markdown 记忆节点，持续索引、链接、整理。论文 *Remember Me, Refine Me* 已中 **Findings of ACL 2026**。

### 4.1 ReMe 四大 Job

| Job | 功能 | 产出 |
|-----|------|------|
| `auto_memory` | 对话→记忆（LLM 蒸馏） | `daily/<date>/<session>.md` + `session/dialog/*.jsonl` |
| `auto_resource` | 资料→记忆节点 | daily 卡片 |
| `auto_dream` / `digest` | 记忆自进化（夜间整理、建关联、生成兴趣） | digest 节点 + `interests.yaml` |
| `search` | 渐进式混合检索 | 召回片段 + 链接展开 |

### 4.2 记忆存储结构（已实探 `data/reme/`）

```
workspace/
├── daily/<date>/<session>.md    # 记忆卡片（frontmatter + wikilink）
├── digest/                      # 长期沉淀节点
├── resource/                    # 原始资料
├── session/dialog/*.jsonl       # 原始对话轨迹
├── metadata/                    # 索引（file_catalog/file_graph/keyword_index）
└── MEMORY.md                    # 长期记忆索引
```

### 4.3 search 的渐进式混合检索（核心，课题基座）

```
query + limit
  → candidates = min(200, limit × candidate_multiplier[5])
      → 向量搜索（embedding，默认关）
      → 关键词搜索（BM25 倒排，默认开）
  → RRF 融合（score = 0.7/(60+v_rank) + 0.3/(60+k_rank)）
  → min_score 过滤 → 截断 limit
  → wikilink 展开（每向 ≤10 条邻居，默认开）
```

> 这就是**检索基座要中性化**（D-12）的具体来源：RRF/wikilink/min_score 都是 ReMe 自带的"检索算法决策"，要与我们的调度算法分开。

### 4.4 ReMe 与 AgentScope 的连接：ReMeMiddleware

ReMeMiddleware 就是"把 ReMe 接进 AgentScope 的那根线"（源码 `agentscope/middleware/_longterm_memory/_reme/_middleware.py` 85 行起，已核验）：

| 关键部分 | 内容 |
|---------|------|
| **构造函数** | `ReMeMiddleware(workspace_dir, parameters=Parameters(chat_model, embedding_model, mode, top_k))` |
| **模式** | `mode="static_control"`（自动检索注入，无工具）/ `"agent_control"`（无自动检索，有工具）/ `"both"`（都要） |
| **on_reply** | 快照 pre_ids + 启动异步检索任务 + yield 链 + finally 写回增量 |
| **on_reasoning** | 轮询检索任务.done()，完成则 append HintBlock(name="memory") 到 context |
| **on_system_prompt** | 追加 memory_search 工具说明（非 static_control 时） |
| **list_tools** | 返回 memory_search 工具（非 static_control 时） |
| **_search/query** | `self._run_job("search", query=query, limit=top_k)` |
| **_write_back** | `self._run_job("auto_memory", messages=[...], session_id=...)` |

> **课题的关键认知**：我们不是"用 ReMe"，而是"在 ReMe 之上做调度增强"——ReMeMiddleware 就是我们的**基座**，我们的 `MemorySchedulingMiddleware` 是**套在它外面的第二层中间件**（D-1 方案 A）。

---

## 5. 上手练习（5 个，由浅入深）

### 练习 1：跑通现有 `reme_demo.py`
```bash
# 已可运行（DEEPSEEK_API_KEY 已配置）
python examples/reme_demo.py
```
**观察**：3 轮对话→auto_memory 自动写记忆→第 3 轮检索回忆。**注意**：当前实测三次检索 0 命中（跨会话未验证），这是我们要修的 P0-1。

### 练习 2：装配一个"带 Tracing 的裸 Agent"
写 `examples/agent_basic.py`：用 `Agent` + `DeepSeekChatModel` + `TracingMiddleware`，不带记忆，跑一次回复。
**目的**：理解 Agent 最小装配 + Tracing 自动采集。

### 练习 3：读一遍中间件链源码
按顺序读：`_base.py`(68-279 行 7 个 hook) → `_reme/_middleware.py`(324 行 on_reply / 397 行 on_reasoning) → `agent/_agent.py`(633 行 execute_chain)。
**目的**：把"调用路径"（第 3 节图）和源码一一对上。

### 练习 4：手写一个最小 Middleware
写 `examples/logging_mw.py`：子类 `MiddlewareBase`，只实现 `on_reply`，打印"回复前/后"，挂到一个 Agent 上。
**目的**：亲手体验中间件 hook 机制——我们调度中间件就是这个的"加强版"。

### 练习 5：实现一个最小 MemoryCoordinator 原型
在 `srtp_memory/` 下搭骨架，实现 `_recall_candidates`（取 ReMe 原始分，D-12）+ 一个最简单的 `_time_score`，挂到 demo 上跑。
**目的**：把工程文档第 4.2/4.6 章"方法级设计"跑成能执行的代码。

---

## 6. 关键源码路径索引

```
agentscope/
├── agent/_agent.py                # Agent；reply/reply_stream/_reply/execute_chain/_reasoning
├── middleware/_base.py            # MiddlewareBase（7 hooks + is_implemented + list_tools）
├── middleware/_longterm_memory/_reme/
│   ├── _middleware.py             # ReMeMiddleware（on_reply/on_reasoning/on_system_prompt/_search/_write_back/_build_memory_message）
│   ├── _tools.py                  # _MemorySearchTool（memory_search 工具）
│   └── _utils.py                  # _extract_query_text / _extract_memory_texts
├── middleware/_tracing/_trace.py  # TracingMiddleware（on_reply/on_model_call/on_acting）
├── message/_block.py              # TextBlock/HintBlock/ToolCallBlock/ToolResultBlock
├── state/_state.py                # AgentState.session_id / reply_id
├── model/_base.py                 # ChatModelBase.__call__ / generate_structured_output
├── embedding/_embedding_base.py   # EmbeddingModelBase.__call__（批量+重试）
├── embedding/_dashscope/_model.py # DashScopeEmbeddingModel（text-embedding-v4, dimensions=1024）
└── rag/
    ├── _knowledge.py              # KnowledgeBase（search/insert_document）
    └── __init__.py                # Parser/Chunker/VectorStore 导出

reme/
├── application.py                 # ReMe 应用（run_job / update_component）
├── config/default.yaml            # search: limit=5, min_score=0.0, vector_weight=0.7,
│                                  #   candidate_multiplier=5.0, expand_links=true
└── components/                    # file_store / embedding_store / agent_wrapper 等
```

---

## 7. 课题视角：ReMe 好在哪、局限在哪

### 7.1 好在哪（不可替代的）
- **Memory as File**：记忆是磁盘上的 Markdown，可 git、可人工编辑、可 diff——**书签定位 `path:line`** 依赖这个
- **混合检索**：BM25+向量 RRF + wikilink 关系——我们四维注意力的**任务头/频率头**依赖可解释的分词命中
- **auto_memory 蒸馏**：对话→记忆节点的 LLM 提炼，纯 RAG 做不了
- **学术叙事**：ACL 2026 Findings 基座

### 7.2 局限在哪（要用 D-12 绕开的）
- **检索算法固化**：RRF/wikilink/min_score 是 ReMe 的决策，与我们的调度算法**不可叠加**（否则归因不清）→ D-12 中性化
- **文档解析弱**：多格式文档（PDF/Excel/图片）用 AgentScope RAG 的 parser 更合适 → 作为"资料入库管道"
- **单机 file store**：大规模/分布式场景不够 → 课题暂不涉及

---

**学习路径建议**：先做练习 1-4（1-2 天），读一遍第 2-4 节 + 源码，再做练习 5（把工程文档第 4 章跑成代码）。遇到任何"这里为什么这么设计"，回来看本文档第 2 节组件速览。

---

## 8. 2.0.6 实战补充：内置工具、权限与沙箱（2026-08-27 实测）

> 本节记录把智能体**真正跑起来**（而非只读文档）时用到的能力与踩到的坑，
> 均经本机 conda `srtp-memory` 环境（agentscope 2.0.6 + 真实 DeepSeek）实测验证。

### 8.1 调用方式（最容易踩的坑）

```python
from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel
from agentscope.agent import Agent
from agentscope.message import UserMsg, TextBlock
import asyncio

model = DeepSeekChatModel(credential=DeepSeekCredential(api_key=key), model="deepseek-v4-flash")
agent = Agent(name="MainAgent", system_prompt="...", model=model)

resp = asyncio.run(agent.reply(UserMsg(name="User", content="你好")))    # ✅ 正确
# agent(UserMsg(...))                                                   # ❌ TypeError: 不可调用
text = "".join(b.text for b in resp.content if isinstance(b, TextBlock))
```

- `Agent` 实例**不可**当函数调用，必须用 `asyncio.run(agent.reply(...))`（`reply` 是 async 方法）。
- `Msg.content` 是 **ContentBlock 列表**，取文本要遍历 `TextBlock`；构造用 `UserMsg(name=..., content=str)` 最省事。
- 工具模块名是 **`agentscope.tool`**（单数），**没有** `agentscope.toolkit` / `service` / `tools`。

### 8.2 内置工具清单（开箱即用，不必自己写）

| 工具 | 作用 |
| --- | --- |
| `Read` / `Write` / `Edit` | 文件读取 / 写入 / 修改 |
| `Glob` / `Grep` | 文件搜索 / 内容搜索 |
| `Bash` / `PowerShell` | 执行命令（Windows 下 Bash 走 `cmd /c`） |
| `TaskCreate` / `TaskGet` / `TaskList` / `TaskUpdate` | 任务管理（计划—执行—跟踪） |

```python
from agentscope.tool import Toolkit, Read, Write, Edit, Bash, Glob, Grep
toolkit = Toolkit([Read(), Write(), Edit(), Glob(), Grep(), Bash(cwd=项目根)])
agent = Agent(name=..., system_prompt=..., model=model, toolkit=toolkit, state=state)
```

内置工具自带 `dangerous_files` / `dangerous_directories` 保护（`.env`、`.ssh`、`.git` 等默认不可写），
即使权限放开也拦得住。ReAct 循环默认开启（`max_iters=20`），Agent 在推理中自动调用工具并继续。

**实测结果**：`Write` 真实创建文件（16s：Write→Read→中文总结）、`Bash` 真实执行 `dir /b` 并正确列出文件（2.5s）。

### 8.3 权限系统（PermissionMode）

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| `DEFAULT` | 逐项询问（无人应答会卡住） | 默认、最安全 |
| `ACCEPT_EDITS` | 工作目录内文件读写自动放行 | 用户在场、快速迭代 |
| `EXPLORE` | 只读，任何修改被拒 | 探索代码库 |
| `BYPASS` | 跳过权限检查（工具自带危险保护仍生效） | **本地可信自用 / 沙箱内** |
| `DONT_ASK` | 所有 ASK 转 **DENY** | 无人值守；**交互场景会卡死**，勿用 |

```python
from agentscope.state import AgentState          # 注意：在 agentscope.state，不在 agentscope.agent
from agentscope.permission import PermissionContext, PermissionMode
state = AgentState(permission_context=PermissionContext(mode=PermissionMode.BYPASS))
```

### 8.4 沙箱（workspace）

`agentscope.workspace` 提供 `LocalWorkspace` / `DockerWorkspace` / `BubblewrapWorkspace` /
`DaytonaWorkspace` / `E2BWorkspace` / `K8sWorkspace` / `OpenSandboxWorkspace`。
本课题是**本地单机自用**（D3），用 `LocalWorkspace` + `BYPASS` 即可；需要真隔离时再挂 Docker / E2B。

### 8.5 Windows 特有问题

- `Bash` 在 Windows 走 `cmd /c`：`echo` / `dir` / `python --version` 正常，
  但 `python -c "带引号的命令"` **输出为空**（cmd 引号传递坑）——改用脚本文件，或提示 Agent 少用复杂引号。
- `Toolkit.get_tool_schemas()` 是 async，需 `await` / `asyncio.run`。
- 工具任务经多轮 ReAct，耗时十几秒至数十秒属正常，前端需有加载态。
