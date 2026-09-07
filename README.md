# 基于注意力引导和 LLMs 可学习权重的多智能体记忆共享调度方法研究

> SRTP 大学生科研训练项目 · 福州大学（2026）
>
> Multi-Agent Memory Sharing Scheduling via Attention Guidance and LLMs Learnable Weights

## 项目简介

多智能体场景下，主线 Agent 拥有完整记忆池，副线 Agent 运行在隔离沙箱中。本项目研究如何通过「四维注意力 + LLM 可学习权重」实现跨智能体的**记忆共享调度**：协调层从主线完整记忆池中挑选最有价值的记忆（≤10 条）注入副线，让副线在保持上下文隔离的同时获得记忆增益。

### 核心创新点

1. **多智能体记忆共享调度**：主线完整池（只读）+ 副线隔离沙箱 + 协调层共享池（≤10 条）
2. **四维注意力打分**：时间（指数衰减）/ 语义（余弦相似）/ 频率（访问次数）/ 任务（标签匹配）
3. **LLM 可学习权重**：混合权重 = α·LLM 先验 + β·可学习（轻量 MLP），两阶段学习：离线蒸馏（MSE）+ 在线反馈（RL 式更新）
4. **条件化个性化**：per-user 条件嵌入，语义/任务维度随用户与场景漂移，时间/频率保持全局
5. **务实插件式架构**：固定内核（双池语义/协调管线/数据模型/埋点）+ 四层插件面（算法/基础设施/消费层/基准桩）+ 配置即组合（消融 = 5 份 YAML）

## 仓库结构

```
app_dashboard.py                     # 控制台后端（stdlib http.server，端口 8787，零新依赖）
dashboard.html                       # 控制台前端（7 阶段调度流 + 双池 PCA + 副线书签 + SSE 流式）
main.py                              # 装配入口：加载 srtp.yaml → 构建主副线 → 跑演示/消融
srtp_memory/                         # 核心包（headless，不依赖 UI）
├── middleware.py                    # MemorySchedulingMiddleware（调度总入口）
├── attention.py / weights.py        # 四维注意力打分 / 条件化混合权重（α·LLM先验 + β·可学习）
├── resident_pool.py / shared_pool.py# 双池：常驻完整池 / 共享池（≤10 条）
├── retriever.py / selector.py       # 5 种检索器 / 动作选择器
├── coordinator.py / condition.py    # 协调层（持有双池）/ 条件嵌入
├── agents.py / agentscope_runtime.py# Agent 工厂 / 真实 AgentScope 2.0.6 运行时（含工具系统）
└── monitor/ train/ plugins/         # 埋点监控 / 权重训练 / 插件注册表
config/                              # srtp.yaml + ablation/ 5 组消融 YAML（唯一真源）
analysis/                            # 实验脚本：消融、指标计算、流程 trace
tests/                               # 8 个单测模块
docs/                                # 工程文档（核心交付物）
├── 多智能体记忆共享调度系统工程实现文档.md   # 主工程文档 v2.1
├── 需求清单.md / 产品形态定义.md / 研究推进方案.md
├── AgentScope与ReMe学习指南.md / 权重训练集标注规范.md
└── 课题材料.md
examples/                            # reme_demo.py / sched_demo.py
```

> **目录约定**：`data/` 只放**实验数据产物**（埋点 JSONL、run_manifest、embedding 缓存、trace），
> `logs/` 只放**运行日志**（控制台日志、ReMe 应用日志）；`dev-tools-srtp/`（本地模型权重与包缓存）、
> `.workbuddy/`、`.env` 均为本地生成物，不入库。

## 技术栈

- Python · PyTorch（轻量 MLP 权重模型，CPU 可训）
- ReMe 0.4.1.6（记忆框架）· **AgentScope 2.0.6**（多智能体框架，已真实接入并启用工具系统）
- DashScope text-embedding-v4（1024 维向量）· DeepSeek（`deepseek-v4-flash`）
- faiss-cpu（ANN 索引）· jieba（BM25 中文分词）

## 快速开始

```bash
# 1) 安装依赖（conda 环境 srtp-memory 已具备全部依赖）
pip install -r requirements.txt

# 2) 配置密钥：.env 写 DASHSCOPE_API_KEY（embedding），DEEPSEEK_API_KEY 由 shell 注入
cp .env.example .env

# 3) 启动控制台（日志统一写 logs/）
python app_dashboard.py > logs/dashboard.log 2>&1 &
#    打开 http://127.0.0.1:8787/

# 4) 命令行跑一次调度 / 5 组消融
python main.py --demo
python main.py --ablation
```

控制台能力：输入问题 → 7 阶段调度流实时推进 → 双池向量库 PCA 投影（严格区分「本轮调度」
与「历史留存」记忆）→ 副线书签（真实 AgentScope Agent 产出）→ 主线流式回答。勾选
「运行副线 Agent」可让两条副线真实执行；智能体还装配了 `Read/Write/Edit/Glob/Grep/Bash`
工具，可真实读写文件与执行命令。未装 agentscope 或缺 key 时自动降级 headless，页面顶部标注。

## 文档导航

- [项目全景与全流程重要节点](docs/项目全景与全流程重要节点.md) — **入门首选**：项目是什么 + 一次问答背后的 13 个节点 + 上手路线图
- [主工程文档 v2.1](docs/多智能体记忆共享调度系统工程实现文档.md) — 需求、架构、模块设计、消融实验、埋点、验收全链路
- [需求清单](docs/需求清单.md) — FR/NFR/REQ 编号需求
- [产品形态定义](docs/产品形态定义.md) — D1~D5 产品锚点
- [研究推进方案](docs/研究推进方案.md) — 阶段规划
- [AgentScope 与 ReMe 学习指南](docs/AgentScope与ReMe学习指南.md) — 框架 API 与踩坑记录
- [权重训练集标注规范](docs/权重训练集标注规范.md) — 可学习权重训练数据标准

## 说明

本项目为学术研究用途。当前状态：主工程文档 **v2.1**（2026-08-19 定稿）；控制台与真实
AgentScope 主副线接入于 2026-08 落地，Agent 工具系统于 2026-08-27 接入，
目录清理与日志规范于 2026-09-07 统一。
