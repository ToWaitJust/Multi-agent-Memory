"""演示画布的主副线模拟数据：一个问答回合 = 一个节点。

- build_demo() 生成主线 6 轮 + 3 条副线（各自 2-3 轮）的模拟聊天记录；
- load_or_create() 落盘 data/demo/canvas_history.json，重启后历史仍在；
- append_round() 把画布里的新追问追加进历史（演示时可现场长出节点）；
- CANVAS_MEMORY_SEEDS 把聊天内容抽成记忆，随 _get_mid() 灌进常驻池 → 调度可召回。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srtp_memory.attention import MemoryCandidate  # noqa: E402

HISTORY_PATH = ROOT / "data" / "demo" / "canvas_history.json"


def _ts(minutes_ago: float) -> float:
    return time.time() - minutes_ago * 60


def build_demo() -> dict:
    base = _ts(3 * 24 * 60)  # 模拟聊天发生在 3 天前到 10 分钟前
    step = 5.5 * 60  # 相邻回合约 5.5 小时

    main_rounds = [
        {"round_id": "R1", "title": "技术栈盘点", "ts": base,
         "user": "我们的项目现在用什么技术栈？",
         "assistant": "当前技术栈：AgentScope 2.0.6（多智能体运行时）+ ReMe 0.4.1.6（记忆底座）"
                      "+ DashScope text-embedding-v4（1024 维语义向量）+ DeepSeek deepseek-v4-flash"
                      "（轻量 LLM）。检索侧用 faiss-cpu 做 ANN 索引，环境是 conda 的 srtp-memory（Python 3.13）。"},
        {"round_id": "R2", "title": "双池架构", "ts": base + step,
         "user": "双池架构是怎么设计的？",
         "assistant": "主线 MainAgent 挂常驻完整池（全量只读），副线 SubAgent 用隔离沙箱，"
                      "只注入共享池精选的 ≤10 条记忆。调度中间件挂在 ReMe 之前：on_reply 先调度、"
                      "把入选记忆以 HintBlock 注入上下文，保证副线回答不脱离记忆库。"},
        {"round_id": "R3", "title": "四维打分与权重", "ts": base + 2 * step,
         "user": "四维注意力是怎么打分的？权重怎么混合？",
         "assistant": "四个维度：time 指数衰减、semantic（余弦+向量+BM25 原始分融合）、"
                      "frequency 访问次数、task 标签匹配。混合权重 final = α·LLM先验 + β·可学习MLP，"
                      "α=0.6、β=0.4 固定不学——为了消融归因干净。"},
        {"round_id": "R4", "title": "消融实验设计", "ts": base + 3 * step,
         "user": "消融实验设计好了吗？",
         "assistant": "5 组：baseline_naive（全量传输下界）/ baseline_reme（wrap ReMe 原生）/ "
                      "+attention（四维增量）/ +weight（混合权重增量）/ full_ours（完整方案）。"
                      "配置即组合，YAML 是唯一手写真源，manifest 由脚本自动生成。"},
        {"round_id": "R5", "title": "检索器选型", "ts": base + 4 * step,
         "user": "检索器为什么最后选了向量检索？",
         "assistant": "full_ours 从 bm25 升级为 retriever.vector：真实 DashScope embedding + "
                      "落盘缓存保证可复现。BM25 侧装 jieba 做中文分词当对照组。"
                      "注意归因问题：五组检索基座要统一，否则增量会混入检索器变化。"},
        {"round_id": "R6", "title": "下一步计划", "ts": base + 5 * step,
         "user": "下一步计划是什么？",
         "assistant": "两条线：主线先修复消融退化（offline 下 naive≡reme、+attn≡+weight），"
                      "换在线模式重跑 200 episodes；同时 topic 图 idea 已开独立工作区做 POC 验证，"
                      "真实数据上多源调度已跑赢单源。"},
    ]

    subs = [
        {"session_id": "sub_weight", "derived_from": "R3", "title": "副线 · 权重训练",
         "inherited": [
             "四维打分：time 指数衰减 / semantic 余弦+向量+BM25 / frequency 访问数 / task 标签匹配",
             "混合权重 final = α·LLM先验 + β·MLP，α=0.6 β=0.4 固定保归因干净",
             "MLP 训练目标：MSE 蒸馏 LLM 先验权重，正则防偏离；反馈只更新 condition 切片",
             "条件化个性化：一份 MLP 拼 user/scenario embedding，仅语义/任务维随场景漂移",
         ],
         "rounds": [
             {"round_id": "S1-1", "title": "训练目标", "ts": base + 2.2 * step,
              "user": "副线任务启动：把可学习权重模型训出来。",
              "assistant": "从主线继承了相关记忆（四维打分定义、混合权重公式 α·先验+β·MLP）。"
                           "本线独立推进：训练目标定为 MSE 蒸馏 LLM 先验权重（多次采样平均做标签），"
                           "加正则防偏离先验；不动 α 和主线模型。"},
             {"round_id": "S1-2", "title": "训练集", "ts": base + 2.6 * step,
              "user": "训练集从哪来？",
              "assistant": "按权重训练集标注规范：真实交互 + 模拟生成，蒸馏标签取 LLM 先验的"
                           "多次采样平均。样本进入前先过条件化（拼 user/scenario embedding）。"},
             {"round_id": "S1-3", "title": "反馈闭环", "ts": base + 2.9 * step,
              "user": "在线反馈怎么接进训练？",
              "assistant": "点赞/点踩 = reward(+1/-1)，bandit 式奖励加权；只更新 condition 对应"
                           "切片（语义/任务头），时间/频率保持全局。"},
         ]},
        {"session_id": "sub_ablation", "derived_from": "R4", "title": "副线 · 消融数据集",
         "inherited": [
             "消融 5 组：baseline_naive / baseline_reme / +attention / +weight / full_ours，YAML 唯一真源",
             "消融数据集：确定性合成 200 行带 ground-truth；计划 AgentScope 自建多模块真实语料",
             "消融指标：recall_acc / task_completion / token_reduction / compression_ratio / latency",
             "消融判据：+weight→full_ours 的增量必须只含选择器变化，检索基座五组统一",
         ],
         "rounds": [
             {"round_id": "S2-1", "title": "数据集构建", "ts": base + 3.2 * step,
              "user": "副线任务启动：把消融数据集做出来。",
              "assistant": "从主线继承了消融 5 组设计与指标口径的记忆。本线产出：确定性合成"
                           "生成器 200 行带 ground-truth；后续计划用 AgentScope 多模块项目场景"
                           "自建真实语料（todo-cli 场景已跑通）。"},
             {"round_id": "S2-2", "title": "标注口径", "ts": base + 3.6 * step,
              "user": "ground truth 怎么标才不踩坑？",
              "assistant": "锚点关键词 + 人工抽检。教训：相关集一旦超过注入上限 K，所有组的"
                           "recall 都被 10/|relevant| 天花板压死——标注必须与 |relevant| ≤ K 校验联动。"},
         ]},
        {"session_id": "sub_retriever", "derived_from": "R5", "title": "副线 · 检索器对比",
         "inherited": [
             "检索器决策：full_ours 用 retriever.vector（真实 DashScope embedding + 落盘缓存可复现）",
             "BM25 中文场景必须 jieba 分词，否则召回接近随机、对照组失真",
             "归因纪律：五组消融检索基座必须统一，否则增量混入检索器变化",
         ],
         "rounds": [
             {"round_id": "S3-1", "title": "对比实验", "ts": base + 4.2 * step,
              "user": "副线任务启动：把 bm25 和 vector 比清楚。",
              "assistant": "从主线继承了检索器决策与 embedding 缓存机制的记忆。本线独立跑对比："
                           "中文场景 bm25 依赖分词质量，向量检索对同义改写更稳，两者是可组合的插件原语。"},
             {"round_id": "S3-2", "title": "jieba 结论", "ts": base + 4.5 * step,
              "user": "分词器结论？",
              "assistant": "默认分词器对中文基本无效，不装 jieba 的 bm25 召回接近随机，对照组失真。"
                           "结论：bm25 对照组必须 jieba 分词。"},
             {"round_id": "S3-3", "title": "归因结论", "ts": base + 4.8 * step,
              "user": "检索器升级会污染归因吗？",
              "assistant": "会。full_ours 单独换检索器，+weight→full_ours 增量就混入检索器变化。"
                           "本线结论：五组消融统一 retriever.vector，归因才干净。"},
         ]},
    ]

    return {"main": {"session_id": "canvas_main", "title": "主线 · SRTP 项目推进",
                     "rounds": main_rounds},
            "subs": subs, "saved_at": time.strftime("%F %T")}


# 从模拟聊天抽出的记忆：聊天内容 → 常驻池（演示时调度可真实召回）
CANVAS_MEMORY_SEEDS = [
    ("mc_r1a", "技术栈定型：AgentScope 2.0.6 + ReMe 0.4.1.6 + DashScope text-embedding-v4（1024 维）+ DeepSeek", "tech_stack", 72),
    ("mc_r1b", "运行环境：Python 3.13 conda srtp-memory，faiss-cpu 做 ANN 检索", "tech_stack", 72),
    ("mc_r2a", "双池设计：主线常驻完整池只读，副线只注入共享池 ≤10 条精选记忆", "architecture", 60),
    ("mc_r2b", "调度时机：中间件 on_reply 先调度，入选记忆以 HintBlock 注入副线上下文", "architecture", 60),
    ("mc_r3a", "四维打分：time 指数衰减 / semantic 余弦+向量+BM25 / frequency 访问数 / task 标签匹配", "architecture", 48),
    ("mc_r3b", "混合权重 final = α·LLM先验 + β·MLP，α=0.6 β=0.4 固定保归因干净", "architecture", 48),
    ("mc_s3a", "MLP 训练目标：MSE 蒸馏 LLM 先验权重，正则防偏离；反馈只更新 condition 切片", "research", 40),
    ("mc_s3b", "条件化个性化：一份 MLP 拼 user/scenario embedding，仅语义/任务维随场景漂移", "research", 40),
    ("mc_r4a", "消融 5 组：baseline_naive / baseline_reme / +attention / +weight / full_ours，YAML 唯一真源", "research", 36),
    ("mc_s1a", "消融数据集：确定性合成 200 行带 ground-truth；计划 AgentScope 自建多模块真实语料", "research", 30),
    ("mc_s1b", "消融指标：recall_acc / task_completion / token_reduction / compression_ratio / latency", "research", 30),
    ("mc_r5a", "检索器决策：full_ours 用 retriever.vector（真实 DashScope embedding + 落盘缓存可复现）", "architecture", 20),
    ("mc_s2a", "BM25 中文场景必须 jieba 分词，否则召回接近随机、对照组失真", "tech_stack", 16),
    ("mc_s2b", "归因纪律：五组消融检索基座必须统一，否则增量混入检索器变化", "research", 12),
    ("mc_r6a", "下一步：修消融退化后在线重跑 200 episodes；topic 图 idea 走 POC 验证", "research", 6),
    # ── 大幅扩充：mc2_*（覆盖项目里程碑 / 调度细节 / 消融数据 / 检索 / 运行时 / 偏好 / 干扰）──
    ("mc2_p01", "课题全称：基于注意力引导和 LLMs 可学习权重的多智能体记忆共享调度方法研究（SRTP）", "project", 700),
    ("mc2_p02", "项目里程碑：端到端链路已跑通（AgentScope + ReMe + faiss + DashScope + DeepSeek）", "project", 600),
    ("mc2_p03", "项目里程碑：真实 AgentScope 运行时接入，主/副/审三 Agent 装配 Toolkit 与 BYPASS 权限", "project", 500),
    ("mc2_p04", "项目里程碑：检索器升级为真实 DashScope embedding 语义向量检索 + 落盘缓存", "project", 400),
    ("mc2_p05", "项目里程碑：topic 图 idea POC 两轮跑通，真实数据上多源调度优于单源", "project", 300),
    ("mc2_p06", "项目里程碑：演示控制台与会话画布上线，支持画布追问与创建副线", "project", 200),
    ("mc2_p07", "工程约定：YAML 是唯一手写真源，manifest 一律由脚本生成禁止手改", "project", 500),
    ("mc2_p08", "工程约定：data/ 放实验数据，logs/ 放进程日志，两者严格分开", "project", 480),
    ("mc2_a01", "调度五阶段：召回候选(50) → 四维打分 → 混合权重 → 排序选择(≤10) → 共享池写入", "architecture", 300),
    ("mc2_a02", "候选集规模 50：由 retriever 在常驻池上一次召回，避免全量打分开销", "architecture", 260),
    ("mc2_a03", "四维启用掩码 enabled_dims：关闭维的权重强制置 0 后归一化，保证 Σ=1", "architecture", 200),
    ("mc2_a04", "语义头内部系数 w_cos=0.5 / w_vec=0.3 / w_bm25=0.2，与全局 α/β 无关", "architecture", 180),
    ("mc2_a05", "时间维：指数衰减 exp(-λ·t)，λ=0.01/小时；频率维：1-exp(-0.1·访问次数)", "architecture", 160),
    ("mc2_a06", "任务维：标签匹配得 1.0 / 不匹配 0.3 / 双方为空中性 0.5", "architecture", 150),
    ("mc2_a07", "共享池淘汰：超 max_shared 按 final 分从低到高淘汰，淘汰条目回到常驻池不丢", "architecture", 140),
    ("mc2_a08", "调度日志逐轮写 data/metrics/schedule.jsonl，含 condition/reward/latency 字段", "architecture", 130),
    ("mc2_a09", "插件注册表：@register(name, kind) + PLUGINS 字典，配置即组合", "architecture", 400),
    ("mc2_a10", "向量索引：faiss IndexFlatIP + L2 归一化 = 余弦，1000 条记忆检索 <1ms", "architecture", 350),
    ("mc2_a11", "中间件挂载顺序：调度中间件在 ReMeMiddleware 之前，on_reply 先调度后注入", "architecture", 300),
    ("mc2_a12", "记忆主键 memory_id 用 uuid，跨会话引用与去重都靠它", "architecture", 280),
    ("mc2_a13", "检索原语四件套：full / bm25 / vector / rrf，彻底禁用 wikilink 与 min_score", "architecture", 240),
    ("mc2_a14", "配置漂移防护：CI 断言 ablation_manifest 字段 ⊆ YAML，防手改 JSON", "architecture", 220),
    ("mc2_r01", "消融判据：+weight→full_ours 的增量必须只含选择器变化，检索基座五组统一", "research", 200),
    ("mc2_r02", "消融数据集 200 行可能不够，统计显著性要求至少 5 组 × 多次重复", "research", 180),
    ("mc2_r03", "评估指标口径：recall 按 kept 集合算，precision=kept 中相关条目占比", "research", 160),
    ("mc2_r04", "topic 图 POC 发现：填充式配额与封顶式配额在 Σquota=K 时数学等价", "research", 150),
    ("mc2_r05", "topic 图 POC 发现：ground truth 过松会制造 10/|relevant| 召回天花板假象", "research", 140),
    ("mc2_r06", "topic 图真实数据结果：G4 多源 recall 0.726 > G2 单源 0.661，源覆盖 1.0 vs 0.583", "research", 120),
    ("mc2_r07", "两线诊断汇合点：G1 写入序在真实数据上打分最高，四维打分器疑被频率维拖累", "research", 100),
    ("mc2_r08", "软偏置 quota_mu 首选 0.2；硬配额无区分度时才考虑", "research", 90),
    ("mc2_r09", "边强度是静态属性：仅建边/用户反馈/summary 重算三时机变化，不随 query 变", "research", 500),
    ("mc2_r10", "源打分公式：score_source = 0.7·strength + 0.3·cos(emb(topic), emb(q))", "research", 400),
    ("mc2_r11", "冷启动两阶段建边：粗识别用 topic 标题+首轮 query，精识别等 summary 生成后覆盖", "research", 380),
    ("mc2_t01", "embedding 落盘缓存：data/embed_cache/DashScopeEmbedding.jsonl，同文本零重复调用", "tech_stack", 200),
    ("mc2_t02", "conda 环境 srtp-memory 是项目唯一运行环境，managed python 缺 numpy 不可用", "tech_stack", 300),
    ("mc2_t03", "openai SDK 已装入 srtp-memory 环境（3.8.0），模型适配器都走 OpenAI 兼容协议", "tech_stack", 250),
    ("mc2_t04", "DashScope 兼容端点：https://dashscope.aliyuncs.com/compatible-mode/v1，模型 qwen-plus", "tech_stack", 200),
    ("mc2_t05", "faiss 不可用时自动线性余弦兜底，管线不中断（D-11 降级链）", "tech_stack", 180),
    ("mc2_t06", "控制台启动：python app_dashboard.py，端口 8787，stdlib 零依赖", "tech_stack", 100),
    ("mc2_t07", "会话画布数据：data/demo/canvas_history.json，追问与副线创建实时落盘", "tech_stack", 50),
    ("mc2_t08", "AgentScope 调用：asyncio.run(agent.reply(UserMsg(...)))，reply 是 async 方法", "tech_stack", 400),
    ("mc2_t09", "权限模式用 BYPASS；DONT_ASK 会把所有询问变成拒绝导致卡死", "tech_stack", 380),
    ("mc2_u01", "用户协作偏好：动手前逐项确认疑点，要求批判性审查而非顺从", "profile", 600),
    ("mc2_u02", "用户偏好：盘点进度诚实不夸大；对凭据泄漏高度敏感", "profile", 550),
    ("mc2_u03", "演示偏好：可视化要占满屏、字体不放大、UI 对齐整洁", "profile", 30),
    ("mc2_d01", "本周计划：完成 5 组消融在线重跑并产出 run_manifest", "daily", 48),
    ("mc2_d02", "近期阻塞：消融结果退化（naive≡reme、+attn≡+weight）待诊断", "daily", 24),
    ("mc2_d03", "下午有组会，需要准备演示动线：控制台 → 画布 → 创建副线", "daily", 6),
    ("mc2_d04", "天气不错，傍晚适合跑步放松", "daily", 30),
    ("mc2_d05", "团队周会定了下阶段目标：论文实验部分先行", "daily", 200),
    ("mc2_d06", "咖啡库存不足，记得补货", "daily", 500),
    ("mc2_d07", "昨天把画布演示给师兄看过，反馈是副线继承机制讲得清楚", "daily", 20),
    ("mc2_d08", "导师提醒：论文创新点表述要聚焦调度算法，图只是结构先验", "daily", 100),
    ("mc2_d09", "记得把 .env 的密钥注释检查一遍，绝不能进仓库", "daily", 400),
]


def load_or_create() -> dict:
    """读历史文件；不存在（或损坏）则生成模拟数据并落盘。"""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_PATH.exists():
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            if data.get("main") and data.get("subs"):
                return data
        except Exception:
            pass
    data = build_demo()
    HISTORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return data


def append_round(session_id: str, user: str, assistant: str) -> dict:
    """把画布新追问追加进历史并落盘；返回新回合。"""
    data = load_or_create()
    sess = None
    if data["main"]["session_id"] == session_id:
        sess = data["main"]
    else:
        for s in data["subs"]:
            if s["session_id"] == session_id:
                sess = s
                break
    if sess is None:
        raise ValueError(f"unknown session: {session_id}")

    n = len(sess["rounds"])
    if sess is data["main"]:
        rid = f"R{n + 1}"
    else:
        prefix = sess["rounds"][0]["round_id"].split("-")[0] if sess["rounds"] else "S"
        rid = f"{prefix}-{n + 1}"
    rnd = {"round_id": rid, "title": (user[:8] + "…") if len(user) > 8 else user,
           "ts": time.time(), "user": user, "assistant": assistant}
    sess["rounds"].append(rnd)
    data["saved_at"] = time.strftime("%F %T")
    HISTORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return rnd


def create_sub(title: str, derived_from: str, first_round: dict) -> tuple[dict, dict]:
    """创建新副线（独立可问答线路）：继承源 = 主线回合，第一回合由调用方（真实调度）产出。

    first_round 需含 user/assistant；inherited（继承的记忆文本列表）挂在会话上供画布展示。
    返回 (新会话, 第一回合)。
    """
    data = load_or_create()
    prefix = f"S{len(data['subs']) + 1}"
    sess = {"session_id": f"sub_{int(time.time())}", "derived_from": derived_from,
            "title": title, "inherited": first_round.get("inherited", []),
            "rounds": [dict(first_round, round_id=f"{prefix}-1")]}
    data["subs"].append(sess)
    data["saved_at"] = time.strftime("%F %T")
    HISTORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return sess, sess["rounds"][0]


def save_layout(positions: dict) -> dict:
    """保存节点拖拽后的位置偏移 {session|round: {dx,dy}}，落盘。"""
    data = load_or_create()
    lay = data.setdefault("layout", {})
    lay.update(positions or {})
    data["saved_at"] = time.strftime("%F %T")
    HISTORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return lay


def seed_memories(mid) -> dict:
    """把画布聊天记忆灌进常驻池（复用已持久化向量，只 embed 一次）。"""
    now = time.time()
    n_new, n_reuse = 0, 0
    for mid_, text, tag, ago_h in CANVAS_MEMORY_SEEDS:
        existing = mid.resident_pool.get(mid_)
        if existing is not None and existing.embedding is not None:
            emb, n_reuse = existing.embedding, n_reuse + 1
        else:
            emb, n_new = mid.embedding.encode(text), n_new + 1
        mid.resident_pool.upsert(MemoryCandidate(
            memory_id=mid_, text=text, path=f"canvas/chat/{mid_}.md",
            timestamp=now - ago_h * 3600, task_tag=tag,
            user_id="u_tu", session_id="chat_vis", embedding=emb))
    return {"seeded": len(CANVAS_MEMORY_SEEDS), "embedded": n_new, "reused": n_reuse}


if __name__ == "__main__":  # 手动重建模拟历史：python demo_canvas.py --reset
    import sys
    if "--reset" in sys.argv and HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    d = load_or_create()
    print("main rounds:", len(d["main"]["rounds"]),
          "| subs:", [(s["title"], len(s["rounds"])) for s in d["subs"]],
          "\nsaved ->", HISTORY_PATH)
