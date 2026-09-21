"""图结构消融数据集生成器（Q4 阻塞项落地）。

与 `gen_ablation_dataset.py`（扁平语料）的区别：本生成器产出**带真实 topic 结构**的 episode ——
每个 episode 是一张**多模块项目图**：节点 = 模块会话，边 = typed 依赖关系，
记忆按 `owner_node` 打标记，查询带**人工标注的相关记忆与应参考源节点**。

设计约束（来自 POC 两轮教训 + 本轮实测坑，必须遵守）
----------------------------------------------------
1. **四类 typed 边齐全**：上一轮真实数据 7 条边全是 `depends_on` → 强度零方差 → G3≡G4，
   "边强度是否有效"根本无法检验。本生成器强制混合边型，并**故意留噪**：
   真实依赖中有 1 条落在弱边（`references`），干扰源中有 1 条落在强边（`depends_on`）。
2. **不构造"最强边恰好指向最相关节点"**（POC 第一轮 G2 作弊的来源）：
   相关源由**任务语义**人工标注，与边强度**独立**。
3. **ground truth 必须语义化**（本轮实测坑）：第一版按 memory_id 前缀取"相关记忆"，
   与查询语义无关 → 语义检索再好也对不上 → recall 退化为随机。现改为**逐查询手工标注**。
4. **任务标签是任务级、不是节点级**（本轮实测坑）：若 `task_tag` 写成节点 id，
   跨源记忆会被任务头判为"离题"（task=0.3），recall 被系统性清零。
5. **报告天花板**：`ceiling = min(1, K/|relevant|)`。本数据集 `|rel| <= 3 < K=10`，
   天花板恒为 1.0 —— 即"没有截断"，这本身就是要报的结论。
6. 完全确定性（固定 seed + 固定时间基准），同命令任意次数结果一致（NFR-6）。

用法
----
    python -m analysis.gen_graph_dataset
输出
----
    data/graph_dataset.jsonl   (data/ 在 .gitignore 内，按需重生成)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 固定时间基准（避免 wall-clock 依赖 → 结果完全可复现，NFR-6）
TIME_BASE = 1_700_000_000.0
HOUR = 3600.0
#: 每个节点占 6 小时（模拟"项目按模块顺序推进"），节点内每条记忆错开半小时
NODE_SPAN_H = 6.0
#: 任务标签是**任务级**（同一项目任务），**不是**节点级 —— 见模块 docstring 约束 4
TASK_TAG = "research"

# -- 模块表：node_id / 主题 / 4 条记忆 --------------------------------------------------
MODULES: list[dict] = [
    {"id": "n_config", "topic": "配置与参数管理",
     "mems": ["调度阈值 threshold 默认 0.6，共享池上限 max_shared 固定 10",
              "候选集 candidate_override 统一为 50，作为消融控制变量",
              "配置文件 srtp.yaml 是唯一手写真源，插件按 _impl 字段装配",
              "embedding 维度锁定 1024，与注意力维度保持一致"]},
    {"id": "n_embed", "topic": "文本向量化与落盘缓存",
     "mems": ["向量化后端用 DashScope text-embedding-v4，输出 1024 维",
              "embedding 首次计算写 data/embed_cache，重启后直接复用",
              "无 API key 时降级为确定性占位向量，保证管线不中断",
              "占位向量不落盘，避免日后补 key 取到陈旧占位"]},
    {"id": "n_retrieval", "topic": "检索基座与召回原语",
     "mems": ["检索器共五种：full、bm25、vector、rrf、reme，全部可插拔",
              "vector 检索器优先走 faiss 内积索引，缺失时退化为线性余弦",
              "BM25 采用字符级中文分词，未引入 jieba 以保证确定性",
              "候选集统一返回 50 条，保证五组消融可比"]},
    {"id": "n_attention", "topic": "四维注意力打分器",
     "mems": ["四个打分头是时间、语义、频率、任务，默认权重 0.25/0.35/0.15/0.25",
              "时间头用指数衰减 exp(-0.01t)，t 以小时计",
              "频率头用 1-exp(-0.1n) 的饱和增长，避免高频记忆垄断",
              "语义头内部融合系数是余弦 0.5、向量分 0.3、BM25 分 0.2"]},
    {"id": "n_weights", "topic": "混合权重与条件化",
     "mems": ["混合权重等于 0.6 乘 LLM 先验加 0.4 乘可学习权重",
              "可学习侧是条件化 MLP，输入为 query 向量拼接 32 维条件向量",
              "只有语义维和任务维随 user 与 scenario 漂移，时间与频率保持全局",
              "alpha 与 beta 固定不学习，保证消融归因干净"]},
    {"id": "n_feedback", "topic": "反馈闭环与在线学习",
     "mems": ["反馈奖励取显式点赞加一与点踩减一，bandit 式一步更新",
              "在线更新只改 condition 对应的切片，不动 alpha 与 LLM 先验",
              "更新后向 LLM 先验回归，防止权重学偏",
              "训练集来自实际交互数据与模拟生成数据两条来源"]},
    {"id": "n_pool", "topic": "双池与共享池拓扑",
     "mems": ["常驻完整池负责全量存储，共享池只承载被选中的记忆",
              "共享池按分数有序插入，超过 10 条时淘汰最低分",
              "共享池不做二次检索，保证排序与选择是唯一变量",
              "记忆元数据直接进 resident_pool.jsonl，不设独立侧表"]},
    {"id": "n_graph", "topic": "图结构记忆调度",
     "mems": ["节点是一次主题会话而不是一条记忆，这是与知识图谱的根本区别",
              "边强度是静态量，只在建边、用户反馈、summary 重算三个时机变化",
              "Stage A 先选源再选记忆，形成多对多筛选",
              "边强度与查询相关度按 0.7 与 0.3 融合成一个源分"]},
    {"id": "n_ablation", "topic": "消融实验设计",
     "mems": ["五组消融是 G0 邻域全量、G1 图加均匀、G2 图加单源、G3 多源均匀、G4 完整方案",
              "主副线架构等价于 G2 单源，作为图方案的下界对照",
              "五组检索基座统一为 vector，修正此前单独换检索器的归因污染",
              "必须报告相关集规模与召回天花板，否则指标不可解释"]},
    {"id": "n_dataset", "topic": "数据集与场景设计",
     "mems": ["数据来自多模块项目会话，节点天然形成有向无环图",
              "要求至少十个节点十五条边且含一个分叉结构",
              "typed 边必须四类齐全，否则边强度零方差无法检验",
              "ground truth 需人工标注应参考的源节点集合"]},
    {"id": "n_demo", "topic": "演示壳与会话画布",
     "mems": ["会话画布一回合一节点，点击节点跳转到节点控制台",
              "主线可创建副线，调度按副线第一问执行并展示继承记忆",
              "画布内嵌 JavaScript 必须做语法检查，一次多余括号就整页空白",
              "双池视图用 PCA 投影到二维，虚线表示记忆继承"]},
    {"id": "n_deploy", "topic": "服务器部署与运维",
     "mems": ["演示服务部署在腾讯云 150.158.26.158 的 8787 端口",
              "systemd 单元名为 srtp-demo，绑定 0.0.0.0 并随开机启动",
              "公网端口不通优先检查云厂商安全组",
              "ssh 前台安装大体积依赖会超时截断，必须后台安装再轮询"]},
    {"id": "n_metrics", "topic": "指标与可复现性",
     "mems": ["核心指标包括召回准确率、压缩比、平均响应时间与 token 降低率",
              "多源场景需要新增源召回命中率与多源覆盖率",
              "每次实验登记 run_manifest 记录配置与指标便于审计",
              "全链路延迟要求小于一秒，打分单条小于十毫秒"]},
    {"id": "n_paper", "topic": "论文叙事与投稿",
     "mems": ["创新点是多对多筛选而不是主副线结构本身",
              "论文必须写明与知识图谱的边界区别",
              "需要报告图结构贡献与算法贡献的归因结果",
              "目标投稿窗口集中在 2026 年十月至 2027 年三月"]},
]

# -- 边表：(src, dst, type)。混合 typed；含多个分叉；故意留噪 -----------------------------
EDGES: list[tuple[str, str, str]] = [
    # 真实依赖（强边）
    ("n_config", "n_embed", "depends_on"),
    ("n_embed", "n_retrieval", "depends_on"),
    ("n_retrieval", "n_attention", "depends_on"),
    ("n_attention", "n_weights", "depends_on"),
    ("n_weights", "n_feedback", "derives_from"),
    ("n_embed", "n_pool", "depends_on"),
    ("n_retrieval", "n_graph", "depends_on"),
    ("n_pool", "n_graph", "depends_on"),
    ("n_attention", "n_ablation", "depends_on"),
    ("n_weights", "n_ablation", "depends_on"),
    ("n_graph", "n_demo", "depends_on"),
    ("n_pool", "n_demo", "depends_on"),
    ("n_demo", "n_deploy", "derives_from"),
    ("n_ablation", "n_dataset", "depends_on"),
    ("n_graph", "n_paper", "depends_on"),
    ("n_dataset", "n_paper", "depends_on"),
    ("n_metrics", "n_paper", "references"),
    # 干扰源（弱边，其记忆大多与下游查询无关）—— 故意留噪
    ("n_config", "n_deploy", "similar_to"),
    ("n_feedback", "n_metrics", "references"),
    ("n_deploy", "n_paper", "similar_to"),
    ("n_feedback", "n_demo", "similar_to"),
]

# -- 查询表（**人工标注 ground truth**，与边强度独立）-----------------------------------
# 每项：当前节点 / 查询 / 相关记忆 id（来自其上游源节点的池）。
# 多源查询（相关记忆跨 >=2 个上游源）是检验"多对多优于一对一"的关键样本。
QUERIES: list[tuple[str, str, list[str]]] = [
    # —— 单源（G2 可以完整覆盖）——
    ("n_embed", "调度阈值和共享池上限的默认值是多少", ["n_config_m1"]),
    ("n_embed", "配置是怎么驱动插件装配的、向量维度锁在多少", ["n_config_m3", "n_config_m4"]),
    ("n_retrieval", "文本向量化用哪个模型、输出多少维", ["n_embed_m1"]),
    ("n_retrieval", "embedding 缓存怎么复用、没 key 时怎么办", ["n_embed_m2", "n_embed_m3"]),
    ("n_attention", "检索基座一共有哪几种召回原语", ["n_retrieval_m1"]),
    ("n_attention", "向量检索底层用什么索引、候选集多大", ["n_retrieval_m2", "n_retrieval_m4"]),
    ("n_weights", "四维注意力都有哪四个打分头、默认权重是多少", ["n_attention_m1"]),
    ("n_weights", "时间衰减和频率饱和的公式分别是什么", ["n_attention_m2", "n_attention_m3"]),
    ("n_feedback", "混合权重的公式是什么", ["n_weights_m1"]),
    ("n_feedback", "条件化作用在哪几维、alpha 会不会被学习", ["n_weights_m3", "n_weights_m4"]),
    ("n_pool", "向量化后端和维度是多少", ["n_embed_m1"]),
    ("n_pool", "没有 key 的时候向量化怎么降级、占位向量存不存盘", ["n_embed_m3", "n_embed_m4"]),
    ("n_deploy", "会话画布的节点和连线分别表示什么", ["n_demo_m1"]),
    ("n_deploy", "画布内嵌脚本踩过什么坑", ["n_demo_m3"]),
    ("n_dataset", "五组消融分别是什么、主副线对应哪一组", ["n_ablation_m1", "n_ablation_m2"]),
    ("n_dataset", "为什么必须报告相关集规模与召回天花板", ["n_ablation_m4"]),
    ("n_metrics", "用户反馈怎么更新权重、奖励怎么取", ["n_feedback_m1"]),
    ("n_metrics", "在线更新会不会把权重学偏、怎么防", ["n_feedback_m3"]),
    # —— 多源（相关记忆跨 >=2 个上游源 → G2 单源必然漏源，核心对照样本）——
    ("n_graph", "检索候选集多大、共享池容量上限是多少", ["n_retrieval_m4", "n_pool_m2"]),
    ("n_graph", "向量检索用什么索引、共享池为什么不做二次检索", ["n_retrieval_m2", "n_pool_m3"]),
    ("n_ablation", "四维打分和混合权重是怎么串起来的", ["n_attention_m1", "n_weights_m1"]),
    ("n_ablation", "时间衰减与频率饱和的公式是什么、alpha 能不能学",
     ["n_attention_m2", "n_weights_m4"]),
    ("n_demo", "图里节点指什么、共享池最多放几条", ["n_graph_m1", "n_pool_m2"]),
    ("n_demo", "边强度是不是静态的、共享池会不会二次检索", ["n_graph_m2", "n_pool_m3"]),
    ("n_paper", "创新点落在哪里、核心指标有哪些", ["n_graph_m1", "n_metrics_m1"]),
    ("n_paper", "数据集对图的节点和边有什么硬性要求", ["n_dataset_m1", "n_dataset_m2"]),
]


def _check_dag(nodes: list[str], edges: list[tuple[str, str, str]]) -> None:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for s, d, _t in edges:
        adj[s].append(d)
        indeg[d] += 1
    q = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while q:
        cur = q.pop()
        seen += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if seen != len(nodes):
        raise ValueError("边表不是 DAG，请检查 EDGES")


def build_graph_payload() -> tuple[list[dict], list[dict]]:
    """构造全图的 nodes / corpus（所有 episode 共用同一张图）。"""
    corpus: list[dict] = []
    for ni, m in enumerate(MODULES):
        for i, text in enumerate(m["mems"], 1):
            corpus.append({
                "memory_id": f"{m['id']}_m{i}",
                "text": text,
                "task_tag": TASK_TAG,
                "owner_node": m["id"],
                # 时间梯度：模块列表靠后 = 更晚产生（更"新"）
                "timestamp": TIME_BASE + ni * NODE_SPAN_H * HOUR + (i - 1) * 0.5 * HOUR,
            })
    nodes = [{"node_id": m["id"], "topic": m["topic"],
              "summary": f"{m['topic']}：" + "；".join(m["mems"])} for m in MODULES]
    return nodes, corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/graph_dataset.jsonl")
    ap.add_argument("--shuffle-relevant", action="store_true",
                    help="打乱相关记忆顺序（默认保持标注顺序）")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    node_ids = [m["id"] for m in MODULES]
    _check_dag(node_ids, EDGES)
    nodes, corpus = build_graph_payload()
    by_id = {c["memory_id"]: c for c in corpus}
    rng = random.Random(args.seed)

    episodes: list[dict] = []
    for idx, (cur_node, query, rel_mems) in enumerate(QUERIES):
        missing = [m for m in rel_mems if m not in by_id]
        if missing:
            raise ValueError(f"{cur_node} 的标注引用不存在的记忆: {missing}")
        # 应参考源节点 = 相关记忆的 owner_node（自动派生，保证与标注一致无歧义）
        rel_sources = sorted({by_id[m]["owner_node"] for m in rel_mems})
        # 上游源必须覆盖全部标注源（否则题目本身矛盾）
        preds = {s for s, d, _t in EDGES if d == cur_node}
        bad = [s for s in rel_sources if s not in preds]
        if bad:
            raise ValueError(f"{cur_node} 的标注源不在其上游: {bad}（上游={sorted(preds)}）")
        rel = list(rel_mems)
        if args.shuffle_relevant:
            rng.shuffle(rel)
        episodes.append({
            "episode_id": f"gep_{idx:04d}",
            "task_tag": TASK_TAG,
            "user_id": "u_tu",
            "scenario": "multi_module_project",
            "business": "srtp",
            "query": query,
            "cur_node": cur_node,
            "now": TIME_BASE + len(MODULES) * NODE_SPAN_H * HOUR + HOUR,
            "nodes": nodes,
            "edges": [{"src": s, "dst": d, "type": t} for s, d, t in EDGES],
            "corpus": corpus,
            "relevant": sorted(rel),
            "relevant_sources": rel_sources,
        })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # 结构自检报告（论文里要报的图统计）
    types: dict[str, int] = {}
    for _s, _d, t in EDGES:
        types[t] = types.get(t, 0) + 1
    forked = [n for n in node_ids if sum(1 for s, _d, _t in EDGES if s == n) >= 2]
    multi = [e for e in episodes if len(e["relevant_sources"]) >= 2]
    print(f"OK {out}  ({len(episodes)} episodes)")
    print(f"   nodes={len(node_ids)} edges={len(EDGES)} memories={len(corpus)} types={types}")
    print(f"   forked(outdeg>=2): {forked}")
    print(f"   |relevant| set: {sorted({len(e['relevant']) for e in episodes})} (K=10 -> ceiling=1.0)")
    print(f"   multi-source queries: {len(multi)}/{len(episodes)}"
          f" (sources={sorted({len(e['relevant_sources']) for e in multi})})")


if __name__ == "__main__":
    main()
