"""POC Step 1：合成 topic 图 + 记忆 + ground-truth（04_poc_plan.md 的 Step 1）。

合成图只能验证调度逻辑，不能当论文证据（结论说服力打折）。
后续换成方案 A（AgentScope 跑多模块项目自建）时，只需保持本模块输出的数据形状不变。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

EDGE_DEFAULT_STRENGTH = {
    "derives_from": 0.9,
    "depends_on": 0.8,
    "references": 0.5,
    "similar_to": 0.3,
}


@dataclass
class PocNode:
    node_id: str
    topic: str
    keywords: list[str]
    task_tag: str
    summary: str = ""


@dataclass
class PocEdge:
    src: str
    dst: str
    type: str
    strength: float


@dataclass
class PocMemory:
    memory_id: str
    node_id: str
    text: str
    timestamp: float
    access_count: int
    task_tag: str


@dataclass
class PocQuery:
    query_id: str
    cur_node: str
    text: str
    task_tag: str
    relevant_nodes: list[str] = field(default_factory=list)


# 3 个簇 × 3~4 个节点，主题间有真实依赖（子模块产物 → 下游输入）
TOPICS: list[tuple[str, list[str], str]] = [
    ("双池架构与常驻池设计", ["双池", "常驻池", "共享池", "隔离", "写入"], "arch"),
    ("四维注意力打分器", ["注意力", "时间衰减", "语义余弦", "频率", "任务标签"], "arch"),
    ("可学习权重与条件化 MLP", ["可学习权重", "条件化", "蒸馏", "正则", "个性化"], "arch"),
    ("消融实验设计与指标", ["消融", "指标", "召回", "精确率", "对照"], "eval"),
    ("消融数据集生成", ["消融", "数据集", "合成", "标注", "样本"], "eval"),
    ("评测脚本与 run_manifest", ["评测", "脚本", "manifest", "落盘", "复现"], "eval"),
    ("AgentScope 运行时接入", ["AgentScope", "运行时", "工具", "权限", "沙箱"], "infra"),
    ("embedding 服务与缓存", ["embedding", "向量", "缓存", "落盘", "维度"], "infra"),
    ("检索器插件化", ["检索器", "插件", "注册", "基座", "召回"], "infra"),
]

# 有向边：src → dst 表示"dst 依赖/派生自 src"。手工构造，保证 DAG。
EDGES: list[tuple[str, str, str]] = [
    ("T0", "T1", "derives_from"),
    ("T0", "T6", "references"),
    ("T1", "T2", "depends_on"),
    ("T2", "T3", "depends_on"),
    ("T3", "T4", "derives_from"),
    ("T3", "T5", "depends_on"),
    ("T4", "T5", "references"),
    ("T0", "T8", "references"),
    ("T6", "T8", "depends_on"),
    ("T6", "T7", "references"),
    ("T7", "T8", "depends_on"),
    ("T1", "T3", "references"),
    ("T2", "T5", "references"),
]

# 查询：cur_node + 问句 + 相关源节点（ground truth，必须 ⊆ cur 的入边邻居）
QUERIES: list[tuple[str, str, list[str]]] = [
    ("Q1", "T5", "消融指标对照怎么设计，数据集怎么合成标注", ["T3", "T4"]),
    ("Q2", "T3", "四维注意力打分和可学习权重蒸馏怎么做", ["T1", "T2"]),
    ("Q3", "T8", "双池常驻池设计和 AgentScope 工具权限", ["T0", "T6"]),
    ("Q4", "T8", "AgentScope 运行时接入 embedding 向量缓存", ["T6", "T7"]),
    ("Q5", "T2", "注意力语义余弦和时间衰减怎么实现", ["T1"]),
    ("Q6", "T7", "AgentScope 工具权限沙箱怎么配", ["T6"]),
]

NOISE = ["登录", "样式", "端口", "日志", "部署", "缓存清理", "界面", "文档", "评审", "周报"]


def build_graph(seed: int = 42):
    rng = random.Random(seed)
    nodes = {
        tid: PocNode(node_id=tid, topic=title, keywords=kws, task_tag=tag,
                     summary=f"{title}：围绕 {'、'.join(kws)} 的讨论与产物。")
        for tid, (title, kws, tag) in zip([f"T{i}" for i in range(len(TOPICS))], TOPICS)
    }
    edges = [PocEdge(s, d, t, EDGE_DEFAULT_STRENGTH[t]) for s, d, t in EDGES]

    # 每节点 10 条记忆：3 个本节点关键词 + 0~2 个噪声词，模拟"主题相对纯净但非 100%"
    memories: list[PocMemory] = []
    mid = 0
    base_ts = 1_700_000_000.0
    for nid, node in nodes.items():
        for k in range(10):
            kws = rng.sample(node.keywords, k=3)
            noise = rng.sample(NOISE, k=rng.randint(0, 2))
            text = "、".join(kws + noise) + f"（{node.topic} 第{k + 1}条记录）"
            memories.append(PocMemory(
                memory_id=f"m{mid:04d}", node_id=nid, text=text,
                timestamp=base_ts + mid * 3600.0,
                access_count=rng.randint(0, 5), task_tag=node.task_tag,
            ))
            mid += 1

    queries = [PocQuery(qid, cur, text, nodes[cur].task_tag, rel)
               for qid, cur, text, rel in QUERIES]

    return nodes, edges, memories, queries


def is_dag(nodes: dict[str, PocNode], edges: list[PocEdge]) -> bool:
    """Kahn 拓扑排序验证无环（P0 约束的机器可查版）。"""
    indeg = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for e in edges:
        adj[e.src].append(e.dst)
        indeg[e.dst] += 1
    queue = [n for n, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return seen == len(nodes)
