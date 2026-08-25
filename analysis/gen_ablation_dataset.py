"""确定性合成消融数据集生成器（V2.1：供 5 组消融跑批）。

设计要点
--------
- 完全确定性：固定 seed，同一条命令跑一万次结果一致（可复现，NFR-6）。
- 自包含 episode：每条 episode 自带一个小型语料(corpus) + 一条查询(query)
  + 相关记忆标签(relevant)。后续运行器对每条 episode：播种 corpus -> 调度 query
  -> 拿 kept 记忆 -> 对照 relevant 算 recall@k / precision@k。
- 语料围绕本 SRTP 课题（多智能体记忆共享调度）的真实中文内容，覆盖 6 个任务域：
  tech_stack / progress / decision / bug / meeting / paper。
- 每条 episode 的"相关记忆"来自查询所属域，干扰项来自其它域 —— 形成可区分的
  排序问题，使 5 组消融（尤其检索器/注意力/权重差异）能体现相对增益。

用法
----
    python -m analysis.gen_ablation_dataset            # 默认 200 条, seed=20260825
    python -m analysis.gen_ablation_dataset --n 500 --seed 7
输出
----
    data/ablation_dataset.jsonl  (data/ 在 .gitignore 内，按需重生成)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# ── 内容库：6 个任务域 × 6 个主题，每主题 = {记忆文本, 对应查询} ──────────────
BANK: dict[str, list[dict[str, str]]] = {
    "tech_stack": [
        {"memory": "注意力头维度设为 1024，与 embedding 向量维度一致", "query": "注意力头和 embedding 的维度是多少"},
        {"memory": "文本向量化后端用 DashScope text-embedding-v4 模型", "query": "我们的文本向量化用的哪个模型"},
        {"memory": "混合权重由一个 4 输入 4 输出的轻量 MLP 算出", "query": "四维混合权重是怎么算出来的"},
        {"memory": "向量检索底层用 faiss 做内积索引近似余弦", "query": "向量检索底层用的什么索引"},
        {"memory": "LLM 先验接入 DeepSeek 作为实时小模型", "query": "LLM 先验用的是哪个大模型"},
        {"memory": "共享记忆池上限固定为 10 条记忆", "query": "共享记忆池最多能放几条"},
    ],
    "progress": [
        {"memory": "本周完成了插件注册表与 17 个插件实现", "query": "这周插件层做到哪了"},
        {"memory": "向量记忆池已搭好，1000 条查询耗时 0.44 毫秒", "query": "向量池性能达标了吗"},
        {"memory": "工程文档已定稿到 V2.1 并推上 GitHub", "query": "文档现在到第几版了"},
        {"memory": "消融实验还差造数据集和跑通 5 组", "query": "消融还差什么没做"},
        {"memory": "AgentScope 完整运行时还没真正实例化多智能体", "query": "多智能体运行时接了吗"},
        {"memory": "离线集成测试 62 个用例全部通过", "query": "单测通过率怎么样"},
    ],
    "decision": [
        {"memory": "决定 full_ours 用语义向量检索器而不是 BM25", "query": "完整方案最后选了哪种检索器"},
        {"memory": "消融向量统一用真实 DashScope 加落盘缓存保证可复现", "query": "embedding 怎么保证可复现"},
        {"memory": "condition 个性化只在语义和任务两个注意力头做", "query": "个性化权重作用在哪些维度"},
        {"memory": "alpha 固定 0.6、beta 0.4，不设为可学习", "query": "LLM 先验和可学习权重怎么融合"},
        {"memory": "jieba 不装，BM25 保持字符级中文分词", "query": "中文分词为什么没用 jieba"},
        {"memory": "多智能体运行时按完整方案实例化 Main 加 Sub 乘 N", "query": "智能体运行时打算做到多深"},
    ],
    "bug": [
        {"memory": "修过 faiss 维度硬编码 1024 导致自适应维度失败", "query": "faiss 遇到过什么维度问题"},
        {"memory": "修过 condition 哈希字节当浮点导致数值溢出", "query": "condition 嵌入最早踩过什么坑"},
        {"memory": "修过 middleware 硬编码导致 baseline 组 selector 选不出记忆", "query": "之前 baseline 组为什么选不出记忆"},
        {"memory": "修过 retriever 前缀不兼容 retriever. 命名风格", "query": "检索器注册名出过什么 bug"},
        {"memory": "修过向量索引删除后 id 映射不同步", "query": "向量索引删除有什么坑"},
        {"memory": "修过 embedding 缓存只存内存导致重启丢失", "query": "embedding 缓存为什么加落盘"},
    ],
    "meeting": [
        {"memory": "周一例会说要先把已验证改动提交防丢失", "query": "例会对提交有什么要求"},
        {"memory": "评审建议消融每组共用同一套向量保证可比", "query": "评审对消融可比性提了啥"},
        {"memory": "答辩要准备 live demo 展示共享池协调", "query": "答辩演示计划展示什么"},
        {"memory": "组会决定真实环境接入分批做降低风险", "query": "组会怎么安排真实环境"},
        {"memory": "周报要写四维注意力加可学习权重的创新点", "query": "周报创新点怎么写"},
        {"memory": "导师要求文档不确定点逐项确认再动手", "query": "导师对工程文档有什么要求"},
    ],
    "paper": [
        {"memory": "参考了 ReMe 的长短期记忆与读写原语", "query": "记忆机制参考了哪篇工作"},
        {"memory": "参考了 AgentScope 的多智能体编排框架", "query": "多智能体框架参考了什么"},
        {"memory": "对标了 MEM0 的向量检索与去重方案", "query": "检索方案对标了哪个系统"},
        {"memory": "借鉴了 DeepSeek harness 的插件式架构", "query": "插件架构借鉴了谁"},
        {"memory": "相关文献提到注意力可解释性对记忆有用", "query": "注意力为什么对记忆重要"},
        {"memory": "文献综述覆盖记忆共享与调度两类方法", "query": "文献综述覆盖哪些方向"},
    ],
}

DOMAINS = list(BANK.keys())
SCENARIOS = ["research", "dev", "demo"]
DISTRACTOR_PER_EPISODE = 8


def gen_episode(rng: random.Random, idx: int) -> dict:
    domain = DOMAINS[idx % len(DOMAINS)]
    topic = rng.choice(BANK[domain])
    relevant_text = topic["memory"]
    query = topic["query"]

    # 干扰项：从其它域的主题里抽，保留各自真实 task_tag（让 task 头有区分信号）
    other_topics: list[tuple[str, str]] = []
    for d in DOMAINS:
        if d == domain:
            continue
        other_topics.extend((d, m["memory"]) for m in BANK[d])
    distractors = rng.sample(other_topics, min(DISTRACTOR_PER_EPISODE, len(other_topics)))

    corpus_items = [(domain, relevant_text)] + distractors
    rng.shuffle(corpus_items)

    corpus = []
    relevant_ids: list[str] = []
    for j, (d, txt) in enumerate(corpus_items):
        mid = f"m{j}"
        corpus.append({"memory_id": mid, "text": txt, "task_tag": d})
        if txt == relevant_text:
            relevant_ids.append(mid)

    return {
        "episode_id": f"ep_{idx:04d}",
        "task_tag": domain,
        "user_id": "u_tu",
        "scenario": rng.choice(SCENARIOS),
        "business": "srtp",
        "query": query,
        "corpus": corpus,
        "relevant": relevant_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="episode 数量")
    ap.add_argument("--seed", type=int, default=20260825, help="随机种子（确定性）")
    ap.add_argument("--out", default="data/ablation_dataset.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(__file__).resolve().parents[1]
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    episodes = [gen_episode(rng, i) for i in range(args.n)]
    with open(out, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    # 简单统计
    tag_counts: dict[str, int] = {}
    for ep in episodes:
        tag_counts[ep["task_tag"]] = tag_counts.get(ep["task_tag"], 0) + 1
    print(f"生成 {len(episodes)} 条 episode -> {out}")
    print(f"任务域分布: {tag_counts}")
    print(f"文件 sha256[:12] = {digest}  (确定性：同 seed 重跑应一致)")


if __name__ == "__main__":
    main()
