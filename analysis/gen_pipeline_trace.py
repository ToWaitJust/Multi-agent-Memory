"""全流程可视化埋点运行器（Studio 看板增强，§M3）。

用真实组件跑一次完整调度，逐阶段抓取中间值，产出 data/pipeline_trace.json。
覆盖用户要求的全部可见节点：
  ① 记忆入库（多少条、何时进常驻池）
  ② 候选召回
  ③ 四维初步打分（time/semantic/frequency/task）
  ④ 权重生成（LLM 先验 + 可学习 MLP + 混合公式 α·prior+β·learnable）
  ⑤ 最终得分排行榜
  ⑥ 保留 / 丢回
  ⑦ 双池架构（常驻池→共享池→LLM API→回答）
  ⑧ 过程计时 / token 消耗 / 压缩 / token 节省

真实 API 不可达时优雅降级（embedding/llm 走 fallback），并在 trace.api_status 标注模式。
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import srtp_memory  # noqa: F401 - 触发 .env 加载
from srtp_memory.attention import ALL_DIMS
from srtp_memory.config import SchedulingConfig
from srtp_memory.condition import ConditionKey
from srtp_memory.middleware import MemorySchedulingMiddleware
from srtp_memory.plugins import get_plugin

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 中文友好的确定性 token 估算（仅用于"节省"相对对比，非真实分词）
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_cjk = re.sub(r"[\u4e00-\u9fff]", " ", text)
    word_toks = re.findall(r"[A-Za-z0-9]+", non_cjk)
    non_cjk_chars = max(0, len(non_cjk) - sum(len(t) for t in word_toks))
    return cjk + len(word_toks) + non_cjk_chars // 4


# ---------------------------------------------------------------------------
# 演示语料（围绕 SRTP 课题，使可视化自解释、打分有区分度）
# ---------------------------------------------------------------------------
DEMO_CORPUS = [
    ("基于注意力引导的多智能体记忆共享调度方法研究", "main_task", 2.0, 5),
    ("四维注意力头：时间衰减、语义相似、访问频率、任务匹配", "attention", 5.0, 3),
    ("LLM 可学习权重经 MLP 融合先验与学习信号，α=0.6 β=0.4", "weight", 24.0, 2),
    ("ReMe 原生记忆管理：自动记忆与检索原语", "reme", 48.0, 1),
    ("DashScope text-embedding-v4 输出 1024 维语义向量", "embedding", 10.0, 4),
    ("共享池约束：精选 ≤10 条高相关记忆注入副线 workspace", "share", 1.0, 6),
    ("faiss 向量索引支持余弦相似度快速检索", "retrieval", 12.0, 2),
    ("消融实验五组：naive / reme / +attention / +weight / full_ours", "ablation", 72.0, 1),
    ("双池架构：常驻完整池与共享池解耦，主副线逻辑隔离", "architecture", 3.0, 3),
]

DEFAULT_QUERY = "多智能体记忆共享调度如何为副线精选高相关记忆并注入 LLM？"


def build_trace(config_name: str, query: str, corpus) -> dict:
    tmp = tempfile.mkdtemp()
    import yaml
    cfg = SchedulingConfig.model_validate(
        yaml.safe_load((ROOT / "config" / "ablation" / f"{config_name}.yaml").read_text(encoding="utf-8"))
    )
    mw = MemorySchedulingMiddleware(
        config=cfg, user_id="u_tu", session_id="viz_sess",
        resident_pool_path=str(Path(tmp) / "rp.jsonl"),
        shared_pool_path=str(Path(tmp) / "sp.json"),
        log_path=str(Path(tmp) / "m.jsonl"),
        llm_prior_fn=None,  # 让中间件按 config.model_impl 接真实轻量模型
    )

    now = time.time()
    t0 = time.perf_counter()
    for i, (text, tag, age_h, access) in enumerate(corpus):
        cand = mw.add_memory(text=text, memory_id=f"m{i}", task_tag=tag,
                            timestamp=now - age_h * 3600.0)
        cand.access_count = access  # 频率注意力头需要访问次数（add_memory 不收该字段）
    seed_ms = (time.perf_counter() - t0) * 1000.0

    # ---- ③ 权重生成：分别抓 LLM 先验 与 可学习 MLP ----
    t0 = time.perf_counter()
    query_emb = mw.embedding.encode(query)
    embed_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    prior = mw.weights.llm_prior(query)
    prior_ms = (time.perf_counter() - t0) * 1000.0

    cond = ConditionKey.parse(cfg.condition_key)
    learnable = mw.weights._learnable_forward(query_emb, cond)
    hybrid = mw.weights.calculate(query_emb, cond)  # 与协调器内部一致
    # 把先验缓存给协调器（避免重复调 LLM）
    mw.weights.set_prior(prior)

    # ---- 协调器逐阶段调度（复用内部分级计时）----
    res = mw.coordinator.schedule(
        query=query, query_emb=query_emb, task_tag=cfg.task_tag,
        now=now, enabled_dims=None, condition=cond, top_k=cfg.candidate_override,
    )
    lat = res.latency_ms

    # ---- ③ 四维打分明细 ----
    scoring = []
    for item in res.scored:
        mem = item["memory"]
        sc = item["score"]
        scoring.append({
            "memory_id": mem.memory_id,
            "text": mem.text,
            "task_tag": mem.task_tag,
            "access_count": mem.access_count,
            "age_hours": round((now - mem.timestamp) / 3600.0, 1),
            "scores": {
                "time": round(sc["time"], 4),
                "semantic": round(sc["semantic"], 4),
                "frequency": round(sc["frequency"], 4),
                "task": round(sc["task"], 4),
            },
            "final": round(sc["final"], 4),
        })

    # ---- ⑤ 排行榜（按 final 降序）----
    ranking = sorted(scoring, key=lambda x: x["final"], reverse=True)
    for i, r in enumerate(ranking):
        r["rank"] = i + 1
        r["kept"] = r["memory_id"] in {m.memory_id for m in res.kept}

    # ---- ⑥ 保留 / 丢回 ----
    kept_ids = [m.memory_id for m in res.kept]
    discarded_ids = [m.memory_id for m in res.discarded]

    # ---- ⑦ 双池 + LLM 回答 ----
    shared_items = mw.shared_pool.top()
    resident_total = mw.resident_pool.size()
    api_status = mw.api_status()

    # 真实 LLM 回答（注入共享池精选记忆）
    kept_texts = [it["text"] for it in shared_items]
    context_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(kept_texts))
    prompt = (
        "你是多智能体记忆调度系统的研究助手。以下是调度层从共享池精选并注入的高相关记忆：\n"
        f"{context_block}\n\n"
        f"请基于以上记忆，简洁回答用户问题：{query}"
    )
    prompt_tokens = estimate_tokens(prompt)
    t0 = time.perf_counter()
    answer, answer_real = _call_llm_answer(cfg.model_impl, prompt)
    llm_ms = (time.perf_counter() - t0) * 1000.0
    answer_tokens = estimate_tokens(answer)

    # ---- token 节省（对比"假设原记忆量"=召回集）----
    all_mem = mw.resident_pool.all()
    resident_total_tokens = sum(estimate_tokens(m.text) for m in all_mem)
    recall_tokens = sum(estimate_tokens(m.text) for m in res.candidates)
    kept_tokens = sum(estimate_tokens(it["text"]) for it in shared_items)
    original_assumed = recall_tokens  # 无调度时送入 LLM 的"原记忆量"
    saved_tokens = max(0, original_assumed - kept_tokens)
    saved_pct = round(100.0 * saved_tokens / original_assumed, 1) if original_assumed else 0.0

    trace = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config_name,
            "query": query,
            "condition_key": cfg.condition_key,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "conditioning_dims": list(cfg.conditioning_dims),
            "candidate_count": len(res.candidates),
            "kept_count": len(res.kept),
            "api_status": {
                "embedding": api_status["embedding"]["mode"],
                "llm_prior": api_status["llm_prior"]["mode"],
                "llm_answer": "real" if answer_real else "fallback",
            },
        },
        "resident_pool": {
            "total": resident_total,
            "seed_ms": round(seed_ms, 1),
            "memories": [
                {"memory_id": m.memory_id, "text": m.text, "task_tag": m.task_tag,
                 "access_count": m.access_count,
                 "age_hours": round((now - m.timestamp) / 3600.0, 1)}
                for m in all_mem
            ],
        },
        "recall": {
            "count": len(res.candidates),
            "memory_ids": [m.memory_id for m in res.candidates],
            "note": "向量检索器在常驻完整池上召回候选集（top_k≤50）",
        },
        "scoring": scoring,
        "weights": {
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "condition_key": cfg.condition_key,
            "conditioning_dims": list(cfg.conditioning_dims),
            "prior": {k: round(v, 4) for k, v in prior.items()},
            "learnable": {k: round(v, 4) for k, v in learnable.items()},
            "hybrid": {k: round(v, 4) for k, v in hybrid.items()},
            "formula": "w_d = α·prior_d + β·learnable_d （归一化）",
        },
        "ranking": [
            {"rank": r["rank"], "memory_id": r["memory_id"], "text": r["text"],
             "final": r["final"], "kept": r["kept"]}
            for r in ranking
        ],
        "selection": {
            "kept_ids": kept_ids,
            "discarded_ids": discarded_ids,
            "kept_count": len(kept_ids),
            "discarded_count": len(discarded_ids),
            "compression_ratio": round(res.compression_ratio, 3),
        },
        "dual_pool": {
            "resident_total": resident_total,
            "shared_items": [
                {"memory_id": it["memory_id"], "text": it["text"], "score": round(it["score"], 4)}
                for it in shared_items
            ],
            "shared_max": cfg.max_shared,
            "llm_call": {
                "prompt_tokens": prompt_tokens,
                "context_memory_ids": [it["memory_id"] for it in shared_items],
                "answer": answer,
                "answer_tokens": answer_tokens,
                "real": answer_real,
                "llm_ms": round(llm_ms, 1),
            },
        },
        "metrics": {
            "timing": {
                "seed_ms": round(seed_ms, 1),
                "embedding_ms": round(embed_ms, 1),
                "prior_ms": round(prior_ms, 1),
                "recall_ms": round(lat.get("recall", 0.0), 1),
                "weights_ms": round(lat.get("weights", 0.0), 1),
                "scoring_ms": round(lat.get("scoring", 0.0), 1),
                "select_ms": round(lat.get("select", 0.0), 1),
                "llm_ms": round(llm_ms, 1),
                "total_ms": round(seed_ms + embed_ms + prior_ms
                                  + lat.get("recall", 0) + lat.get("weights", 0)
                                  + lat.get("scoring", 0) + lat.get("select", 0) + llm_ms, 1),
            },
            "tokens": {
                "resident_total_tokens": resident_total_tokens,
                "recall_tokens": recall_tokens,
                "kept_tokens": kept_tokens,
                "original_assumed_tokens": original_assumed,
                "saved_tokens": saved_tokens,
                "saved_pct": saved_pct,
            },
            "compression_ratio": round(res.compression_ratio, 3),
        },
    }
    return trace


def _call_llm_answer(model_impl: str, prompt: str):
    """真实 LLM 回答；失败降级为演示合成回答。返回 (answer, real)。"""
    try:
        client = get_plugin(model_impl)
        if getattr(client, "api_key", None):
            raw = client.complete(prompt, max_tokens=320, temperature=0.3, thinking=True)
            if raw and raw.strip():
                return raw.strip(), True
    except Exception:  # noqa: BLE001 - 任何异常降级，可视化不中断
        pass
    # 合成（演示）回答
    synth = (
        "调度系统先由向量检索器从常驻完整池召回候选记忆，再用四维注意力"
        "（时间/语义/频率/任务）逐条打分，并以「α·LLM先验 + β·可学习权重」"
        "混合得到动态权重，最终按分数排序精选 ≤10 条高相关记忆写入共享池，"
        "注入副线 LLM 上下文。相比把全部记忆直接塞入上下文，本方法显著压缩了"
        "token 占用、突出高相关记忆，从而提升回答质量与效率。"
    )
    return synth, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full_ours")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--out", default=str(ROOT / "data" / "pipeline_trace.json"))
    args = ap.parse_args()

    trace = build_trace(args.config, args.query, DEMO_CORPUS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    m = trace["meta"]
    print(f"✓ trace 写入 {args.out}")
    print(f"  配置={m['config']}  查询='{m['query']}'")
    print(f"  API: embedding={m['api_status']['embedding']} "
          f"prior={m['api_status']['llm_prior']} answer={m['api_status']['llm_answer']}")
    print(f"  常驻池={trace['resident_pool']['total']}条  召回={trace['recall']['count']}条  "
          f"保留={trace['selection']['kept_count']}条")
    print(f"  token 原假设={trace['metrics']['tokens']['original_assumed_tokens']}  "
          f"保留={trace['metrics']['tokens']['kept_tokens']}  "
          f"节省={trace['metrics']['tokens']['saved_pct']}%")
    print(f"  总耗时={trace['metrics']['timing']['total_ms']}ms")


if __name__ == "__main__":
    main()
