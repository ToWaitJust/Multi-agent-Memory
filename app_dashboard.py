# -*- coding: utf-8 -*-
"""可视化冒烟测试控制台（真实 API 链路）。

用法:  python app_dashboard.py
打开:  http://127.0.0.1:8787/

后端: stdlib http.server（零新依赖）。POST /api/smoke 用真实 DashScope embedding +
DeepSeek LLM 先验跑一轮调度冒烟，返回全流程 trace；前端渲染流水线可视化。
真实密钥从 .env 读取；未配置时自动降级（D-11），界面标注 mode=fallback。
"""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import numpy as np

from srtp_memory.attention import MemoryCandidate
from srtp_memory.config import SchedulingConfig
from srtp_memory.middleware import MemorySchedulingMiddleware
from srtp_memory.plugins import register_all
from srtp_memory.weights import DEFAULT_WEIGHTS
from srtp_memory.agentscope_runtime import SchedulingRuntime, AGENTSCOPE_AVAILABLE

register_all()

PORT = 8787
SMOKE_DIR = ROOT / "data" / "smoke"

# ---- 冒烟数据集：50 条贴近项目/用户背景的中文记忆（含干扰项） ----
# 覆盖 profile / tech_stack / architecture / research / daily 五类 task_tag，
# 时龄从几分钟到几天不等，用于考察四维注意力的区分度。
SEED_MEMORIES = [
    # ── profile：用户/团队背景（10） ──
    ("m_profile", "用户涂伟健：信息管理专业，正在做多智能体记忆调度课题", "profile", 3600),
    ("m_ps_utf8", "用户偏好 PowerShell 5，命令前设置 UTF-8 编码防乱码", "profile", 86400),
    ("m_navicat", "用户习惯用 Navicat 图形界面观察数据库表，而非直接写 SQL", "profile", 172800),
    ("m_learn_style", "用户学习方式：动手实践、边做边学、注重项目实际操作", "profile", 259200),
    ("m_outteam", "用户管理外包团队开发标准和 Git 协作流程", "profile", 604800),
    ("m_md_docs", "用户要求学习进度与操作步骤文档化到 Markdown 文件", "profile", 259200),
    ("m_research_dir", "用户研究方向含多智能体系统记忆调度算法与 AI 集成", "profile", 43200),
    ("m_yudao", "用户同时进行 Yudao 框架开发与测试", "profile", 86400),
    ("m_step_guide", "用户偏好清晰、分步的操作指南，而非泛泛而谈", "profile", 259200),
    ("m_wf_integration", "用户当前关注 workflow 模块集成与 AI 模块开发", "profile", 21600),
    # ── tech_stack：技术选型/踩坑（12） ──
    ("m_ts_main", "技术栈：AgentScope 2.0 + ReMe + DashScope text-embedding-v4", "tech_stack", 7200),
    ("m_old_bert", "项目早期用 BERT-base 做语义向量（已弃用）", "tech_stack", 172800),
    ("m_deepseek", "DeepSeek 调用需 thinking_enable=False，否则推理 token 吃满 max_tokens", "tech_stack", 7200),
    ("m_maven_npm", "后端 Java 用 Maven 打包，前端用 npm（pnpm 会触发沙箱限制）", "tech_stack", 43200),
    ("m_faiss", "知识库检索用 faiss ANN 索引做召回", "tech_stack", 3600),
    ("m_req_txt", "依赖管理用 requirements.txt，Python 环境在 conda 的 srtp-memory", "tech_stack", 86400),
    ("m_embed_v4", "DashScope embedding 用 text-embedding-v4（1024 维）", "tech_stack", 3600),
    ("m_llm_flash", "LLM 先验用 DeepSeek 轻量小模型 deepseek-v4-flash", "tech_stack", 3600),
    ("m_reme_ver", "ReMe 0.4.1.6 安装于 conda 环境 srtp-memory", "tech_stack", 604800),
    ("m_cache_d", "pip 和 HF 模型缓存重定向到 D 盘 dev-tools-srtp 防 C 盘爆满", "tech_stack", 43200),
    ("m_vuetsc", "前端用 vue-tsc --noEmit 类型检查，需加大内存 --max-old-space-size=8192", "tech_stack", 129600),
    ("m_mvn_am", "Maven 编译通知模块必须带 -am 参数，否则找不到依赖符号", "tech_stack", 129600),
    # ── architecture：调度系统架构（12） ──
    ("m_shared_cap", "共享池容量上限 max_shared=20，超限按 final 分淘汰", "architecture", 5400),
    ("m_attention", "四维注意力权重：time/semantic/frequency/task，LLM 先验+可学习权重混合", "architecture", 1800),
    ("m_pipeline", "调度中间件 schedule_once：检索→打分→混合权重→选择→共享池", "architecture", 600),
    ("m_mount", "中间件挂载在 ReMeMiddleware 之前：on_reply 先调度、on_reasoning 注入 HintBlock", "architecture", 900),
    ("m_dual_ws", "主智能体用常驻完整池、副智能体用共享池，双工作区隔离", "architecture", 2700),
    ("m_retrieve", "检索由 BaseRetriever 在常驻池上接管，ReMe 仅写回", "architecture", 3600),
    ("m_composition", "调度配置 config-as-composition：按 retriever_impl/weight_impl 注册表装配", "architecture", 10800),
    ("m_d11", "无 key 时 embedding 降级为确定性哈希占位向量（D-11），pipeline 不中断", "architecture", 1800),
    ("m_prior_fb", "LLM 先验解析失败回退默认权重，测试/离线不触发网络", "architecture", 1800),
    ("m_obs", "可观测性：插件记录 last_error，中间件 api_status() 暴露给控制台", "architecture", 600),
    ("m_log", "调度日志写入 data/metrics/schedule.jsonl", "architecture", 21600),
    ("m_headless", "调度核心为 headless 骨架，提供 schedule_once 同步入口供 CLI/测试驱动", "architecture", 43200),
    # ── research：实验/评估（10） ──
    ("m_ablation", "消融实验 5 组：baseline_naive / baseline_reme / +attention / +weight / full_ours", "research", 10800),
    ("m_metric", "评估指标含跨会话召回率与中文完整性", "research", 21600),
    ("m_fusion", "权重融合公式：final = α×LLM先验 + β×可学习权重", "research", 5400),
    ("m_dataset", "需扩大消融数据集以提升统计显著性", "research", 7200),
    ("m_ratio", "调度压缩比衡量记忆精选程度，越小越精选", "research", 21600),
    ("m_itest", "集成测试覆盖跨会话召回、智能体调度、失败降级、并发隔离", "research", 43200),
    ("m_smoke", "实时链路冒烟验证 embedding 语义区分度与 LLM 先验解析", "research", 3600),
    ("m_baseline", "对照基线含 naive 与 ReMe 原生检索", "research", 43200),
    ("m_stats", "四维打分结果用统计量汇总做消融对比", "research", 21600),
    ("m_vis", "可视化冒烟控制台展示调度流水线与权重变化", "research", 1800),
    # ── daily：日常/干扰项（6） ──
    ("m_walk", "今天天气不错适合出门散步", "daily", 7200),
    ("m_weekly", "昨天下午开完项目周会，确定了下一阶段目标", "daily", 86400),
    ("m_overtime", "用户近期在加班推进冒烟测试与集成测试", "daily", 3600),
    ("m_standup", "团队早上有每日站会同步进度", "daily", 43200),
    ("m_coffee", "用户习惯喝咖啡提神，下午效率更高", "daily", 259200),
    ("m_reme_demo", "上周完成 ReMe demo 端到端验证，产出用户画像", "daily", 604800),
]
TEST_QUERIES = [
    ("我们的项目现在用什么技术栈？", "tech_stack"),
    ("用户有哪些个人背景信息？", "profile"),
    ("调度中间件内部是怎么工作的？", "architecture"),
    ("共享池的容量和淘汰策略是什么？", "architecture"),
    ("今天天气怎么样适合出门散步吗？", "daily"),
]

# ---- 回答生成（REQ-502：副线吃共享池精选记忆生成回答，防"胡言乱语"） ----
ANSWER_PROMPT = (
    "你是基于记忆库回答的助手。下面是调度精选出的相关记忆：\n{ctx}\n\n"
    "用户问题：{query}\n"
    "请仅基于以上记忆简洁作答（2~3 句），不要编造记忆中没有的信息；"
    "记忆不足以回答时如实说明。"
)
ANSWER_MAX_TOKENS = 512


def _answer_with(client, query: str, ctx: str) -> str:
    """用 LLM client 基于精选记忆生成回答（client 复用中间件装配的真实模型）。

    deepseek-v4-flash 偶发返回空内容：空回复自动重试一次，仍空则标注，避免页面留白。
    """
    prompt = ANSWER_PROMPT.format(ctx=ctx, query=query)
    text = client.complete(prompt, max_tokens=ANSWER_MAX_TOKENS).strip()
    if not text:
        text = client.complete(prompt, max_tokens=ANSWER_MAX_TOKENS).strip()
    return text or "(模型返回空回复)"


def _answer_stream(client, query: str, ctx: str):
    """流式回答生成器：yield 文本片段（供 SSE 逐字推送到前端）。"""
    prompt = ANSWER_PROMPT.format(ctx=ctx, query=query)
    try:
        yield from client.complete_stream(prompt, max_tokens=ANSWER_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001 - 流式中断也要给用户一个交代
        yield f"\n[流式回答中断: {type(e).__name__}: {e}]"


def _get_llm_client(mid: MemorySchedulingMiddleware):
    """从中间件取出已装配的真实 LLM client（供回答生成复用）。"""
    _prior_fn = getattr(mid.weights, "_llm_prior_fn", None)
    return getattr(_prior_fn, "client", None) if _prior_fn else None


# ---- 单例中间件（聊天接口复用，避免每次请求重建池） ----
_MID: MemorySchedulingMiddleware | None = None
_RT: "SchedulingRuntime | None" = None


def _get_mid() -> MemorySchedulingMiddleware:
    """懒初始化单例中间件（首次调用种入 50 条种子记忆）。"""
    global _MID
    if _MID is not None:
        return _MID
    cfg = SchedulingConfig.load(ROOT / "config" / "srtp.yaml")
    mid = MemorySchedulingMiddleware(
        config=cfg,
        user_id="u_tu", session_id="chat_vis",
        log_path=str(SMOKE_DIR / "schedule.jsonl"),
        resident_pool_path=str(SMOKE_DIR / "resident_pool.jsonl"),
        shared_pool_path=str(SMOKE_DIR / "shared_pool.json"),
    )
    # 种入种子记忆（复用已入库向量）
    now = time.time()
    for mid_, text, tag, ago in SEED_MEMORIES:
        existing = mid.resident_pool.get(mid_)
        if existing is not None and existing.embedding is not None:
            emb = existing.embedding
        else:
            emb = mid.embedding.encode(text)
        mid.resident_pool.upsert(MemoryCandidate(
            memory_id=mid_, text=text, path=f"daily/mem/{mid_}.md",
            timestamp=now - ago, task_tag=tag,
            user_id="u_tu", session_id="chat_vis",
            embedding=emb))
    _MID = mid
    return mid


def _get_runtime() -> "SchedulingRuntime":
    """懒初始化主副线运行时（与中间件共用单例）。"""
    global _RT
    if _RT is None:
        _RT = SchedulingRuntime(_get_mid(), SchedulingConfig.load(ROOT / "config" / "srtp.yaml"))
    return _RT


def _pool_layout(mid: MemorySchedulingMiddleware) -> dict:
    """双池 2D 语义布局（PCA 投影，坐标归一化到 [0,1]）。

    常驻池：对其全部 embedding 做 SVD 取 top-2 主成分；共享池条目复用同一变换投影。
    缺 embedding / 样本不足时退回网格布局，保证页面永远有图。
    返回 {resident:[{id,text,tag,age_h,access,x,y}], shared:[{id,text,score,x,y}]}。
    """
    res = mid.resident_pool.all()
    mats, meta = [], []
    for m in res:
        if getattr(m, "embedding", None) is not None:
            mats.append(np.asarray(m.embedding, dtype=np.float32).ravel())
            meta.append(m)
    shared_items = mid.shared_pool.top()
    shared_emb = {}
    for it in shared_items:
        rec = mid.resident_pool.get(it.get("memory_id"))
        if rec is not None and getattr(rec, "embedding", None) is not None:
            shared_emb[it.get("memory_id")] = np.asarray(rec.embedding, dtype=np.float32).ravel()

    now = time.time()
    resident = []
    if len(mats) >= 2:
        X = np.stack(mats)
        mean = X.mean(0)
        Xc = X - mean
        # SVD top-2（numpy 自带，无需 sklearn）
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comp = Vt[:2]
        P = Xc @ comp.T
        lo, hi = P.min(0), P.max(0)
        rng = (hi - lo) + 1e-9
        coords = (P - lo) / rng
        for m, (cx, cy) in zip(meta, coords):
            resident.append({
                "id": m.memory_id, "text": m.text, "tag": m.task_tag or "",
                "age_h": round((now - m.timestamp) / 3600.0, 1),
                "access": getattr(m, "access_count", 0),
                "x": round(float(cx), 4), "y": round(float(cy), 4),
            })
        # 共享池用同一 mean/comp 投影
        shared = []
        for it in shared_items:
            emb = shared_emb.get(it.get("memory_id"))
            if emb is not None:
                pc = (emb - mean) @ comp.T
                sx = (pc[0] - lo[0]) / rng[0]
                sy = (pc[1] - lo[1]) / rng[1]
            else:
                sx = sy = 0.5
            shared.append({
                "id": it.get("memory_id"), "text": it.get("text", ""),
                "score": round(float(it.get("score", 0.0)), 4),
                "x": round(float(min(1.0, max(0.0, sx))), 4),
                "y": round(float(min(1.0, max(0.0, sy))), 4),
            })
    else:
        # 退化网格布局
        n = max(1, len(meta))
        cols = max(1, int(n ** 0.5))
        for i, m in enumerate(meta):
            resident.append({
                "id": m.memory_id, "text": m.text, "tag": m.task_tag or "",
                "age_h": round((now - m.timestamp) / 3600.0, 1),
                "access": getattr(m, "access_count", 0),
                "x": round((i % cols) / max(1, cols - 1), 4) if cols > 1 else 0.5,
                "y": round((i // cols) / max(1, (n // cols)), 4) if (n // cols) > 0 else 0.5,
            })
        shared = [{
            "id": it.get("memory_id"), "text": it.get("text", ""),
            "score": round(float(it.get("score", 0.0)), 4),
            "x": round((i % 3) / 3.0, 4), "y": round((i // 3) / 3.0, 4),
        } for i, it in enumerate(shared_items)]
    return {"resident": resident, "shared": shared}


def _run_smoke() -> dict:
    """跑一轮真实链路冒烟，返回全流程 trace（不抛异常，异常记录进 trace）。"""
    t_start = time.time()
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = SchedulingConfig.load(ROOT / "config" / "srtp.yaml")

    mid = MemorySchedulingMiddleware(
        config=cfg,
        user_id="u_tu", session_id="smoke_vis",
        log_path=str(SMOKE_DIR / "schedule.jsonl"),
        resident_pool_path=str(SMOKE_DIR / "resident_pool.jsonl"),
        shared_pool_path=str(SMOKE_DIR / "shared_pool.json"),
        # llm_prior_fn=None → 中间件自动装配真实 DeepSeek 先验（.env 有 key 时）
    )

    emb_mode = "real" if getattr(mid.embedding, "api_key", None) else "fallback"
    llm_mode = "real" if getattr(mid.weights, "_llm_prior_fn", None) else "fallback"

    # 1) 种入真实中文记忆（复用已入库向量，仅新记忆/缺向量时 embed 一次）
    # 记忆的 embedding 在首次 upsert 时已持久化进 resident_pool.jsonl；后续轮次直接
    # 复用存量向量，不再重复调用 embedding API（与"初次入库即可复用"的设计一致）。
    now = time.time()
    n_embedded, n_reused = 0, 0
    for mid_, text, tag, ago in SEED_MEMORIES:
        existing = mid.resident_pool.get(mid_)
        if existing is not None and existing.embedding is not None:
            emb, n_reused = existing.embedding, n_reused + 1
        else:
            emb, n_embedded = mid.embedding.encode(text), n_embedded + 1
        mid.resident_pool.upsert(MemoryCandidate(
            memory_id=mid_, text=text, path=f"daily/mem/{mid_}.md",
            timestamp=now - ago, task_tag=tag,
            user_id="u_tu", session_id="smoke_vis",
            embedding=emb))

    # 2) 逐条跑真实调度
    queries = []
    api_errors: dict[str, dict] = {}  # 聚合 API 错误（供前端告警横幅）
    for i, (q, tag) in enumerate(TEST_QUERIES, 1):
        step = {"index": i, "query": q, "task_tag": tag}
        try:
            res = mid.schedule_once(q, task_tag=tag, now=now)
            # 暴露该轮 API 运行状态（含 last_error，如"余额不足"）——不再静默降级
            step["api_status"] = mid.api_status()
            for scope, st in step["api_status"].items():
                if st.get("error") and scope not in api_errors:
                    api_errors[scope] = st["error"]
            prior = getattr(mid.weights, "_cached_prior", None) or dict(DEFAULT_WEIGHTS)
            step["llm_prior"] = {k: round(v, 3) for k, v in prior.items()}
            step["is_llm_prior_default"] = prior == dict(DEFAULT_WEIGHTS)
            step["final_weights"] = {k: round(v, 3) for k, v in res.weights.items()}
            kept_ids = {m.memory_id for m in res.kept}
            cands = []
            for item in res.scored:
                m, sc = item["memory"], item["score"]
                cands.append({
                    "id": m.memory_id, "text": m.text,
                    "age_h": round((now - m.timestamp) / 3600, 1),
                    "scores": {d: round(sc[d], 3) for d in ("time", "semantic", "frequency", "task", "final")},
                    "kept": m.memory_id in kept_ids,
                })
            cands.sort(key=lambda c: -c["scores"]["final"])
            step["candidates"] = cands
            step["kept_count"] = len(res.kept)
            step["discarded_count"] = len(res.discarded)
            step["compression_ratio"] = round(res.compression_ratio, 3)
            step["latency_ms"] = {k: round(v, 2) for k, v in res.latency_ms.items()}

            # 回答生成（REQ-502：副线吃调度精选记忆生成回答）——直接展示防"胡言乱语"
            kept_texts = [m.text for m in res.kept]
            step["kept_memories"] = kept_texts
            llm_client = _get_llm_client(mid)
            if llm_client:
                ctx = "\n".join(f"- {t}" for t in kept_texts[:8]) or "（本轮无保留记忆）"
                try:
                    step["answer"] = _answer_with(llm_client, q, ctx)
                except Exception as e:  # noqa: BLE001 - 回答失败不影响调度 trace
                    step["answer"] = f"(回答生成失败: {type(e).__name__}: {e})"
            else:
                step["answer"] = "(未配置 LLM，跳过回答生成)"
        except Exception as e:  # noqa: BLE001 - 单条失败不阻断整轮
            step["error"] = f"{type(e).__name__}: {e}"
        queries.append(step)

    mid.flush()
    shared = [{k: v for k, v in item.items() if k in ("memory_id", "text", "score", "timestamp")}
              for item in mid.shared_pool.top()]

    return {
        "ok": True,
        "config": {
            "retriever": cfg.retriever_impl, "attention": cfg.attention_impl,
            "weight": cfg.weight_impl, "selector": cfg.selector_impl,
            "embedding": cfg.embedding_impl, "model": cfg.model_impl,
            "alpha": cfg.alpha, "beta": cfg.beta, "candidate_override": cfg.candidate_override,
            "threshold": cfg.threshold, "max_shared": cfg.max_shared,
        },
        "api": {"embedding": emb_mode, "llm_prior": llm_mode},
        "api_errors": api_errors,
        "seeded_count": len(SEED_MEMORIES),
        "embed_stats": {"embedded": n_embedded, "reused": n_reused},
        "queries": queries,
        "shared_pool": shared,
        "elapsed_ms": round((time.time() - t_start) * 1000, 1),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, data: str) -> None:
        """向已打开的 SSE 流写一条 event（data: JSON\n\n）。"""
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] in ("/", "/index.html"):
            html = (ROOT / "dashboard.html").read_text(encoding="utf-8")
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.split("?")[0] == "/api/pools":
            try:
                layout = _pool_layout(_get_mid())
                body = json.dumps({"ok": True, "layout": layout,
                                    "agent_mode": _get_runtime().mode},
                                   ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                                           ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
        elif self.path.split("?")[0] == "/api/status":
            try:
                rt = _get_runtime()
                body = json.dumps({
                    "ok": True,
                    "agent_mode": rt.mode,
                    "agentscope_available": AGENTSCOPE_AVAILABLE,
                    "init_error": rt._init_error,
                    "api": _get_mid().api_status(),
                }, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                                           ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path == "/api/smoke":
            try:
                trace = _run_smoke()
                self._send(200, json.dumps(trace, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                                           ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
        elif self.path == "/api/chat":
            self._handle_chat_sse()
        else:
            self._send(404, b"not found", "text/plain")

    def _handle_chat_sse(self):
        """SSE 流式聊天（主副线架构）：POST body={query, task_tag?, run_sub?} → 逐条 SSE event。

        事件顺序：
        1. {type:"schedule", agent_mode, ...}  — 主线调度过程（双池召回/四维打分/权重/选择）
        2. {type:"pools", resident, shared, recalled_ids, kept_ids} — 双池向量库布局
        3. {type:"sublines", items:[...]}      — 副线书签（共享池注入隔离 workspace）
        4. {type:"answer_delta", text}         — 主线 Agent 流式回答（逐字/逐段）
        5. {type:"done", ...}                  — 结束信号
        """
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        query = (payload.get("query") or "").strip()
        task_tag = payload.get("task_tag", "")
        run_sub = bool(payload.get("run_sub", False))

        # SSE 响应头
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        if not query:
            self._send_sse(json.dumps({"type": "error", "message": "查询不能为空"},
                                      ensure_ascii=False))
            return

        try:
            mid = _get_mid()
            rt = _get_runtime()
            now = time.time()
            agent_mode = rt.mode

            # ① 主线调度
            res = mid.schedule_once(query, task_tag=task_tag or None, now=now)
            kept_texts = [m.text for m in res.kept]
            kept_ids = {m.memory_id for m in res.kept}
            recalled_ids = [m.memory_id for m in res.candidates]
            cands = []
            for item in res.scored:
                m, sc = item["memory"], item["score"]
                cands.append({
                    "id": m.memory_id, "text": m.text,
                    "age_h": round((now - m.timestamp) / 3600, 1),
                    "scores": {d: round(sc[d], 3) for d in ("time", "semantic", "frequency", "task", "final")},
                    "kept": m.memory_id in kept_ids,
                })
            cands.sort(key=lambda c: -c["scores"]["final"])
            prior = getattr(mid.weights, "_cached_prior", None) or dict(DEFAULT_WEIGHTS)

            # 主线回答（真实 LLM / AgentScope Agent；先算出完整文本，再切片流式推送）
            main_answer = rt.main_answer(query, kept_texts)

            schedule_event = {
                "type": "schedule",
                "query": query,
                "task_tag": task_tag,
                "agent_mode": rt.mode,
                "candidates": cands,
                "kept_count": len(res.kept),
                "discarded_count": len(res.discarded),
                "compression_ratio": round(res.compression_ratio, 3),
                "latency_ms": {k: round(v, 2) for k, v in res.latency_ms.items()},
                "llm_prior": {k: round(v, 3) for k, v in prior.items()},
                "final_weights": {k: round(v, 3) for k, v in res.weights.items()},
                "kept_memories": kept_texts,
                "api_status": mid.api_status(),
            }
            self._send_sse(json.dumps(schedule_event, ensure_ascii=False))

            # ② 双池向量库布局
            layout = _pool_layout(mid)
            self._send_sse(json.dumps({
                "type": "pools",
                "resident": layout["resident"],
                "shared": layout["shared"],
                "recalled_ids": recalled_ids,
                "kept_ids": list(kept_ids),
                "max_shared": mid.config.max_shared,
            }, ensure_ascii=False))

            # ③ 副线书签（共享池注入隔离 workspace；run_sub 时运行真实副线 Agent，
            #    reviewer 已拿到主线回答做一致性校验）
            sub_items = rt.sub_lines(query, kept_texts, main_answer, run_sub)
            self._send_sse(json.dumps({"type": "sublines", "items": sub_items},
                                      ensure_ascii=False))

            # ④ 主线回答流式推送（预计算文本切片）
            for chunk in _slice_text(main_answer):
                self._send_sse(json.dumps({"type": "answer_delta", "text": chunk},
                                          ensure_ascii=False))

            # ⑤ 收尾
            mid.flush()
            self._send_sse(json.dumps({"type": "done",
                                       "agent_mode": rt.mode,
                                       "kept_count": len(res.kept),
                                       "compression_ratio": round(res.compression_ratio, 3)},
                                      ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self._send_sse(json.dumps({"type": "error",
                                       "message": f"{type(e).__name__}: {e}"},
                                      ensure_ascii=False))


def _slice_text(text: str, size: int = 4):
    """把完整文本切成小段用于 SSE 推送（每 size 字一段，兼容中英文）。"""
    if not text:
        return [""]
    out = []
    for i in range(0, len(text), size):
        out.append(text[i:i + size])
    return out or [""]

    def log_message(self, *a):  # noqa: D401 - 静默请求日志
        pass


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"可视化冒烟控制台已启动: http://127.0.0.1:{PORT}/")
    print("（真实 API 链路 · 密钥读取自 .env；Ctrl+C 停止）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
