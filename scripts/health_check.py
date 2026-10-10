"""API 健康体检（明天演示前跑一次，全绿才敢上台）。

检查四层：
  L1 外部 API   —— DashScope embedding / DeepSeek LLM / OpenAI，逐个真实发请求
  L2 依赖库     —— torch / faiss / dashscope / openai / agentscope / reme / jieba
  L3 后端接口   —— app_dashboard 全部 GET/POST 路由，含 SSE
  L4 降级状态   —— api_status() 报的 real/fallback，防止"能跑但在偷偷降级"

用法：
  E:/dev-tools/envs/srtp-memory/Scripts/python.exe scripts/health_check.py
  加 --http 才测运行中的服务（默认只测 L1/L2/L4，避免依赖服务存活）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# D-063：体检前的画布状态快照，跑完用来把脏数据还原回去
_probe_state: dict = {}
sys.path.insert(0, ROOT)

BASE = "http://127.0.0.1:8787"

# 展示用颜色（Windows 终端）
OK, BAD, WARN, DIM, RST = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"

results: list[tuple[str, str, bool, str, float]] = []  # (层, 名称, 通过, 详情, 秒)


def rec(layer: str, name: str, ok: bool | None, detail: str = "", cost: float = 0.0) -> None:
    results.append((layer, name, ok, detail, cost))
    mark = "未测" if ok is None else ("PASS" if ok else "FAIL")
    color = DIM if ok is None else (OK if ok else BAD)
    print(f"  [{color}{mark}{RST}] {name:<34} {DIM}{detail}{RST}"
          + (f"  ({cost:.2f}s)" if cost >= 0.01 else ""))


def head(title: str) -> None:
    print(f"\n{title}\n" + "-" * 72)


# ---------------------------------------------------------------- L2 依赖库
def check_deps() -> None:
    """区分「必需」与「可选」：可选缺失不算故障，只提示。

    依据是代码里是否真的 import/调用它——
    jieba 全项目零 import（BM25 走自研字符分词），sklearn 全项目零 import（SVD 用 numpy）。
    agentscope/reme 缺失会让mode 降级成 headless（功能仍在，但少了真实Agent 装配）。
    """
    head("L2 依赖库")
    # (库名, 是否必需, 缺失时的实际影响)
    mods = [
        ("numpy", True, ""),
        ("faiss", False, "向量索引退化线性余弦，结果仍可用"),
        ("torch", True, ""),
        ("dashscope", True, ""),
        ("openai", True, ""),
        ("yaml", True, ""),
        ("jieba", False, "未被 import，BM25 走自研字符分词（无影响）"),
        ("sklearn", False, "未被 import，SVD 用 numpy 实现（无影响）"),
        ("agentscope", False, "mode 降级为 headless：功能完整，但非真实 Agent 装配"),
        ("reme", False, "baseline-reme 检索器基线不可用（仅跑对照实验时需要）"),
    ]
    for mod, required, impact in mods:
        t0 = time.time()
        try:
            m = __import__(mod)
            rec("L2", mod, True, f"v{getattr(m, '__version__', '?')}", time.time() - t0)
        except Exception as e:  # noqa: BLE001
            if required:
                rec("L2", mod, False, f"必需但缺失：{type(e).__name__}")
            else:
                rec("L2", mod, None, f"可选未装（{impact}）", time.time() - t0)


# ------------------------------------------------------------ L1 外部 API
def check_env_keys() -> None:
    """只DASHSCOPE 是必需的。

    DEEPSEEK_API_KEY 没配是**设计好的自动降级**（app_dashboard._get_mid() 会把
    model.deepseek 换成 model.dashscope/qwen-plus），不是故障。
    OPENAI 纯属备选，没配不影响任何功能。
    """
    head("L1-0 环境变量（只有 DASHSCOPE 是必需的）")
    import srtp_memory  # noqa: F401  触发 .env 加载

    v = os.environ.get("DASHSCOPE_API_KEY")
    rec("L1", "DASHSCOPE_API_KEY  必需", bool(v),
        f"len={len(v)} 前缀={v[:6]}****" if v else "缺失 → embedding 退化为占位向量，语义检索完全失效")

    v = os.environ.get("DEEPSEEK_API_KEY")
    rec("L1", "DEEPSEEK_API_KEY  可选", None if not v else True,
        f"len={len(v)}" if v else "未配 → 自动降级 qwen-plus（app_dashboard:169），功能不受影响")

    v = os.environ.get("OPENAI_API_KEY")
    rec("L1", "OPENAI_API_KEY  可选", None if not v else True,
        f"len={len(v)}" if v else "未配 → 纯备选后端，不影响任何功能")


def check_dashscope_embed() -> None:
    head("L1-1 DashScope embedding (text-embedding-v4)")
    from srtp_memory.plugins.infra.embedding_impls import DashScopeEmbedding

    # 用一个绝不会命中缓存的文本，强制走真实 API
    probe = f"__healthcheck_{int(time.time())}__"
    emb = DashScopeEmbedding()# 不传 cache_dir，绕过落盘缓存
    if not emb.api_key:
        rec("L1", "dashscope embed", False, "无 api_key，将降级为占位向量（语义检索失效）")
        return
    t0 = time.time()
    try:
        v = emb._api_encode(probe)  # 直接调真实 API，不走 encode 的兜底
        dt = time.time() - t0
        if v is None:
            rec("L1", "dashscope embed", False, "_api_encode 返回 None")
        else:
            import numpy as np
            n = float(np.linalg.norm(v))
            rec("L1", "dashscope embed", True,
                f"dim={v.shape[0]} |v|={n:.4f}", dt)
    except Exception as e:  # noqa: BLE001
        rec("L1", "dashscope embed", False,
            f"{type(e).__name__}: {str(e)[:120]} | last_error={emb.last_error}", time.time() - t0)


def check_llm(adapter_name: str, label: str, required: bool = False) -> None:
    from srtp_memory.plugins.infra import model_adapters
    cls = {
        "model.deepseek": model_adapters.DeepSeekAdapter,
        "model.openai": model_adapters.OpenAIAdapter,
        "model.dashscope": model_adapters.DashScopeModelAdapter,
    }[adapter_name]
    ad = cls()
    if not ad.api_key:
        rec("L1", label, False if required else None,
            f"未配置 {ad.env_var}" + ("" if required else "（可选后端）"))
        return
    t0 = time.time()
    try:
        out = ad.complete("用一句话回答：1+1等于几？", max_tokens=32, thinking=False)
        dt = time.time() - t0
        ok = bool(out and out.strip())
        rec("L1", label, ok, (out.strip()[:60] if ok else "空返回"), dt)
    except Exception as e:  # noqa: BLE001
        rec("L1", label, False,
            f"{type(e).__name__}: {str(e)[:110]} | last_error={ad.last_error}",
            time.time() - t0)


def check_llm_stream() -> None:
    """SSE 真流式依赖的 complete_stream —— 单独测，坏在这里明天页面就卡住。

    注意测的是**实际在用的那个后端**（无 DEEPSEEK key 时是 qwen-plus），
    否则测了个没配key 的 DeepSeek，等于什么都没验。
    """
    import srtp_memory  # noqa: F401
    from srtp_memory.plugins.infra import model_adapters

    use_dashscope = not os.environ.get("DEEPSEEK_API_KEY")
    cls = model_adapters.DashScopeModelAdapter if use_dashscope else model_adapters.DeepSeekAdapter
    who = "qwen-plus(实际在用)" if use_dashscope else "deepseek"
    ad = cls()
    if not ad.api_key:
        rec("L1", f"LLM stream [{who}]", False, f"无 key（{ad.env_var}）")
        return
    t0 = time.time()
    try:
        chunks = list(ad.complete_stream("数到三", max_tokens=32, thinking=False))
        dt = time.time() - t0
        rec("L1", f"LLM stream [{who}]", len(chunks) > 0,
            f"{len(chunks)} chunks: {''.join(chunks)[:50]!r}", dt)
    except Exception as e:  # noqa: BLE001
        rec("L1", f"LLM stream [{who}]", False,
            f"{type(e).__name__}: {str(e)[:110]}", time.time() - t0)


# ------------------------------------------------------------ L4 降级状态
def check_pipeline_health() -> None:
    """模块可导入性：只确认管线代码本身能加载（真实降级状态看 /api/status）。"""
    head("L4 管线模块可加载性")
    for mod in ("srtp_memory.middleware", "srtp_memory.retriever",
                "srtp_memory.agentscope_runtime", "srtp_memory.config"):
        t0 = time.time()
        try:
            __import__(mod)
            rec("L4", mod, True, "导入成功", time.time() - t0)
        except Exception as e:  # noqa: BLE001
            rec("L4", mod, False, f"{type(e).__name__}: {e}", time.time() - t0)


def check_api_status_via_http() -> dict | None:
    """从运行中的服务读 api_status()——最真实的降级状态。"""
    try:
        with urllib.request.urlopen(f"{BASE}/api/status", timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  {DIM}读 /api/status 失败: {type(e).__name__}: {e}{RST}")
        return None


# ------------------------------------------------------------ L3 后端接口
def _req(method: str, path: str, body: dict | None = None, timeout: int = 90) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def _sse_post(path: str, body: dict, timeout: int = 240) -> list[str]:
    """POST 一个 SSE 端点，返回收到的 event type 列表。

    ⚠️ create_sub / chat 都是 text/event-stream，不能按普通 JSON 读——
    curl -o 落盘看着像"空响应"，其实是流。
    """
    r = urllib.request.Request(f"{BASE}{path}",
                               data=json.dumps(body).encode("utf-8"),
                               method="POST", headers={"Content-Type": "application/json"})
    types: list[str] = []
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        buf = b""
        while True:
            b1 = resp.read(1)
            if not b1:
                break
            buf += b1
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                s = line.decode("utf-8", "replace")
                if s.startswith("data:"):
                    try:
                        types.append(json.loads(s[5:]).get("type", "?"))
                    except Exception:  # noqa: BLE001
                        types.append("?")
    return types


def check_http() -> None:
    head("L3 后端接口（运行中的服务）")

    t0 = time.time()
    code, txt = _req("GET", "/")
    rec("L3", "GET /  控制台", code == 200, f"HTTP {code}, {len(txt)}B", time.time() - t0)

    for path, label in [("/canvas", "GET /canvas  会话画布"),
                        ("/graph", "GET /graph  图拓扑"),
                        ("/static/app.css", "GET /static/app.css"),
                        ("/static/theme.js", "GET /static/theme.js")]:
        t0 = time.time()
        code, txt = _req("GET", path)
        rec("L3", label, code == 200, f"HTTP {code}, {len(txt)}B", time.time() - t0)

    # 静态资源 —— 缺 vendor 文件会导致图拓扑直接白屏，明天最致命
    head("L3 静态资源（缺文件 = 页面白屏）")
    assets = ["app.css", "canvas.css", "dashboard.css", "theme.js",
              "canvas.js", "dashboard.js", "graph.js", "vendor/cytoscape.min.js"]
    bad_assets = []
    for a in assets:
        code, txt = _req("GET", f"/static/{a}")
        if code != 200 or len(txt) < 100:
            bad_assets.append(a)
            rec("L3", f"static/{a}", False, f"HTTP {code}, {len(txt)}B（可能白屏）")
        else:
            rec("L3", f"static/{a}", True, f"{len(txt)}B")

    # /api/graph —— 走真实选源与 embedding
    t0 = time.time()
    code, txt = _req("GET", "/api/graph", timeout=120)
    ok = code == 200
    detail = f"HTTP {code}, {len(txt)}B"
    if ok:
        try:
            g = json.loads(txt).get("graph", {})
            rec("L3", "GET /api/graph", True,
                f"节点 {len(g.get('nodes', []))} / 边 {len(g.get('edges', []))}", time.time() - t0)
        except Exception as e:  # noqa: BLE001
            rec("L3", "GET /api/graph", False, f"JSON 解析失败 {e}", time.time() - t0)
    else:
        rec("L3", "GET /api/graph", False, f"{detail} {txt[:100]}", time.time() - t0)

    # /api/pools —— 记忆池（依赖 embedding 相似度）
    t0 = time.time()
    code, txt = _req("GET", "/api/pools", timeout=120)
    ok = code == 200
    if ok:
        try:
            lay = json.loads(txt).get("layout", {})
            rec("L3", "GET /api/pools", True,
                f"HTTP 200, {len(txt)}B, mode={json.loads(txt).get('agent_mode')}",
                time.time() - t0)
        except Exception as e:  # noqa: BLE001
            rec("L3", "GET /api/pools", False, f"JSON 解析失败 {e}", time.time() - t0)
    else:
        rec("L3", "GET /api/pools", False, f"HTTP {code} {txt[:100]}", time.time() - t0)

    # /api/stage_a —— 实算 Stage A 选源
    t0 = time.time()
    code, txt = _req("GET", "/api/stage_a?node=main&q=%E8%AE%B0%E5%BF%86%E5%85%B1%E4%BA%AB", timeout=120)
    rec("L3", "GET /api/stage_a  选源", code == 200, f"HTTP {code}, {len(txt)}B", time.time() - t0)

    # /api/canvas/history —— 记录体检前的回合数，供后面还原
    t0 = time.time()
    code, txt = _req("GET", "/api/canvas/history")
    rec("L3", "GET /api/canvas/history", code == 200, f"HTTP {code}, {len(txt)}B", time.time() - t0)
    try:
        _probe_state["main_rounds"] = len(json.loads(txt)["history"]["main"]["rounds"])
    except Exception:  # noqa: BLE001
        pass

    # POST /api/canvas/layout —— 布局微调（D-063 支持 reset，体检用空 body 不改动数据）
    t0 = time.time()
    code, txt = _req("POST", "/api/canvas/layout", {})
    rec("L3", "POST /api/canvas/layout", code == 200, f"HTTP {code}, {len(txt)}B", time.time() - t0)

    # POST /api/smoke —— 冒烟（可能触发 LLM）
    t0 = time.time()
    code, txt = _req("POST", "/api/smoke", {}, timeout=180)
    rec("L3", "POST /api/smoke", code == 200, f"HTTP {code}, {len(txt)}B", time.time() - t0)

    # ---- SSE 端点（最容易挂，单独验证）----
    head("L3-POST SSE 流式端点（text/event-stream）")
    print(f"  {DIM}注意：这两个端点返回事件流，用普通 JSON 方式读会误判为「空响应」{RST}")

    t0 = time.time()
    try:
        types = _sse_post("/api/chat", {"query": "健康检查：请用一句话说明记忆共享调度"})
        rec("SSE", "POST /api/chat", "done" in types,
            f"{len(types)} 事件 types={sorted(set(types))}", time.time() - t0)
        rec("SSE", "  ↳ done 收尾事件", "done" in types,
            "已收到" if "done" in types else "未收到 → 前端会卡在「调度中」")
    except Exception as e:  # noqa: BLE001
        rec("SSE", "POST /api/chat", False, f"{type(e).__name__}: {str(e)[:140]}",
            time.time() - t0)

    # create_sub 需要真实 round_id（是 R11 这种字符串，不是序号）
    rid = None
    try:
        code, txt = _req("GET", "/api/canvas/history")
        rounds = json.loads(txt)["history"]["main"]["rounds"]
        rid = rounds[-1].get("round_id")
    except Exception:  # noqa: BLE001
        pass
    if not rid:
        rec("SSE", "POST /api/canvas/create_sub", False, "拿不到有效 round_id，跳过")
    else:
        t0 = time.time()
        try:
            types = _sse_post("/api/canvas/create_sub",
                              {"from_round": rid, "title": "健康检查节点",
                               "query": "记忆共享调度的核心机制是什么"})
            rec("SSE", f"POST /api/canvas/create_sub", "done" in types,
                f"from_round={rid}, {len(types)} 事件 types={sorted(set(types))}",
                time.time() - t0)
        except Exception as e:  # noqa: BLE001
            rec("SSE", "POST /api/canvas/create_sub", False,
                f"{type(e).__name__}: {str(e)[:140]}", time.time() - t0)

    cleanup_probe_data()


def cleanup_probe_data() -> None:
    """★ D-063：体检自己造的脏数据必须自己清掉。

    create_sub 会**真的写进** data/demo/canvas_history.json，以前跑完不清理，
    演示数据里就留下 "健康检查节点" 僵尸下游节点（历史上堆到过 7 个），
    招新演示时一眼就能看出是假数据。
    服务端每次请求都重新读文件（load_or_create），所以直接改文件即可生效。
    """
    path = os.path.join(ROOT, "data", "demo", "canvas_history.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  {DIM}清理跳过（读不到画布历史）：{e}{RST}")
        return
    removed = []
    subs = data.get("subs", [])
    kept = [s for s in subs if s.get("title") != "健康检查节点"]
    if len(kept) != len(subs):
        data["subs"] = kept
        removed.append(f"节点 {len(subs) - len(kept)} 个")

    # /api/chat、/api/smoke 也会往主节点追加回合 —— 一并裁回体检前的长度
    n0 = _probe_state.get("main_rounds")
    rounds = data.get("main", {}).get("rounds", [])
    if n0 is not None and len(rounds) > n0:
        data["main"]["rounds"] = rounds[:n0]
        removed.append(f"主节点回合 {len(rounds) - n0} 条")

    if removed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {DIM}已清理体检数据：{'、'.join(removed)}{RST}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="同时测运行中的 HTTP 服务")
    args = ap.parse_args()

    print(f"{'=' * 72}\nAPI 健康体检  {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 72}")

    check_deps()
    check_env_keys()
    check_dashscope_embed()
    # 实际主用后端 = 配了 DEEPSEEK key 就走 DeepSeek，否则 app_dashboard:169 自动降级 qwen
    import srtp_memory  # noqa: F401
    main_llm = "model.deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "model.dashscope"
    main_tag = "deepseek（实际主用）" if main_llm == "model.deepseek" else "qwen-plus（实际主用）"
    other = ("model.dashscope", "qwen-plus（降级备用）") if main_llm == "model.deepseek" \
        else ("model.deepseek", "deepseek（未配 key）")
    check_llm(main_llm, main_tag, required=True)
    check_llm_stream()
    check_llm(*other)
    check_llm("model.openai", "openai  complete（备选）")
    check_pipeline_health()

    if args.http:
        check_http()
        st = check_api_status_via_http()
        if st:
            head("L4 服务端自报 api_status()")
            api = st.get("api", {})
            for mode, info in api.items():
                err = info.get("error")
                if err:
                    rec("L4", f"{mode} 模式", False,
                        f"fallback ← {err.get('kind')} / {str(err.get('message'))[:80]}")
                else:
                    rec("L4", f"{mode} 模式", True, "real（真实 API，无降级）")

    # ---------------- 汇总
    head("汇总")
    fail = [r for r in results if r[2] is False]
    warn = [r for r in results if r[2] is None]
    passed = [r for r in results if r[2] is True]
    print(f"  {OK}PASS {len(passed)}{RST}{DIM}（含 1 项可选未装）{RST}"
          f"{'   ' if warn else ''}{WARN}可选缺失 {len(warn)}{RST}   "
          f"{BAD}FAIL {len(fail)}{RST}\n")
    if warn:
        print(f"  {WARN}可选缺失（不影响明天演示，列出仅为知情）：{RST}")
        for layer, name, _, detail, _ in warn:
            print(f"    · [{layer}] {name}：{detail}")
        print()
    if fail:
        print(f"  {BAD}不可用清单（明天演示前必须处理）：{RST}")
        for layer, name, _, detail, _ in fail:
            print(f"    · [{layer}] {name}\n        {detail}")
    else:
        print(f"  {OK}{'=' * 40}{RST}")
        print(f"  {OK}所有必需 API 与接口均可用，可以演示。{RST}")
    return 1 if fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
