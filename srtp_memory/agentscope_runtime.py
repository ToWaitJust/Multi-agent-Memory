"""主副线运行时（真实 AgentScope 接入点，§4.8 / D3）。

SchedulingRuntime 统一封装「主副线」执行，对外暴露与具体后端无关的接口：
- main_answer(query, kept_texts)：主线智能体基于共享池精选记忆作答。
- sub_lines(query, kept_texts, main_answer, run_sub)：副线隔离 workspace 的书签 +
  （可选）真实副线 Agent 的侧任务产物。

后端选择（启动时定一次，页面标注 agent_mode）：
- agentscope：环境装有 agentscope 且 DeepSeek key 可用时，主线/副线用真实 AgentScope
  2.0.6 Agent 实例驱动（每个副线 Agent 独立实例 = 逻辑隔离沙箱，D3）。
- headless：缺 agentscope 或缺 key 时，降级为 headless 中间件（仍是真实 DashScope
  embedding + DeepSeek LLM，仅「Agent 编排层」用我们自己的适配器），页面标注清楚。

所有 agentscope 调用都做隔离 try/except；任何失败都回退 headless，保证页面永远可跑。
真实 AgentScope 执行需要运行环境装有 agentscope（项目 conda 环境 srtp-memory）。
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from .config import SchedulingConfig

# ---- agentscope 可用性探测（隔离，不影响无包环境）----
AGENTSCOPE_AVAILABLE = False
try:
    import agentscope  # noqa: F401
    AGENTSCOPE_AVAILABLE = True
except Exception:  # noqa: BLE001 - 探测失败即视为不可用
    AGENTSCOPE_AVAILABLE = False

# 真实 AgentScope 2.0.6 消息原语（仅在可用时导入，避免无包环境报错）。
# 注意：2.0.6 的 Msg.content 是 ContentBlock 列表，需通过 UserMsg 便捷构造器
# 把字符串包成 TextBlock，并遍历 content 取 .text，不能再用旧版的
# Msg(name=, content=str, role=) 或 agentscope.init()。
if AGENTSCOPE_AVAILABLE:
    try:
        from agentscope.message import UserMsg, TextBlock  # noqa: F401
    except Exception:  # noqa: BLE001
        UserMsg = None
        TextBlock = None
else:
    UserMsg = None
    TextBlock = None


# 副线规格（书签定位 + 隔离 workspace，FR-8 / D3）
SUB_LINE_SPECS = [
    ("sub1", "检索副线", "基于共享池记忆做一条补充侧摘（≤2 句），与主回答互补。"),
    ("reviewer", "校验副线", "基于共享池记忆对主线回答做一致性校验（≤2 句），指出偏差。"),
]


class SchedulingRuntime:
    """主副线执行运行时（单例，随中间件一同懒初始化）。"""

    def __init__(self, middleware, config: SchedulingConfig):
        self.mid = middleware
        self.config = config
        self.mode = "agentscope" if (AGENTSCOPE_AVAILABLE and os.environ.get("DEEPSEEK_API_KEY")) else "headless"
        self._agents: dict[str, object] = {}
        self._init_error: Optional[str] = None
        if self.mode == "agentscope":
            try:
                self._build_agentscope()
            except Exception as e:  # noqa: BLE001 - 任何失败都回退 headless
                self.mode = "headless"
                self._agents = {}
                self._init_error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ #
    # 真实 AgentScope 装配（best-effort，失败即降级）
    # ------------------------------------------------------------------ #
    def _build_agentscope(self) -> None:
        """用 AgentScope 2.0.6 真实原语装配主/副线 Agent。

        2.0.6 关键差异（与旧文档/示例不符，已实测）：
        - 没有 agentscope.init、没有 agentscope.agents 模块、没有 DialogAgent。
        - 唯一 Agent 类为 agentscope.agent.Agent，直接传 model 实例（ChatModelBase）。
        - 模型用 agentscope.model.DeepSeekChatModel + agentscope.credential.DeepSeekCredential。
        - 调用需传 UserMsg（content 会被包成 TextBlock 列表），返回 Msg.content 仍是
          ContentBlock 列表，文本需遍历取 .text。
        - 工具系统：agentscope.tool.Toolkit([Read(), Write(), Edit(), Bash(), Glob(), Grep()])，
          直接作为 Agent(toolkit=...) 传入；ReAct 循环默认开启（max_iters=20），Agent 会在
          推理过程中自动调用工具并继续。
        - 权限系统：PermissionMode.BYPASS=跳过权限检查（适合本地可信自用，工具自带
          .env/.ssh 等 dangerous_files 保护仍生效）；DONT_ASK=所有请求转 DENY（会卡住）。
          权限通过 AgentState(permission_context=...) 传入。
        """
        if UserMsg is None or TextBlock is None:  # 无 agentscope 消息原语，直接降级
            raise RuntimeError("agentscope.message 不可用")

        key = os.environ.get("DEEPSEEK_API_KEY", "")
        from agentscope.credential import DeepSeekCredential
        from agentscope.model import DeepSeekChatModel
        from agentscope.agent import Agent
        from agentscope.state import AgentState
        from agentscope.tool import Toolkit, Read, Write, Edit, Bash, Glob, Grep
        from agentscope.permission import PermissionContext, PermissionMode

        # 与项目既有 model.deepseek 适配器对齐：deepseek-v4-flash / https://api.deepseek.com
        model = DeepSeekChatModel(
            credential=DeepSeekCredential(api_key=key),
            model="deepseek-v4-flash",
        )

        # 项目根目录：Bash 工具的默认工作目录，限制命令跑在项目内
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def _make_toolkit() -> Toolkit:
            return Toolkit([
                Read(),
                Write(),
                Edit(),
                Glob(),
                Grep(),
                Bash(cwd=root_dir),
            ])

        def _make_state() -> AgentState:
            # BYPASS：本地可信自用（研究者本人），工具级 dangerous_files 保护仍生效
            return AgentState(
                permission_context=PermissionContext(mode=PermissionMode.BYPASS),
            )

        tool_hint = (
            "你配备了真实工具：Read/Write/Edit（读写改文件）、Glob/Grep（搜索）、"
            "Bash（执行命令，工作目录为项目根）。当用户要求查看/修改代码、生成文件、"
            "执行命令、运行脚本时，必须真实调用对应工具并依据工具返回结果回答，"
            "不要假装完成。普通记忆问答直接回答即可，无需调用工具。"
        )

        # 主线 Agent：共享池记忆作为上下文注入系统提示
        self._agents["main"] = Agent(
            name="MainAgent",
            system_prompt=(
                "你是多智能体记忆共享调度系统的【主线智能体】。你拥有常驻完整记忆池的"
                "全量只读上下文，并会收到调度层从共享池精选注入的高相关记忆。请仅基于"
                "这些记忆简洁作答，不要编造记忆中没有的信息；记忆不足时如实说明。\n"
                + tool_hint
            ),
            model=model,
            toolkit=_make_toolkit(),
            state=_make_state(),
        )
        # 副线 Agent：各自独立实例 = 逻辑隔离沙箱（D3），只吃共享池记忆
        self._agents["sub1"] = Agent(
            name="SubAgent-sub1",
            system_prompt=(
                "你是【检索副线】智能体，运行在隔离 workspace。你只看到调度层注入的共享池"
                "精选记忆，看不到主线全量上下文。请基于这些记忆生成一条补充侧摘（≤2 句）。\n"
                + tool_hint
            ),
            model=model,
            toolkit=_make_toolkit(),
            state=_make_state(),
        )
        self._agents["reviewer"] = Agent(
            name="SubAgent-reviewer",
            system_prompt=(
                "你是【校验副线】智能体，运行在隔离 workspace。你只看到调度层注入的共享池"
                "精选记忆与主线回答。请基于记忆对主线回答做一致性校验（≤2 句），指出可能的偏差。\n"
                + tool_hint
            ),
            model=model,
            toolkit=_make_toolkit(),
            state=_make_state(),
        )

    # ------------------------------------------------------------------ #
    # 主线回答
    # ------------------------------------------------------------------ #
    def main_answer(self, query: str, kept_texts: list[str]) -> str:
        """主线智能体基于共享池记忆作答，返回完整文本。"""
        ctx = "\n".join(f"- {t}" for t in kept_texts[:8]) or "（本轮无保留记忆）"
        if self.mode == "agentscope":
            try:
                agent = self._agents["main"]
                prompt = (
                    f"【共享池精选记忆】\n{ctx}\n\n"
                    f"【用户问题】{query}\n请基于以上记忆回答。"
                )
                # 2.0.6 的 Agent.reply 是 async 方法，需在事件循环里 await
                msg = asyncio.run(agent.reply(UserMsg(name="User", content=prompt)))
                text = _extract_text(msg)
                if text.strip():
                    return text.strip()
            except Exception as e:  # noqa: BLE001 - AgentScope 调用失败降级
                self.mode = "headless"
        # headless 路径：复用中间件已装配的真实 LLM 适配器（DeepSeekAdapter.complete）
        client = _get_llm_client(self.mid)
        if client:
            prompt = (
                f"【共享池精选记忆】\n{ctx}\n\n"
                f"【用户问题】{query}\n请基于以上记忆回答。"
            )
            try:
                text = client.complete(prompt, max_tokens=512, temperature=0.3, thinking=False)
                return text.strip() if isinstance(text, str) else str(text)
            except Exception:  # noqa: BLE001
                return "(LLM 调用失败)"
        return "(未配置 LLM，跳过回答生成)"

    # ------------------------------------------------------------------ #
    # 副线（书签 + 可选真实副线 Agent）
    # ------------------------------------------------------------------ #
    def sub_lines(self, query: str, kept_texts: list[str],
                  main_answer: str, run_sub: bool) -> list[dict]:
        """返回副线列表：每个含 name/role/desc/injected(共享记忆)/note(可选侧任务产物)。

        run_sub=False 时只产出架构书签（展示「副线吃共享池」的运作，不耗 token）；
        run_sub=True 且有真实 LLM/Agent 时，额外调用副线 Agent 生成 note。
        """
        items: list[dict] = []
        ctx = "\n".join(f"- {t}" for t in kept_texts[:8]) or "（本轮无保留记忆）"
        client = _get_llm_client(self.mid)
        for name, role, desc in SUB_LINE_SPECS:
            entry = {
                "name": name, "role": role, "desc": desc,
                "injected_count": len(kept_texts[:8]),
                "injected": kept_texts[:8],
                "note": None,
            }
            if run_sub:
                note = self._run_sub_agent(name, ctx, query, main_answer)
                if note is None and client:  # headless 兜底：用适配器生成
                    note = _sub_note_headless(client, name, ctx, query, main_answer)
                entry["note"] = note
            items.append(entry)
        return items

    def _run_sub_agent(self, name: str, ctx: str, query: str, main_answer: str) -> Optional[str]:
        if self.mode != "agentscope" or name not in self._agents:
            return None
        try:
            agent = self._agents[name]
            if name == "reviewer":
                prompt = (
                    f"【共享池精选记忆】\n{ctx}\n\n"
                    f"【主线回答】{main_answer}\n请基于记忆校验主线回答一致性（≤2 句）。"
                )
            else:
                prompt = (
                    f"【共享池精选记忆】\n{ctx}\n\n"
                    f"【用户问题】{query}\n请基于记忆生成一条补充侧摘（≤2 句）。"
                )
            msg = asyncio.run(agent.reply(UserMsg(name="User", content=prompt)))
            text = _extract_text(msg)
            return text.strip() or None
        except Exception:  # noqa: BLE001 - 副线失败不影响主线
            return None


# ---------------------------------------------------------------------- #
# 复用的小工具（与 app_dashboard 解耦，避免循环依赖放在此处）
# ---------------------------------------------------------------------- #
def _extract_text(msg) -> str:
    """从 AgentScope 2.0.6 的 Msg（content 为 ContentBlock 列表）提取纯文本。"""
    if TextBlock is None:
        # 兜底：旧式 / 非 agentscope 消息
        content = getattr(msg, "content", "") or ""
        return content if isinstance(content, str) else str(content)
    parts: list[str] = []
    for block in getattr(msg, "content", None) or []:
        if isinstance(block, TextBlock):
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def _get_llm_client(mid):
    _prior_fn = getattr(mid.weights, "_llm_prior_fn", None)
    return getattr(_prior_fn, "client", None) if _prior_fn else None


def _sub_note_headless(client, name: str, ctx: str, query: str, main_answer: str) -> Optional[str]:
    if name == "reviewer":
        prompt = (
            "你是校验副线智能体。请仅基于下列共享池记忆，对主线回答做一致性校验（≤2 句），"
            "指出可能的偏差或遗漏。\n【共享池记忆】\n" + ctx +
            "\n\n【主线回答】" + main_answer
        )
    else:
        prompt = (
            "你是检索副线智能体。请仅基于下列共享池记忆，生成一条与主回答互补的补充侧摘"
            "（≤2 句），不要重复主回答已说的内容。\n【共享池记忆】\n" + ctx +
            "\n\n【用户问题】" + query
        )
    try:
        text = client.complete(prompt, max_tokens=160, temperature=0.3, thinking=False)
        return text.strip() or None
    except Exception:  # noqa: BLE001
        return None
