# -*- coding: utf-8 -*-
"""AgentScope 2.0.6 + ReMe 记忆模块最小示例

用途：
  1. 验证 AgentScope 2.0 的 ReMeMiddleware 能正确初始化（无需 LLM Key）
  2. 有 LLM API Key 时，演示完整流程：
     用户对话 → auto_memory 自动写入记忆 → memory_search 检索回忆

运行：
  conda activate srtp-memory
  python examples/reme_demo.py

说明：
  - 无 API Key 时只验证中间件初始化与 workspace 目录创建
  - 有 API Key 时（DASHSCOPE_API_KEY / OPENAI_API_KEY 任选），
    走完整对话 + 记忆写入 + 记忆检索流程
"""

import asyncio
import os
import sys

# 保证脚本可从项目根目录或 examples 目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentscope.middleware import ReMeMiddleware, TracingMiddleware
from agentscope.tool import Toolkit


def _pick_model():
    """根据环境变量选择可用的 LLM，未配置返回 None

    支持顺序：DeepSeek（最便宜，优先）→ DashScope → OpenAI
    """
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")

    if deepseek_key:
        from agentscope.credential import DeepSeekCredential
        from agentscope.model import DeepSeekChatModel
        return DeepSeekChatModel(
            credential=DeepSeekCredential(api_key=deepseek_key),
            model="deepseek-v4-flash",          # 2026-07-24 后 deepseek-chat 已弃用
            max_retries=1,                      # 降低成本：失败不反复重试
            parameters=DeepSeekChatModel.Parameters(
                max_tokens=200,                  # 控制成本：限制输出长度
                thinking_enable=False,           # 关闭思考模式，省钱
            ),
        ), "deepseek(v4-flash)"

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    if dashscope_key:
        from agentscope.model import DashScopeChatModel
        return DashScopeChatModel(
            model_name="qwen-max",
            api_key=dashscope_key,
        ), "dashscope(qwen-max)"

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from agentscope.model import OpenAIChatModel
        return OpenAIChatModel(
            model_name="gpt-4o-mini",
            api_key=openai_key,
        ), "openai(gpt-4o-mini)"
    return None, None


def main():
    # ── 0. 环境与依赖检查 ────────────────────────────────────────────
    print("=" * 60)
    print("AgentScope 2.0.6 + ReMe 记忆模块最小示例")
    print("=" * 60)
    print(f"[1] 检查 reme 依赖: ", end="")
    try:
        import reme  # noqa: F401
        print("OK")
    except ImportError as e:
        print(f"FAIL: {e}")
        print("    请先执行: pip install \"agentscope[memory-reme]\"")
        return

    # ── 1. 选择模型（DeepSeek 优先，最便宜） ─────────────────────────
    model, model_desc = _pick_model()

    # ── 2. 初始化 ReMeMiddleware ─────────────────────────────────────
    print("[2] 初始化 ReMeMiddleware ...")
    # workspace 指向项目文件夹下 data/reme，避免污染系统目录
    workdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reme")
    os.makedirs(workdir, exist_ok=True)

    mw = ReMeMiddleware(
        workspace_dir=workdir,
        parameters=ReMeMiddleware.Parameters(
            chat_model=model,  # 注入 LLM 给 ReMe 内部组件（auto_memory 蒸馏用）
            mode="both",       # 自动检索注入 + memory_search 工具
            top_k=5,
        ),
    )
    print(f"    workspace_dir = {workdir}")
    print(f"    mode = both, top_k = 5, chat_model = {model_desc}")

    # ── 3. 列出中间件提供的工具 ──────────────────────────────────────
    print("[3] 中间件提供的工具:", end=" ")
    tools = asyncio.run(mw.list_tools())
    print([t.name for t in tools] if tools else "(无，static_control 模式)")

    # ── 4. 检测 LLM API Key，决定运行模式 ────────────────────────────
    if model is None:
        print("\n[⚠] 未检测到 LLM API Key（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY）。")
        print("    已完成【中间件初始化】验证。")
        print("    如需完整记忆流程演示，请设置 API Key 后重跑：")
        print("      $env:DEEPSEEK_API_KEY = \"sk-xxx\"")
        print("      python examples/reme_demo.py")
        return

    print(f"\n[4] 使用模型: {model_desc}")
    print("    开始完整记忆流程演示 ...")

    # ── 5. 构建带记忆的 Agent ────────────────────────────────────────
    from agentscope.agent import Agent
    from agentscope.message import Msg, TextBlock

    agent = Agent(
        name="demo_assistant",
        system_prompt="你是一个友好的对话助手，会记住与用户的对话内容。",
        model=model,
        middlewares=[
            TracingMiddleware(),   # 全链路追踪（可选）
            mw,                    # ReMe 记忆中间件
        ],
        toolkit=Toolkit(tools=tools),  # memory_search 工具
    )

    def _user_msg(text: str) -> Msg:
        """构造用户消息（2.0 的 content 是 block 列表）"""
        return Msg(name="user", role="user", content=[TextBlock(text=text)])

    # ── 6. 第一轮对话：写入记忆 ───────────────────────────────────────
    print("\n[5] 第一轮对话（写入记忆）:")
    r1 = asyncio.run(agent.reply(_user_msg(
        "我叫张三，是一名信息管理专业的学生，正在做多智能体记忆调度课题。")))
    print(f"    Agent: {r1.get_text_content()[:80]}")

    print("\n[6] 第二轮对话（写入记忆）:")
    r2 = asyncio.run(agent.reply(_user_msg(
        "我们的技术栈确定用 AgentScope 2.0，向量库用 Chroma。")))
    print(f"    Agent: {r2.get_text_content()[:80]}")

    # ── 7. 第三轮对话：触发记忆检索 ───────────────────────────────────
    print("\n[7] 第三轮对话（触发记忆检索）:")
    r3 = asyncio.run(agent.reply(_user_msg(
        "你还记得我叫什么名字吗？我是学什么专业的？")))
    print(f"    Agent: {r3.get_text_content()[:120]}")

    # ── 8. 查看 workspace 产物 ────────────────────────────────────────
    print("\n[8] ReMe workspace 产物:")
    for root, dirs, files in os.walk(workdir):
        depth = root[len(workdir):].count(os.sep)
        if depth > 3:
            continue
        indent = "    " * depth
        print(f"{indent}{os.path.basename(root) or workdir}/")
        for f in sorted(files)[:8]:
            print(f"{indent}    {f}")

    print("\n✅ 演示完成")


if __name__ == "__main__":
    main()
