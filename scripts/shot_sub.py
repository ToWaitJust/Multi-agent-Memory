"""下游节点创建全流程验证（D-055）：走 SSE 路径，截「流式过渡动画」与「进入下游节点」两张图。

用法：
    E:/dev-tools/envs/webshot/Scripts/python.exe scripts/shot_sub.py [base_url] [outdir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

QUERY = "下游节点要展开哪些检索增强的工作？"


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/shots")
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(f"[{m.type}] {m.text}")
              if m.type == "error" else None)

        pg.goto(base + "/", wait_until="load")
        pg.wait_for_timeout(7000)

        # 下游节点创建入口只在「主节点回合」下出现
        pg.fill("#nsTitle", "检索增强下游节点")
        pg.fill("#nsQuery", QUERY)
        pg.click("#nsGo")

        # 过渡动画期间（流式回答正在推）
        pg.wait_for_selector("#flowov.show", timeout=30000)
        pg.wait_for_timeout(700)
        pg.screenshot(path=str(outdir / "40_sub_flow.png"))
        print("OK 40_sub_flow")
        print("  过渡层可见:", pg.eval_on_selector("#flowov", "e=>e.classList.contains('show')"))
        print("  已点亮步骤:", pg.eval_on_selector_all(
            "#flowov .fstep.on .t", "es=>es.map(e=>e.textContent)"))
        print("  第三步文案:", (pg.inner_text("#fs3d") or "")[:60])

        # 动画结束 → 进入下游节点
        pg.wait_for_timeout(6000)
        pg.screenshot(path=str(outdir / "41_sub_console.png"))
        print("OK 41_sub_console")
        print("  当前会话:", pg.eval_on_selector("#sessbar .stab.active", "e=>e.textContent"))
        print("  消息气泡数:", pg.eval_on_selector_all("#messages .msg", "es=>es.length"))
        print("  首卡(用户输入)文案:", pg.inner_text("#stepper .node[data-s='input'] .d"))

        print("\n".join(errs[:6]) if errs else "  no console errors")
        pg.close()
        b.close()


if __name__ == "__main__":
    main()
