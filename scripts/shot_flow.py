"""产品线视觉评估（交互态）：走一遍真实使用流程再截图。

对应涂总重点关注的两个画面：
  · 对话框窗口（控制台发问后的三栏 + 会话画布）
  · 图结构可放缩可编辑（选源后的拓扑 + 侧栏）

用法：
    E:/dev-tools/envs/webshot/Scripts/python.exe scripts/shot_flow.py [base_url] [outdir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

QUERY = "我上周说要改的那个数据库配置，后来怎么定的？"


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/shots")
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        b = p.chromium.launch()

        def new_page():
            # 每个页面独立 context：服务端是单线程 HTTP server，
            # 一次交互（如发问触发后端调度）会阻塞其它请求，必须隔离。
            return b.new_context(viewport={"width": 1600, "height": 1000},
                                 device_scale_factor=2).new_page()

        # ---- 图拓扑：计算选源（纯 GET，不会阻塞）----
        pg = new_page()
        pg.goto(base + "/graph", wait_until="load")
        pg.wait_for_timeout(8000)
        gq = pg.query_selector("#q")
        if gq:
            gq.fill(QUERY)
            pg.wait_for_timeout(200)
            btn = pg.query_selector("#saBtn")
            if btn:
                btn.click()
                pg.wait_for_timeout(9000)
        pg.screenshot(path=str(outdir / "13_graph_selected.png"))
        print("OK 13_graph_selected")

        cbtn = pg.query_selector("button[data-layout='circle']")
        if cbtn:
            cbtn.click()
            pg.wait_for_timeout(2500)
            pg.screenshot(path=str(outdir / "14_graph_circle.png"))
            print("OK 14_graph_circle")
        pg.close()

        # ---- 会话画布 ----
        pg = new_page()
        pg.goto(base + "/canvas", wait_until="load")
        pg.wait_for_timeout(9000)
        pg.screenshot(path=str(outdir / "12_canvas_filled.png"))
        print("OK 12_canvas_filled")
        pg.close()

        # ---- 控制台：发问 → 等回复（放最后，会阻塞服务端）----
        pg = new_page()
        pg.goto(base + "/", wait_until="load")
        pg.wait_for_timeout(6000)
        box = pg.query_selector("#q, #query, textarea, input[type=text]")
        if box:
            box.fill(QUERY)
            pg.wait_for_timeout(300)
            for sel in ("#send", "button:has-text('发送')", "button:has-text('送')"):
                btn = pg.query_selector(sel)
                if btn:
                    btn.click()
                    break
            pg.wait_for_timeout(15000)
        pg.screenshot(path=str(outdir / "11_dashboard_filled.png"))
        print("OK 11_dashboard_filled")
        pg.close()
        b.close()


if __name__ == "__main__":
    main()
