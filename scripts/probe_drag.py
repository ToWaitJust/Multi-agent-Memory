"""D-063 诊断：画布节点拖拽后位置是否持久化。

链路：mousedown/mousemove/mouseup(前端) → POST /api/canvas/layout → 落盘 json → 重载页面。
逐步打印节点 x 坐标与网络回包，定位"复原"发生在哪一步。
"""
from __future__ import annotations

import json
import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8787"
SEL_MAIN = 'g.node[data-sid="canvas_main"][data-rid="R3"]'
SEL_SUB = 'g.node[data-sid="sub_retriever"][data-rid="S3-2"]'


def node_x(page, sel):
    """读节点 rect 的 x（SVG 用户坐标，与 layout() 同一坐标系）。"""
    return page.eval_on_selector(sel + " rect", "el => +el.getAttribute('x')")


def probe(page, sel, label):
    x0 = node_x(page, sel)
    box = page.locator(sel + " rect").bounding_box()
    if not box:
        print(f"[{label}] 找不到节点 {sel}")
        return
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    for i in range(1, 11):
        page.mouse.move(cx + 8 * i, cy + 4 * i)
        time.sleep(0.012)
    page.mouse.up()
    time.sleep(0.35)
    x1 = node_x(page, sel)
    time.sleep(1.2)
    x2 = node_x(page, sel)
    print(f"[{label}] 拖前 x={x0}  松手后 x={x1}  静置1.2s后 x={x2}")
    return x2


def main():
    logs, posts = [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))

        def on_resp(r):
            if "/api/canvas/layout" in r.url:
                try:
                    posts.append((r.status, r.text()[:200]))
                except Exception:
                    posts.append((r.status, "<no body>"))

        pg.on("response", on_resp)

        pg.goto(BASE + "/canvas", wait_until="load")
        pg.wait_for_selector("g.node", timeout=15000)
        time.sleep(0.6)

        probe(pg, SEL_MAIN, "主节点R3")
        probe(pg, SEL_SUB, "下游S3-2")

        # 重载：验证落盘是否生效
        pg.reload(wait_until="load")
        pg.wait_for_selector("g.node", timeout=15000)
        time.sleep(0.8)
        print(f"[重载后] 主节点R3 x={node_x(pg, SEL_MAIN)}  下游S3-2 x={node_x(pg, SEL_SUB)}")
        b.close()

    print("\n--- POST /api/canvas/layout 回包 ---")
    for st, body in posts:
        print(f"  HTTP {st}  {body}")
    print("\n--- console ---")
    for l in logs[:20]:
        print("  " + l)

    hist = json.load(open("data/demo/canvas_history.json", encoding="utf-8"))
    print("\n--- 文件内 layout ---")
    print(" ", json.dumps(hist.get("layout"), ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
