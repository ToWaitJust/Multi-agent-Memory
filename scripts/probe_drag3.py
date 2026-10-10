"""D-063 第三轮：只跑「拖拽 → 点节点进控制台 → 返回画布」往返，带事件日志。"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8787"
SEL = 'g.node[data-sid="canvas_main"][data-rid="R2"]'


def x_of(pg):
    return pg.eval_on_selector(SEL + " rect", "el => +el.getAttribute('x')")


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append("pageerror: " + str(e)))
        pg.on("framenavigated", lambda f: logs.append("nav -> " + f.url[:90]))

        pg.goto(BASE + "/canvas", wait_until="load")
        pg.wait_for_selector("g.node", timeout=15000)
        time.sleep(0.5)

        x0 = x_of(pg)
        box = pg.locator(SEL + " rect").bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        print("drag start at", round(cx), round(cy))

        pg.mouse.move(cx, cy)
        pg.mouse.down()
        pg.mouse.move(cx + 80, cy + 25, steps=8)
        pg.mouse.up()
        time.sleep(0.5)
        print(f"拖拽: {x0} -> {x_of(pg)}")

        # 在节点当前位置单击
        box2 = pg.locator(SEL + " rect").bounding_box()
        nx, ny = box2["x"] + box2["width"] / 2, box2["y"] + box2["height"] / 2
        print("click at", round(nx), round(ny))
        pg.mouse.move(nx, ny)
        pg.mouse.down()
        pg.mouse.up()
        time.sleep(1.5)
        print("url after click:", pg.url)

        if "/canvas" in pg.url:
            pg.go_back(wait_until="load")
        pg.wait_for_selector("g.node", timeout=15000)
        time.sleep(0.8)
        print(f"返回后 x={x_of(pg)}  url={pg.url[:70]}")

        print("\n--- events ---")
        for l in logs[:25]:
            print(" ", l)
        b.close()


if __name__ == "__main__":
    sys.exit(main())
