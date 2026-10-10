"""D-063 画布拖拽布局持久化 —— 回归校验（exit code 即通过与否）。

三条断言：
  ① 持久化：拖拽 → 刷新页面，偏移仍在（且落盘到 data/demo/canvas_history.json）；
  ② 钳制：往左上猛拖，节点坐标不得 <8（否则落到 viewBox 外 = 节点消失）；
  ③ 重置：点「恢复默认布局」→ 偏移清空、坐标回到默认值。
另：保存成功必须有可见提示（原来是 .catch() 静默吞掉失败）。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8787"
HIST = "data/demo/canvas_history.json"
SEL = 'g.node[data-sid="canvas_main"][data-rid="R2"]'

fails = []


def ck(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def api_reset():
    req = urllib.request.Request(
        BASE + "/api/canvas/layout",
        data=json.dumps({"reset": True}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=8).read()


def pos(pg):
    return pg.eval_on_selector(SEL + " rect",
                               "el => ({x:+el.getAttribute('x'), y:+el.getAttribute('y')})")


def center(pg):
    b = pg.locator(SEL + " rect").bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def drag(pg, dx, dy, steps=10):
    cx, cy = center(pg)
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    pg.mouse.move(cx + dx, cy + dy, steps=steps)
    pg.mouse.up()
    time.sleep(0.45)


def load(pg):
    pg.goto(BASE + "/canvas", wait_until="load")
    pg.wait_for_selector("g.node", timeout=15000)
    time.sleep(0.5)


def main():
    api_reset()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("dialog", lambda d: d.accept())
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))

        load(pg)
        base = pos(pg)

        # ① 持久化
        drag(pg, 120, 40)
        after = pos(pg)
        ck(after["x"] == base["x"] + 120 and after["y"] == base["y"] + 40,
           f"① 拖拽生效 {base} -> {after}")
        toast = pg.eval_on_selector("#layoutToast", "el => el.textContent") \
            if pg.query_selector("#layoutToast") else ""
        ck("已保存" in (toast or ""), f"① 保存有可见提示（toast={toast!r}）")

        pg.reload(wait_until="load")
        pg.wait_for_selector("g.node", timeout=15000)
        time.sleep(0.5)
        reloaded = pos(pg)
        ck(reloaded == after, f"① 刷新后仍在 {reloaded}")

        # ② 钳制（往左上猛拖，不能跑出 viewBox）
        drag(pg, -3000, -3000)
        cl = pos(pg)
        ck(cl["x"] >= 8 and cl["y"] >= 8,
           f"② 拖到左上被钳制在画布内 {cl}（不得 <8，否则节点消失）")
        inner = pg.evaluate("""() => {
            const s = document.getElementById('cv');
            const vb = s.getAttribute('viewBox').split(' ').map(Number);
            return {W: vb[2], H: vb[3]};
        }""")
        ck(cl["x"] + 152 <= inner["W"] and cl["y"] + 50 <= inner["H"],
           f"② 仍在 viewBox 内 viewBox={inner}")

        # ③ 重置
        pg.click("#resetLayoutBtn")
        time.sleep(0.8)
        rst = pos(pg)
        ck(rst == base, f"③ 重置回到默认 {rst}（默认 {base}）")
        pg.reload(wait_until="load")
        pg.wait_for_selector("g.node", timeout=15000)
        time.sleep(0.5)
        ck(pos(pg) == base, f"③ 重置后刷新仍是默认 {pos(pg)}")

        ck(not errs, f"无 JS 运行时错误 {errs[:2]}")
        b.close()

    lay = json.load(open(HIST, encoding="utf-8")).get("layout") or {}
    ck(lay == {}, f"落盘 layout 已清空 {lay}")

    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
