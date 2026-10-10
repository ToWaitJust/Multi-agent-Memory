"""D-063 第二轮：覆盖真实操作路径，找出"复原"发生在哪一步。"""
from __future__ import annotations

import json
import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8787"
SEL = 'g.node[data-sid="canvas_main"][data-rid="R2"]'


def x_of(pg, sel=SEL):
    return pg.eval_on_selector(sel + " rect", "el => +el.getAttribute('x')")


def y_of(pg, sel=SEL):
    return pg.eval_on_selector(sel + " rect", "el => +el.getAttribute('y')")


def center(pg, sel=SEL):
    b = pg.locator(sel + " rect").bounding_box()
    return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2


def fresh(pg):
    pg.goto(BASE + "/canvas", wait_until="load")
    pg.wait_for_selector("g.node", timeout=15000)
    time.sleep(0.5)


def case_single_move(pg):
    """① 只发一个 mousemove（快速甩拖）"""
    fresh(pg)
    x0 = x_of(pg)
    cx, cy = center(pg)
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    pg.mouse.move(cx + 90, cy + 30)
    pg.mouse.up()
    time.sleep(0.4)
    print(f"① 一次move: {x0} -> {x_of(pg)}")


def case_up_outside(pg):
    """② 拖到画布外/窗口边缘松手"""
    fresh(pg)
    x0 = x_of(pg)
    cx, cy = center(pg)
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    pg.mouse.move(cx + 60, cy + 200, steps=8)
    pg.mouse.move(1439, 899, steps=4)
    pg.mouse.up()
    time.sleep(0.4)
    print(f"② 边缘松手: {x0} -> {x_of(pg)}")


def case_scrolled(pg):
    """③ 先滚动 stage 再拖"""
    fresh(pg)
    pg.eval_on_selector("#stage", "el => { el.scrollLeft = 300; el.scrollTop = 120; }")
    time.sleep(0.3)
    x0 = x_of(pg)
    cx, cy = center(pg)
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    pg.mouse.move(cx + 70, cy + 20, steps=6)
    pg.mouse.up()
    time.sleep(0.4)
    print(f"③ 滚动后拖: {x0} -> {x_of(pg)}")


def case_nav_roundtrip(pg):
    """④ 拖完 → 点节点进控制台 → 浏览器返回画布"""
    fresh(pg)
    x0 = x_of(pg)
    cx, cy = center(pg)
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    pg.mouse.move(cx + 80, cy + 25, steps=8)
    pg.mouse.up()
    time.sleep(0.5)
    x1 = x_of(pg)
    # 单击该节点 → 进控制台
    pg.mouse.move(cx + 80, cy + 25)
    pg.mouse.down()
    pg.mouse.up()
    time.sleep(1.2)
    print(f"   ④ 已进入: {pg.url[:70]}")
    pg.go_back(wait_until="load")
    pg.wait_for_selector("g.node", timeout=15000)
    time.sleep(0.8)
    print(f"④ 往返控制台: 拖前{x0} 拖后{x1} 返回后{x_of(pg)}  url={pg.url[:60]}")


def case_click_then_check(pg):
    """⑤ 拖完立刻再拖一次（验证 base 叠加是否正确，不是覆盖回原位）"""
    fresh(pg)
    x0 = x_of(pg)
    for k in range(2):
        cx, cy = center(pg)
        pg.mouse.move(cx, cy)
        pg.mouse.down()
        pg.mouse.move(cx + 40, cy + 10, steps=5)
        pg.mouse.up()
        time.sleep(0.5)
    print(f"⑤ 连续两次拖(+40 each): {x0} -> {x_of(pg)}  (期望 +80)")


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        case_single_move(pg)
        case_up_outside(pg)
        case_scrolled(pg)
        case_nav_roundtrip(pg)
        case_click_then_check(pg)
        print("pageerror:", errs[:5])
        b.close()


if __name__ == "__main__":
    sys.exit(main())
