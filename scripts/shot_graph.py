"""图页验证：走深链 ?node=&q= 自动跑选源，截「动效后」与「暗色」两张图。

用法：
    E:/dev-tools/envs/webshot/Scripts/python.exe scripts/shot_graph.py [base_url] [outdir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# 深链：page 载入即选中节点并跑 Stage A，等价于用户「点节点 → 点计算选源」
NODE = "n_deploy"
QUERY = "注意力机制怎么用"


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/shots")
    outdir.mkdir(parents=True, exist_ok=True)
    url = f"{base}/graph?node={NODE}&q={QUERY}"

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        pg.goto(url, wait_until="load")
        pg.wait_for_timeout(9000)
        pg.screenshot(path=str(outdir / "20_graph_stageA.png"))
        print("OK 20_graph_stageA")

        # 暗色
        pg.click("#themeBtn")
        pg.wait_for_timeout(1400)
        pg.screenshot(path=str(outdir / "21_graph_dark.png"))
        print("OK 21_graph_dark")

        # 亮色 + 环形布局（看缓动后落位是否铺满）
        pg.click("#themeBtn")
        pg.wait_for_timeout(900)
        pg.click("button[data-layout='circle']")
        pg.wait_for_timeout(2600)
        pg.screenshot(path=str(outdir / "22_graph_circle.png"))
        print("OK 22_graph_circle")

        # 隐藏 similar_to 边（降噪）
        pg.click("button[data-layout='breadthfirst']")
        pg.wait_for_timeout(2200)
        row = pg.query_selector(".map-legend .row[data-t='similar_to']")
        if row:
            row.click()
            pg.wait_for_timeout(900)
            pg.screenshot(path=str(outdir / "23_graph_nonoise.png"))
            print("OK 23_graph_nonoise")
        pg.close()
        b.close()


if __name__ == "__main__":
    main()
