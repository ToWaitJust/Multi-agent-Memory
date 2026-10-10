"""控制台验证：空状态 → 选会话 → 真实发问 → 调度明细 → 暗色。

用法：
    E:/dev-tools/envs/webshot/Scripts/python.exe scripts/shot_dash.py [base_url] [outdir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

QUERY = "注意力机制怎么用？"


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/shots")
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(f"[{m.type}] {m.text}")
              if m.type == "error" else None)

        pg.goto(base + "/", wait_until="load")
        pg.wait_for_timeout(7000)
        pg.screenshot(path=str(outdir / "30_dash_empty.png"))
        print("OK 30_dash_empty")

        # 真实发问（选主节点会话后提交）
        pg.fill("#q", QUERY)
        pg.click("#send")
        pg.wait_for_timeout(1200)
        pg.screenshot(path=str(outdir / "31_dash_pending.png"))
        print("OK 31_dash_pending")

        pg.wait_for_selector("#detail .rank", timeout=60000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(outdir / "32_dash_answered.png"))
        print("OK 32_dash_answered")

        # 暗色
        pg.click("#themeBtn")
        pg.wait_for_timeout(1200)
        pg.screenshot(path=str(outdir / "33_dash_dark.png"))
        print("OK 33_dash_dark")

        print("\n".join(errs[:8]) if errs else "no console errors")
        pg.close()
        b.close()


if __name__ == "__main__":
    main()
