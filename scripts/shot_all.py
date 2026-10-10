"""全站回归：确认三页都无 JS 报错，并输出亮/暗两套六张图。

用法：
    E:/dev-tools/envs/webshot/Scripts/python.exe scripts/shot_all.py [base_url] [outdir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [("dash", "/"), ("canvas", "/canvas"), ("graph", "/graph")]


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/shots")
    outdir.mkdir(parents=True, exist_ok=True)
    bad = 0

    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, path in PAGES:
            ctx = b.new_context(viewport={"width": 1600, "height": 1000},
                                device_scale_factor=2)
            pg = ctx.new_page()
            errs: list[str] = []
            pg.on("pageerror", lambda e, n=name: errs.append(f"{n}: {e}"))
            pg.on("console", lambda m, n=name: errs.append(f"{n}: [{m.type}] {m.text}")
                  if m.type == "error" else None)
            pg.goto(base + path, wait_until="load")
            pg.wait_for_timeout(8000)
            pg.screenshot(path=str(outdir / f"4x_{name}_light.png"))
            # 切暗色（canvas/graph 的按钮是 svg，dashboard 的由 JS 注入）
            tb = pg.query_selector("#themeBtn")
            if tb:
                tb.click()
                pg.wait_for_timeout(1400)
                pg.screenshot(path=str(outdir / f"4x_{name}_dark.png"))
            status = "OK " if not errs else "ERR"
            if errs:
                bad += 1
            print(f"{status} {name}: {len(errs)} error(s)")
            for e in errs[:4]:
                print("    ", e)
            ctx.close()
        b.close()
    print("\n结论：" + ("三页零报错" if bad == 0 else f"{bad} 页有报错"))


if __name__ == "__main__":
    main()
