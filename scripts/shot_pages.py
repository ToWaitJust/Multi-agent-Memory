"""产品线视觉评估：对三页各截一张图，供人工比对。

用法：
    E:/dev-tools/envs/webshot/Scripts/python.exe scripts/shot_pages.py [base_url] [outdir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [
    ("01_dashboard", "/", 9000),
    ("02_canvas", "/canvas", 9000),
    ("03_graph", "/graph", 11000),
]


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/shots")
    outdir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, path, wait in PAGES:
            pg = b.new_page(viewport={"width": 1600, "height": 1000},
                            device_scale_factor=2)
            pg.goto(base + path, wait_until="load")
            pg.wait_for_timeout(wait)
            f = outdir / f"{name}.png"
            pg.screenshot(path=str(f))
            print(f"OK {f}")
            pg.close()
        b.close()


if __name__ == "__main__":
    main()
