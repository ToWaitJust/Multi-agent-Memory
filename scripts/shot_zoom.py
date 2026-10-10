"""图页局部放大：按「选中源 → 当前节点」的真实屏幕坐标裁剪，
用来看清虚线与箭头尖端的衔接是否错位。

用法：
    E:/dev-tools/envs/webshot/Scripts/python.exe scripts/shot_zoom.py [base_url] [outdir]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

NODE = "n_deploy"
QUERY = "注意力机制怎么用"


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/shots")
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=3)
        pg.goto(f"{base}/graph?node={NODE}&q={QUERY}", wait_until="load")
        pg.wait_for_timeout(9000)

        # 取「当前节点 + 所有入选源」的屏幕坐标并集
        clip = pg.evaluate("""() => {
          const cy = window.__srtpCy;
          if (!cy) return null;
          const ids = ['n_deploy'].concat(
            cy.nodes('.src').map(n => n.id()));
          let x1=1e9,y1=1e9,x2=-1e9,y2=-1e9;
          ids.forEach(id=>{
            const n = cy.getElementById(id);
            if(!n || n.empty()) return;
            const b = n.renderedBoundingBox();
            x1=Math.min(x1,b.x1); y1=Math.min(y1,b.y1);
            x2=Math.max(x2,b.x2); y2=Math.max(y2,b.y2);
          });
          const pad = 90;
          return {x:Math.max(0,x1-pad), y:Math.max(0,y1-pad),
                  width:Math.min(innerWidth, x2-x1+pad*2),
                  height:Math.min(innerHeight, y2-y1+pad*2),
                  sources: cy.nodes('.src').map(n=>n.id())};
        }""")
        if not clip:
            print("!! 拿不到 cy 实例")
            pg.close(); b.close(); return
        print("入选源:", clip.pop("sources"), "| 裁剪区:", {k: round(v) for k, v in clip.items()})
        pg.screenshot(path=str(outdir / "50_zoom_flow.png"), clip=clip)
        print("OK 50_zoom_flow")
        pg.close()
        b.close()


if __name__ == "__main__":
    main()