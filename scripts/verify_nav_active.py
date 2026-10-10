"""D-064 三页顶栏导航选中态 —— 回归校验（exit code 即通过与否）。

背景：canvas.html 的 <header> 漏了 class="topbar"，而 app.css 的导航样式
（含 .navlink.on 选中态）全部挂在 .topbar 下 → 画布页顶栏布局正常但**选中态永远不生效**。

断言：
  ① 三页都存在唯一一个 .navlink.on，且指向当前页；
  ② 画布页 .navlink.on 的计算样式（背景/文字/边框）与控制台页**一致**（证明走的是同一套 .topbar 规则）；
  ③ 暗色、亮色、高对比三套主题下均生效（不是只有默认主题碰巧对）。
"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8787"
PAGES = {"/": "/", "/canvas": "/canvas", "/graph": "/graph"}
PROPS = ["backgroundColor", "color", "borderTopColor", "fontWeight"]
fails = []


def ck(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def on_style(pg):
    return pg.evaluate("""(props) => {
        const a = document.querySelector('.navlink.on');
        if (!a) return null;
        const cs = getComputedStyle(a);
        const o = {href: a.getAttribute('href')};
        props.forEach(p => o[p] = cs[p]);
        return o;
    }""", PROPS)


def set_theme(pg, theme, contrast=None):
    pg.evaluate("""([t, c]) => {
        try {
            localStorage.setItem('srtp.theme', t);
            if (c) localStorage.setItem('srtp.contrast', c);
            else localStorage.removeItem('srtp.contrast');
        } catch (e) {}
    }""", [theme, contrast])
    pg.reload(wait_until="load")
    time.sleep(0.6)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))

        # ① + ② 默认（暗色）主题
        styles = {}
        for path, href in PAGES.items():
            pg.goto(BASE + path, wait_until="load")
            time.sleep(0.6)
            st = on_style(pg)
            ck(st is not None, f"① {path} 存在 .navlink.on")
            if not st:
                continue
            ck(st["href"] == href, f"① {path} 选中项指向自己（href={st['href']}）")
            styles[path] = st

        if "/canvas" in styles and "/" in styles:
            same = all(styles["/canvas"][k] == styles["/"][k] for k in PROPS)
            ck(same, f"② 画布页选中态与控制台页一致 {[styles['/canvas'][k] for k in PROPS[:3]]}")
            ck(styles["/canvas"]["backgroundColor"] not in ("rgba(0, 0, 0, 0)", "transparent"),
               f"② 画布页选中态有背景色 {styles['/canvas']['backgroundColor']}")

        # ③ 三套主题
        for theme, contrast in [("dark", None), ("light", None),
                                ("dark", "high"), ("light", "high")]:
            pg.goto(BASE + "/canvas", wait_until="load")
            set_theme(pg, theme, contrast)
            st = on_style(pg)
            tag = f"{theme}{'+hc' if contrast else ''}"
            ok = (st and st["backgroundColor"] not in ("rgba(0, 0, 0, 0)", "transparent")
                  and st["fontWeight"] == "600")
            ck(ok, f"③ 画布页 {tag} 选中态生效 {st and [st[k] for k in PROPS[:3]]}")

        ck(not errs, f"无 JS 运行时错误 {errs[:2]}")
        b.close()

    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
