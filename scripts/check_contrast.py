"""D-061验证脚本：在真实浏览器里量测四套主题的对比度。

为什么必须实测：令牌值写对≠ 页面对。玻璃层有 alpha 叠加、
子样式表有映射、组件有硬编码色 —— 只有getComputedStyle 读到
最终渲染值才算数。

产出 data/shots/contrast_report.json + 三页 × 四态截图。
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = "http://127.0.0.1:8787"
OUT = os.path.join(ROOT, "data", "shots")

# 要检查的令牌：(CSS 变量名, 是否正文级[True=需>=4.5])
TOKENS = [
    ("--ink", True), ("--ink-2", True), ("--sub", True), ("--mute", True),
    ("--brand", True), ("--accent", True), ("--e-similar", False),
]

PAGES = [("/", "dash"), ("/canvas", "canvas"), ("/graph", "graph")]

JS_MEASURE = """
() => {
  const cs = getComputedStyle(document.documentElement);
  const cv = n => (cs.getPropertyValue(n) || '').trim();
  // 逐层解析 rgba() 叠到不透明底色上，算出真实呈现色
  const parse = s => {
    const m = String(s).match(/rgba?\\(([^)]+)\\)/);
    if (!m) return null;
    const p = m[1].split(/[,\\s\\/]+/).filter(Boolean).map(Number);
    return {r:p[0], g:p[1], b:p[2], a: p.length > 3 ? p[3] : 1};
  };
  const over = (fg, bg) => ({
    r: Math.round(fg.r*fg.a + bg.r*(1-fg.a)),
    g: Math.round(fg.g*fg.a + bg.g*(1-fg.a)),
    b: Math.round(fg.b*fg.a + bg.b*(1-fg.a)), a:1});
  const hex = c => '#' + [c.r,c.g,c.b].map(v => v.toString(16).padStart(2,'0')).join('');
  const bodyBg = parse(getComputedStyle(document.body).backgroundColor)
                || {r:255,g:255,b:255,a:1};
  // 卡片实际底色（body 背景之上）
  const el = document.querySelector('.card, .side, .glass');
  let cardBg = bodyBg;
  if (el) {
    const cs2 = getComputedStyle(el);
    const bg = parse(cs2.backgroundColor);
    if (bg) cardBg = over(bg, bodyBg);
  }
  const out = {tokens:{}, theme:document.documentElement.getAttribute('data-theme'),
               contrast:document.documentElement.getAttribute('data-contrast'),
               cardHex: hex(cardBg)};
  for (const n of TOKENS_JS) {
    const raw = cv(n);
    const p = parse(raw);
    out.tokens[n] = p ? hex(over(p, cardBg)) : (raw || '');
  }
  return out;
}
"""


RENDER_PROBE = r"""() => {
  const why = [];
  const txt = document.body.innerText || '';
  if (txt.includes('载入失败')) why.push('出现"载入失败"');
  if (txt.includes('undefined') && txt.includes('is not defined')) why.push('JS 常量未定义');
  const cy = window.__srtpCy;
  let cyOk = null;
  let cyStale = null;
  if (cy) {
    const n = cy.nodes().length, e = cy.edges().length;
    // 节点必须有实际尺寸与标签，否则等于没渲染
    const s = n ? cy.nodes()[0].style('background-color') : null;
    const lb = n ? cy.nodes()[0].style('label') : null;
    if (!n) why.push('图无节点');
    else if (!s) why.push('样式表未生效(background-color 为空)');
    else if (!lb) why.push('节点无标签');
    cyOk = (n > 0 && e > 0 && !!s);

    /* D-061 最关键的一条：图配色必须**跟着主题走**。
       用 renderedStyle（画布真实渲染值）比对，不能用 style（样式表计算值）——
       样式表重建失败时后者仍返回旧值，看着"有值"其实没换。 */
    const th = document.documentElement.getAttribute('data-theme');
    const hi = document.documentElement.getAttribute('data-contrast') === 'high';
    const cs = getComputedStyle(document.documentElement);
    const want = (cs.getPropertyValue('--card-solid') || '').trim();
    const got = cy.nodes()[0].renderedStyle('background-color');
    const toRgb = s => {
      /* 不用正则：Python raw string 里的 \\d 传到 JS 会变成字面反斜杠，
         正则直接语法错误（实测踩过）。split + map 更稳。 */
      const t = String(s).replace(/[^0-9,.]/g, '');
      const p = t.split(',').map(Number);
      return p.length >= 3 ? 'rgb(' + p[0] + ', ' + p[1] + ', ' + p[2] + ')' : '';
    };
    const w = toRgb(want), g = toRgb(got);
    // 高对比亮色 card-solid 是纯白；暗色是 #141a2a / #0d1220
    if (w && g && w !== g) {
      cyStale = '图底色 ' + g + ' != 主题 card-solid ' + w;
      why.push('图配色未跟随主题(' + th + (hi ? '/high' : '') + ')');
    }
  }
  return {ok: why.length === 0, why: why.join('; '), cyOk: cyOk, cyStale: cyStale,
          nodes: cy ? cy.nodes().length : 0,
          edges: cy ? cy.edges().length : 0,
          theme: document.documentElement.getAttribute('data-theme'),
          contrast: document.documentElement.getAttribute('data-contrast') || 'normal'};
}"""


def run() -> int:
    from playwright.sync_api import sync_playwright

    os.makedirs(OUT, exist_ok=True)
    token_names = [t[0] for t in TOKENS]
    measure = JS_MEASURE.replace("TOKENS_JS", json.dumps(token_names))

    report = {}
    errors = []

    def lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    def lum(h):
        h = h.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

    def cr(a, b):
        l1, l2 = lum(a), lum(b)
        hi, lo = max(l1, l2), min(l1, l2)
        return (hi + 0.05) / (lo + 0.05)

    with sync_playwright() as pw:
        br = pw.chromium.launch()
        for path, tag in PAGES:
            for theme in ("light", "dark"):
                for contrast in ("normal", "high"):
                    key = f"{tag}/{theme}/{contrast}"
                    pg = br.new_page(viewport={"width": 1560, "height": 900})
                    errs = []
                    pg.on("pageerror", lambda e: errs.append(str(e)))
                    pg.on("console", lambda m: errs.append(m.text)
                          if m.type == "error" else None)
                    try:
                        pg.goto(BASE + path, wait_until="networkidle", timeout=45000)
                        #走**真实 UI 路径**：点按钮而不是直接改属性。
                        # 直接 setAttribute 跳过了按钮的 JS 绑定，
                        # 图拓扑不会重建样式表 → 截到"主题变了但图没变"的过渡态。
                        if theme == "light":
                            pg.evaluate("() => window.SRTPTheme.apply('light')")
                            pg.wait_for_timeout(700)
                        if contrast == "high":
                            pg.evaluate("() => window.SRTPTheme.setContrast(true)")
                            pg.wait_for_timeout(700)
                        pg.wait_for_timeout(1400)
                        # ⚠️ D-061：图页是 cose 力导向布局，**渐进收敛**，
                        #  1.4s 时还在跑 → 截图会拍到空白/半散状态，
                        #  曾据此误判"图崩了"。必须等布局停。
                        if tag == "graph":
                            try:
                                pg.wait_for_function(
                                    "() => window.__srtpCy && "
                                    "window.__srtpCy.nodes()[0] && "
                                    "window.__srtpCy.nodes()[0].position().x !== 0",
                                    timeout=20000)
                            except Exception:  # noqa: BLE001
                                pass
                            pg.wait_for_timeout(3200)
                            # 终检：图是否真的跟着主题走了。
                            # 断言用 **renderedStyle**（画布实际渲染值）而非 style
                            # （样式表计算值）—— 后者在重建失败时仍会返回旧值，
                            # 看着"有值"其实没生效。
                            pg.evaluate("() => window.SRTPTheme.apply("
                                        + json.dumps(theme) + ")")
                            pg.wait_for_timeout(300)
                            if contrast == "high":
                                pg.evaluate("() => window.SRTPTheme.setContrast(true)")
                            pg.wait_for_timeout(1100)
                        data = pg.evaluate(measure)
                        # ---- 渲染断言：D-061 教训"对比度全绿但整页崩溃" ----
                        # 只量令牌不算数，必须确认页面真的画出来了。
                        probe = pg.evaluate(RENDER_PROBE)
                        data["render"] = probe
                        if not probe.get("ok"):
                            data["render_error"] = probe.get("why", "")
                        data["page_errors"] = errs
                        report[key] = data
                        shot = os.path.join(OUT, f"contrast_{tag}_{theme}_{contrast}.png")
                        pg.screenshot(path=shot, full_page=False)
                        if errs:
                            errors.extend([f"{key}: {e}" for e in errs])
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{key}: {type(e).__name__}: {e}")
                        report[key] = {"error": str(e)[:200]}
                    finally:
                        pg.close()
        br.close()

    # ---- 判定 ----
    print("=" * 78)
    print("D-061 真实浏览器对比度实测")
    print("=" * 78)
    bad = []
    for key, d in report.items():
        if "error" in d:
            print(f"\n[{key}] 页面错误: {d['error'][:100]}")
            continue
        theme = d["theme"] + ("/high" if d["contrast"] == "high" else "")
        need = 7.0 if d["contrast"] == "high" else 4.5
        print(f"\n[{key}]  card底色={d['cardHex']}  阈值>={need}")
        for name, is_body in TOKENS:
            hexv = d["tokens"].get(name, "")
            if not hexv.startswith("#") or len(hexv) != 7:
                print(f"    {name:<12} {hexv:<9} ⚠ 未取到")
                bad.append((key, name, 0, need))
                continue
            r = cr(hexv, d["cardHex"])
            # 非正文级（装饰性边线）用 3.0 阈值
            lim = need if is_body else min(need, 3.0)
            flag = "OK" if r >= lim else "FAIL"
            if r < lim:
                bad.append((key, name, r, lim))
            print(f"    {name:<12} {hexv:<9} {r:6.2f}  {flag}")

    print("\n" + "=" * 78)
    render_bad = [k for k, d in report.items()
                  if "error" not in d and not d.get("render", {}).get("ok", True)]
    if render_bad:
        print(f"❌ {len(render_bad)} 个页面渲染异常：")
        for k in render_bad:
            print(f"    {k}: {report[k].get('render_error')}")
    else:
        print("✅ 所有页面渲染正常（图有节点/边、样式生效、无载入失败）")

    if bad:
        print(f"\n❌ {len(bad)} 项对比度不达标：")
        for k, n, r, l in bad:
            print(f"    {k}  {n}  {r:.2f} < {l}")
    else:
        print("✅ 四套主题（亮/暗 × 普通/高对比）全部达标")

    if errors:
        print(f"\n⚠ 页面报错 {len(errors)} 条：")
        for e in errors[:12]:
            print("   ", e[:160])

    rep = os.path.join(OUT, "contrast_report.json")
    with open(rep, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {rep}")
    return 1 if (bad or render_bad) else 0


if __name__ == "__main__":
    sys.exit(run())
