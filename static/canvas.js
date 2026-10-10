/* 会话画布（canvas）脚本 —— 迁移自 canvas.html 内联 <script> */

let H = null;
const NS = "http://www.w3.org/2000/svg";
const NW = 152, NH = 50, GAP = 186;          // 主节点（像素固定）
const SW = 134, SH = 46, SGAP = 160;         // 下游节点
const MAIN_Y = 52, LANE0 = 196, LANE_GAP = 116;
const NODE_POS = {};                          // session|round -> {x,y} 供平移定位

function el(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}

/* D-057 主题感知配色：原先全部硬编码浅色（#e1f5ee/#0f9d76…），
   切暗色时节点仍是白卡片 —— 与整体主题割裂。改为从 CSS 变量读取。 */
function cssVar(name, fallback){
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
function palette(){
  return {
    mainFill:   cssVar('--brand-soft', '#6b74f016'),
    mainStroke: cssVar('--brand',      '#6b74f0'),
    mainText:   cssVar('--brand-d',    '#8b93ff'),
    text:       cssVar('--ink-2',      '#b9c3dd'),
    muted:      cssVar('--mute',       '#5b6482'),
    subFill:    cssVar('--violet-soft','#a78bfa24'),
    subStroke:  cssVar('--violet',     '#a78bfa'),
    subText:    cssVar('--violet',     '#a78bfa'),
    line:       cssVar('--line-2',     'rgba(255,255,255,.14)'),
    curFill:    cssVar('--accent-soft','#22d3ee24'),
    curStroke:  cssVar('--accent',     '#22d3ee'),
  };
}

function fmtTs(ts){const d=new Date(ts*1000);
  return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`}
function short(t,n){return t.length>n ? t.slice(0,n)+"…" : t}
function sessionById(sid){
  if (H.main.session_id === sid) return H.main;
  return H.subs.find(s => s.session_id === sid);
}

/* 布局：主节点一行；每个下游节点独立泳道，首节点对齐其共享来源。
   每个节点可带用户拖拽产生的偏移（H.layout["sid|rid"]={dx,dy}），落盘后重启仍在。

   ★ D-063：偏移必须**钳制在画布内**。原实现直接 `44 + i*GAP + o.dx`，
     往左上拖时 x/y 会变成负数 → 坐标落到 viewBox（0 0 W Hh）之外 →
     **节点整个消失**，用户看到的就是"拖完位置复原 / 节点没了"，
     而且没有任何手段把它拖回来（画布不会向左上扩展）。
     现在改成两遍计算：
       ① 先用原始偏移求包围盒，画布**跟着长大**（拖多远都滚得到）；
       ② 再把每个节点钳回 [8, 画布尺寸-节点尺寸-8]，保证永远可见可拖回。 */
function layout(){
  const OFFS = H.layout || {};
  const off = k => { const o = OFFS[k]; return o ? {dx:o.dx||0, dy:o.dy||0} : {dx:0, dy:0}; };
  const mains = H.main.rounds;

  // ① 原始位置（未钳制）
  const rawM = mains.map((r, i) => {
    const o = off("canvas_main|" + r.round_id);
    return {x: 44 + i * GAP + o.dx, y: MAIN_Y + o.dy};
  });

  // ② 画布尺寸 = 默认布局 与 拖拽后包围盒 的**并集**
  let W  = 44 + mains.length * GAP + 60;
  let Hh = LANE0 + Math.max(0, H.subs.length - 1) * LANE_GAP + SH + 56;
  const grow = (x, y, w, h) => {
    W  = Math.max(W,  x + w + 40);
    Hh = Math.max(Hh, y + h + 56);
  };
  rawM.forEach(p => grow(p.x, p.y, NW, NH));

  // ③ 主节点钳制（下游首节点要跟随父节点的**最终**位置，所以先定主节点）
  const clampX = (x, w) => Math.max(8, Math.min(x, Math.max(8, W  - w - 8)));
  const clampY = (y, h) => Math.max(8, Math.min(y, Math.max(8, Hh - h - 8)));
  const mx = rawM.map(p => ({x: clampX(p.x, NW), y: clampY(p.y, NH)}));

  // ④ 下游节点泳道（首节点对齐父节点最终位置）
  const lanes = H.subs.map((s, li) => {
    const y0 = LANE0 + li * LANE_GAP;
    const pIdx = mains.findIndex(r => r.round_id === (s.derived_from || s.parent));
    const x0 = Math.max(44, (pIdx >= 0 ? mx[pIdx].x : 44) - 6);
    const xs = s.rounds.map((r, i) => {
      const o = off(s.session_id + "|" + r.round_id);
      return {x: x0 + i * SGAP + o.dx, y: y0 + o.dy};
    });
    xs.forEach(p => grow(p.x, p.y, SW, SH));   // 拖远了画布跟着长
    return {y0, xs, pIdx};
  }).map(l => ({y0: l.y0, pIdx: l.pIdx,
                xs: l.xs.map(p => ({x: clampX(p.x, SW), y: clampY(p.y, SH)}))}));

  return {mx, lanes, W, Hh};
}

function draw(){
  const P = palette();          // D-057 主题感知配色
  const svg = document.getElementById("cv");
  svg.innerHTML = "";
  const {mx, lanes, W, Hh} = layout();
  // D-058 试过「内容垂直居中」，**两次都失败，已回退**：
  //   ① 把 viewBox 高度撑到容器高 → 缩放比例被压到 fit 高度，节点缩成一团；
  //   ② 只上移起点保持高度 → 底部节点被 viewBox 裁掉（更糟：内容不可见）。
  // 底部留白只是观感问题，**内容被裁是功能问题**，不值得为前者牺牲后者。
  // 若要真正居中，正确做法是在 draw 时给每个节点的 y 统一加偏移量（改坐标系），
  // 而不是动 viewBox —— 留作后续单独处理。
  svg.setAttribute("viewBox", `0 0 ${Math.max(W, 1100)} ${Hh}`);
  svg.setAttribute("width", Math.max(W, 1100));
  svg.setAttribute("height", Hh);
  const mains = H.main.rounds;

  // 主节点泳道标签：与首个主节点左对齐
  svg.appendChild(el("text", {x:44, y:MAIN_Y-14, class:"maintag"},
    "主节点 · 私有记忆全量只读"));

  /* D-062：主节点推进轴**改为相邻卡片之间串接**，不再一笔画到底。
     原实现是一条从首节点中心直连到末节点的直线：
       x1 = mx[0].x + NW/2  →  x2 = mx.at(-1).x + NW/2
       y  = mx[0].y + NH/2 + 14
     两个致命问题叠加：
       ① y 落在 NH/2+14 = 39（卡片高 50）→ **线从卡片内部第二行文字穿过**；
       ② 跨度是首到末，而节点宽 152 / 间距 186，中间只有 34px 空隙，
          根本没有容纳连线的空间 → 中间的卡片全被穿透，视觉上非常突兀。
     正确画法：**逐段连接相邻卡片**（左卡片右缘 → 右卡片左缘），
     既表达"同一会话按时间推进"，又完全不碰卡片本体；
     拖拽产生 dy 偏移时用斜线连接，保持跟随。 */
  for (let i = 1; i < mx.length; i++) {
    const a = mx[i - 1], b = mx[i];
    const x1 = a.x + NW, y1 = a.y + NH / 2;
    const x2 = b.x,      y2 = b.y + NH / 2;
    // 间距过窄或被拖成重叠时不画，避免出现反向/零长的怪线
    if (x2 - x1 >= 6) {
      svg.appendChild(el("line", {x1, y1, x2, y2, class:"ml"}));
      /* 箭头：间隙只有 34px，单画一条线读不出方向。
         用三角 marker 明确「后一回合 ← 前一回合」的推进方向。
         ⚠️ 不能靠 stroke-dasharray 做流动 —— 与 D-056 同因：
            dash 按原路径长度铺，箭头标记会与线端错位。 */
      const ux = (x2 - x1), uy = (y2 - y1);
      const len = Math.hypot(ux, uy) || 1;
      const dx = ux / len, dy = uy / len;   // 单位方向向量
      const tipX = x2 - 1, tipY = y2;
      const w = 3.2, h = 5;
      // 箭头两翼：沿方向向量回退 h，再各向法线偏 w
      svg.appendChild(el("polygon", {
        points: [
          `${tipX},${tipY}`,
          `${tipX - dx * h - dy * w},${tipY - dy * h + dx * w}`,
          `${tipX - dx * h + dy * w},${tipY - dy * h - dx * w}`
        ].join(" "),
        class: "ml-arrow"
      }));
    }
  }

  mains.forEach((r, i) => {
    const x = mx[i].x, y = mx[i].y;
    NODE_POS["canvas_main|"+r.round_id] = {x, y};
    const g = el("g", {class:"node", "data-sid":"canvas_main", "data-rid":r.round_id});
    g.appendChild(el("rect", {x, y, width:NW, height:NH, rx:10, fill:P.mainFill,
      stroke:P.mainStroke, "stroke-width":1.2}));
    g.appendChild(el("text", {x:x+10, y:y+19, "font-size":12, "font-weight":600, fill:P.mainText}, r.round_id));
    g.appendChild(el("text", {x:x+44, y:y+19, "font-size":12, fill:P.text}, short(r.title,6)));
    g.appendChild(el("text", {x:x+10, y:y+38, "font-size":10.5, fill:P.text}, fmtTs(r.ts)));
    svg.appendChild(g);
  });

  // 下游节点泳道：标签与共享曲线跟随首节点（含拖拽偏移）
  H.subs.forEach((s, li) => {
    const {xs, pIdx} = lanes[li];
    const inhN = (s.inherited && s.inherited.length) ? ` · 共享 ${s.inherited.length} 条记忆` : "";
    svg.appendChild(el("text", {x:xs[0].x, y:xs[0].y-11, class:"lanetag"},
      `${s.title} · 共享自 ${s.derived_from || s.parent}${inhN}`));

    const par = pIdx >= 0 ? mx[pIdx] : {x:44, y:MAIN_Y};
    const px = par.x + NW/2, py = par.y + NH;
    const tx = xs[0].x + SW/2, ty = xs[0].y;
    svg.appendChild(el("path", {d:`M${px} ${py} C ${px} ${py+40}, ${tx} ${ty-40}, ${tx} ${ty}`, class:"dl"}));
    svg.appendChild(el("circle", {cx:px, cy:py, r:3, fill:P.subStroke}));
    svg.appendChild(el("text", {x:(px+tx)/2, y:(py+ty)/2, class:"tip", fill:P.curStroke,
      "text-anchor":"middle"}, "共享记忆"));

    s.rounds.forEach((r, i) => {
      const x = xs[i].x, y = xs[i].y;
      NODE_POS[s.session_id+"|"+r.round_id] = {x, y};
      if (i > 0) svg.appendChild(el("line", {x1:xs[i-1].x+SW, y1:xs[i-1].y+SH/2,
        x2:x, y2:y+SH/2, class:"lane"}));
      const g = el("g", {class:"node", "data-sid":s.session_id, "data-rid":r.round_id});
      g.appendChild(el("rect", {x, y, width:SW, height:SH, rx:9, fill:P.subFill,
        stroke:P.subStroke, "stroke-width":1.2}));
      g.appendChild(el("text", {x:x+9, y:y+18, "font-size":11, "font-weight":600, fill:P.subText}, r.round_id));
      g.appendChild(el("text", {x:x+40, y:y+18, "font-size":11, fill:P.text}, short(r.title,6)));
      g.appendChild(el("text", {x:x+9, y:y+36, "font-size":9.5, fill:P.text}, fmtTs(r.ts)));
      svg.appendChild(g);
    });
  });
}

/* D-063：布局保存的**可见反馈**。
   原来是 `fetch(...).catch(() => {})` —— 请求失败被完全静默，
   用户只能看到"位置复原"，根本分不清是没保存、还是保存了没生效。
   现在成功/失败都给浮层提示，失败还要说明原因。 */
function flash(msg, bad){
  let t = document.getElementById("layoutToast");
  if (!t){
    t = document.createElement("div");
    t.id = "layoutToast";
    t.className = "floattip";           // 复用画布浮条样式
    t.style.cssText = "left:auto;right:12px;bottom:10px;transition:opacity .4s";
    (document.querySelector(".stagewrap") || document.body).appendChild(t);
  }
  t.textContent = msg;
  t.style.color = bad ? "var(--danger, #ff8f8f)" : "";
  t.style.opacity = "1";
  clearTimeout(t._tm);
  t._tm = setTimeout(() => { t.style.opacity = "0"; }, 1900);
}

/* 布局落盘：合并 pending 后一次 POST（连续拖多个节点不会打一堆请求），
   pagehide 时强制 flush —— 防止"拖完立刻刷新页面"时请求还没发出去。 */
const _pending = {};
let _saveTimer = null;
function flushLayout(){
  clearTimeout(_saveTimer); _saveTimer = null;
  const keys = Object.keys(_pending);
  if (!keys.length) return;
  const pos = {};
  keys.forEach(k => { pos[k] = _pending[k]; delete _pending[k]; });
  return fetch("/api/canvas/layout", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({positions: pos})
  }).then(r => r.json()).then(j => {
    if (j && j.ok) flash("布局已保存 · 刷新后仍在");
    else flash("布局保存失败：" + ((j && j.error) || "服务端未返回 ok"), true);
  }).catch(e => flash("布局保存失败（网络）：" + e, true));
}
function queueLayoutSave(key, val){
  _pending[key] = val;
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(flushLayout, 150);
}
window.addEventListener("pagehide", flushLayout);

/* 节点拖拽：按住节点拖动重排位置（连接线实时跟随），松手落盘；
   位移 <5px 视为点击 → 打开该节点会话控制台。 */
(function(){
  const stage = document.getElementById("stage");
  let drag = null; // {sid, rid, key, sx, sy, base, moved}
  stage.addEventListener("mousedown", e => {
    const g = e.target.closest("g.node");
    if (!g) return;
    const key = g.dataset.sid + "|" + g.dataset.rid;
    const base = (H.layout && H.layout[key]) || {dx:0, dy:0};
    drag = {sid: g.dataset.sid, rid: g.dataset.rid, key,
            sx: e.pageX, sy: e.pageY, base, moved: false};
    e.preventDefault();
  });
  window.addEventListener("mousemove", e => {
    if (!drag) return;
    const dx = e.pageX - drag.sx, dy = e.pageY - drag.sy;
    if (!drag.moved && Math.hypot(dx, dy) < 5) return;
    drag.moved = true;
    if (!H.layout) H.layout = {};
    H.layout[drag.key] = {dx: drag.base.dx + dx, dy: drag.base.dy + dy};
    document.getElementById("cv").style.cursor = "grabbing";
    draw();  // 重绘让连接线/标签实时跟随（节点量级 ~40，重绘开销可忽略）
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    const d = drag; drag = null;
    document.getElementById("cv").style.cursor = "";
    if (!d.moved) { enterConsole(d.sid, d.rid); return; }  // 未拖动 = 点击 → 进入节点控制台
    // 落盘该节点的最新偏移（D-063：成功/失败都有提示，不再静默）
    queueLayoutSave(d.key, H.layout[d.key]);
  });
})();

/* D-063：恢复默认布局。
   没有这个入口时，节点一旦被拖乱/拖出可视区就再也回不去（只能手改 json）。 */
(function(){
  const rb = document.getElementById("resetLayoutBtn");
  if (!rb) return;
  rb.addEventListener("click", () => {
    const n = Object.keys(H && H.layout ? H.layout : {}).length;
    if (!n) { flash("当前已是默认布局"); return; }
    if (!confirm(`恢复默认布局？将清空已保存的 ${n} 个节点位置。`)) return;
    fetch("/api/canvas/layout", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({reset: true})
    }).then(r => r.json()).then(j => {
      if (j && j.ok){ H.layout = {}; draw(); flash("已恢复默认布局"); }
      else flash("重置失败：" + ((j && j.error) || "服务端未返回 ok"), true);
    }).catch(e => flash("重置失败（网络）：" + e, true));
  });
})();

function focusNode(sid, rid){
  const p = NODE_POS[sid + "|" + rid];
  const stage = document.getElementById("stage");
  if (!p) return;
  stage.scrollTo({left: Math.max(0, p.x - 160), top: Math.max(0, p.y - 120), behavior: "smooth"});
}

/* 点击节点：节点脉冲放大 → 跳转节点控制台（携带 session/round 上下文） */
function enterConsole(sid, rid){
  const g = document.querySelector(`g.node[data-sid="${sid}"][data-rid="${rid}"]`);
  if (g) g.classList.add("enter");
  setTimeout(() => {
    location.href = "/?session=" + encodeURIComponent(sid) + "&round=" + encodeURIComponent(rid);
  }, 430);
}

/* 从控制台返回时 ?hl=session|round → 对应节点脉冲高亮提示 */
function highlightFromParam(){
  const hl = new URLSearchParams(location.search).get("hl");
  if (!hl) return;
  const [sid, rid] = hl.split("|");
  const g = document.querySelector(`g.node[data-sid="${sid}"][data-rid="${rid}"]`);
  if (g){ g.classList.add("hl"); focusNode(sid, rid); }
}

async function refreshHistory(){
  H = (await (await fetch("/api/canvas/history")).json()).history;
  // 有会话数据时隐藏空态引导（否则会与画布内容重叠）
  // ⚠️ H 是 `{main, subs}` **对象**，不是数组 —— 原判断写的是 `H.length`，
  //    对对象恒为 undefined → 空态永远显示，与节点叠在一起（实测截图确认）。
  const mainN = (H && H.main && H.main.rounds) ? H.main.rounds.length : 0;
  const subN  = (H && H.subs) ? H.subs.length : 0;
  const emp = document.getElementById("stageEmpty");
  if (emp) emp.style.display = (mainN || subN) ? "none" : "block";
}

async function boot(){
  // D-053：主题切换（三页统一）
  const tb=document.getElementById("themeBtn");
  if(tb && window.SRTPTheme){
    tb.addEventListener("click", function(){ SRTPTheme.toggle(); draw(); });
    window.addEventListener("srtp:theme", draw);
  }
  // D-061：高对比模式。画布配色从 CSS 变量取，切高对比后必须重绘才生效。
  const cb=document.getElementById("contrastBtn");
  if(cb && window.SRTPTheme){
    cb.addEventListener("click", function(){ SRTPTheme.toggleContrast(); draw(); });
    window.addEventListener("srtp:contrast", draw);
    if(SRTPTheme.contrastOn()) cb.classList.add("on");
    window.addEventListener("srtp:contrast", function(e){ cb.classList.toggle("on", !!e.detail); });
  }
  await refreshHistory();
  draw();
  const hl = new URLSearchParams(location.search).get("hl");
  if (hl){ highlightFromParam(); }
  window.addEventListener("resize", draw);
}
boot();

