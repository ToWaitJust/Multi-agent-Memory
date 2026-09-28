/* 会话画布（canvas）脚本 —— 迁移自 canvas.html 内联 <script> */

let H = null;
const NS = "http://www.w3.org/2000/svg";
const NW = 152, NH = 50, GAP = 186;          // 主线节点（像素固定）
const SW = 134, SH = 46, SGAP = 160;         // 副线节点
const MAIN_Y = 52, LANE0 = 196, LANE_GAP = 116;
const NODE_POS = {};                          // session|round -> {x,y} 供平移定位

function el(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
function fmtTs(ts){const d=new Date(ts*1000);
  return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`}
function short(t,n){return t.length>n ? t.slice(0,n)+"…" : t}
function sessionById(sid){
  if (H.main.session_id === sid) return H.main;
  return H.subs.find(s => s.session_id === sid);
}

/* 布局：主线一行；每条副线独立泳道，首节点对齐其继承源。
   每个节点可带用户拖拽产生的偏移（H.layout["sid|rid"]={dx,dy}），落盘后重启仍在。 */
function layout(){
  const OFFS = H.layout || {};
  const off = k => { const o = OFFS[k]; return o ? {dx:o.dx||0, dy:o.dy||0} : {dx:0, dy:0}; };
  const mains = H.main.rounds;
  const mx = mains.map((r, i) => {
    const o = off("canvas_main|" + r.round_id);
    return {x: 44 + i * GAP + o.dx, y: MAIN_Y + o.dy};
  });
  let W = 44 + mains.length * GAP + 60;
  let maxY = MAIN_Y + NH;
  const lanes = [];
  H.subs.forEach((s, li) => {
    const y0 = LANE0 + li * LANE_GAP;
    const pIdx = mains.findIndex(r => r.round_id === (s.derived_from || s.parent));
    const x0 = Math.max(44, (pIdx >= 0 ? mx[pIdx].x : 44) - 6);
    const xs = s.rounds.map((r, i) => {
      const o = off(s.session_id + "|" + r.round_id);
      return {x: x0 + i * SGAP + o.dx, y: y0 + o.dy};
    });
    W = Math.max(W, xs.at(-1).x + SGAP + 40);
    maxY = Math.max(maxY, ...xs.map(p => p.y + SH));
    lanes.push({y0, xs, pIdx});
  });
  maxY = Math.max(maxY, ...mx.map(p => p.y + NH));
  return {mx, lanes, W, Hh: Math.max(maxY + 56, LANE0 + Math.max(0, H.subs.length - 1) * LANE_GAP + SH + 56)};
}

function draw(){
  const svg = document.getElementById("cv");
  svg.innerHTML = "";
  const {mx, lanes, W, Hh} = layout();
  svg.setAttribute("viewBox", `0 0 ${Math.max(W, 1100)} ${Hh}`);
  svg.setAttribute("width", Math.max(W, 1100));
  svg.setAttribute("height", Hh);
  const mains = H.main.rounds;

  // 主线泳道标签：与主线首节点左对齐
  svg.appendChild(el("text", {x:44, y:MAIN_Y-14, class:"maintag"},
    "主线 MAIN · 常驻完整池（全量只读）"));
  svg.appendChild(el("line", {x1:mx[0].x+NW/2, y1:mx[0].y+NH/2+14,
    x2:mx.at(-1).x+NW/2, y2:mx.at(-1).y+NH/2+14, class:"ml"}));

  mains.forEach((r, i) => {
    const x = mx[i].x, y = mx[i].y;
    NODE_POS["canvas_main|"+r.round_id] = {x, y};
    const g = el("g", {class:"node", "data-sid":"canvas_main", "data-rid":r.round_id});
    g.appendChild(el("rect", {x, y, width:NW, height:NH, rx:10, fill:"#e1f5ee",
      stroke:"#0f9d76", "stroke-width":1.2}));
    g.appendChild(el("text", {x:x+10, y:y+19, "font-size":12, "font-weight":600, fill:"#0c5c49"}, r.round_id));
    g.appendChild(el("text", {x:x+44, y:y+19, "font-size":12, fill:"#334155"}, short(r.title,6)));
    g.appendChild(el("text", {x:x+10, y:y+38, "font-size":10.5, fill:"#64748b"}, fmtTs(r.ts)));
    svg.appendChild(g);
  });

  // 副线泳道：标签与继承曲线跟随首节点（含拖拽偏移）
  H.subs.forEach((s, li) => {
    const {xs, pIdx} = lanes[li];
    const inhN = (s.inherited && s.inherited.length) ? ` · ${s.inherited.length} 条继承记忆` : "";
    svg.appendChild(el("text", {x:xs[0].x, y:xs[0].y-11, class:"lanetag"},
      `${s.title} · 独立会话（继承主线 ${s.derived_from || s.parent}${inhN}）`));

    const par = pIdx >= 0 ? mx[pIdx] : {x:44, y:MAIN_Y};
    const px = par.x + NW/2, py = par.y + NH;
    const tx = xs[0].x + SW/2, ty = xs[0].y;
    svg.appendChild(el("path", {d:`M${px} ${py} C ${px} ${py+40}, ${tx} ${ty-40}, ${tx} ${ty}`, class:"dl"}));
    svg.appendChild(el("circle", {cx:px, cy:py, r:3, fill:"#7fa8c9"}));
    svg.appendChild(el("text", {x:(px+tx)/2, y:(py+ty)/2, class:"tip", fill:"#5b8db3",
      "text-anchor":"middle"}, "继承记忆"));

    s.rounds.forEach((r, i) => {
      const x = xs[i].x, y = xs[i].y;
      NODE_POS[s.session_id+"|"+r.round_id] = {x, y};
      if (i > 0) svg.appendChild(el("line", {x1:xs[i-1].x+SW, y1:xs[i-1].y+SH/2,
        x2:x, y2:y+SH/2, class:"lane"}));
      const g = el("g", {class:"node", "data-sid":s.session_id, "data-rid":r.round_id});
      g.appendChild(el("rect", {x, y, width:SW, height:SH, rx:9, fill:"#e6f1fb",
        stroke:"#378add", "stroke-width":1.2}));
      g.appendChild(el("text", {x:x+9, y:y+18, "font-size":11, "font-weight":600, fill:"#0c447c"}, r.round_id));
      g.appendChild(el("text", {x:x+40, y:y+18, "font-size":11, fill:"#334155"}, short(r.title,6)));
      g.appendChild(el("text", {x:x+9, y:y+36, "font-size":9.5, fill:"#64748b"}, fmtTs(r.ts)));
      svg.appendChild(g);
    });
  });
}

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
    // 落盘该节点的最新偏移（失败不阻断，仅提示）
    fetch("/api/canvas/layout", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({positions: {[d.key]: H.layout[d.key]}})
    }).catch(() => {});
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
  const emp = document.getElementById("stageEmpty");
  if (emp) emp.style.display = (H && H.length) ? "none" : "block";
}

async function boot(){
  await refreshHistory();
  draw();
  const hl = new URLSearchParams(location.search).get("hl");
  if (hl){ highlightFromParam(); }
  window.addEventListener("resize", draw);
}
boot();

