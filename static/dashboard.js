/* 控制台（dashboard）脚本 —— 迁移自 dashboard.html 内联 <script> */
(function(){
'use strict';

"use strict";
const $ = s => document.querySelector(s);
const svgns = "http://www.w3.org/2000/svg";
/* D-057：**标签维度废弃**，记忆按「所属 topic 节点」着色。
   理由：标签（daily/profile/tech_stack…）是早期"记忆分类"设计的遗留，
   而真实结构是"记忆归属某个会话节点、可被多个节点共享"。按 owner 着色
   才能让记忆池和图拓扑页对得上。*/
/* D-061：记忆池按「所属节点」着色（D-057）的调色板改为**读 CSS 变量**。
   原来写死暗色系（#22d3ee 青在亮色下仅 3.6:1，强光下发灰），
   且完全切不开data-contrast —— 高对比模式对池子的点无效。
   现在每个槽位绑一个令牌名，切主题/切高对比自动跟随。 */
const NODE_COL_VAR = ["--brand","--accent","--violet","--orange","--rose","--blue",
                      "--green-ne","--pink-ne","--amber-ne","--cyan-ne","--purple-ne","--coral-ne"];
const NODE_COL_FALLBACK = ["#6b74f0","#22d3ee","#a78bfa","#f5a524","#fb7185","#60a5fa",
                           "#34d399","#f472b6","#facc15","#38bdf8","#c084fc","#fb923c"];
function cssTok(name, fb){
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  return (v && v.trim()) || fb;
}
function nodeColor(owner){
  if(!owner) return cssTok('--mute', '#5b6482');
  let h=0; for(let i=0;i<owner.length;i++) h=(h*31+owner.charCodeAt(i))>>>0;
  const i = h % NODE_COL_VAR.length;
  return cssTok(NODE_COL_VAR[i], NODE_COL_FALLBACK[i]);
}

let H = null;                       // 画布历史（会话 + 回合）
let CUR = null;                     // {session_id, round_id}
const params = new URLSearchParams(location.search);

function esc(s){ return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

/* D-053 极简 Markdown 渲染。
   原因：Agent 的回答里大量使用 `**粗体**` / `- 列表` / `` `标识符` ``，
   之前一律走 textContent，结果满屏都是星号和反引号 —— 对话窗口最影响观感。
   ⚠️ 安全：先做 HTML 转义再插入标记，**不注入原始 HTML**，
      所以 LLM 返回的 <script> 只会被原样显示为文本。 */
function mdInline(t){
  return esc(t)
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
}
function mdRender(src){
  const lines = String(src==null?'':src).split('\n');
  let html = '', inList = false;
  const closeList = ()=>{ if(inList){ html += '</ul>'; inList = false; } };
  for (const raw of lines){
    const line = raw.replace(/\s+$/,'');
    if (!line.trim()){ closeList(); continue; }
    const li = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)$/);
    if (li){
      if(!inList){ html += '<ul>'; inList = true; }
      html += '<li>' + mdInline(li[1]) + '</li>';
      continue;
    }
    closeList();
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h){
      const lv = Math.min(4, h[1].length + 2);
      html += '<h' + lv + '>' + mdInline(h[2]) + '</h' + lv + '>';
      continue;
    }
    html += '<p>' + mdInline(line) + '</p>';
  }
  closeList();
  return html;
}
function sessionById(sid){
  if (H.main.session_id === sid) return H.main;
  return H.subs.find(s => s.session_id === sid);
}
function isMain(sid){ return H.main.session_id === sid; }
function fmtTs(ts){const d=new Date(ts*1000);
  return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`}

function setStage(name){
  const order=["input","recall","score","weight","rank","select","answer"];
  const i=order.indexOf(name);
  document.querySelectorAll('#stepper .node').forEach(n=>{
    const j=order.indexOf(n.dataset.s);
    n.classList.remove('active','done','cur');
    if(j<i) n.classList.add('done');
    else if(j===i) n.classList.add('active','cur');
  });
  // D-055：「用户输入」卡原来是占位文案「节点内追问」，看不出这轮问了什么。
  // 调度一开始就把真实 query 填进去。
  if(name!=='input'){
    const d=$('#stepper .node[data-s="input"] .d');
    if(d && window.__curQuery) d.textContent = window.__curQuery;
  }
}
function resetStage(){ setStage('input'); }

// ---- 左侧：会话标签 + 上下文历史 ----
function renderTabs(){
  const bar=$('#sessbar'); bar.innerHTML='';
  const mk=(sid,title,n)=>{
    const b=document.createElement('button');
    b.className='stab'+(CUR&&CUR.session_id===sid?' active':'');
    b.innerHTML=`${esc(title)}<span class="n">${n}轮</span>`;
    b.onclick=()=>selectSession(sid);
    bar.appendChild(b);
  };
  mk(H.main.session_id, '主节点', H.main.rounds.length);
  H.subs.forEach(s=>mk(s.session_id, s.title.replace(/^(副线|下游节点) · /,''), s.rounds.length));
}

function selectSession(sid, rid){
  const sess=sessionById(sid);
  if(!sess) return;
  CUR={session_id:sid, round_id: rid || sess.rounds[sess.rounds.length-1].round_id};
  // 同步 URL：刷新/分享都能回到当前节点控制台
  history.replaceState(null, '', '/?session='+encodeURIComponent(sid)+'&round='+encodeURIComponent(CUR.round_id));
  renderTabs();
  const main=isMain(sid);
  const box=$('#messages'); box.innerHTML='';
  const cap=document.createElement('div'); cap.className='ctxcap';
  cap.innerHTML = main
    ? `<b>主节点会话</b> · 节点私有记忆全量只读；每次追问会按 query 选源，把上游命中节点的记忆共享进来。`
    : `<b>${esc(sess.title)}</b> · 下游节点，创建时从 ${esc(sess.derived_from||'')} 共享了一批精选记忆；它也可再向上下游节点共享。`;
  box.appendChild(cap);
  // 下游节点：记忆来源卡（源节点可点击跳回上游该回合）
  if(!main){
    const src=document.createElement('div'); src.className='srccard';
    const srcRound=H.main.rounds.find(r=>r.round_id===(sess.derived_from||sess.parent));
    const inherited=(sess.inherited && sess.inherited.length) ? sess.inherited
      : ((sess.rounds[0]&&sess.rounds[0].inherited)||[]);
    let html=`<div class="cap">记忆来源 · 记忆继承链路</div>
      <span class="srcnode" onclick="selectSession('${H.main.session_id}','${esc(sess.derived_from||sess.parent||'')}')">
        <span class="arr">↩</span>上游节点 ${esc(sess.derived_from||sess.parent||'?')}${srcRound? ' · '+esc(srcRound.title||''):''}</span>`;
    if(inherited.length){
      html+=`<div class="srclen">创建时按第一问「${esc((sess.rounds[0]&&sess.rounds[0].user)||'')}」选源并共享 <b>${inherited.length}</b> 条记忆：</div>
        <div class="srcchips">${inherited.map(t=>`<div class="c">· ${esc(t)}</div>`).join('')}</div>`;
    }else{
      html+=`<div class="srcnote">演示数据未附共享清单 —— 在主节点选中回合现场创建下游节点，可看到完整的「按第一问选源 → 跨节点共享」效果。</div>`;
    }
    src.innerHTML=html;
    box.appendChild(src);
  }
  sess.rounds.forEach(r=>{
    box.appendChild(roundCard(sess, r, r.round_id===CUR.round_id));
  });
  // 创建下游节点入口：仅主节点回合
  const sc=$('#subcreate');
  if(main){ sc.classList.add('show'); $('#subFromRid').textContent=CUR.round_id; }
  else sc.classList.remove('show');
  const curEl=box.querySelector('.msg.cur');
  if(curEl) curEl.scrollIntoView({block:'center'});
  box.scrollTop=box.scrollHeight;
  // D-058：若该回合带了**预置的调度明细**，直接渲染到右侧。
  // 否则进站时右侧明细区是一片空白 —— 展示场景下这等于"没有证据"。
  let curRound=null;
  sess.rounds.forEach(function(r){ if(r.round_id===CUR.round_id) curRound=r; });
  if(curRound && curRound.detail){
    renderDetail(curRound.detail);
    animateDetail();
    refreshPools((curRound.detail.candidates||[]).map(function(c){return c.id;}),
                 curRound.detail.kept_ids||[]).catch(function(){});
  }
  // D-053：回合卡片依次入场（只对刚切进来的这一屏，切换会话时也有）
  if(window.Fx) window.Fx.rise(box.querySelectorAll('.ctxcap, .srccard, .msg'), 1);
  syncComposer();
  // 返回画布时高亮当前节点
  $('#canvasLink').href='/canvas?hl='+encodeURIComponent(CUR.session_id+'|'+CUR.round_id);
  // 双池视图随会话刷新：下游节点显示其共享自上游的记忆连线
  updatePoolView();
}

function roundCard(sess, r, isCur){
  const main=isMain(sess.session_id);
  const d=document.createElement('div');
  d.className='msg '+(main?'assistant':'assistant')+(isCur?' cur':'');
  const who=main?'节点 Agent（私有 + 共享记忆）':'节点 Agent（共享自上游）';
  const w=document.createElement('div'); w.className='who';
  w.innerHTML=`<span class="rid">${esc(r.round_id)}</span><span>${esc(r.title||'')}</span><span>${fmtTs(r.ts)}</span>`;
  d.appendChild(w);
  const u=document.createElement('div'); u.className='msg user';
  const uw=document.createElement('div'); uw.className='who'; uw.textContent='用户';
  const ub=document.createElement('div'); ub.className='bubble'; ub.innerHTML=mdRender(r.user);
  u.appendChild(uw); u.appendChild(ub); d.appendChild(u);
  const ab=document.createElement('div'); ab.className='bubble md'; ab.innerHTML=mdRender(r.assistant);
  const aw=document.createElement('div'); aw.className='who'; aw.textContent=who;
  d.appendChild(aw); d.appendChild(ab);
  if(r.inherited && r.inherited.length){
    const ih=document.createElement('div'); ih.className='inh';
    ih.textContent='继承的记忆：\n'+r.inherited.map(t=>'· '+t).join('\n');
    d.appendChild(ih);
  }
  return d;
}

// ---- 右侧：双池语义地图 ----
function scaleX(zx,isShared){ return isShared? 530+zx*430 : 20+zx*450; }
function scaleY(zy){ return 40+zy*450; }
/* ------------------------------------------------------------------
     D-058 记忆飞行动画 —— 让"跨节点共享"从**抽象状态**变成**可见动作**。

     为什么必须做：此前共享只表现为「散点变色 + 虚线连接」，
     同学看不出"记忆飞过去了"这件事，而它正是整个系统的立意。

     实现：私有侧被选中的散点，沿二次贝塞尔曲线**逐个飞抵**共享侧，
     到达时泛起一圈光晕确认。错峰延迟（每条 +70ms）形成"陆续到达"的节奏。

     ⚠️ 兜底：动画期间若用户切页/重绘，flightRAF 必须能被取消；
        且动画**只读坐标不改布局**，中途失败也只是没动画，不影响功能。
     ------------------------------------------------------------------ */
  let flightRAF = null;

  function cancelFlight(){ if (flightRAF){ cancelAnimationFrame(flightRAF); flightRAF = null; } }

  function playFlight(resident, shared, keptIds){
    cancelFlight();
    if (window.Fx && window.Fx.reduced) return;
    const svg = $('#poolsvg');
    if (!svg) return;
    const layer = document.createElementNS(svgns, 'g');
    layer.setAttribute('id', 'flightLayer');
    svg.appendChild(layer);

    const kep = new Set(keptIds || []);
    const dstOf = {};
    shared.forEach(function (m){ dstOf[m.id] = { x: scaleX(m.x, true), y: scaleY(m.y, true) }; });

    // 只飞"本轮选中"的记忆；历史留存的（已经在共享侧）不飞
    const flights = [];
    resident.forEach(function (r){
      if (!kep.has(r.id)) return;
      const d = dstOf[r.id];
      if (!d) return;
      flights.push({
        s: { x: scaleX(r.x, false), y: scaleY(r.y, false) },
        d: d,
        // 控制点：水平中点 + 向上抬起，三条错开高度避免重叠成一条直线
        c: null, delay: flights.length * 55,
      });
    });
    if (!flights.length){ layer.remove(); return; }

    flights.forEach(function (f, i){
      f.c = {
        x: (f.s.x + f.d.x) / 2,
        y: Math.min(f.s.y, f.d.y) - 46 - (i % 3) * 26,
      };
      // 轨迹虚线（比飞行点先出现，给人"路径已规划"的感觉）
      const path = document.createElementNS(svgns, 'path');
      path.setAttribute('d', `M${f.s.x} ${f.s.y} Q${f.c.x} ${f.c.y} ${f.d.x} ${f.d.y}`);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', cssTok('--accent','#0891b2'));
      path.setAttribute('stroke-width', '1.8');
      path.setAttribute('stroke-dasharray', '5 6');
      path.setAttribute('opacity', '0');
      layer.appendChild(path);
      f.path = path;

      const halo = document.createElementNS(svgns, 'circle');
      halo.setAttribute('r', '11');
      halo.setAttribute('fill', cssTok('--accent','#22d3ee'));
      halo.setAttribute('opacity', '0');
      layer.appendChild(halo);
      f.halo = halo;
      const dot = document.createElementNS(svgns, 'circle');
      dot.setAttribute('r', '6');
      dot.setAttribute('fill', cssTok('--accent-hi','#7ff3e4'));
      dot.setAttribute('opacity', '0');
      layer.appendChild(dot);
      f.dot = dot;

      const ring = document.createElementNS(svgns, 'circle');
      ring.setAttribute('cx', f.d.x); ring.setAttribute('cy', f.d.y);
      ring.setAttribute('r', '6'); ring.setAttribute('fill', 'none');
      ring.setAttribute('stroke', cssTok('--accent','#22d3ee')); ring.setAttribute('stroke-width', '1.6');
      ring.setAttribute('opacity', '0');
      layer.appendChild(ring);
      f.ring = ring;
    });

    const DUR = 860;
    function qbez(s, c, d, t){
      const u = 1 - t;
      return { x: u*u*s.x + 2*u*t*c.x + t*t*d.x, y: u*u*s.y + 2*u*t*c.y + t*t*d.y };
    }
    // ⚠️ 起始时刻必须用**闭包变量**记一次。原写法 `if (!T0) var _t0 = ts;`
    //    里的 T0 是常量 0，条件恒真 → 每帧都把 _t0 重置为当前 ts，
    //    elapsed 永远是 0 → 所有飞行点停在起点，动画等于没跑（实测）。
    let t0 = null;
    function frame(ts){
      if (!flightRAF) return;
      if (t0 === null) t0 = ts;
      const elapsed = ts - t0;
      let alive = false;
      flights.forEach(function (f){
        const local = (elapsed - f.delay) / DUR;
        if (local < 0) { alive = true; return; }
        if (local > 1.35) {                    // 落位后淡出轨迹
          if (f.path && f.path.parentNode) f.path.remove();
          if (f.halo && f.halo.parentNode) f.halo.remove();
          if (f.dot && f.dot.parentNode) f.dot.remove();
          if (f.ring) f.ring.setAttribute('opacity', String(Math.max(0, 1 - (local - 1) / 0.35)));
          if (f.ring && local > 1.35 && f.ring.parentNode) f.ring.remove();
          return;
        }
        alive = true;
        const t = Math.min(1, local);
        const k = 1 - Math.pow(1 - t, 3);       // easeOutCubic
        const p = qbez(f.s, f.c, f.d, k);
        f.dot.setAttribute('cx', p.x); f.dot.setAttribute('cy', p.y);
        f.dot.setAttribute('opacity', '1');
        if (f.halo){ f.halo.setAttribute('cx', p.x); f.halo.setAttribute('cy', p.y);
          f.halo.setAttribute('opacity', String(0.20 * Math.sin(Math.PI * t))); }
        if (f.path) f.path.setAttribute('opacity', String(0.8 * Math.sin(Math.PI * t)));
      });
      if (alive || flights.some(function (f) { return f.dot && f.dot.parentNode; })){
        flightRAF = requestAnimationFrame(frame);
      } else {
        cancelFlight();
        if (layer.parentNode) layer.remove();
      }
    }
    flightRAF = requestAnimationFrame(frame);
  }

function renderPools(resident, shared, recalledIds, keptIds, maxShared, extra){
  const rg=$('#residentDots'), sg=$('#sharedDots'), fg=$('#flows');
  rg.innerHTML=''; sg.innerHTML=''; fg.innerHTML='';
  const rec=new Set(recalledIds), kep=new Set(keptIds);
  const coord={};
  resident.forEach(m=>{
    const cx=scaleX(m.x,false), cy=scaleY(m.y);
    coord[m.id]={x:cx, y:cy};
    const recalled=rec.has(m.id), kept=kep.has(m.id);
    const col=nodeColor(m.owner);
    const c=document.createElementNS(svgns,'circle');
    c.setAttribute('cx',cx); c.setAttribute('cy',cy);
    // D-057：召回候选更大更亮；保留的记忆加光晕
    c.setAttribute('r', kept? 9 : (recalled? 7.5 : 4.6));
    c.setAttribute('fill', col);
    c.setAttribute('fill-opacity', kept? 1 : (recalled? .92 : .62));
    /* D-061：描边色走令牌 —— 原来写死 '#22d3ee' / 'rgba(255,255,255,.55)'，
       亮色下白描边等于隐形、且完全切不开高对比模式。 */
    c.setAttribute('stroke', kept? cssTok('--accent','#22d3ee')
                                 : cssTok('--pool-stroke','rgba(255,255,255,.55)'));
    c.setAttribute('stroke-width', kept? 2.4 : 1);
    if(kept) c.setAttribute('filter','url(#ptGlow)');
    else if(recalled) c.setAttribute('class','recalled');
    const t=document.createElementNS(svgns,'title');
    t.textContent=`${m.text}\n归属节点：${m.owner||'—'} · 访问 ${m.access} 次 · ${m.age_h}h`
      + (kept? '\n◆ 本轮选中并共享给当前节点' : (recalled? '\n● 本轮召回候选' : ''));
    c.appendChild(t); rg.appendChild(c);
  });
  // 跨节点共享连线：私有记忆 → 共享侧虚位（虚线，随所选节点刷新）
  (extra||[]).forEach(m=>{
    const from=coord[m.id];
    if(!from) return;
    const cx=scaleX(m.x,true), cy=scaleY(m.y);
    const ln=document.createElementNS(svgns,'line');
    ln.setAttribute('x1',from.x); ln.setAttribute('y1',from.y);
    ln.setAttribute('x2',cx); ln.setAttribute('y2',cy);
    ln.setAttribute('stroke', cssTok('--inherit-line','#378add')); ln.setAttribute('stroke-width','1.7');
    ln.setAttribute('stroke-dasharray','6 5'); ln.setAttribute('class','inheritline');
    fg.appendChild(ln);
    const c=document.createElementNS(svgns,'circle');
    c.setAttribute('cx',cx); c.setAttribute('cy',cy); c.setAttribute('r',6);
    c.setAttribute('fill', cssTok('--inherit-dot','#e6f1fb'));
    c.setAttribute('stroke', cssTok('--inherit-line','#378add')); c.setAttribute('stroke-width','1.8');
    c.setAttribute('class','inheritdot');
    const t=document.createElementNS(svgns,'title');
    t.textContent=`当前节点共享自上游的记忆：${m.text||m.id}`;
    c.appendChild(t); sg.appendChild(c);
    const lab=document.createElementNS(svgns,'text');
    lab.setAttribute('x',cx+9); lab.setAttribute('y',cy+4); lab.setAttribute('font-size','11');
    lab.setAttribute('fill', cssTok('--inherit-line','#378add')); lab.textContent='继';
    sg.appendChild(lab);
  });
  shared.slice().sort((a,b)=>b.score-a.score).forEach((m,idx)=>{
    const cx=scaleX(m.x,true), cy=scaleY(m.y);
    const active=kep.has(m.id);
    if(active){
      const from=resident.find(r=>r.id===m.id);
      if(from){
        const ln=document.createElementNS(svgns,'line');
        ln.setAttribute('x1',scaleX(from.x,false)); ln.setAttribute('y1',scaleY(from.y));
        ln.setAttribute('x2',cx); ln.setAttribute('y2',cy);
        ln.setAttribute('stroke', cssTok('--accent','#0891b2')); ln.setAttribute('stroke-width','1.6');
        ln.setAttribute('stroke-dasharray','5 4'); ln.setAttribute('class','flowline');
        fg.appendChild(ln);
      }
    }
    const c=document.createElementNS(svgns,'circle');
    c.setAttribute('cx',cx); c.setAttribute('cy',cy);
    c.setAttribute('r', active? 6.5 : 5.5);
    if(active){ c.setAttribute('fill', cssTok('--accent','#0891b2')); c.setAttribute('class','shareddot'); }
    else{ c.setAttribute('fill', cssTok('--card-solid','#ffffff')); c.setAttribute('stroke', cssTok('--blue','#2563eb'));
      c.setAttribute('stroke-width','1.6'); c.setAttribute('class','sharedold'); }
    const t=document.createElementNS(svgns,'title');
    t.textContent=`#${idx+1} ${m.text} (score=${m.score})`
      + (active? ' · 本轮保留' : ' · 历史留存');
    c.appendChild(t); sg.appendChild(c);
    const lab=document.createElementNS(svgns,'text');
    lab.setAttribute('x',cx+9); lab.setAttribute('y',cy+4); lab.setAttribute('font-size','11');
    lab.setAttribute('fill', active? cssTok('--accent','#0d8a6b') : cssTok('--mute','#94a3b8'));
    lab.textContent=`#${idx+1}`;
    sg.appendChild(lab);
  });
  if(shared.length){
    const cap=document.createElementNS(svgns,'text');
    cap.setAttribute('x',755); cap.setAttribute('y',500); cap.setAttribute('text-anchor','middle');
    cap.setAttribute('font-size','11'); cap.setAttribute('fill', cssTok('--blue','#2563eb'));
    cap.textContent=`已用 ${shared.length} / ${maxShared||10}`; sg.appendChild(cap);
  }
}
async function refreshPools(recalledIds, keptIds, extraIds){
  const qs=(extraIds&&extraIds.length)? ('?ids='+encodeURIComponent(extraIds.join(','))) : '';
  const p=await fetch('/api/pools'+qs).then(r=>r.json());
  window.__residentCache = p.layout.resident||[];
  renderPools(window.__residentCache, p.layout.shared||[], recalledIds||[], keptIds||[],
    p.layout.max_shared||10, p.layout.extra||[]);
  // D-058：只有「本轮 kept 与上一轮不同」才放飞行动画 —— 否则切节点/重绘时
  // 每次都放一遍，动画就变成了噪声，反而看不出这是"新一轮共享"。
  const sig = (keptIds||[]).slice().sort().join(',');
  if (sig && sig !== window.__lastKeptSig){
    window.__lastKeptSig = sig;
    playFlight(window.__residentCache, p.layout.shared||[], keptIds||[]);
  } else if (!sig){
    window.__lastKeptSig = '';
  }
}
/* 随所选会话刷新双池视图：主节点=真实共享记忆状态；下游节点=其共享来源连线视图 */
async function updatePoolView(){
  if(!CUR || !H) return;
  if(isMain(CUR.session_id)){ await refreshPools(); return; }
  const sess=sessionById(CUR.session_id);
  const inherited=(sess.inherited&&sess.inherited.length)? sess.inherited
    : ((sess.rounds[0]&&sess.rounds[0].inherited)||[]);
  const res=window.__residentCache||[];
  const byText={}; res.forEach(m=>{ byText[m.text]=m.id; });
  const ids=inherited.map(t=>byText[t]).filter(Boolean);
  await refreshPools([], [], ids);
}

// ---- 右侧：调度明细 ----
function renderDetail(d){
  const w=d.weights||{}, pw=d.llm_prior||{};
  const wchips=Object.entries(w).map(([k,v])=>`<span class="wchip">${k} <b>${(+v).toFixed(2)}</b></span>`).join('');
  const pwchips=Object.entries(pw).map(([k,v])=>`<span class="wchip">${k} <b>${(+v).toFixed(2)}</b></span>`).join('');
  const rows=(d.candidates||[]).map(c=>{
    const s=c.scores;
    return `<tr class="${c.kept?'kept':''}"><td>${c.kept?'◆':'·'}</td><td>${esc(c.text)}</td>`+
      `<td>${s.time}</td><td>${s.semantic}</td><td>${s.frequency}</td><td>${s.task}</td><td><b>${s.final}</b></td></tr>`;
  }).join('');
  $('#detail').innerHTML=`
    <h3>本回合调度明细</h3>
    <div class="kv">查询：<b>${esc(d.query||'')}</b>　保留 <b>${d.kept_count}</b> 条 → 共享池　压缩比 <b>${d.compression_ratio}</b></div>
    <div style="margin-top:8px" class="mini">LLM 先验权重：</div><div class="wbar">${pwchips}</div>
    <div class="mini">混合权重（α·先验 + β·可学习）：</div><div class="wbar">${wchips}</div>
    <div class="mini" style="margin-top:6px">四维打分排行榜（final 降序）：</div>
    <table class="rank"><thead><tr><th></th><th>记忆</th><th>时间</th><th>语义</th><th>频率</th><th>任务</th><th>final</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// ---- 追问（真实调度） ----
let busy=false;

/* D-053 修「发问静默失效」：
   原代码 `if(busy || !CUR) return;` 直接吞掉请求 —— 未选中会话节点时，
   用户能打字、点了发送，但**什么都没发生、也没有任何提示**，
   首次使用会以为页面坏了。改为：发送按钮显式置灰 + 占位文案说明原因。 */
function syncComposer(){
  const btn=$('#send'), inp=$('#q');
  const noSession = !CUR;
  btn.disabled = busy || noSession;
  btn.textContent = busy ? '调度中…' : (noSession ? '先选会话' : '发送');
  inp.placeholder = noSession
    ? '请先在上方选择一个会话节点'
    : '在该节点所在会话继续追问（会走真实调度流水线）';
  inp.disabled = busy;
}

/* D-054 SSE 客户端。
   为什么不用 EventSource：它只支持 GET、不能带 body，
   而追问必须 POST {session_id, query}。所以走 fetch + ReadableStream 手动解 SSE。
   解法：按 \n\n 切事件块，每块取 `data:` 行 JSON.parse。 */
async function ssePost(url, body, onEvent, signal){
  const resp = await fetch(url, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body),
    signal,
  });
  if(!resp.ok || !resp.body){
    throw new Error('HTTP '+resp.status);
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder('utf-8');
  let buf = '', finished = false;
  try{
    for(;;){
      const {done, value} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      let idx;
      while((idx = buf.indexOf('\n\n')) >= 0){
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for(const line of raw.split('\n')){
          if(!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if(!data) continue;
          let ev;
          try{ ev = JSON.parse(data); }
          catch(err){ continue; }        // 单条坏事件不该中断整条流
          if(ev && (ev.type === 'done' || ev.type === 'error')) finished = true;
          onEvent(ev);
        }
      }
      // ⚠️ 必须主动 cancel：服务端是 BaseHTTPRequestHandler（HTTP/1.0），
      // 响应头又发了 Connection: keep-alive，结果 socket 不会被关闭 ——
      // 浏览器 reader.read() 会一直挂着，整个 UI 卡在"调度中"。
      if(finished){ try{ await reader.cancel(); }catch(e){} return; }
    }
    // 收尾时若缓冲区还有未成块的尾巴
    if(buf.trim()){
      for(const line of buf.split('\n')){
        if(line.startsWith('data:')){
          try{ onEvent(JSON.parse(line.slice(5).trim())); }catch(e){}
        }
      }
    }
  }finally{
    try{ reader.releaseLock(); }catch(e){}
  }
}

$('#composer').addEventListener('submit', async e=>{
  e.preventDefault();
  if(busy) return;
  if(!CUR){ flashHint('请先在上方选择一个会话节点，再开始追问'); return; }
  const q=$('#q').value.trim(); if(!q){ flashHint('请输入问题'); return; }
  busy=true; syncComposer();
  window.__curQuery = q;                 // D-055：供 setStage 填「用户输入」卡
  resetStage();
  setStage('input');
  const pending=showPending(q);          // 立即给出「调度中」气泡，不再干等

  // 流式气泡：调度结果一到就先渲染四维表，LLM 回答随后逐字追加
  let streamBox=null, acc='';
  const ensureStream=()=>{
    if(streamBox) return streamBox;
    const d=document.createElement('div');
    d.className='msg pending streaming';
    d.innerHTML=`<div class="who"><span class="rid">…</span><span>回答生成中</span></div>
      <div class="bubble md"><span class="cursor-blink"></span></div>`;
    $('#messages').appendChild(d);
    const box=$('#messages'); box.scrollTop=box.scrollHeight;
    if(window.Fx) window.Fx.enter(d,'bottom');
    streamBox=d;
    return d;
  };
  const appendDelta=(t)=>{
    const d=ensureStream();
    acc += t;
    const b=d.querySelector('.bubble');
    b.innerHTML = mdRender(acc) + '<span class="cursor-blink"></span>';
    const box=$('#messages'); box.scrollTop=box.scrollHeight;
  };

  let payload=null, round=null, err=null;
  try{
    await ssePost('/api/canvas/ask', {session_id:CUR.session_id, query:q}, ev=>{
      if(ev.type==='stage'){ setStage(ev.stage); }
      else if(ev.type==='round'){ payload=ev; }
      else if(ev.type==='delta'){ appendDelta(ev.text||''); }
      else if(ev.type==='warn'){ flashHint(ev.message||'提示'); }
      else if(ev.type==='error'){ err=new Error(ev.message||'调度失败'); }
      else if(ev.type==='done'){ round=ev.round; }
    });
  }catch(err0){ err=err0; }

  if(streamBox) streamBox.remove();
  if(pending) pending.remove();

  if(err){
    flashHint('追问失败：'+err.message);
    busy=false; $('#q').value=''; syncComposer();
    return;
  }
  // 阶段动画补完（流式下阶段是实时收到的，这里只做收尾确认）
  ['recall','score','weight','rank','select'].forEach((s,i)=>setTimeout(()=>setStage(s), i*120));
  setTimeout(()=>setStage('answer'), 700);

  // 更新上下文历史（新回合长出来）
  if(round){
    const sess=sessionById(CUR.session_id);
    sess.rounds.push(round);
    selectSession(CUR.session_id, round.round_id);
  }
  if(payload) renderDetail(Object.assign({query:q}, payload));
  animateDetail();
  try{
    await refreshPools((payload&&payload.candidates||[]).map(c2=>c2.id),
                       (payload&&payload.kept_ids)||[]);
  }catch(e2){ /* 池视图刷新失败不影响主流程 */ }
  busy=false; $('#q').value=''; syncComposer();
});

/* 轻量提示条（替代 alert —— alert 打断观感，且演示时很丑） */
let hintTimer=null;
function flashHint(msg){
  let el=document.getElementById('wbHint');
  if(!el){
    el=document.createElement('div');
    el.id='wbHint';
    el.className='wb-hint';
    document.body.appendChild(el);
  }
  el.textContent=msg;
  el.classList.add('show');
  clearTimeout(hintTimer);
  hintTimer=setTimeout(()=>el.classList.remove('show'), 2600);
  if(window.Fx) window.Fx.enter(el,'top');
}

/* 「调度中」占位气泡：真实调度要跑 embedding + 权重 + 选源，没有它就是干等 */
function showPending(q){
  const box=$('#messages'); if(!box) return null;
  const d=document.createElement('div');
  d.className='msg pending';
  d.innerHTML=`<div class="who"><span class="rid">…</span><span>正在调度</span></div>
    <div class="bubble user">${String(q).replace(/[&<>]/g,'')}</div>
    <div class="bubble think"><span class="tdot"></span><span class="tdot"></span>
      <span class="tdot"></span> 正在召回 · 四维打分 · 混合权重 · 选源</div>`;
  box.appendChild(d);
  box.scrollTop=box.scrollHeight;
  if(window.Fx) window.Fx.enter(d,'bottom');
  return d;
}

/* D-053：调度明细逐块入场 + 排行榜逐行 stagger，让"算出了什么"可读 */
function animateDetail(){
  if(!window.Fx) return;
  const box=$('#detail'); if(!box) return;
  window.Fx.rise(box.querySelectorAll('h3, .kv, .mini, .wbar, .rank tbody tr'), 1);
}

// ---- 创建下游节点（全流程过渡：当前节点 → 选源与共享 → 节点就绪） ----
function showFlowTransition(fromRid, query, subTitle, keptCount, streamed){
  const ov=$('#flowov');
  $('#flowCap').textContent='创建下游节点 · 全流程';
  $('#fs1d').textContent=`当前节点 ${fromRid} · 共享起点`;
  $('#fs2d').textContent=`按第一问「${query.slice(0,24)}${query.length>24?'…':''}」精选 ${keptCount} 条记忆`;
  // D-055：若流式阶段已把前两步点亮/填好，就不重置（避免进度条回退）
  if(!$('#fs1').classList.contains('on')) $('#fs1').classList.add('on');
  if(!$('#fs2').classList.contains('on')){
    $('#fs2').classList.add('on');
    document.querySelectorAll('.fsep')[0].classList.add('on');
  }
  // 第三步：优先展示流式收到的回答开头，让"下游节点第一问答了什么"可见
  const head=(streamed||'').trim().slice(0,60);
  $('#fs3d').textContent = head ? `${subTitle} · ${head}${streamed.trim().length>60?'…':''}` : subTitle;
  ov.classList.add('show');
  if(window.Fx) window.Fx.enter(ov,'top');
  return new Promise(res=>{
    setTimeout(()=>{$('#fs3').classList.add('on');
      document.querySelectorAll('.fsep')[1].classList.add('on');},420);
    setTimeout(()=>{
      ov.classList.remove('show');
      // 复位：下次创建时重新走完整动画
      ['fs1','fs2','fs3'].forEach(id=>$('#'+id).classList.remove('on'));
      document.querySelectorAll('.fsep').forEach(s=>s.classList.remove('on'));
      res();
    },1600);
  });
}

$('#nsGo').addEventListener('click', async ()=>{
  if(!CUR || !isMain(CUR.session_id)){ flashHint('请先在左侧选中一个主节点回合'); return; }
  const title=$('#nsTitle').value.trim(), query=$('#nsQuery').value.trim();
  if(!query){ flashHint('请填写下游节点的第一问（调度按它选源并共享）'); return; }
  const btn=$('#nsGo'); btn.disabled=true; btn.textContent='调度中…';
  const fromRound=CUR.round_id;
  window.__curQuery = query;              // D-055：顶部「用户输入」卡显示下游节点第一问
  resetStage(); setStage('input');

  // D-055：创建下游节点也走 SSE —— 原来是一次性 JSON，而它内部同样要跑
  // schedule_once + LLM 回答，前端是干等好几秒。现在把"记忆调度 → 回答"逐段推过来，
  // 并用已有的三步全流程动画（当前节点 → 选源共享 → 节点就绪）实时填充说明文案。
  let roundData=null, keptN=0, err=null, acc='';
  try{
    await ssePost('/api/canvas/create_sub',
      {from_round:fromRound, title, query},
      (ev)=>{
        if(ev.type==='stage'){
          setStage(ev.stage);              // D-055：创建下游节点也驱动顶部流水线
          if(ev.stage==='recall'){
            $('#flowCap').textContent='创建下游节点 · 全流程';
            $('#fs1d').textContent=`当前节点 ${fromRound} · 共享起点`;
            $('#fs1').classList.add('on');
          }else if(ev.stage==='answer' || ev.stage==='score'){
            $('#fs2d').textContent = keptN ? `按第一问「${query.slice(0,24)}」精选 ${keptN} 条记忆`
                                           : `按第一问「${query.slice(0,24)}」调度记忆…`;
            if(keptN){ $('#fs2').classList.add('on'); document.querySelectorAll('.fsep')[0].classList.add('on'); }
          }
        }
        else if(ev.type==='round'){ keptN = ev.kept_count || 0; }
        else if(ev.type==='delta'){ acc += ev.text || ''; }
        else if(ev.type==='error'){ err = new Error(ev.message||'创建失败'); }
        else if(ev.type==='done'){ roundData = ev; }
      });
  }catch(e){ err = e; }

  if(err){
    flashHint('创建下游节点失败：'+err.message);
    btn.disabled=false; btn.textContent='创建';
    return;
  }
  const d = roundData || {};
  try{
    H = (await (await fetch('/api/canvas/history')).json()).history;
  }catch(e){ /* 历史刷新失败不阻断跳转 */ }
  $('#nsTitle').value=''; $('#nsQuery').value='';
  btn.textContent='创建成功 · 正在进入节点';
  // 复用全流程动画补完最后一步（文案已在前面的事件里实时填好了）
  await showFlowTransition(fromRound, query, (d.session&&d.session.title)||'下游节点',
                           keptN, acc);
  if(d.session && d.round) selectSession(d.session.session_id, d.round.round_id);
  btn.disabled=false; btn.textContent='创建';
});

// ---- 启动 ----
async function boot(){
  // D-053：主题切换（三页统一）+ 顶部流水线节点补实时数值
  const tb=document.getElementById('themeBtn');
  if(tb && window.SRTPTheme){
    tb.addEventListener('click', function(){ SRTPTheme.toggle(); });
    tb.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '+
      'stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  }
  // D-061：高对比模式（一键把弱对比令牌推到 AAA，强光/投影下更清晰）
  const cb=document.getElementById('contrastBtn');
  if(cb && window.SRTPTheme){
    cb.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '+
      'stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/>'+
      '<path d="M12 3.5v17a8.5 8.5 0 0 0 0-17z" fill="currentColor" stroke="none"/></svg>';
    cb.addEventListener('click', function(){ SRTPTheme.toggleContrast(); });
    // 状态回显：刷新后按钮要反映当前是否高对比
    if(SRTPTheme.contrastOn()) cb.classList.add('on');
    window.addEventListener('srtp:contrast', function(e){
      cb.classList.toggle('on', !!e.detail);
    });
  }
  /* D-061：记忆池是 SVG，点的**填色是运行时从 CSS 变量取值**写进去的
     （不是 CSS 控制），所以切主题/切高对比后必须整体重绘，否则颜色停在旧值。 */
  const repaintPools = function(){ if(window.__residentCache) updatePoolView().catch(function(){}); };
  window.addEventListener('srtp:theme', repaintPools);
  window.addEventListener('srtp:contrast', repaintPools);
  try{
    const st=await fetch('/api/status').then(r=>r.json());
    const b=$('#modeBadge');
    if(st.agent_mode==='agentscope'){b.textContent='AgentScope 真实 Agent';b.className='badge agentscope';}
    else {b.textContent='真实 LLM · Headless 中间件';b.className='badge headless';}
  }catch(e){}
  try{
    const p=await fetch('/api/pools').then(r=>r.json());
    window.__maxShared=p.layout.max_shared||10;
    fillStepperStats(p);
  }catch(e){}
  H=(await (await fetch('/api/canvas/history')).json()).history;
  await refreshPools();   // 先缓存节点私有记忆（共享连线需要 text→id 匹配）
  const sid=params.get('session'), rid=params.get('round');
  if(sid && sessionById(sid)) selectSession(sid, rid);
  else selectSession(H.main.session_id);
  syncComposer();
}

/* D-053：顶部 6 个流水线节点原来全是文字描述（"向量检索候选"/"α·先验+β·学习"），
   看不出系统现在什么状态。改成**实时数值**：私有记忆规模 / 共享上限 / 本轮候选数等，
   一眼就知道系统跑到哪一步、有没有在调度。 */
function fillStepperStats(pools){
  const set=(s,val)=>{ const n=document.querySelector(`#stepper .node[data-s="${s}"] .d`);
    if(n) n.textContent=val; };
  try{
    const r=(pools&&pools.resident)||[], sh=(pools&&pools.shared)||[];
    set('recall', `${r.length} 条私有记忆 · 全量只读`);
    set('select', `≤${(pools&&pools.layout&&pools.layout.max_shared)||10} 条精选写入`);
    set('answer', `共享池现有 ${sh.length} 条`);
  }catch(e){}
}
boot();

})();
