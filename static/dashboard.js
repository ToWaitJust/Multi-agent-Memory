/* 控制台（dashboard）脚本 —— 迁移自 dashboard.html 内联 <script> */
(function(){
'use strict';

"use strict";
const $ = s => document.querySelector(s);
const svgns = "http://www.w3.org/2000/svg";
const TAGCOL = {profile:"#6366f1",tech_stack:"#0ea5e9",architecture:"#10b981",research:"#f59e0b",daily:"#9ca3af",
  project:"#8b5cf6",main_task:"#8b5cf6",attention:"#10b981",weight:"#f59e0b",reme:"#0ea5e9",embedding:"#0ea5e9",
  share:"#ef4444",retrieval:"#0ea5e9",ablation:"#f59e0b",architecture2:"#10b981"};

let H = null;                       // 画布历史（会话 + 回合）
let CUR = null;                     // {session_id, round_id}
const params = new URLSearchParams(location.search);

function esc(s){ return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
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
  mk(H.main.session_id, '主线', H.main.rounds.length);
  H.subs.forEach(s=>mk(s.session_id, s.title.replace(/^副线 · /,''), s.rounds.length));
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
    ? `<b>主线会话</b> · 常驻完整池全量只读，追问基于调度精选记忆作答，每回合是一个画布节点。`
    : `<b>${esc(sess.title)}</b> · 独立副线，创建时从主线 ${esc(sess.derived_from||'')} 继承了一批精选记忆。`;
  box.appendChild(cap);
  // 副线：记忆来源卡（源节点可点击跳回主线该回合）
  if(!main){
    const src=document.createElement('div'); src.className='srccard';
    const srcRound=H.main.rounds.find(r=>r.round_id===(sess.derived_from||sess.parent));
    const inherited=(sess.inherited && sess.inherited.length) ? sess.inherited
      : ((sess.rounds[0]&&sess.rounds[0].inherited)||[]);
    let html=`<div class="cap">记忆来源 · 记忆继承链路</div>
      <span class="srcnode" onclick="selectSession('${H.main.session_id}','${esc(sess.derived_from||sess.parent||'')}')">
        <span class="arr">↩</span>主线 ${esc(sess.derived_from||sess.parent||'?')}${srcRound? ' · '+esc(srcRound.title||''):''}</span>`;
    if(inherited.length){
      html+=`<div class="srclen">创建时按第一问「${esc((sess.rounds[0]&&sess.rounds[0].user)||'')}」调度继承 <b>${inherited.length}</b> 条记忆：</div>
        <div class="srcchips">${inherited.map(t=>`<div class="c">· ${esc(t)}</div>`).join('')}</div>`;
    }else{
      html+=`<div class="srcnote">演示数据未附继承清单 —— 在主线选中回合现场创建副线，可看到完整的"按第一问调度 → 继承记忆"效果。</div>`;
    }
    src.innerHTML=html;
    box.appendChild(src);
  }
  sess.rounds.forEach(r=>{
    box.appendChild(roundCard(sess, r, r.round_id===CUR.round_id));
  });
  // 创建副线入口：仅主线回合
  const sc=$('#subcreate');
  if(main){ sc.classList.add('show'); $('#subFromRid').textContent=CUR.round_id; }
  else sc.classList.remove('show');
  const curEl=box.querySelector('.msg.cur');
  if(curEl) curEl.scrollIntoView({block:'center'});
  box.scrollTop=box.scrollHeight;
  // 返回画布时高亮当前节点
  $('#canvasLink').href='/canvas?hl='+encodeURIComponent(CUR.session_id+'|'+CUR.round_id);
  // 双池视图随会话刷新：副线显示其继承记忆连线
  updatePoolView();
}

function roundCard(sess, r, isCur){
  const main=isMain(sess.session_id);
  const d=document.createElement('div');
  d.className='msg '+(main?'assistant':'assistant')+(isCur?' cur':'');
  const who=main?'主线 Agent':'副线 Agent（基于继承/共享池记忆）';
  const w=document.createElement('div'); w.className='who';
  w.innerHTML=`<span class="rid">${esc(r.round_id)}</span><span>${esc(r.title||'')}</span><span>${fmtTs(r.ts)}</span>`;
  d.appendChild(w);
  const u=document.createElement('div'); u.className='msg user';
  const uw=document.createElement('div'); uw.className='who'; uw.textContent='用户';
  const ub=document.createElement('div'); ub.className='bubble'; ub.textContent=r.user;
  u.appendChild(uw); u.appendChild(ub); d.appendChild(u);
  const ab=document.createElement('div'); ab.className='bubble'; ab.textContent=r.assistant;
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
function renderPools(resident, shared, recalledIds, keptIds, maxShared, extra){
  const rg=$('#residentDots'), sg=$('#sharedDots'), fg=$('#flows');
  rg.innerHTML=''; sg.innerHTML=''; fg.innerHTML='';
  const rec=new Set(recalledIds), kep=new Set(keptIds);
  const coord={};
  resident.forEach(m=>{
    const cx=scaleX(m.x,false), cy=scaleY(m.y);
    coord[m.id]={x:cx, y:cy};
    const recalled=rec.has(m.id), kept=kep.has(m.id);
    const c=document.createElementNS(svgns,'circle');
    c.setAttribute('cx',cx); c.setAttribute('cy',cy);
    c.setAttribute('r', recalled? 7 : 4.5);
    c.setAttribute('fill', TAGCOL[m.tag]||'#9ca3af');
    c.setAttribute('stroke', kept? '#10a37f' : '#ffffff');
    c.setAttribute('stroke-width', kept? 2.5 : 1);
    if(recalled) c.setAttribute('class','recalled');
    const t=document.createElementNS(svgns,'title');
    t.textContent=`${m.text} [${m.tag}] age=${m.age_h}h`
      + (kept? ' · 本轮保留进共享池' : (recalled? ' · 本轮召回候选' : ''));
    c.appendChild(t); rg.appendChild(c);
  });
  // 副线继承连线：常驻池 → 共享池侧虚位（蓝色虚线，随所选副线刷新）
  (extra||[]).forEach(m=>{
    const from=coord[m.id];
    if(!from) return;
    const cx=scaleX(m.x,true), cy=scaleY(m.y);
    const ln=document.createElementNS(svgns,'line');
    ln.setAttribute('x1',from.x); ln.setAttribute('y1',from.y);
    ln.setAttribute('x2',cx); ln.setAttribute('y2',cy);
    ln.setAttribute('stroke','#378add'); ln.setAttribute('stroke-width','1.7');
    ln.setAttribute('stroke-dasharray','6 5'); ln.setAttribute('class','inheritline');
    fg.appendChild(ln);
    const c=document.createElementNS(svgns,'circle');
    c.setAttribute('cx',cx); c.setAttribute('cy',cy); c.setAttribute('r',6);
    c.setAttribute('fill','#e6f1fb'); c.setAttribute('stroke','#378add'); c.setAttribute('stroke-width','1.8');
    c.setAttribute('class','inheritdot');
    const t=document.createElementNS(svgns,'title');
    t.textContent=`当前副线继承记忆：${m.text||m.id}`;
    c.appendChild(t); sg.appendChild(c);
    const lab=document.createElementNS(svgns,'text');
    lab.setAttribute('x',cx+9); lab.setAttribute('y',cy+4); lab.setAttribute('font-size','11');
    lab.setAttribute('fill','#378add'); lab.textContent='继';
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
        ln.setAttribute('stroke','#10a37f'); ln.setAttribute('stroke-width','1.6');
        ln.setAttribute('stroke-dasharray','5 4'); ln.setAttribute('class','flowline');
        fg.appendChild(ln);
      }
    }
    const c=document.createElementNS(svgns,'circle');
    c.setAttribute('cx',cx); c.setAttribute('cy',cy);
    c.setAttribute('r', active? 6.5 : 5.5);
    if(active){ c.setAttribute('fill','#10a37f'); c.setAttribute('class','shareddot'); }
    else{ c.setAttribute('fill','#ffffff'); c.setAttribute('stroke','#2563eb');
      c.setAttribute('stroke-width','1.6'); c.setAttribute('class','sharedold'); }
    const t=document.createElementNS(svgns,'title');
    t.textContent=`#${idx+1} ${m.text} (score=${m.score})`
      + (active? ' · 本轮保留' : ' · 历史留存');
    c.appendChild(t); sg.appendChild(c);
    const lab=document.createElementNS(svgns,'text');
    lab.setAttribute('x',cx+9); lab.setAttribute('y',cy+4); lab.setAttribute('font-size','11');
    lab.setAttribute('fill', active? '#0d8a6b' : '#94a3b8'); lab.textContent=`#${idx+1}`;
    sg.appendChild(lab);
  });
  if(shared.length){
    const cap=document.createElementNS(svgns,'text');
    cap.setAttribute('x',755); cap.setAttribute('y',500); cap.setAttribute('text-anchor','middle');
    cap.setAttribute('font-size','11'); cap.setAttribute('fill','#2563eb');
    cap.textContent=`已用 ${shared.length} / ${maxShared||10}`; sg.appendChild(cap);
  }
}
async function refreshPools(recalledIds, keptIds, extraIds){
  const qs=(extraIds&&extraIds.length)? ('?ids='+encodeURIComponent(extraIds.join(','))) : '';
  const p=await fetch('/api/pools'+qs).then(r=>r.json());
  window.__residentCache = p.layout.resident||[];
  renderPools(window.__residentCache, p.layout.shared||[], recalledIds||[], keptIds||[],
    p.layout.max_shared||10, p.layout.extra||[]);
}
/* 随所选会话刷新双池视图：主线=真实共享池状态；副线=其继承记忆的连线视图 */
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
$('#composer').addEventListener('submit', async e=>{
  e.preventDefault();
  if(busy || !CUR) return;
  const q=$('#q').value.trim(); if(!q) return;
  busy=true; $('#send').disabled=true;
  resetStage();
  try{
    const resp=await fetch('/api/canvas/ask',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({session_id:CUR.session_id, query:q})});
    const d=await resp.json();
    if(!d.ok) throw new Error(d.error||'失败');
    // 阶段动画（数据已到，快速走完流水线视觉）
    ['recall','score','weight','rank','select'].forEach((s,i)=>setTimeout(()=>setStage(s), i*180));
    setTimeout(()=>setStage('answer'), 180*5);
    // 更新上下文历史（新回合长出来）
    const sess=sessionById(CUR.session_id);
    sess.rounds.push(d.round);
    selectSession(CUR.session_id, d.round.round_id);
    renderDetail(Object.assign({query:q}, d));
    await refreshPools((d.candidates||[]).map(c=>c.id), d.kept_ids||[]);
  }catch(err){ alert('追问失败：'+err.message); }
  busy=false; $('#send').disabled=false; $('#q').value='';
});

// ---- 创建副线（含全流程过渡动画：主线节点 → 记忆调度 → 副线控制台就绪） ----
function showFlowTransition(fromRid, query, subTitle, keptCount){
  const ov=$('#flowov');
  $('#flowCap').textContent='副线创建中 · 全流程';
  $('#fs1d').textContent=`主线 ${fromRid} · 继承源节点`;
  $('#fs2d').textContent=`按第一问「${query.slice(0,24)}${query.length>24?'…':''}」精选 ${keptCount} 条记忆`;
  $('#fs3d').textContent=`${subTitle}`;
  ['fs1','fs2','fs3'].forEach(id=>$('#'+id).classList.remove('on'));
  document.querySelectorAll('.fsep').forEach(s=>s.classList.remove('on'));
  ov.classList.add('show');
  return new Promise(res=>{
    setTimeout(()=>{$('#fs1').classList.add('on'); document.querySelectorAll('.fsep')[0].classList.add('on');},350);
    setTimeout(()=>{$('#fs2').classList.add('on'); document.querySelectorAll('.fsep')[1].classList.add('on');},1000);
    setTimeout(()=>{$('#fs3').classList.add('on');},1650);
    setTimeout(()=>{ov.classList.remove('show'); res();},2350);
  });
}

$('#nsGo').addEventListener('click', async ()=>{
  if(!CUR || !isMain(CUR.session_id)) return;
  const title=$('#nsTitle').value.trim(), query=$('#nsQuery').value.trim();
  if(!query){ alert('请填写副线的第一问（记忆调度按它进行）'); return; }
  const btn=$('#nsGo'); btn.disabled=true; btn.textContent='调度中…';
  try{
    const resp=await fetch('/api/canvas/create_sub',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({from_round:CUR.round_id, title, query})});
    const d=await resp.json();
    if(!d.ok) throw new Error(d.error||'创建失败');
    H = (await (await fetch('/api/canvas/history')).json()).history;
    $('#nsTitle').value=''; $('#nsQuery').value='';
    btn.textContent='创建成功 · 正在进入副线';
    await showFlowTransition(CUR.round_id, query, d.session.title, d.kept_count);
    // 主动跳转到新副线的控制台（URL 同步更新，可刷新/分享）
    selectSession(d.session.session_id, d.round.round_id);
  }catch(err){ alert('创建副线失败：'+err.message); }
  btn.disabled=false; btn.textContent='创建';
});

// ---- 启动 ----
async function boot(){
  try{
    const st=await fetch('/api/status').then(r=>r.json());
    const b=$('#modeBadge');
    if(st.agent_mode==='agentscope'){b.textContent='AgentScope 真实 Agent';b.className='badge agentscope';}
    else {b.textContent='真实 LLM · Headless 中间件';b.className='badge headless';}
  }catch(e){}
  try{
    const p=await fetch('/api/pools').then(r=>r.json());
    window.__maxShared=p.layout.max_shared||10;
  }catch(e){}
  H=(await (await fetch('/api/canvas/history')).json()).history;
  await refreshPools();   // 先缓存常驻池（副线继承连线需要 text→id 匹配）
  const sid=params.get('session'), rid=params.get('round');
  if(sid && sessionById(sid)) selectSession(sid, rid);
  else selectSession(H.main.session_id);
}
boot();

})();
