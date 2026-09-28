/* ============================================================================
   图拓扑视图（Cytoscape.js 驱动）
   替代原先"手写 SVG + 字符串模板拼 DOM"的实现 —— 那个版本没有缩放/平移、
   样式无法按数据映射、加功能要改一大坨模板，且存在点击后空白的 JS 异常。

   本文件职责：
     - 用 Cytoscape 承载图（内置缩放/平移/拖拽/框选/布局切换/样式选择器）
     - 右侧面板：图状态仪表 → 节点详情 → Stage A 选源明细（三态切换）
     - Stage A 选源可视化：源节点高亮 + 边流动 + src_score 标签 + 保持源路径聚焦
   ========================================================================== */
(function () {
  'use strict';

  var TYPE_COLOR = {depends_on:'#3b82f6', derives_from:'#0f9d76',
                    references:'#e08a3c', similar_to:'#9aa9a3'};
  var TYPE_LABEL = {depends_on:'依赖', derives_from:'派生', references:'引用', similar_to:'相近'};

  var cy = null, G = {nodes:[], edges:[], source:'empty'}, SA = null, SEL = null;
  var side = document.getElementById('side');
  var tip = document.getElementById('tip');

  /* ---------------- 工具 ---------------- */
  function esc(s){
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }
  function edgeId(e){ return e.src + '→' + e.dst; }

  /* ---------------- 构建 Cytoscape 图 ---------------- */
  function buildElements(){
    var els = [];
    var outDeg = {};
    G.edges.forEach(function (e) {
      if (e.status !== 'cycle_rejected') outDeg[e.src] = (outDeg[e.src] || 0) + 1;
    });
    G.nodes.forEach(function (n) {
      var nm = n.n_memories || 0;
      els.push({data: {
        id: n.node_id,
        // 两行标签：主题（主）+ node_id（次），避免"只看主题不知道数据键"的信息损失
        label: (n.topic || n.node_id) + '\n' + n.node_id,
        mem: nm,
        forked: (outDeg[n.node_id] || 0) >= 2,
        kind: 'node',
        summary: n.summary || ''
      }});
    });
    G.edges.forEach(function (e) {
      els.push({data: {
        id: edgeId(e),
        source: e.src, target: e.dst,
        type: e.type, strength: (typeof e.strength === 'number' ? e.strength : 0.5),
        rejected: e.status === 'cycle_rejected',
        color: TYPE_COLOR[e.type] || '#9aa9a3',
        sim: (typeof e.similarity === 'number' ? e.similarity : null),
        ev: e.evidence || '',
        inter: e.interactions || 0,
        kind: 'edge'
      }});
    });
    return els;
  }

  var CY_STYLE = [
    {selector:'node', style:{
      'shape':'round-rectangle', 'width':150, 'height':48,
      'background-color':'#ffffff', 'background-opacity':1,
      'border-width':1.4, 'border-color':'#cfdcd6',
      'label':'data(label)', 'font-size':11.5, 'font-weight':600, 'color':'#152b23',
      'text-valign':'center', 'text-halign':'center', 'text-wrap':'wrap',
      'text-max-width':134, 'line-height':1.3,
      'shadow-blur':14, 'shadow-color':'rgba(16,42,33,.13)',
      'shadow-opacity':0.55, 'shadow-offset-y':3, 'overlay-opacity':0,
      'transition-property':'border-color,border-width,background-color',
      'transition-duration':'130ms'
    }},
    {selector:'node[forked]', style:{'border-color':'#b9d3c8','border-width':1.8}},
    {selector:'node.dim', style:{'opacity':0.28}},
    {selector:'node.sel', style:{
      'border-color':'#0f9d76', 'border-width':3,
      'shadow-blur':26, 'shadow-color':'#0f9d76', 'shadow-opacity':0.65
    }},
    {selector:'node.src', style:{
      'border-color':'#e08a3c', 'border-width':3, 'background-color':'#fff8f0',
      'shadow-blur':22, 'shadow-color':'#e08a3c', 'shadow-opacity':0.5
    }},
    {selector:'node.up', style:{'border-color':'#3b82f6','border-width':2.2}},
    {selector:'edge', style:{
      'curve-style':'bezier', 'width':'mapData(strength,0.1,1,1.2,5.5)',
      'line-color':'data(color)', 'target-arrow-color':'data(color)',
      'target-arrow-shape':'triangle', 'arrow-scale':0.85, 'opacity':0.72,
      'transition-property':'opacity,line-color,width', 'transition-duration':'130ms',
      'overlay-opacity':0
    }},
    {selector:'edge[?rejected]', style:{
      'line-style':'dashed', 'opacity':0.35, 'target-arrow-shape':'none'
    }},
    {selector:'edge.dim', style:{'opacity':0.06}},
    {selector:'edge.hot', style:{'opacity':1, 'width':5.5}},
    {selector:'edge.flow', style:{
      'line-style':'dashed', 'line-dash-pattern':[7,5], 'opacity':1, 'width':5.5
    }}
  ];

  function initCy(){
    cy = cytoscape({
      container: document.getElementById('cy'),
      elements: buildElements(),
      style: CY_STYLE,
      wheelSensitivity: 0.22,
      minZoom: 0.25, maxZoom: 2.6,
      boxSelectionEnabled: true,
      layout: {name:'breadthfirst', directed:true, spacingFactor:1.12,
               padding:46, animate:false, grid:false, avoidOverlap:true}
    });
    cy.on('tap', 'node', function (ev) {
      var d = ev.target.data();
      select(d.id);
      fetchStageA(d.id);        // 后台预取选源（点按钮即可秒显）
    });
    cy.on('tap', 'edge', function (ev) {
      showEdgeDetail(ev.target.data());
    });
    cy.on('tap', function (ev) { if (ev.target === cy) select(null); });
    cy.on('mouseover', 'edge', function (ev) { showTip(ev.originalEvent, ev.target.data()); });
    cy.on('mousemove', 'edge', function (ev) { moveTip(ev.originalEvent); });
    cy.on('mouseout', 'edge', function () { tip.classList.remove('show'); });
  }

  /* ---------------- 节点选中与高亮 ---------------- */
  function select(id){
    SEL = id;
    cy.elements().removeClass('sel dim up src');
    if (!id){ renderDashboard(); return; }
    var node = cy.getElementById(id);
    if (!node || node.empty()) { renderDashboard(); return; }
    node.addClass('sel');
    var ins = node.incomers('edge[status != "cycle_rejected"]');
    var outs = node.outgoers('edge');
    var near = node.union(ins).union(outs).union(ins.sources()).union(outs.targets());
    cy.elements().difference(near).addClass('dim');
    ins.sources().addClass('up');
    renderNode(id);
  }

  /* ---------------- Stage A 选源 ---------------- */
  function fetchStageA(nodeId){
    var q = (document.getElementById('q').value || '').trim();
    if (!nodeId || !q) return Promise.resolve(null);
    return fetch('/api/stage_a?node=' + encodeURIComponent(nodeId) + '&q=' + encodeURIComponent(q))
      .then(function (r) { return r.json(); })
      .then(function (r) { return r && r.ok ? r.stage_a : null; })
      .catch(function () { return null; });
  }

  function runStageA(){
    var nodeId = SEL;
    if (!nodeId){
      flash('先在图上点一个节点作为「当前节点」');
      return;
    }
    if (!(document.getElementById('q').value || '').trim()){
      flash('先在输入框填写查询');
      return;
    }
    var btn = document.getElementById('saBtn');
    btn.disabled = true; btn.textContent = '计算中…';
    fetchStageA(nodeId).then(function (sa) {
      btn.disabled = false; btn.textContent = '计算选源';
      if (!sa){ flash('选源失败：请确认后端可用'); return; }
      SA = sa; paintStageA();
    }).catch(function (e) {
      btn.disabled = false; btn.textContent = '计算选源';
      flash('选源失败：' + (e && e.message ? e.message : e));
    });
  }

  function paintStageA(){
    cy.elements().removeClass('src flow hot dim');
    if (!SA){ renderDashboard(); return; }
    var cur = cy.getElementById(SA.cur_node);
    cur.addClass('sel');
    var keep = cur.closedNeighborhood();
    SA.sources.forEach(function (s) {
      var n = cy.getElementById(s.node_id);
      if (n.empty()) return;
      n.addClass('src');
      var e = cy.getElementById(s.node_id + '→' + SA.cur_node);
      if (!e.empty()) e.addClass('flow');
      keep = keep.union(n).union(e);
    });
    cy.elements().difference(keep).addClass('dim');
    // 不 fit：保持用户当前视野，避免自动缩放到"当前节点+源"的小范围后
    // 整图变小看不清、失去上下文（用户想聚焦时可点「适应窗口」）。
    renderStagePanel();
  }

  /* ---------------- 右侧面板三态 ---------------- */
  function renderDashboard(){
    var n = G.nodes.length, m = G.edges.length, rej = 0, sSum = 0, sN = 0;
    var byType = {};
    G.edges.forEach(function (e) {
      byType[e.type] = (byType[e.type] || 0) + 1;
      if (e.status === 'cycle_rejected') rej++;
      if (typeof e.strength === 'number'){ sSum += e.strength; sN++; }
    });
    var forked = 0;
    var od = {};
    G.edges.forEach(function (e) { if (e.status !== 'cycle_rejected') od[e.src] = (od[e.src]||0)+1; });
    Object.keys(od).forEach(function (k) { if (od[k] >= 2) forked++; });
    var trows = Object.keys(TYPE_COLOR).map(function (t) {
      var c = byType[t] || 0, w = m ? Math.round(c / m * 100) : 0;
      return '<div class="trow"><i class="dot" style="background:' + TYPE_COLOR[t] + '"></i>' +
        '<span class="nm">' + t + '</span>' +
        '<span class="bar" style="flex:1"><i style="width:' + w + '%;background:' +
        TYPE_COLOR[t] + '"></i></span><span class="num">' + c + '</span></div>';
    }).join('');
    side.innerHTML =
      '<h2>图状态仪表</h2>' +
      '<div class="kpi"><span>节点（会话）</span><b>' + n + '</b></div>' +
      '<div class="kpi"><span>边（typed 关系）</span><b>' + m + '</b></div>' +
      '<div class="kpi"><span>分叉节点（出度≥2）</span><b>' + forked + '</b></div>' +
      '<div class="kpi"><span>环拒绝边</span><b>' + rej + '</b></div>' +
      '<div class="kpi"><span>平均边强度</span><b>' + (sN ? (sSum / sN).toFixed(3) : '—') + '</b></div>' +
      '<h3>边类型分布</h3>' + trows +
      (G.source === 'dataset'
        ? '<h3>数据来源</h3><div class="meta" style="margin:0">消融数据集固定图 —— ' +
          '图结构由数据集注入，未跑演示服务也能查看</div>'
        : '<h3>数据来源</h3><div class="meta" style="margin:0">演示运行时持久化的图</div>');
  }

  function renderNode(id){
    var n = null;
    G.nodes.forEach(function (x) { if (x.node_id === id) n = x; });
    if (!n){ renderDashboard(); return; }
    var ins = G.edges.filter(function (e) { return e.dst === id; });
    var outs = G.edges.filter(function (e) { return e.src === id; });
    var row = function (e, dir) {
      var w = Math.round((e.strength || 0) * 100);
      return '<div class="elist">' +
        '<span class="etype" style="background:' + (TYPE_COLOR[e.type] || '#9aa9a3') + '">' +
        (TYPE_LABEL[e.type] || e.type) + '</span>' +
        '<span class="nm">' + (dir === 'in' ? '← ' : '→ ') + esc(dir === 'in' ? e.src : e.dst) + '</span>' +
        '<span class="bar" style="flex:1"><i style="width:' + w + '%;background:' +
        (TYPE_COLOR[e.type] || '#9aa9a3') + '"></i></span>' +
        '<span class="val">' + (e.strength || 0).toFixed(2) + '</span></div>';
    };
    side.innerHTML =
      '<h2>节点详情</h2>' +
      '<div class="title">' + esc(n.topic || id) + '</div>' +
      '<div class="meta">' + esc(id) + ' · 记忆 ' + (n.n_memories || 0) + ' 条</div>' +
      (n.summary ? '<div class="box">' + esc(n.summary) + '</div>' : '') +
      '<h3>上游源（' + ins.length + '）</h3>' +
      (ins.length ? ins.map(function (e) { return row(e, 'in'); }).join('')
                  : '<div class="empty">无上游（图根节点）</div>') +
      '<h3>下游依赖方（' + outs.length + '）</h3>' +
      (outs.length ? outs.map(function (e) { return row(e, 'out'); }).join('')
                   : '<div class="empty">无下游</div>') +
      '<a class="back-dash" href="#" onclick="return false" data-back="1">← 返回图状态仪表</a>';
    bindBack();
  }

  function renderStagePanel(){
    if (!SA){ renderDashboard(); return; }
    var cards = SA.sources.map(function (s) {
      var w = Math.round(s.src_score * 100);
      return '<div class="card' + (s.src_score >= 0.6 ? ' hot' : '') + '">' +
        '<div class="hd"><b>' + esc(s.node_id) + '</b>' +
        '<span class="pill' + (s.quota > 0 ? '' : ' g') + '">' +
        (s.quota > 0 ? '配额 ' + s.quota : '软偏置') + '</span></div>' +
        '<div class="elist"><span class="nm" style="min-width:64px;color:var(--sub)">src_score</span>' +
        '<span class="bar" style="flex:1"><i style="width:' + w + '%;background:#e08a3c"></i></span>' +
        '<span class="val">' + s.src_score.toFixed(3) + '</span></div>' +
        '<div class="subline" style="margin-left:0">' +
        '静态边强度 ' + s.strength.toFixed(3) + ' · query 相关度 ' + s.relevance.toFixed(3) +
        '</div></div>';
    }).join('');
    side.innerHTML =
      '<h2>Stage A 选源（实算）</h2>' +
      '<div class="title">' + esc(SA.cur_node) + '</div>' +
      '<div class="meta">查询：' + esc(SA.query) + '<br>模式 ' + esc(SA.mode) +
      ' · λ=' + SA.lambda + '（边强度与 query 相关度的融合比）</div>' +
      '<h3>入选源（' + SA.sources.length + '）</h3>' +
      (cards || '<div class="empty">无上游源（图根节点）</div>') +
      '<h3>怎么看</h3><div class="meta" style="margin:0">' +
      '橙色节点 = 本轮入选的上游源，流动虚线 = 参与选源的边。<br>' +
      '<b>配额</b> 是该源在本轮注入中的带宽上限；<b>src_score</b> 决定它排多前。' +
      '</div><a class="back-dash" href="#" onclick="return false" data-back="1">← 返回图状态仪表</a>';
    bindBack();
  }

  function showEdgeDetail(d){
    side.innerHTML =
      '<h2>边详情</h2>' +
      '<div class="title">' + esc(d.source) + ' → ' + esc(d.target) + '</div>' +
      '<div class="meta">' +
      '类型 <b>' + (TYPE_LABEL[d.type] || d.type) + '</b><br>' +
      '强度 <b>' + (d.strength != null ? d.strength.toFixed(3) : '—') + '</b>' +
      (d.sim != null ? '<br>建边相似度 <b>' + d.sim.toFixed(3) + '</b>' : '') +
      (d.inter ? '<br>行为证据（被下游引用） <b>' + d.inter + '</b> 次' : '') +
      '</div>' +
      (d.rejected ? '<div class="box">该边在环检测中被降级（status=cycle_rejected），' +
        '不参与邻域扩散 —— 这是"强制 DAG"设计决策的可见痕迹。</div>' : '') +
      (d.ev ? '<h3>建边依据</h3><div class="box">' + esc(d.ev) + '</div>' : '') +
      '<a class="back-dash" href="#" onclick="return false" data-back="1">← 返回图状态仪表</a>';
    bindBack();
  }

  function bindBack(){
    var a = side.querySelector('[data-back]');
    if (a) a.addEventListener('click', function () { select(null); SA = null;
      cy.elements().removeClass('src flow dim hot'); });
  }

  /* ---------------- 工具条交互 ---------------- */
  /* ---------------- 布局：自算 DAG 分层（自左向右） ----------------
     不用 breadthfirst+旋转的取巧做法（实测 rotate 不生效/被后续 fit 覆盖）：
     直接按「层号 = 最长上游路径」算坐标 —— 上游在左、下游在右，完全可控。 */
  function layeredPositions(){
    var ids = G.nodes.map(function (n) { return n.node_id; });
    var indeg = {}, adj = {}, lvl = {};
    ids.forEach(function (i) { indeg[i] = 0; adj[i] = []; lvl[i] = 0; });
    G.edges.forEach(function (e) {
      if (e.status === 'cycle_rejected') return;
      if (adj[e.src] === undefined || indeg[e.dst] === undefined) return;
      adj[e.src].push(e.dst);
      indeg[e.dst]++;
    });
    // Kahn 逐层剥离：lvl[v] = max(lvl[u] + 1)
    var q = ids.filter(function (i) { return indeg[i] === 0; });
    var seen = 0;
    while (q.length) {
      var cur = q.shift(); seen++;
      adj[cur].forEach(function (nxt) {
        if (lvl[cur] + 1 > lvl[nxt]) lvl[nxt] = lvl[cur] + 1;
        if (--indeg[nxt] === 0) q.push(nxt);
      });
    }
    // 有环时的兜底（正常图不会走到）
    if (seen < ids.length) ids.forEach(function (i) { if (lvl[i] === undefined) lvl[i] = 0; });
    var byLvl = {};
    ids.forEach(function (i) { (byLvl[lvl[i]] = byLvl[lvl[i]] || []).push(i); });
    var pos = {}, COL = 138, ROW = 72;
    Object.keys(byLvl).sort(function (a, b) { return a - b; }).forEach(function (L) {
      var col = byLvl[L];
      col.forEach(function (id, r) {
        // 竖直居中：同一层的节点围绕中线分布，避免长尾层把图拉偏
        pos[id] = {x: (+L) * COL, y: (r - (col.length - 1) / 2) * ROW};
      });
    });
    return pos;
  }

  function applyLayered(animate){
    var pos = layeredPositions();
    cy.nodes().forEach(function (n) {
      var p = pos[n.id()];
      if (p) n.position(p);
    });
    var fit = {fit: {eles: cy.elements(), padding: 56}, duration: animate ? 420 : 0};
    if (animate) cy.animate(fit); else cy.fit(cy.elements(), 56);
  }

  function setLayout(name){
    // 分层 = 自算坐标；其余布局交给 Cytoscape 内置
    if (name === 'breadthfirst'){ applyLayered(true); return; }
    var opts = {name: name, padding: 52, animate: true, animationDuration: 420};
    if (name === 'circle') opts.spacingFactor = 1.25;
    if (name === 'grid') opts.avoidOverlap = true;
    cy.layout(opts).run();
  }

  function showTip(ev, d){
    if (!ev) return;
    tip.innerHTML = '<b>' + esc(d.source) + ' → ' + esc(d.target) + '</b><br>' +
      (TYPE_LABEL[d.type] || d.type) + ' · 强度 ' + (d.strength || 0).toFixed(2) +
      (d.sim != null ? ' · 相似度 ' + d.sim.toFixed(3) : '') +
      (d.inter ? '<br>被下游引用 ' + d.inter + ' 次' : '') +
      (d.ev ? '<br><span style="opacity:.8">' + esc(String(d.ev).slice(0, 90)) + '</span>' : '');
    tip.classList.add('show');
    moveTip(ev);
  }
  function moveTip(ev){
    if (!ev) return;
    var x = ev.clientX + 15, y = ev.clientY + 15;
    tip.style.left = Math.min(x, window.innerWidth - 340) + 'px';
    tip.style.top = Math.min(y, window.innerHeight - 130) + 'px';
  }

  function flash(msg){
    var t = document.getElementById('flash');
    t.textContent = msg;
    t.style.opacity = '1';
    clearTimeout(flash._t);
    flash._t = setTimeout(function () { t.style.opacity = '0'; }, 2600);
  }

  /* ---------------- 启动 ---------------- */
  function boot(){
    fetch('/api/graph').then(function (r) { return r.json(); }).then(function (r) {
      if (!r || !r.ok){ side.innerHTML = '<div class="empty">载入失败</div>'; return; }
      G = r.graph || G;
      if (!G.nodes.length){
        side.innerHTML = '<div class="empty">暂无图数据：先跑一次消融或发起一轮会话后刷新</div>';
        return;
      }
      initCy();
      applyLayered(false);       // 自算 DAG 分层（自左向右，上游在左）
      renderDashboard();
      var ft = document.getElementById('ftip');
      ft.textContent = '节点 = 一次主题会话 · 箭头「被依赖方 → 依赖方」· ' +
        '滚轮缩放 / 拖拽平移 / 框选 · 点节点看详情与选源';
      // ⚠️ 必须先取到 DOM 引用再使用（此前深链代码写在这之前，var 提升导致
      //    `q` 为 undefined → 抛异常 → 右侧面板变空白，正是"点击后空白"的同类根因）
      var q = document.getElementById('q');
      var saBtn = document.getElementById('saBtn');
      if (!q.value) q.value = '演示服务上线前需要确认哪些前置条件';
      saBtn.addEventListener('click', runStageA);
      q.addEventListener('keydown', function (e) { if (e.key === 'Enter') runStageA(); });
      document.getElementById('fitBtn').addEventListener('click', function () {
        cy.animate({fit: {eles: cy.elements(), padding: 56}, duration: 320});
      });
      document.querySelectorAll('.seg button').forEach(function (b) {
        b.addEventListener('click', function () {
          document.querySelectorAll('.seg button').forEach(function (x) { x.classList.remove('on'); });
          b.classList.add('on');
          setLayout(b.dataset.layout);
        });
      });
      // 支持 ?node=<id>&q=<query> 深链：直接落到某节点的选源视图
      // （既是分享能力，也让无头浏览器可自动化验证选源渲染）
      var sp = new URLSearchParams(location.search);
      var qp = sp.get('q'), np = sp.get('node');
      if (qp) q.value = qp;
      if (np){
        select(np);
        fetchStageA(np).then(function (sa) {
          if (sa){ SA = sa; paintStageA(); }
          else { flash('选源未返回结果：请确认后端 /api/stage_a 可用'); }
        });
      }
    }).catch(function (e) {
      side.innerHTML = '<div class="empty">载入失败：' + esc(e && e.message) + '</div>';
    });
  }

  boot();
})();
