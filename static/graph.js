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
        label: (n.topic || n.node_id),
        sub: n.node_id,
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
      'shape':'round-rectangle', 'width':152, 'height':50,
      'background-color':'#ffffff', 'background-opacity':1,
      'border-width':1.4, 'border-color':'#cfdcd6',
      'label':'data(label)', 'font-size':12.5, 'font-weight':600, 'color':'#152b23',
      'text-valign':'center', 'text-halign':'center', 'text-wrap':'ellipsis',
      'text-max-width':130, 'text-margin-y':-6,
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
    cy.animate({fit:{eles:keep, padding:70}, duration:420});
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
  function setLayout(name){
    var opts = {name: name, padding: 46, animate: true, animationDuration: 380};
    if (name === 'breadthfirst'){ opts.directed = true; opts.spacingFactor = 1.12; opts.avoidOverlap = true; }
    if (name === 'circle'){ opts.spacingFactor = 1.3; }
    if (name === 'grid'){ opts.avoidOverlap = true; }
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
      renderDashboard();
      var ft = document.getElementById('ftip');
      ft.textContent = '节点 = 一次主题会话 · 箭头「被依赖方 → 依赖方」· ' +
        '滚轮缩放 / 拖拽平移 / 框选 · 点节点看详情与选源';
      var q = document.getElementById('q');
      if (!q.value) q.value = '演示服务上线前需要确认哪些前置条件';
      document.getElementById('saBtn').addEventListener('click', runStageA);
      q.addEventListener('keydown', function (e) { if (e.key === 'Enter') runStageA(); });
      document.getElementById('fitBtn').addEventListener('click', function () {
        cy.animate({fit:{padding:46}, duration:320});
      });
      document.querySelectorAll('.seg button').forEach(function (b) {
        b.addEventListener('click', function () {
          document.querySelectorAll('.seg button').forEach(function (x) {
            x.classList.remove('on');
          });
          b.classList.add('on');
          setLayout(b.dataset.layout);
        });
      });
    }).catch(function (e) {
      side.innerHTML = '<div class="empty">载入失败：' + esc(e && e.message) + '</div>';
    });
  }

  boot();
})();
