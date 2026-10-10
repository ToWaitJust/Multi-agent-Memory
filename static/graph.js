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

  /* D-061：边配色**改为读 CSS 变量**，不再硬编码。
     原来这里写死两套 JS 常量（TYPE_COLOR 亮 / DARK_COLOR 暗），结果：
       · app.css 调色（尤其新增的 data-contrast="high"）完全传不进图里；
       · DARK_COLOR 还留着上一代绿松石配色（#3ecf9a/#f0a45c），
         与 D-057 定稿的靛蓝→青→紫体系不一致。
     现在统一从令牌取，四类边的语义色与图例、边标签天然一致。 */
  var TYPE_VAR = {depends_on:'--e-depends', derives_from:'--e-derives',
                  references:'--e-refs', similar_to:'--e-similar'};
  var TYPE_LABEL = {depends_on:'依赖', derives_from:'派生', references:'引用', similar_to:'相近'};

  function cssVar(name, fallback){
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v && v.trim()) || fallback;
  }
  function isDark(){ return document.documentElement.getAttribute('data-theme') === 'dark'; }
  function colors(){
    var c = {};
    for (var k in TYPE_VAR) c[k] = cssVar(TYPE_VAR[k], '#888');
    return c;
  }

  var cy = null, G = {nodes:[], edges:[], source:'empty'}, SA = null, SEL = null;
  var side = document.getElementById('side');
  var tip = document.getElementById('tip');
  /* D-053：用户手动拖动过的节点位置，重排布局后还原（见 relayout） */
  var USER_MOVED = {};
  /* D-053：图例开关状态，按边类型过滤（降噪用） */
  var HIDE = {};
  /* 首次 cose 布局用 randomize 自由展开，之后重排保持稳定 */
  var FIRST_LAYOUT = true;

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
        color: (colors()[e.type] || colors().similar_to),
        sim: (typeof e.similarity === 'number' ? e.similarity : null),
        ev: e.evidence || '',
        inter: e.interactions || 0,
        kind: 'edge'
      }});
    });
    return els;
  }

  /* D-053：主题切换时原地更新配色，不重建图（重建会丢失缩放/平移/拖动位置）
     D-061：改为**重建整张样式表**再update()。
     原来用 `cy.style().selector().style().update()` 链式覆盖，实测不可靠 ——
     背景色变了但文字色没变（D-057 已记录）；高对比涉及十几个令牌，逐个覆盖更不可控。
     `cyStyle()` 每次都从 CSS 变量现取，直接整体替换最稳。 */
  function applyThemeToCy(){
    if (!cy) return;
    rebuildStyle();
    var c = colors();
    cy.edges().forEach(function (e){
      var t = e.data('type');
      if (c[t]) e.data('color', c[t]);
    });
  }

  /* D-057/D-061：节点文字/描边/底色全部**读 CSS 变量**。
     原来写死 '#152b23'（深色文字）→ 暗色下变"深底深字"；
     DARK_COLOR 还留着上一代绿松石配色，与 D-057 定稿的靛蓝体系不一致。
     改读令牌后，高对比模式只需动 app.css，图自动跟随。 */
  /* D-061：所有值必须是**字符串**（Cytoscape 的 `fromJson` 拒绝数字/对象，
     会抛 "A selector must be created from a string"）。
     `shadow-*` 系列当前 vendor 版不支持，会被静默丢弃 —— 不再写入。 */
  function cyStyle(){
   var V = cssVar;
   return [
    {selector:'node', style:{
      'shape':'round-rectangle', 'width':'190px', 'height':'58px',
      'background-color': V('--card-solid', '#fff'), 'background-opacity':'1',
      /* ⚠️ 边框**不要用 --line-2**（暗色下是半透明白）→ 节点描边会变成刺眼白框。
         单独给图用中性描边令牌，两套主题都已定义。 */
      'border-width':'1.6px', 'border-color': V('--node-stroke', '#3a4260'),
      'label':'data(label)', 'font-size':'13px', 'font-weight':'600',
      'color': V('--ink', '#111'),
      'text-valign':'center', 'text-halign':'center', 'text-wrap':'wrap',
      'text-max-width':'174px', 'line-height':'1.34',
      'overlay-opacity':'0',
      'transition-property':'border-color border-width background-color',
      'transition-duration':'130ms'
    }},
    {selector:'node[forked]', style:{
      'border-color':V('--brand','#6b74f0'), 'border-width':'2px'}},
    /* D-061：`dim` 原用 opacity 0.45 —— 亮色下会同时拉低**文字**对比度
       （白底深字被整体淡化，正好违背"强光可读"目标）。
       亮色改为「不透明 + 文字降到 --sub + 底色转浅灰」：节点仍可辨识、不抢焦点，
       但正文对比度不塌。
       ⚠️ 不能写成 `[data-theme="light"] node.dim` —— Cytoscape 选择器只认
       元素自身的 class / data，不认祖先 DOM 属性，会被静默忽略。 */
    {selector:'node.dim', style: isDark() ? {'opacity':'0.45'} : {
      'opacity':'1',
      'background-color': V('--panel', 'rgba(255,255,255,.6)'),
      'color': V('--sub', '#4e5872')
    }},
    {selector:'node.sel', style:{
      'border-color':V('--brand-d','#8b93ff'), 'border-width':'3.2px',
      'background-color': V('--brand-soft', 'rgba(107,116,240,.16)')
    }},
    {selector:'node.src', style:{
      'border-color':V('--accent','#22d3ee'), 'border-width':'3px',
      'background-color': V('--accent-soft', 'rgba(34,211,238,.14)')
    }},
    {selector:'node.up', style:{
      'border-color':V('--blue','#60a5fa'), 'border-width':'2.2px'}},
    {selector:'edge', style:{
      'curve-style':'bezier', 'width':'mapData(strength,0.1,1,1.2,5.5)',
      'line-color':'data(color)', 'target-arrow-color':'data(color)',
      'target-arrow-shape':'triangle', 'arrow-scale':'0.85', 'opacity':'0.72',
      'transition-property':'opacity line-color width', 'transition-duration':'130ms',
      'overlay-opacity':'0'
    }},
    {selector:'edge[?rejected]', style:{
      'line-style':'dashed', 'opacity':'0.35', 'target-arrow-shape':'none'
    }},
    /* D-061：亮色下edge.dim 用 0.06 透明度等于消失（玻璃背景上一根白线都看不到），
       提到0.22 保持"存在但不抢眼"；暗色沿用 0.06。 */
    {selector:'edge.dim', style: isDark() ? {'opacity':'0.06'} : {'opacity':'0.22'}},
    {selector:'edge.hot', style:{'opacity':'1', 'width':'5.5px'}},
    /* 选源命中的边（D-056）：**必须用实线**。
       此前是 dashed + 箭头，用户反馈"虚线和箭头错位"——
       根因是 Cytoscape 的 `arrow-scale` 会把线的终点缩短给箭头留位，
       而 `line-dash-pattern` 仍按**原路径长度**铺 dash，最后一段被错开，
       于是虚线断开、箭头孤立在节点边缘。虚线留给"次要"语义（成环降级 / 被隐藏），
       且那两类都不带箭头。 */
    {selector:'edge.flow', style:{
      'line-style':'solid', 'opacity':'1', 'width':'4.2px',
      'line-color':V('--accent','#22d3ee'), 'target-arrow-color':V('--accent','#22d3ee'),
      'arrow-scale':'1.0'
    }}
   ];
  }

  /* D-061：整体替换样式表。
     ⚠️ **Cytoscape 3.30.2 没有 `fromArray`** —— 只有 `fromJson` / `fromString`。
     之前误用 `fromArray(...)` 会抛 `is not a function`，若外面套了 try/catch
     就会**静默失败**：切主题时页面不变、图还是旧配色（实测踩过）。
     正解：`fromJson(json())` —— `json()` 取出整张表，改完再`fromJson` 灌回去。 */
  function rebuildStyle(){
    if (!cy) return;
    try {
      cy.style().fromJson(cyStyle()).update();
    } catch (err){
      /* 兜底：整表替换失败时退回逐元素重刷，至少让节点底色跟着主题走。
         静默吞掉异常会让"主题没生效"变成查不到原因的玄学问题，故 console 留痕。 */
      if (window.console) console.warn('[graph] rebuildStyle 失败:', err);
      var v = cssVar;
      cy.nodes().style({
        'background-color': v('--card-solid', '#fff'),
        'color': v('--ink', '#111'),
        'border-color': v('--node-stroke', '#ccc')
      });
    }
  }

  /* ------------------------------------------------------------------
     D-058 进站展开动画：节点从中心向外生长 + 边逐条点亮。
     招新展示的第一眼决定印象 —— 静态图看起来像"截图"，生长动画才像"活着"。
     做法：初始全部透明 → layoutstop 后按**距中心距离**由近到远错峰淡入
     （比按 id 顺序更自然，视觉上像波纹扩散）；边比节点晚出发。
     ------------------------------------------------------------------ */
  function playIntro(){
    var n = cy.nodes().length;
    if (!n) return;
    // ⚠️ 兜底原则：**动画失败绝不能留空白**。
    //    任何一步出错都直接恢复全可见，宁可没有动画也不能没有内容。
    try {
      if (window.Fx && window.Fx.reduced) return;
      cy.elements().style('opacity', 0);
      var vw = cy.width() || 1, vh = cy.height() || 1;
      var cx = vw / 2, cyy = vh / 2;
      var list = cy.nodes().map(function (nd){
        var p = nd.position() || { x: 0, y: 0 };
        var dx = (p.x || 0) - cx, dy = (p.y || 0) - cyy;
        var d = Math.sqrt(dx * dx + dy * dy);
        return { nd: nd, d: isFinite(d) ? d : 0 };
      }).sort(function (a, b) { return a.d - b.d; });
      var far = list.length ? list[list.length - 1].d : 1;
      var span = (isFinite(far) && far > 1) ? far : 1;
      list.forEach(function (item, i){
        var k = item.d / span;                     // 0 = 离中心最近
        var delay = 80 + k * 560 + i * 6;
        if (!isFinite(delay)) delay = 80;
        item.nd.animate({ style: { 'opacity': 1 } },
          { duration: 460, delay: delay, easing: 'cubic-bezier(.16,1,.3,1)' });
      });
      var eDelay = 80 + 560 * 0.35;
      cy.edges().forEach(function (e, i){
        var target = e.style('opacity');
        var d2 = eDelay + i * 5;
        if (!isFinite(d2)) d2 = eDelay;
        e.animate({ style: { 'opacity': target == null ? 0.72 : target } },
                  { duration: 360, delay: d2, easing: 'ease-out' });
      });
    } catch (err) {
      showAll();                                   // 兜底：出异常立刻全部显示
    }
    // 兜底 #2：1.6s 后无条件收尾（即使某条 animate 被浏览器丢弃）
    setTimeout(showAll, 1600);
  }

  /** 取消所有隐藏，让全部元素可见（动画兜底 / 深链场景直接用） */
  function showAll(){
    if (!cy) return;
    try { cy.nodes().style('opacity', 1); } catch (err) { /* 兜底中的兜底 */ }
    try {
      cy.edges().each(function (e){
        var soft = e.data('type') === 'similar_to' && !e.data('rejected');
        e.style('opacity', e.data('rejected') ? 0.35 : (soft ? 0.4 : 0.72));
      });
    } catch (err2) { /* 同上 */ }
  }

  /* ------------------------------------------------------------------
     D-058 纵横比校正：cose 在 22 节点 / 46 边时会收敛成**竖长形**
     （实测约 400×800），左右各留一半空白，画面利用率极低。
     做法：布局收敛后算一次 bbox，把非等比的那一轴按目标比例拉伸，
     节点越多效果越明显（5 节点时偏差小，接近 1 就不动）。
     ------------------------------------------------------------------ */
  function fitAspect(){
    var els = cy.elements();
    if (!els.length) return;
    var bb = els.boundingBox();
    var vw = cy.width(), vh = cy.height();
    if (!bb || !vw || !vh || !bb.w || !bb.h) return;
    var target = (vw - 140) / (vh - 140);          // 视口可用区的宽高比
    var cur = bb.w / bb.h;
    var ratio = target / cur;
    if (ratio > 0.95 && ratio < 1.05) return;       // 已经很接近，不动
    // D-058 调参记录（实测对照）：
    //   限幅 1.9  → 横向拉伸近 2 倍，边被拉得很长、节点显小（涂总反馈"连线太长"）
    //   限幅 1.12 → 竖长原始形状保留，左右各留一半空白（比长边更难看）
    //   限幅 1.38 → **平衡点**：边只长约 1/4，配合放大到 188×58 的节点刚好铺满
    var k = Math.max(0.90, Math.min(1.38, ratio));
    var cx = bb.x1 + bb.w / 2, cyy = bb.y1 + bb.h / 2;
    cy.nodes().forEach(function (n){
      var p = n.position();
      n.position({ x: cx + (p.x - cx) * k, y: cyy + (p.y - cyy) * k });
    });
  }

  /* D-058：带 zoom 下限的 fit。
     22 节点 + 46 边铺满全屏时，fit 会把 zoom 压到 ~0.7，
     而 Cytoscape 的节点 width/height 是**样式属性、随 zoom 缩放** ——
     于是卡片连同字号一起被缩小，投影时完全看不清（涂总反馈"节点太小"）。
     这里给 zoom 兜一个下限：宁可让图轻微超出视口（还能拖/缩放看细节），
     也要保证卡片与文字达到可读尺寸。
     下限取值实测：0.82 → 只看得到 6 个节点（过度）；0.72 → 看全大部分且文字可读。 */
  function fitNice(){
    cy.fit(cy.elements(), 16);
    var MIN_Z = 0.72;
    if (cy.zoom() < MIN_Z){
      cy.zoom({
        level: MIN_Z,
        renderedPosition: { x: Math.round(cy.width() / 2), y: Math.round(cy.height() / 2) }
      });
    }
  }

  function initCy(){
    cy = cytoscape({
      container: document.getElementById('cy'),
      elements: buildElements(),
      style: cyStyle(),
      wheelSensitivity: 0.22,
      minZoom: 0.25, maxZoom: 2.6,
      boxSelectionEnabled: true,
      // D-053：spacingFactor 1.12 → 1.9。实测 22 节点挤在画布中部约 25% 面积、
      // 四周大片留白，根因就是层间距太小。DAG 保持 breadthfirst（我们强制 DAG）。
      layout: {name:'breadthfirst', directed:true, spacingFactor:1.9,
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

    // D-053：拖动过的节点记入 USER_MOVED，后续重排布局后把位置还原回去。
    // （等价于 fcose 的 fixedNodeConstraint，但不需要引入外部布局包。）
    cy.on('dragfree', 'node', function (ev) {
      USER_MOVED[ev.target.id()] = { x: ev.target.position('x'), y: ev.target.position('y') };
    });
  }

  /* 重排布局并保留用户手动拖过的节点位置 */
  function relayout(name){ setLayout(name); }

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
    stopFlow();
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
    animateStageA();
  }

  /* ------------------------------------------------------------------
     D-053 选源反馈动效序列 —— 让核心算法在页面上"看得见"。
     原实现只加 class（静态高亮），点完按钮画面变化太弱，等于没有反馈。
     序列：① 源节点呼吸两轮 → ② 路径边虚线流动（表示"数据流向"）→ ③ 面板滑入

     ⚠️ Cytoscape 渲染到 **canvas**，不跑 CSS animation —— 边的流动必须用
     requestAnimationFrame 逐帧改 line-dash-offset，不能指望写 CSS。
     ------------------------------------------------------------------ */
  var flowRAF = null;

  function stopFlow(){
    if (flowRAF){ cancelAnimationFrame(flowRAF); flowRAF = null; }
  }

  /* 路径边"搏动"（D-056）：原实现逐帧改 `line-dash-offset` 做流动，
     但 dash 流动与箭头冲突（虚线断开、箭头孤立），已改为实线 + 宽度呼吸。
     宽度在 [3.6, 6.4] 之间按正弦往复，读作"这条边正在被选中"。 */
  function startFlow(edges){
    stopFlow();
    if (!edges || !edges.length) return;
    if (window.Fx && window.Fx.reduced) return;      // 尊重系统「减少动态效果」
    var t0 = null, PERIOD = 1400;
    function step(ts){
      if (t0 === null) t0 = ts;
      var k = ((ts - t0) % PERIOD) / PERIOD;
      var w = 5.4 + 1.5 * Math.sin(k * Math.PI * 2);
      edges.forEach(function (e){ e.style({ 'width': w }); });
      flowRAF = requestAnimationFrame(step);
    }
    flowRAF = requestAnimationFrame(step);
  }

  function animateStageA(){
    if (window.Fx && window.Fx.reduced) return;
    // ① 源节点呼吸：先加粗描边，再放大→复原，两个来回读作一次"呼吸"。
    //    ⚠️ 只能用 border-width / width / height —— 当前 vendor 的 cytoscape
    //    不支持 shadow-* 与 scale 样式（一直是 invalid 警告，写进去会让
    //    animate() 抛 "reading 'name'"，整个选源渲染链断掉）。
    cy.nodes('.src').forEach(function (n, i){
      var f = function (w, h, bw, dur, delay){
        n.animate({ style: { 'width': w, 'height': h, 'border-width': bw } },
          { duration: dur, delay: delay || 0, easing: 'cubic-bezier(.16,1,.3,1)' });
      };
      f(208, 64, 4.5, 260, i * 90);
      setTimeout(function () { f(190, 58, 3, 620); }, 260 + i * 90);
    });
    // ② 路径边持续流动
    startFlow(cy.edges('.flow'));
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
    /* D-061：用 colors()（现读 CSS 变量），不要再用已删除的 TYPE_COLOR 常量 */
    var TC = colors();
    var trows = Object.keys(TC).map(function (t) {
      var c = byType[t] || 0, w = m ? Math.round(c / m * 100) : 0;
      return '<div class="trow"><i class="dot" style="background:' + TC[t] + '"></i>' +
        '<span class="nm">' + t + '</span>' +
        '<span class="bar" style="flex:1"><i style="width:' + w + '%;background:' +
        TC[t] + '"></i></span><span class="num">' + c + '</span></div>';
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
    /* D-061：同上报，colors() 现读 CSS 变量 */
    var TC2 = colors();
    var row = function (e, dir) {
      var w = Math.round((e.strength || 0) * 100);
      var col = TC2[e.type] || '#888';
      return '<div class="elist">' +
        '<span class="etype" style="background:' + col + '">' +
        (TYPE_LABEL[e.type] || e.type) + '</span>' +
        '<span class="nm">' + (dir === 'in' ? '← ' : '→ ') + esc(dir === 'in' ? e.src : e.dst) + '</span>' +
        '<span class="bar" style="flex:1"><i style="width:' + w + '%;background:' +
        col + '"></i></span>' +
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
      var sw = Math.round((s.strength || 0) * 100);
      var rw = Math.round((s.relevance || 0) * 100);
      // D-055：strength / relevance 原来只是 subline 里的一行文字，看不出比例。
      // 改成两条独立条形 —— 选源是这个页面的核心动作，分解必须一眼可读。
      return '<div class="card' + (s.src_score >= 0.6 ? ' hot' : '') + '">' +
        '<div class="hd"><b>' + esc(s.node_id) + '</b>' +
        '<span class="pill' + (s.quota > 0 ? '' : ' g') + '">' +
        (s.quota > 0 ? '配额 ' + s.quota : '软偏置') + '</span></div>' +
        '<div class="elist"><span class="nm" style="min-width:64px;color:var(--sub)">src_score</span>' +
        '<span class="bar" style="flex:1"><i style="width:' + w + '%;background:#e08a3c"></i></span>' +
        '<span class="val">' + s.src_score.toFixed(3) + '</span></div>' +
        '<div class="elist" style="margin-top:4px">' +
        '<span class="nm" style="min-width:64px;color:var(--sub)">边强度</span>' +
        '<span class="bar" style="flex:1"><i style="width:' + sw + '%;background:#0891b2"></i></span>' +
        '<span class="val">' + (s.strength || 0).toFixed(2) + '</span></div>' +
        '<div class="elist">' +
        '<span class="nm" style="min-width:64px;color:var(--sub)">query 相关</span>' +
        '<span class="bar" style="flex:1"><i style="width:' + rw + '%;background:#3b82f6"></i></span>' +
        '<span class="val">' + (s.relevance || 0).toFixed(3) + '</span></div>' +
        '<div class="subline" style="margin-left:0">src_score = λ×边强度 + (1−λ)×相关度，λ=' +
        SA.lambda + '</div></div>';
    }).join('');
    side.innerHTML =
      '<h2>Stage A 选源（实算）</h2>' +
      '<div class="title">' + esc(SA.cur_node) + '</div>' +
      '<div class="meta">查询：' + esc(SA.query) + '<br>模式 ' + esc(SA.mode) +
      ' · λ=' + SA.lambda + '（边强度与 query 相关度的融合比）</div>' +
      '<h3>入选源（' + SA.sources.length + '）</h3>' +
      (cards || '<div class="empty">无上游源（图根节点）</div>') +
      '<h3>怎么看</h3><div class="meta" style="margin:0">' +
      '<b style="color:var(--accent)">青色节点</b> = 本轮入选的源节点；' +
      '<b>青色实线</b> = 记忆由该源共享给当前节点（线宽会呼吸）。<br>' +
      '每张卡片给出 <b>src_score</b> 及其两项分解（边强度 / query 相关度），' +
      '底部标注融合公式 —— 排序与配额都按它决定。' +
      '</div><a class="back-dash" href="#" onclick="return false" data-back="1">← 返回图状态仪表</a>';
    bindBack();
    // D-053：面板内容逐条滑入 + src_score 数字滚动，让"算出了什么"一眼可读
    if (window.Fx){
      window.Fx.rise(side.querySelectorAll('.card, .side .title, .side .meta'), 1);
      side.querySelectorAll('.card .bar i').forEach(function (bar, i){
        var to = bar.style.width;
        bar.style.width = '0%';
        setTimeout(function () { bar.style.transition = 'width .5s cubic-bezier(.16,1,.3,1)';
          bar.style.width = to; }, 90 + i * 70);
      });
    }
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
  /* ------------------------------------------------------------------
     D-053 布局铺满：原来 COL=138 / ROW=72，22 节点实测被压成
     「12 层 × 3 行」的扁条（宽高比 ≈ 7:1），四周全是空白。
     两个改动：
       ① 列距拉到卡片宽度之上，避免相邻层卡片视觉重叠；
       ② **层内按视口比例分配行距** —— 层数多时自动把图"立"起来，
          宽高比收敛到接近画布（≈1.3:1），而不是一味拉宽。
     ------------------------------------------------------------------ */
  var CARD_W = 150, CARD_H = 48;

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
    if (seen < ids.length) ids.forEach(function (i) { if (lvl[i] === undefined) lvl[i] = 0; });
    var byLvl = {};
    ids.forEach(function (i) { (byLvl[lvl[i]] = byLvl[lvl[i]] || []).push(i); });
    var levels = Object.keys(byLvl).sort(function (a, b) { return a - b; });
    var maxRow = Math.max.apply(null, levels.map(function (L) { return byLvl[L].length; }));

    // 目标宽高比（画布减去侧栏后的可视区，约 1.3:1）
    // 注：22 节点实测是「12 层 × 3 行」的深 DAG，靠调 COL/ROW 无法收敛到方形
    //     （要 1.3:1 得把行距拉到 600+，节点间全是空隙）。所以**默认布局已改为
    //     内置力导向 cose**（能自然铺成方形），分层只作为"看 DAG 层次"的可选视图。
    var TARGET = 1.3;
    var COL = 196, ROW = 110;

    var pos = {};
    levels.forEach(function (L, li) {
      var col = byLvl[L];
      col.forEach(function (id, r) {
        // 竖直居中：同一层围绕中线分布，避免长尾层把图拉偏
        pos[id] = { x: li * COL, y: (r - (col.length - 1) / 2) * ROW };
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

  /* D-053：把用户拖动过的节点位置还原（等价 fcose 的 fixedNodeConstraint） */
  function restoreUserMoved(){
    var ids = Object.keys(USER_MOVED);
    if (!ids.length) return;
    setTimeout(function () {
      ids.forEach(function (id){
        var n = cy.getElementById(id);
        if (n && n.length) { n.position(USER_MOVED[id]); n.clearAnimate(); }
      });
    }, 460);
  }

  function setLayout(name){
    if (name === 'breadthfirst'){ applyLayered(true); }
    else {
      var opts = {name: name, padding: 52, animate: true, animationDuration: 420,
                  // D-053：原来没配缓动，默认线性 → 布局切换"顿"一下
                  animationEasing: 'ease-out-in-out'};
      if (name === 'circle') opts.spacingFactor = 1.25;
      if (name === 'grid') opts.avoidOverlap = true;
      if (name === 'cose'){
        // D-053 默认布局：内置力导向，22 节点实测能在 ~200ms 内收敛并铺满画布。
        // 不用 fcose 扩展是因为本机代理不通、下不到包，而内置 cose 在这个规模够用。
        // D-058：节点放大到 188×58 后，idealEdgeLength 要跟着抬，否则卡片会挤在一起；
        //   但边长仍控制在 150 以内 —— 边太长会让「连线纵横交错」变成一团乱麻。
        opts.idealEdgeLength = function () { return 150; };
        opts.nodeRepulsion = function () { return 8200; };
        opts.nodeOverlap = 24;
        opts.gravity = 45;
        opts.numIter = 900;
        // ⚠️ 首次必须 randomize:true —— 复用 breadthfirst 的初始位置再收敛，
        //    结果形状跟分层几乎一样，右侧照样留白。之后重排用 false 保持稳定，
        //    否则每点一次布局整张图都会乱跳。
        opts.randomize = FIRST_LAYOUT;
        FIRST_LAYOUT = false;
        opts.componentSpacing = 90;
      }
      var lay = cy.layout(opts);
      // 每次重排都重新 fit（等收敛完），否则切换后会留大片空白
      var isFirst = FIRST_LAYOUT;   // 首次的校正/动画已由 initCy 的 handler 负责
      cy.once('layoutstop', function () {
        if (!isFirst && name === 'cose'){ fitAspect(); fitNice(); }
        restoreUserMoved();
      });
      lay.run();
      if (name !== 'cose') restoreUserMoved();
    }
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

  /* D-053：图例可点击 → 按边类型过滤（降噪）。
     46 条边交叉成毛线团是实测最大视觉噪音源；默认把最弱的 similar_to 降到半透明，
     用户可随时开/关任意类型。 */
  function applyEdgeFilter(){
    if (!cy) return;
    var c = colors();
    cy.edges().forEach(function (e){
      var t = e.data('type');
      var hidden = !!HIDE[t];
      var soft = !hidden && t === 'similar_to';
      // D-056：隐藏/弱化的边同时改虚线**且去掉箭头** —— Cytoscape 的 dash pattern
      // 不为箭头预留长度，虚线带箭头必然在终点错位（涂总反馈的原始问题）。
      // 隐藏态本就要弱化，去掉箭头不损失信息。
      e.style({
        'display': hidden ? 'none' : 'element',
        'opacity': hidden ? null : (soft ? 0.4 : null),
        'line-style': (hidden || soft) ? 'dashed' : 'solid',
        'target-arrow-shape': (hidden || soft) ? 'none' : 'triangle'
      });
    });
  }

  function bindLegend(){
    var box = document.getElementById('legendBox');
    if (!box) return;
    var byType = {};
    G.edges.forEach(function (e) {
      if (e.status === 'cycle_rejected') return;
      byType[e.type] = (byType[e.type] || 0) + 1;
    });
    var rej = G.edges.filter(function (e) { return e.status === 'cycle_rejected'; }).length;
    var order = ['depends_on', 'derives_from', 'references', 'similar_to'];
    box.innerHTML = order.map(function (t) {
      return '<div class="row" data-t="' + t + '" title="点击显示/隐藏 ' +
        (TYPE_LABEL[t] || t) + ' 边">' +
        '<i class="sw" style="width:16px;height:3px;border-radius:2px;background:' +
        (colors()[t] || '#888') + '"></i>' +
        '<span>' + (TYPE_LABEL[t] || t) + '</span>' +
        '<span class="cnt">' + (byType[t] || 0) + '</span></div>';
    }).join('') +
      (rej ? '<div class="row" data-rej="1" title="点击显示/隐藏被环检测降级的边">' +
        '<i class="sw" style="width:16px;height:0;border-top:2px dashed #9aa9a3"></i>' +
        '<span>成环降级</span><span class="cnt">' + rej + '</span></div>' : '') +
      '<div class="note">边宽 = 强度 · 点图例可隐藏该类边<br>' +
      '虚线 = 已隐藏或成环降级（不带箭头）</div>';
    box.querySelectorAll('.row').forEach(function (row){
      row.addEventListener('click', function () {
        if (row.dataset.rej){
          cy.edges('[rejected]').forEach(function (e){
            e.style('display', e.style('display') === 'none' ? 'element' : 'none');
          });
        } else {
          var t = row.dataset.t;
          HIDE[t] = !HIDE[t];
        }
        row.classList.toggle('off');
        applyEdgeFilter();
      });
    });
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
      // D-053：默认用内置 cose 力导向（原 applyLayered 自算坐标实测把 22 节点
      // 压成 8:1 的扁条，四周大片空白）。分层降级为可选视图。
      // ⚠️ fit 必须等 layoutstop，不能用 setTimeout —— cose 是渐进收敛的，
      // 定时 fit 会在它还没停时缩放，导致右侧留白、节点堆在底部。
      // D-058：带 ?node= 深链进来时（自动化截图 / 分享链接）跳过展开动画，
      // 否则动画播完再切选源会闪一下。判断放在 layoutstop 里以便拿到 SP。
      var deepLink = new URLSearchParams(location.search).get('node');
      // ⚠️ 只在**首次**布局收敛时做校正 + 动画。setLayout 里也注册了 layoutstop，
      //    若两处都校正会把图越拉越竖（实测），故用 FIRST_LAYOUT 抢标志。
      cy.once('layoutstop', function () {
        FIRST_LAYOUT = false;       // 交给 setLayout 的 handler，不再重复校正
        fitAspect();
        fitNice();
        if (!deepLink) playIntro(); else showAll();  // 深链直接全可见，不播动画
      });
      setLayout('cose');
      renderDashboard();
      bindLegend();
      applyEdgeFilter();
      applyThemeToCy();
      window.addEventListener('srtp:theme', applyThemeToCy);
      // 可观测性：暴露实例，供自动化脚本按节点真实坐标做局部截图/断言
      // （否则只能盲猜像素裁剪，深链 + 深链图是唯一稳定的自动化入口）
      window.__srtpCy = cy;
      var tb = document.getElementById('themeBtn');
      if (tb && window.SRTPTheme) tb.addEventListener('click', function(){ SRTPTheme.toggle(); });
      // D-061：高对比模式。Cytoscape 的节点/边颜色是 build 样式表时从 CSS 变量
      // 取的**快照**，所以切高对比后必须重建样式表，否则图配色不变、只有页面变了。
      var cb = document.getElementById('contrastBtn');
      if (cb && window.SRTPTheme){
        cb.addEventListener('click', function(){ SRTPTheme.toggleContrast(); });
        window.addEventListener('srtp:contrast', applyThemeToCy);
        if (SRTPTheme.contrastOn()) cb.classList.add('on');
        window.addEventListener('srtp:contrast', function (e){
          cb.classList.toggle('on', !!e.detail);
        });
      }
      var ft = document.getElementById('ftip');
      ft.textContent = '节点 = 一次主题会话 · 箭头「被依赖方 → 依赖方」· ' +
        '滚轮缩放 / 拖拽平移 / 框选 / 拖动节点后重排会保留位置 · 点节点看详情与选源';
      // ⚠️ 必须先取到 DOM 引用再使用（此前深链代码写在这之前，var 提升导致
      //    `q` 为 undefined → 抛异常 → 右侧面板变空白，正是"点击后空白"的同类根因）
      var q = document.getElementById('q');
      var saBtn = document.getElementById('saBtn');
      if (!q.value) q.value = '演示服务上线前需要确认哪些前置条件';
      saBtn.addEventListener('click', runStageA);
      q.addEventListener('keydown', function (e) { if (e.key === 'Enter') runStageA(); });
      document.getElementById('fitBtn').addEventListener('click', function () {
        cy.animate({fit: {eles: cy.elements(), padding: 16}, duration: 320,
                   complete: function(){ if (cy.zoom() < 0.72) cy.zoom(0.72); } });
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
