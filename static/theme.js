/* 主题切换（暗色 / 亮色）+ 极小的通用动效助手。
   三页共用；app.css 已有完整令牌体系，这里只负责切 data-theme 与持久化。
   注意：必须在 <head> 内尽早执行，否则首屏会闪一下白底 —— 见各页的 inline 片段。 */
(function (g) {
  'use strict';

  var KEY = 'srtp.theme';
  var CKEY = 'srtp.contrast';   /* D-061：高对比模式（强光/投影用） */
  var root = document.documentElement;

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function cstored() {
    try { return localStorage.getItem(CKEY); } catch (e) { return null; }
  }
  function systemDark() {
    return g.matchMedia && g.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  /* D-057：**默认暗色**（科技感最好的载体）。
     无历史记录时不再看系统偏好，一律暗色；只有用户显式选过 light 才用亮色。
     注意这与各页 <head> 的首屏同步脚本必须保持一致，否则首屏会闪一下。 */
  function resolve() {
    var s = stored();
    return s === 'dark' || s === 'light' ? s : 'dark';
  }
  function apply(t) {
    root.setAttribute('data-theme', t);
    try { localStorage.setItem(KEY, t); } catch (e) { /* 隐私模式下忽略 */ }
    g.dispatchEvent(new CustomEvent('srtp:theme', { detail: t }));
  }

  /* ---- D-061：高对比模式 ----
     独立于主题的第三个维度：data-contrast="high"。
     为什么独立而不是并进 light/dark：
       · 强光下需要的是「更深的文字 + 更实的边框」，暗色投影偏暗时同样需要；
       · 并进主题会让状态变成 4 种（暗/暗高/亮/亮高），按钮与记忆逻辑都变复杂。
     CSS 侧：app.css 里 `[data-contrast="high"]` 提升所有弱对比令牌到 AAA>=7:1。 */
  function contrastOn() { return root.getAttribute('data-contrast') === 'high'; }
  function setContrast(on) {
    if (on) root.setAttribute('data-contrast', 'high');
    else root.removeAttribute('data-contrast');
    try { localStorage.setItem(CKEY, on ? 'high' : 'normal'); } catch (e) { /* 忽略 */ }
    g.dispatchEvent(new CustomEvent('srtp:contrast', { detail: !!on }));
  }
  function initContrast() {
    if (cstored() === 'high') root.setAttribute('data-contrast', 'high');
  }

  g.SRTPTheme = {
    apply: apply,
    toggle: function () {
      apply(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    },
    current: resolve,
    /* 高对比 API */
    contrastOn: contrastOn,
    toggleContrast: function () { setContrast(!contrastOn()); },
    setContrast: setContrast
  };

  /* 首屏同步：避免刷新后主题闪烁（放在各页 <head>，早于样式表生效） */
  g.__srtpInitTheme = function () {
    root.setAttribute('data-theme', resolve());
    initContrast();   /* D-061：高对比也要首屏同步，否则刷新后闪一下低对比 */
  };

  /* ---- 动效助手：统一用 Web Animations API，尊重 reduced-motion ---- */
  var reduced = g.matchMedia
    && g.matchMedia('(prefers-reduced-motion: reduce)').matches;

  g.Fx = {
    reduced: reduced,
    /* 依次入场：给一批元素加 .rise，序号写进 --i 形成 stagger */
    rise: function (els, step) {
      els = Array.prototype.slice.call(els || []);
      if (reduced) { els.forEach(function (e) { e.style.opacity = 1; }); return; }
      els.forEach(function (e, i) {
        e.style.setProperty('--i', i * (step || 1));
        e.classList.add('rise');
      });
    },
    /* 单元素入场 */
    enter: function (el, from) {
      if (!el || reduced) { if (el) el.style.opacity = 1; return; }
      var dx = (from === 'right') ? 16 : (from === 'left' ? -16 : 0);
      var dy = from === 'top' ? -8 : 8;
      el.animate(
        [{ opacity: 0, transform: 'translate(' + dx + 'px,' + dy + 'px)' },
         { opacity: 1, transform: 'none' }],
        { duration: 300, easing: 'cubic-bezier(.16,1,.3,1)' }
      );
    },
    /* 数字滚动（统计值用） */
    countUp: function (el, to, dur, decimals) {
      if (!el) return;
      var d = decimals == null ? 0 : decimals;
      if (reduced) { el.textContent = Number(to).toFixed(d); return; }
      var from = 0, t0 = null;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var k = Math.min(1, (ts - t0) / (dur || 600));
        // easeOutCubic：起步快、收尾稳，读数不跳
        var e = 1 - Math.pow(1 - k, 3);
        el.textContent = (from + (to - from) * e).toFixed(d);
        if (k < 1) g.requestAnimationFrame(step);
      }
      g.requestAnimationFrame(step);
    }
  };
})(window);
