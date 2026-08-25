"""全流程可视化构建器（§M3 增强版）。

读 data/pipeline_trace.json，生成自包含 app/pipeline_viz.html：
  - ChatGPT 官网白昼风（light）
  - 8 个阶段逐步可视化（入库/召回/四维打分/权重生成/排行榜/保留丢回/双池架构/总结）
  - trace 内嵌为 window.TRACE，file:// 直接打开可看；服务端可覆盖为实时重跑
用法：python -m app.build_pipeline_viz [--trace data/pipeline_trace.json] [--out app/pipeline_viz.html]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>多智能体记忆共享调度 · 全流程可视化</title>
<style>
  :root{
    --bg:#ffffff; --fg:#0d0d0d; --muted:#8e8ea0; --line:#e3e3e3;
    --chip:#f7f7f8; --chip-bd:#ececec; --accent:#10a37f; --accent-soft:#e7f6f0;
    --kept:#10a37f; --disc:#b9b9c3; --warn:#d97706; --blue:#3b82f6;
    --shadow:0 1px 2px rgba(0,0,0,.04),0 8px 24px rgba(0,0,0,.06);
    --radius:16px;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{
    background:var(--bg); color:var(--fg);
    font-family:'Söhne','Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',
      'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;
    line-height:1.6; font-size:15px;
  }
  .wrap{max-width:880px;margin:0 auto;padding:0 20px 80px}

  /* ---------- 顶栏 ---------- */
  header{
    position:sticky;top:0;z-index:50;background:rgba(255,255,255,.9);
    backdrop-filter:blur(10px);border-bottom:1px solid var(--line);
  }
  .head-inner{max-width:880px;margin:0 auto;padding:12px 20px;display:flex;
    align-items:center;gap:12px;flex-wrap:wrap}
  .logo{width:34px;height:34px;border-radius:9px;background:var(--accent);
    display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;flex:none}
  .titles{flex:1;min-width:200px}
  .titles h1{font-size:15px;font-weight:600;letter-spacing:.2px}
  .titles p{font-size:12px;color:var(--muted)}
  .btn{
    border:1px solid var(--line);background:#fff;color:var(--fg);
    padding:8px 16px;border-radius:999px;font-size:13px;font-weight:500;
    cursor:pointer;transition:.18s;display:inline-flex;align-items:center;gap:6px;
  }
  .btn:hover{border-color:#c9c9c9;background:#fafafa}
  .btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
  .btn.primary:hover{background:#0d9374}
  .btn:disabled{opacity:.45;cursor:not-allowed}
  .step-chip{font-size:12px;color:var(--muted);border:1px solid var(--line);
    padding:4px 10px;border-radius:999px}

  /* ---------- 查询条 ---------- */
  .querybar{display:flex;gap:10px;margin:28px 0 6px;align-items:center}
  .querybar .q{
    flex:1;border:1px solid var(--line);border-radius:12px;padding:10px 14px;
    font-size:14px;outline:none;background:#fff;color:var(--fg);
  }
  .querybar .q:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(16,163,127,.12)}
  .runbtn{white-space:nowrap}

  /* ---------- 阶段卡片 ---------- */
  .stage{
    margin-top:22px;border:1px solid var(--line);border-radius:var(--radius);
    background:#fff;box-shadow:var(--shadow);overflow:hidden;
    opacity:0;transform:translateY(14px);transition:opacity .55s ease,transform .55s ease;
  }
  .stage.reveal{opacity:1;transform:none}
  .stage-head{display:flex;align-items:flex-start;gap:14px;padding:18px 22px 0}
  .num{
    width:34px;height:34px;border-radius:50%;background:var(--accent-soft);
    color:var(--accent);font-weight:700;font-size:14px;flex:none;
    display:flex;align-items:center;justify-content:center;border:1px solid #cdeee2;
  }
  .stage-head h2{font-size:16px;font-weight:600}
  .stage-head .sub{font-size:12.5px;color:var(--muted);margin-top:2px}
  .stage-body{padding:16px 22px 22px}
  .divider{height:1px;background:var(--line);margin:14px 0}

  /* ---------- 徽章/标签 ---------- */
  .badge{display:inline-flex;align-items:center;gap:6px;background:var(--accent-soft);
    color:#0b7d61;font-size:12.5px;font-weight:600;padding:5px 12px;border-radius:999px;
    border:1px solid #cdeee2}
  .badge.gray{background:#f4f4f5;color:#5c5c6b;border-color:#e6e6e9}
  .badge.blue{background:#eef4ff;color:#2a5db0;border-color:#d8e4ff}
  .tag{font-size:11px;padding:2px 8px;border-radius:999px;background:#f0f0f2;
    color:#6a6a78;border:1px solid #e6e6e9;white-space:nowrap}
  .tag.green{background:var(--accent-soft);color:#0b7d61;border-color:#cdeee2}

  /* ---------- 记忆卡片 ---------- */
  .memgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px;margin-top:12px}
  .mem{
    border:1px solid var(--chip-bd);background:var(--chip);border-radius:12px;
    padding:10px 12px;font-size:13px;
  }
  .mem .txt{color:var(--fg)}
  .mem .meta{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;align-items:center;
    font-size:11px;color:var(--muted)}
  .mem.hit{border-color:var(--accent);background:var(--accent-soft)}

  /* ---------- 打分表 ---------- */
  .sctable{width:100%;border-collapse:separate;border-spacing:0;margin-top:12px;font-size:13px}
  .sctable th{font-size:12px;color:var(--muted);font-weight:600;text-align:center;padding:8px 6px;
    border-bottom:1px solid var(--line)}
  .sctable td{padding:7px 6px;border-bottom:1px solid #f2f2f3;text-align:center;font-variant-numeric:tabular-nums}
  .sctable td.memtxt{text-align:left;max-width:300px;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;color:#333}
  .sctable tr:last-child td{border-bottom:none}
  .heat{border-radius:6px}
  .final-c{font-weight:700;color:#0b7d61}

  /* ---------- 权重条 ---------- */
  .wgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:14px}
  .wcard{border:1px solid var(--chip-bd);border-radius:12px;padding:14px}
  .wcard h4{font-size:13px;font-weight:600;margin-bottom:10px;display:flex;align-items:center;gap:8px}
  .wrow{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:12px}
  .wrow .lb{width:64px;color:var(--muted);flex:none}
  .track{flex:1;height:14px;background:#f0f0f2;border-radius:7px;overflow:hidden}
  .fill{height:100%;border-radius:7px;background:var(--accent);transition:width .8s ease}
  .fill.prior{background:#9aa0a6}
  .fill.learn{background:var(--blue)}
  .fill.hybrid{background:var(--accent)}
  .wrow .vl{width:44px;text-align:right;font-variant-numeric:tabular-nums;color:#444}

  /* ---------- 排行榜 ---------- */
  .rankrow{display:flex;align-items:center;gap:10px;margin-top:9px;font-size:13px}
  .rk{width:26px;height:26px;border-radius:8px;background:#f0f0f2;color:#666;font-weight:700;
    display:flex;align-items:center;justify-content:center;flex:none;font-size:12px}
  .rk.top{background:var(--accent);color:#fff}
  .rankrow .txt{width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#333;flex:none}
  .rtrack{flex:1;height:22px;background:#f5f5f6;border-radius:8px;overflow:hidden;position:relative}
  .rfill{height:100%;border-radius:8px;background:linear-gradient(90deg,#38b992,var(--accent));
    transition:width .9s ease;display:flex;align-items:center;justify-content:flex-end;
    padding-right:8px;color:#fff;font-size:12px;font-weight:600;min-width:34px}
  .rfill.disc{background:linear-gradient(90deg,#d0d0d6,#b9b9c3)}
  .rankrow .ok{color:var(--accent);font-weight:700;flex:none}
  .rankrow .no{color:var(--disc);flex:none}

  /* ---------- 保留/丢回 ---------- */
  .duocol{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}
  .col{border:1px solid var(--chip-bd);border-radius:12px;padding:12px}
  .col h4{font-size:13px;font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:8px}
  .col.kept{border-color:#cdeee2;background:#f5fbf8}
  .col.disc{background:#fafafa}
  .colitem{font-size:12.5px;padding:7px 10px;border-radius:8px;margin-bottom:6px;background:#fff;
    border:1px solid var(--chip-bd)}
  .col.kept .colitem{border-color:#d8f0e7;color:#0b5d47}
  .col.disc .colitem{color:#6a6a78}

  /* ---------- 双池流向 ---------- */
  .flow{display:flex;align-items:stretch;gap:8px;margin:14px 0;flex-wrap:wrap}
  .fbox{border:1.5px solid var(--line);border-radius:12px;padding:10px 12px;flex:1;min-width:130px;
    background:#fff;text-align:center}
  .fbox .ic{font-size:18px}
  .fbox .nm{font-size:13px;font-weight:600;margin-top:2px}
  .fbox .ct{font-size:12px;color:var(--muted);margin-top:2px}
  .fbox.pool{border-color:var(--blue);background:#f5f8ff}
  .fbox.shared{border-color:var(--accent);background:#f2fbf7}
  .fbox.api{border-color:var(--warn);background:#fffaf2}
  .fbox.ans{border-color:#d97706;background:#fffaf2}
  .farrow{display:flex;align-items:center;color:var(--muted);font-size:16px;font-weight:700;padding:0 2px}
  .fstep{font-size:11px;color:var(--muted);text-align:center;margin-top:2px}

  /* ---------- LLM 回答 ---------- */
  .chat{display:flex;gap:12px;margin-top:14px}
  .avatar{width:32px;height:32px;border-radius:50%;background:var(--accent);color:#fff;
    display:flex;align-items:center;justify-content:center;font-weight:700;flex:none;font-size:13px}
  .bubble{flex:1;background:var(--chip);border:1px solid var(--chip-bd);border-radius:14px;
    padding:14px 16px;font-size:14px;white-space:pre-wrap;color:var(--fg)}
  .usercard{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 16px;
    margin-top:14px;font-size:14px;display:flex;gap:10px;align-items:flex-start}
  .usercard .avatar{background:#fff;color:var(--fg);border:1px solid var(--line)}

  /* ---------- 指标 ---------- */
  .metgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:14px}
  .met{border:1px solid var(--chip-bd);border-radius:12px;padding:14px;background:#fff}
  .met .k{font-size:12px;color:var(--muted)}
  .met .v{font-size:22px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
  .met .v small{font-size:12px;color:var(--muted);font-weight:500}
  .met .v.green{color:var(--accent)}
  .met .v.blue{color:var(--blue)}
  .tbar{display:flex;align-items:center;gap:10px;margin-top:8px;font-size:12px}
  .tbar .lb{width:110px;color:var(--muted);flex:none;text-align:right}
  .tbar .track{flex:1;height:12px;background:#f0f0f2;border-radius:6px;overflow:hidden}
  .tbar .fill{background:#9aa0a6}
  .tbar .vl{width:70px;font-variant-numeric:tabular-nums;color:#444}

  .apibadges{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
  .note{font-size:12px;color:var(--muted);margin-top:10px;line-height:1.7}
  .foot{margin-top:40px;text-align:center;font-size:12px;color:var(--muted)}
  @media (max-width:680px){
    .duocol{grid-template-columns:1fr}
    .rankrow .txt{width:150px}
    .sctable td.memtxt{max-width:120px}
  }
</style>
</head>
<body>

<header>
  <div class="head-inner">
    <div class="logo">记</div>
    <div class="titles">
      <h1>多智能体记忆共享调度 · 全流程可视化</h1>
      <p id="subline">基于注意力引导 + LLM 可学习权重的记忆共享调度</p>
    </div>
    <span class="step-chip" id="stepchip">0 / 8</span>
    <button class="btn primary" id="playbtn">▶ 播放全流程</button>
  </div>
</header>

<div class="wrap">
  <!-- 查询条 -->
  <div class="querybar">
    <input class="q" id="qinput" placeholder="输入查询，点击「重新运行」实时重跑全流程…">
    <button class="btn runbtn" id="runbtn">🔄 重新运行</button>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap" id="topbadges"></div>

  <main id="stages"></main>

  <div class="foot">SRTP · 基于注意力引导和 LLMs 可学习权重的多智能体记忆共享调度方法研究 · headless 核心 + 可视化演示</div>
</div>

<script>
window.TRACE = __TRACE_JSON__;
</script>
<script>
"use strict";

/* ================= 工具 ================= */
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const dimNames = {time:"时间", semantic:"语义", frequency:"频率", task:"任务"};
const DIMS = ["time","semantic","frequency","task"];

function heat(v){ // 打分热力色
  const a = Math.max(0, Math.min(1, v));
  return `rgba(16,163,127,${(a*0.82+0.06).toFixed(3)})`;
}
function pct(v, max){ return (max>0 ? Math.round(v/max*100) : 0); }
function stage(num, title, sub, html){
  return `<section class="stage" data-num="${num}">
    <div class="stage-head"><div class="num">${num}</div>
      <div><h2>${title}</h2><div class="sub">${sub}</div></div></div>
    <div class="stage-body">${html}</div></section>`;
}
function wbar(label, val, cls, max=1){
  return `<div class="wrow"><span class="lb">${label}</span>
    <div class="track"><div class="fill ${cls}" style="width:${pct(val,max)}%"></div></div>
    <span class="vl">${val.toFixed(3)}</span></div>`;
}

/* ================= 渲染 ================= */
const T = window.TRACE;
let STAGES = [];

function renderTop(){
  const m = T.meta, ap = m.api_status;
  const badges = [
    `<span class="badge">⚙ 配置 ${esc(m.config)}</span>`,
    `<span class="badge blue">α=${m.alpha} · β=${m.beta}</span>`,
    `<span class="badge gray">条件 ${esc(m.condition_key)}</span>`,
    `<span class="badge ${ap.embedding==="real"?"":"gray"}">embedding: ${ap.embedding==="real"?"真实API":"占位"}</span>`,
    `<span class="badge ${ap.llm_prior==="real"?"":"gray"}">LLM先验: ${ap.llm_prior==="real"?"真实":"默认"}</span>`,
    `<span class="badge ${ap.llm_answer==="real"?"":"gray"}">回答: ${ap.llm_answer==="real"?"真实":"合成"}</span>`,
  ];
  $("#topbadges").innerHTML = badges.join("");
  $("#qinput").value = T.meta.query;
  $("#subline").textContent = `查询：${T.meta.query}`;
}

/* --- S1 入库 --- */
function s1(){
  const p = T.resident_pool;
  const chips = p.memories.map(m => `
    <div class="mem">
      <div class="txt">${esc(m.text)}</div>
      <div class="meta">
        <span class="tag">${esc(m.task_tag||"无标签")}</span>
        <span>访问×${m.access_count}</span>
        <span>${m.age_hours}h 前</span>
      </div>
    </div>`).join("");
  return stage(1, "记忆入库 · 常驻完整池",
    `真实 embedding 后端（${T.meta.api_status.embedding==="real"?"DashScope":"占位"}）写入 ${p.total} 条记忆`,
    `<span class="badge">🧠 常驻完整池 resident_pool · ${p.total} 条记忆</span>
     <span class="badge gray">入库耗时 ${p.seed_ms}ms</span>
     <div class="memgrid">${chips}</div>
     <div class="note">每条记忆入库即计算 1024 维语义向量并写入 faiss 索引，供后续向量检索。标注了任务标签、访问次数与记忆年龄——它们是四维注意力打分的原始输入。</div>`);
}

/* --- S2 召回 --- */
function s2(){
  const r = T.recall, ids = new Set(r.memory_ids);
  const pool = T.resident_pool.memories.map(m =>
    `<div class="mem ${ids.has(m.memory_id)?"hit":""}">
      <div class="txt">${esc(m.text)}</div>
      <div class="meta"><span class="tag ${ids.has(m.memory_id)?"green":""}">${ids.has(m.memory_id)?"✓ 已召回":"未召回"}</span></div>
    </div>`).join("");
  return stage(2, "候选召回 · 向量检索",
    `从常驻完整池召回候选集（top_k ≤ 50）`,
    `<span class="badge blue">🔎 召回 ${r.count} / ${T.resident_pool.total} 条候选</span>
     <div class="memgrid">${pool}</div>
     <div class="note">${esc(r.note)}。召回集是后续打分与排序的候选集合。</div>`);
}

/* --- S3 四维打分 --- */
function s3(){
  const rows = T.scoring.map(s => {
    const sc = s.scores;
    const cells = DIMS.map(d => `<td class="heat" style="background:${heat(sc[d])}">${sc[d].toFixed(2)}</td>`).join("");
    return `<tr>
      <td class="memtxt" title="${esc(s.text)}">${esc(s.text)}</td>
      ${cells}
      <td class="final-c">${s.final.toFixed(3)}</td></tr>`;
  }).join("");
  return stage(3, "四维注意力初步打分",
    `每条候选 × 四个注意力头，输出 4 维原始分（未加权）`,
    `<span class="badge">🎯 ${T.scoring.length} 条候选 × 4 注意力头</span>
     <table class="sctable">
       <tr><th>记忆</th><th>⏱ 时间</th><th>📖 语义</th><th>🔁 频率</th><th>🏷 任务</th><th>最终分</th></tr>
       ${rows}
     </table>
     <div class="note">时间=指数衰减(越新越高)；语义=余弦+向量+BM25 融合；频率=访问越频繁越高；任务=标签匹配。四维分数是"记忆本身的质量"，最终分需再乘以动态权重。</div>`);
}

/* --- S4 权重生成 --- */
function s4(){
  const w = T.weights;
  const bars = (obj, cls) => DIMS.map(d => wbar(dimNames[d], obj[d], cls, 0.5)).join("");
  return stage(4, "权重生成 · LLM 先验 ⊕ 可学习 MLP",
    `混合权重 = α·LLM先验 + β·可学习（α=${w.alpha}, β=${w.beta}）`,
    `<div class="wgrid">
      <div class="wcard"><h4>🤖 LLM 先验权重</h4>${bars(w.prior,"prior")}</div>
      <div class="wcard"><h4>🧠 可学习 MLP 权重</h4>${bars(w.learnable,"learn")}</div>
      <div class="wcard"><h4>⚖ 混合权重（最终）</h4>${bars(w.hybrid,"hybrid")}</div>
    </div>
    <div class="divider"></div>
    <div style="background:var(--accent-soft);border:1px solid #cdeee2;border-radius:12px;padding:12px 14px;font-size:13px">
      <b>混合公式：</b>${esc(w.formula)} &nbsp;·&nbsp; 条件化维度：${esc(w.conditioning_dims.join("、"))}
      &nbsp;·&nbsp; 条件键：<code>${esc(w.condition_key)}</code>
    </div>
    <div class="note">LLM 先验由轻量小模型根据查询文本实时判断四维重要性；可学习权重由 MLP(query_emb⊕condition_emb) 输出，离线蒸馏 LLM 先验、在线按用户反馈微调。两路按 α/β 融合归一化，语义维度占比最高（该查询偏语义检索）。</div>`);
}

/* --- S5 排行榜 --- */
function s5(){
  const maxF = Math.max(...T.ranking.map(r => r.final));
  const rows = T.ranking.map(r => `
    <div class="rankrow">
      <span class="rk ${r.rank<=3?"top":""}">${r.rank}</span>
      <span class="txt" title="${esc(r.text)}">${esc(r.text)}</span>
      <div class="rtrack"><div class="rfill ${r.kept?"":"disc"}" style="width:${pct(r.final,maxF)}%">${r.final.toFixed(2)}</div></div>
      <span class="${r.kept?"ok":"no"}">${r.kept?"✓ 保留":"↩ 丢回"}</span>
    </div>`).join("");
  return stage(5, "最终得分排行榜",
    `final = Σ w_d·score_d（按最终分降序）`,
    `<span class="badge">🏆 排行榜 · ${T.ranking.length} 条</span>
     <div style="margin-top:6px">${rows}</div>
     <div class="note">绿色 = 通过动作选择进入共享池；灰色 = 低于动态阈值，丢回常驻池。可见 #1 与其余候选拉开显著差距——语义、任务、时间、频率四维同时占优。</div>`);
}

/* --- S6 保留/丢回 --- */
function s6(){
  const sel = T.selection;
  const kept = T.ranking.filter(r => r.kept);
  const disc = T.ranking.filter(r => !r.kept);
  const keptHtml = kept.map(r => `<div class="colitem">✓ ${esc(r.text)} <b>${r.final.toFixed(3)}</b></div>`).join("") || `<div class="colitem" style="color:#999">无</div>`;
  const discHtml = disc.map(r => `<div class="colitem">↩ ${esc(r.text)} <b>${r.final.toFixed(3)}</b></div>`).join("") || "";
  return stage(6, "动作选择 · 保留 / 丢回",
    `动态阈值判定，压缩率 ${T.metrics.compression_ratio}（保留/召回）`,
    `<div class="duocol">
      <div class="col kept"><h4>✓ 保留 → 共享池（${sel.kept_count}）</h4>${keptHtml}</div>
      <div class="col disc"><h4>↩ 丢回 → 常驻池（${sel.discarded_count}）</h4>${discHtml}</div>
    </div>
    <div class="note">保留下来的高相关记忆写入<code>共享池</code>（≤${T.dual_pool.shared_max} 条）；其余记忆继续留在<code>常驻完整池</code>，供后续查询再次召回——这就是"双池架构"的调度出口。</div>`);
}

/* --- S7 双池架构 + LLM 回答 --- */
function s7(){
  const dp = T.dual_pool, llm = dp.llm_call;
  const items = dp.shared_items.map((it,i) =>
    `<div class="colitem">${i+1}. ${esc(it.text)} <b>${it.score.toFixed(3)}</b></div>`).join("");
  const flow = `
    <div class="flow">
      <div class="fbox pool"><div class="ic">🗄</div><div class="nm">常驻完整池</div>
        <div class="ct">${dp.resident_total} 条 · 全量只读</div>
        <div class="fstep">召回 · 打分 · 选择</div></div>
      <div class="farrow">→</div>
      <div class="fbox shared"><div class="ic">📤</div><div class="nm">共享池 SharedPool</div>
        <div class="ct">≤${dp.shared_max} 条 · 精选 ${dp.shared_items.length} 条</div>
        <div class="fstep">注入 HintBlock</div></div>
      <div class="farrow">→</div>
      <div class="fbox api"><div class="ic">🌐</div><div class="nm">LLM API</div>
        <div class="ct">${T.meta.api_status.llm_answer==="real"?"真实调用":"合成降级"} · ${llm.llm_ms}ms</div>
        <div class="fstep">上下文 = 共享池记忆</div></div>
      <div class="farrow">→</div>
      <div class="fbox ans"><div class="ic">💬</div><div class="nm">回答</div>
        <div class="ct">${llm.answer_tokens} tokens</div></div>
    </div>`;
  const answer = `
    <div class="usercard"><div class="avatar">你</div><div><b>用户查询</b><div style="color:#444">${esc(T.meta.query)}</div></div></div>
    <div class="chat">
      <div class="avatar">记</div>
      <div class="bubble">${esc(llm.answer)}</div>
    </div>`;
  return stage(7, "双池架构 · 共享池 → LLM API → 回答",
    `从完整池取出 → 打分保留 → 传入新开的共享池 → 注入 LLM 上下文`,
    flow + `
    <div class="divider"></div>
    <span class="badge">📤 注入共享池 ${dp.shared_items.length} 条记忆（上下文 ${llm.prompt_tokens} tokens）</span>
    <div style="margin-top:10px">${items || '<div class="note">共享池为空</div>'}</div>
    <div class="divider"></div>
    ${answer}
    <div class="note">回答由主 LLM 基于<u>共享池精选记忆</u>生成——副线看到的不是全量记忆，而是被调度层裁剪后的高相关上下文，这正是记忆共享调度的价值所在。</div>`);
}

/* --- S8 总结 --- */
function s8(){
  const t = T.metrics.timing, tk = T.metrics.tokens;
  const tRows = [
    ["入库 seed", t.seed_ms],
    ["query embedding", t.embedding_ms],
    ["LLM 先验", t.prior_ms],
    ["召回 recall", t.recall_ms],
    ["权重融合 weights", t.weights_ms],
    ["四维打分 scoring", t.scoring_ms],
    ["动作选择 select", t.select_ms],
    ["LLM 回答", t.llm_ms],
  ];
  const maxT = Math.max(...tRows.map(r => r[1]), 1);
  const bars = tRows.map(([k,v]) =>
    `<div class="tbar"><span class="lb">${k}</span>
      <div class="track"><div class="fill" style="width:${pct(v,maxT)}%"></div></div>
      <span class="vl">${v.toFixed(1)} ms</span></div>`).join("");
  const savedPct = tk.saved_pct;
  return stage(8, "过程指标 · 耗时 / token / 压缩",
    `全流程计时、token 消耗与节省、压缩效果`,
    `<div class="metgrid">
      <div class="met"><div class="k">总耗时（含真实 API）</div><div class="v">${t.total_ms.toFixed(0)}<small> ms</small></div></div>
      <div class="met"><div class="k">压缩率 保留/召回</div><div class="v green">${(T.metrics.compression_ratio*100).toFixed(0)}<small> %</small></div></div>
      <div class="met"><div class="k">假设原记忆量 tokens</div><div class="v blue">${tk.original_assumed_tokens}<small> tok</small></div></div>
      <div class="met"><div class="k">注入 LLM tokens</div><div class="v">${tk.kept_tokens}<small> tok</small></div></div>
      <div class="met"><div class="k">节省 tokens</div><div class="v green">${tk.saved_tokens}<small> tok</small></div></div>
      <div class="met"><div class="k">token 节省率</div><div class="v green">${savedPct}<small> %</small></div></div>
    </div>
    <div class="divider"></div>
    <div style="font-size:13px;font-weight:600;margin-bottom:4px">⏱ 各阶段耗时</div>
    ${bars}
    <div class="apibadges">
      <span class="badge ${T.meta.api_status.embedding==="real"?"":"gray"}">embedding: ${T.meta.api_status.embedding==="real"?"真实 DashScope":"确定性占位"}</span>
      <span class="badge ${T.meta.api_status.llm_prior==="real"?"":"gray"}">LLM 先验: ${T.meta.api_status.llm_prior==="real"?"真实小模型":"默认权重"}</span>
      <span class="badge ${T.meta.api_status.llm_answer==="real"?"":"gray"}">主 LLM 回答: ${T.meta.api_status.llm_answer==="real"?"真实":"合成"}</span>
    </div>
    <div class="note">若不做调度，${tk.original_assumed_tokens} tokens 的记忆将整体塞入 LLM 上下文；调度后仅注入 ${tk.kept_tokens} tokens，节省 ${savedPct}%。耗时大头来自真实外部 API（embedding / LLM），调度核心（召回+打分+选择）为毫秒级。</div>`);
}

/* ================= 装配 ================= */
function build(){
  const m = $("#stages");
  STAGES = [s1(), s2(), s3(), s4(), s5(), s6(), s7(), s8()];
  m.innerHTML = STAGES.join("");
  $("#stepchip").textContent = `0 / ${STAGES.length}`;
  renderTop();
}

/* ================= 播放 ================= */
let playing = false;
function revealUpTo(n){
  document.querySelectorAll(".stage").forEach((el, i) => {
    if (i < n) el.classList.add("reveal");
  });
  $("#stepchip").textContent = `${n} / ${STAGES.length}`;
}
function play(){
  if (playing) return;
  playing = true;
  const btn = $("#playbtn");
  btn.disabled = true; btn.textContent = "⏸ 播放中…";
  const nodes = document.querySelectorAll(".stage");
  let i = 0;
  const timer = setInterval(() => {
    if (i >= nodes.length) {
      clearInterval(timer);
      btn.textContent = "▶ 重播"; btn.disabled = false; playing = false;
      return;
    }
    nodes[i].classList.add("reveal");
    $("#stepchip").textContent = `${i+1} / ${nodes.length}`;
    nodes[i].scrollIntoView({behavior:"smooth", block:"center"});
    i++;
  }, 900);
}
$("#playbtn").addEventListener("click", play);

/* 装配 DOM（必须先 build 再注册滚动观察） */
build();

/* 滚动到即显示 */
if ("IntersectionObserver" in window){
  const io = new IntersectionObserver(es => {
    es.forEach(e => { if (e.isIntersecting) e.target.classList.add("reveal"); });
  }, {threshold: .12});
  document.querySelectorAll(".stage").forEach(el => io.observe(el));
}

/* 初始显示第一阶段 */
document.querySelectorAll(".stage")[0]?.classList.add("reveal");

/* 重新运行：优先服务端实时，否则仅提示 */
$("#runbtn").addEventListener("click", async () => {
  const q = $("#qinput").value.trim() || T.meta.query;
  $("#runbtn").disabled = true; $("#runbtn").textContent = "⏳ 运行中…";
  try {
    const r = await fetch(`/api/trace?q=${encodeURIComponent(q)}`);
    if (!r.ok) throw new Error("no-server");
    const trace = await r.json();
    window.TRACE = trace;
    build();
    $("#subline").textContent = `查询：${trace.meta.query}（实时重跑）`;
  } catch (e) {
    alert("实时重跑需要启动本地服务（app/pipeline_server.py）。当前展示的是内嵌演示数据。");
  } finally {
    $("#runbtn").disabled = false; $("#runbtn").textContent = "🔄 重新运行";
  }
});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(ROOT / "data" / "pipeline_trace.json"))
    ap.add_argument("--out", default=str(ROOT / "app" / "pipeline_viz.html"))
    args = ap.parse_args()

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    json_str = json.dumps(trace, ensure_ascii=False)
    json_str = json_str.replace("</", "<\\/")  # 防 </script> 逃逸
    html = TEMPLATE.replace("__TRACE_JSON__", json_str)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"✓ 可视化已生成: {args.out}  ({len(html)//1024} KB)")
    print(f"  查询: {trace['meta']['query']}")
    print(f"  常驻池={trace['resident_pool']['total']} 保留={trace['selection']['kept_count']} "
          f"token节省={trace['metrics']['tokens']['saved_pct']}%")


if __name__ == "__main__":
    main()
