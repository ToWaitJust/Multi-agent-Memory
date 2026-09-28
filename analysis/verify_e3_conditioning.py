"""E3 · 条件化切片机制验证（申报书创新点②之一）。

验证目标（机制级，不依赖大规模消融）：
  同一查询向量下，改变条件键（user/scenario/business），可学习 MLP 的输出应满足：
    1. **条件化维**（semantic/task）随条件变化 —— 条件化生效的证据；
    2. **全局维**（time/frequency）不随条件变化 —— 切片隔离正确的证据；
    3. 不同条件间的差异方向有意义（不同 user 的语义维差异 > 同 user 换 business）。
若 1 不成立 → 条件化没接通（bug）；若 2 不成立 → 切片泄漏（bug）。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from srtp_memory.condition import ConditionKey  # noqa: E402
from srtp_memory.weights import HybridWeightCalculator  # noqa: E402


def main() -> None:
    calc = HybridWeightCalculator()
    # 固定查询向量（排除查询侧噪声，只看条件的影响）
    qv = np.random.RandomState(7).randn(1024).astype(np.float32)

    conds = {
        "基准   u_tu|research|srtp": ConditionKey.parse("u_tu|research|srtp"),
        "换场景 u_tu|demo|srtp    ": ConditionKey.parse("u_tu|demo|srtp"),
        "换用户 u_wang|research|srtp": ConditionKey.parse("u_wang|research|srtp"),
        "换业务 u_tu|research|club": ConditionKey.parse("u_tu|research|club"),
    }

    print("== E3 · 条件化切片验证（同一查询向量，四个条件）==\n")
    print(f"{'条件':<26s} {'time':>8s} {'semantic':>9s} {'frequency':>10s} {'task':>8s}")
    out: dict[str, dict[str, float]] = {}
    for name, c in conds.items():
        w = calc._learnable_forward(qv, c)
        out[name] = w
        print(f"{name:<26s} {w['time']:>8.4f} {w['semantic']:>9.4f} "
              f"{w['frequency']:>10.4f} {w['task']:>8.4f}")

    base = out["基准   u_tu|research|srtp"]
    cond_dims = ("semantic", "task")
    glob_dims = ("time", "frequency")

    print("\n-- 断言 1：条件化维（semantic/task）应随条件变化 --")
    ok1 = True
    for name in list(conds)[1:]:
        for d in cond_dims:
            diff = abs(out[name][d] - base[d])
            flag = "变化 ✓" if diff > 1e-6 else "未变 ✗"
            if diff <= 1e-6:
                ok1 = False
            print(f"  {name.strip()} · {d:<9s} Δ={diff:.6f}  {flag}")

    print("\n-- 断言 2：全局支 logits 应不随条件变化（切片隔离，结构级断言）--")
    # 注意：不能断言 softmax 后的概率不变——四维走整体 softmax，条件化支 logits 变化
    # 会改变归一化分母，全局维的概率必然有 ~1e-4 的二阶耦合（数学必然，非泄漏）。
    # 结构级断言：双分支下全局支的 **logits** 只由 query 计算，必须严格不变。
    import torch
    ok2 = True
    x_base = calc.mlp._feat_vec(qv, calc.condition_store.embed(conds["基准   u_tu|research|srtp"]))
    with torch.no_grad():
        lg_base = calc.mlp._logits4(torch.from_numpy(x_base)[None])[0].numpy()
    for name in list(conds)[1:]:
        x = calc.mlp._feat_vec(qv, calc.condition_store.embed(conds[name]))
        with torch.no_grad():
            lg = calc.mlp._logits4(torch.from_numpy(x)[None])[0].numpy()
        for i, d in zip((0, 2), glob_dims):
            diff = abs(float(lg[i] - lg_base[i]))
            flag = "不变 ✓" if diff < 1e-9 else "泄漏 ✗"
            if diff >= 1e-9:
                ok2 = False
            print(f"  {name.strip()} · {d:<9s} logits Δ={diff:.2e}  {flag}")

    print("\n-- 断言 3：差异方向（换用户对 semantic 的影响 ≥ 换业务）--")
    d_user = abs(out["换用户 u_wang|research|srtp"]["semantic"] - base["semantic"])
    d_biz = abs(out["换业务 u_tu|research|club"]["semantic"] - base["semantic"])
    ok3 = d_user >= d_biz - 1e-9
    print(f"  换用户 Δsemantic={d_user:.6f}  换业务 Δsemantic={d_biz:.6f}  "
          f"{'✓' if ok3 else '（信息性，不强求）'}")

    print(f"\n== 结论：条件化生效={'是' if ok1 else '否（bug）'}；"
          f"切片隔离={'是' if ok2 else '否（bug）'} ==")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
