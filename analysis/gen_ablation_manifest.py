"""从 config/ablation/*.yaml 生成 ablation_manifest.json（V2.1：YAML 唯一真源，JSON 不手写）。

用法: python -m analysis.gen_ablation_manifest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srtp_memory.monitor.run_registry import gen_ablation_manifest  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    m = gen_ablation_manifest(root / "config/ablation", root / "data/runs/ablation_manifest.json")
    print("生成 ablation_manifest.json:")
    for g, cfg in m["groups"].items():
        print(f"  {g}: {cfg}")
