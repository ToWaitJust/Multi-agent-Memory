"""实验登记（§11.4，FR-10）。run_manifest.csv + ablation_manifest.json 登记。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


class RunRegistry:
    def __init__(self, manifest_path: str | Path = "data/runs/run_manifest.csv"):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def config_hash(config_dict: dict) -> str:
        """srtp.yaml 内容哈希（可复现，NFR-6）。"""
        raw = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def register(self, entry: dict) -> None:
        """每次实验登记一行。列以首次写入为准。"""
        exists = self.manifest_path.exists()
        fieldnames = list(entry.keys())
        with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists or f.tell() == 0:
                writer.writeheader()
            writer.writerow(entry)


def gen_ablation_manifest(ablation_dir: str | Path,
                          out_path: str | Path = "data/runs/ablation_manifest.json") -> dict:
    """从 config/ablation/*.yaml 生成 ablation_manifest.json（V2.1：YAML 唯一真源，JSON 不手写）。"""
    import yaml
    d = Path(ablation_dir)
    groups: dict[str, dict] = {}
    for yf in sorted(d.glob("*.yaml")):
        cfg = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        groups[yf.stem] = {
            "retriever": cfg.get("retriever_impl"),
            "attention": cfg.get("attention_impl"),
            "weight": cfg.get("weight_impl"),
            "selector": cfg.get("selector_impl"),
            "condition": cfg.get("conditioning_dims") or None,
        }
    manifest = {
        "generated_from": f"{d}/",
        "groups": groups,
        "conditioning_dims": ["semantic", "task"],
        "fixed": ["embedding_dim=1024", "alpha=0.6", "beta=0.4", "max_shared=10", "candidate_override=50"],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
