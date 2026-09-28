"""一次性预热 embedding 缓存：把数据集里所有文本编码并写入落盘缓存。

背景（D-045）：扩图到 22 节点 / 490 条记忆后，主流程里边跑边编码会在
网络抖动时卡死整个实验（实测进程被回收）。把编码独立成预热步骤，
实验阶段全部命中缓存，无网络依赖，快且稳。
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

from srtp_memory.plugins import get_plugin, register_all  # noqa: E402


def main() -> None:
    register_all()
    emb = get_plugin("embed.dashscope")
    rows = [json.loads(l) for l in open("data/graph_dataset.jsonl", encoding="utf-8") if l.strip()]
    texts: set[str] = set()
    for e in rows:
        texts.add(e["query"])
        for n in e["nodes"]:
            texts.add(f"{n['topic']}。{n.get('summary', '')}")
        for m in e["corpus"]:
            texts.add(m["text"])
    ordered = sorted(texts)
    print(f"待编码 {len(ordered)} 条", flush=True)
    t0 = time.time()
    for i, t in enumerate(ordered):
        emb.encode(t)
        if i % 50 == 0:
            print(f"  {i}/{len(ordered)}  elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"done {len(ordered)} 条  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
