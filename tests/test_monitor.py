"""监控模块单测（§4.9：ScheduleLogger / ReMeMonitor / RunRegistry）。"""
import json

from srtp_memory.monitor.reme_monitor import ReMeMonitor
from srtp_memory.monitor.run_registry import RunRegistry, gen_ablation_manifest
from srtp_memory.monitor.schedule_logger import ScheduleLogger


class TestScheduleLogger:
    def test_buffered_flush(self, tmp_path):
        p = tmp_path / "schedule.jsonl"
        logger = ScheduleLogger(p, buffer_size=5)  # 缓冲 5，写 3 不自动落盘
        for i in range(3):
            logger.log_schedule({"run_id": f"r{i}"})
        assert not p.exists()  # 未满不落盘
        logger.flush()
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3
        assert rows[0]["event"] == "schedule"
        assert rows[0]["run_id"] == "r0"

    def test_auto_flush_when_full(self, tmp_path):
        p = tmp_path / "s2.jsonl"
        logger = ScheduleLogger(p, buffer_size=2)
        logger.log_schedule({"a": 1})
        logger.log_schedule({"a": 2})  # 达缓冲 → 自动落盘
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2


class TestReMeMonitor:
    def test_record_and_flush(self, tmp_path):
        p = tmp_path / "reme_jobs.jsonl"
        mon = ReMeMonitor(p, buffer_size=2)
        mon.record_job("search", {"query": "x"}, ["m1", "m2"], 12.5)
        mon.record_job("auto_memory", {"session_id": "s1"}, None, 3.2)
        mon.flush()
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[0]["job"] == "search"
        assert rows[0]["result_summary"] == "list[2]"
        assert rows[0]["event"] == "reme_job"


class TestRunRegistry:
    def test_register_and_hash(self, tmp_path):
        p = tmp_path / "run_manifest.csv"
        reg = RunRegistry(p)
        h1 = reg.config_hash({"a": 1, "b": 2})
        h2 = reg.config_hash({"b": 2, "a": 1})  # 顺序无关
        assert h1 == h2
        reg.register({"run_id": "r1", "seed": 42, "model": "deepseek"})
        reg.register({"run_id": "r2", "seed": 43, "model": "dashscope"})
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "run_id,seed,model"  # 表头
        assert len(lines) == 3  # 表头 + 2 行


class TestGenAblationManifest:
    def test_generates_from_yaml(self, tmp_path):
        from pathlib import Path
        d = tmp_path / "ablation"
        d.mkdir()
        (d / "baseline_naive.yaml").write_text(
            "retriever_impl: retriever.full\nattention_impl: attention.noop\n"
            "weight_impl: weight.uniform\nselector_impl: selector.topk\n"
            "conditioning_dims: []\n", encoding="utf-8")
        (d / "full_ours.yaml").write_text(
            "retriever_impl: retriever.bm25\nattention_impl: attention.full\n"
            "weight_impl: weight.full\nselector_impl: selector.full\n"
            "conditioning_dims: [semantic, task]\n", encoding="utf-8")
        out = tmp_path / "manifest.json"
        m = gen_ablation_manifest(d, out)
        assert set(m["groups"].keys()) == {"baseline_naive", "full_ours"}
        assert m["groups"]["full_ours"]["retriever"] == "retriever.bm25"
        assert m["groups"]["full_ours"]["condition"] == ["semantic", "task"]
        assert out.exists()
