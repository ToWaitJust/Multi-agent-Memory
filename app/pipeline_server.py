"""全流程可视化本地服务（可选增强，零第三方依赖）。

启动：python -m app.pipeline_server [--port 8890]
  - GET /              返回 ChatGPT 白昼风可视化页（app/pipeline_viz.html）
  - GET /api/trace?q=  实时重跑一次完整调度，返回全流程 trace JSON
页面「重新运行」按钮即调用 /api/trace，实现"换查询看全流程"。
用标准库 http.server 实现，无需安装 flask。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.gen_pipeline_trace import DEFAULT_QUERY, DEMO_CORPUS, build_trace  # noqa: E402

VIZ_HTML = (ROOT / "app" / "pipeline_viz.html").read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = VIZ_HTML.encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)
        elif parsed.path == "/api/trace":
            qs = parse_qs(parsed.query)
            q = (qs.get("q", [DEFAULT_QUERY])[0]).strip()[:200]
            config = qs.get("config", ["full_ours"])[0]
            try:
                trace = build_trace(config, q, DEMO_CORPUS)
                self._send(200, "application/json; charset=utf-8",
                           json.dumps(trace, ensure_ascii=False).encode("utf-8"))
            except Exception as e:  # noqa: BLE001
                self._send(500, "application/json; charset=utf-8",
                           json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # 静默访问日志
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"● 全流程可视化服务: http://{args.host}:{args.port}   (Ctrl+C 停止)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
