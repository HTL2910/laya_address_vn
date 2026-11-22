"""Lightweight local static HTTP server for Laya Address VN Widget v3 (Zero-CDN)."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


class LayaWidgetHandler(SimpleHTTPRequestHandler):
    """Static file handler serving dist/, sdk/, and demo/ with proper WASM/JS MIME types."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".mjs": "application/javascript; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/demo/index.html")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def create_server(host: str = "127.0.0.1", port: int = 8086) -> ReusableThreadingHTTPServer:
    return ReusableThreadingHTTPServer((host, port), LayaWidgetHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Laya Address VN Widget v3 demo & SDK.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8086, help="Bind port (default: 8086)")
    args = parser.parse_args()

    if not (ROOT_DIR / "demo" / "index.html").exists():
        raise SystemExit(f"Demo index.html not found under {ROOT_DIR}")

    server = create_server(args.host, args.port)
    print(f"Serving @laya/address-vn-widget v3.0.0 from {ROOT_DIR}")
    print(f"Open: http://{args.host}:{args.port}/demo/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
