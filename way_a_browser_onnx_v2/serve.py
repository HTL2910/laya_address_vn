"""Lightweight local static HTTP server for Laya Address VN v2 (Way A - Browser ONNX)."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"


class LayaStaticHandler(SimpleHTTPRequestHandler):
    """Static file handler with explicit WASM/MJS/ONNX MIME types and CORS headers."""

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
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

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


def create_server(host: str = "127.0.0.1", port: int = 8083) -> ReusableThreadingHTTPServer:
    return ReusableThreadingHTTPServer((host, port), LayaStaticHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Laya Address VN v2 (Way A) static web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8083, help="Bind port (default: 8083)")
    args = parser.parse_args()

    if not WEB_DIR.exists():
        raise SystemExit(f"Web directory not found: {WEB_DIR}")

    server = create_server(args.host, args.port)
    print(f"Serving Laya Address VN v2 (Way A) from {WEB_DIR}")
    print(f"Open: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
