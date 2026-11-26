#!/usr/bin/env python3
"""L3MS Local Live Control Dashboard (:8085 on Tailscale).

Stdlib-only ThreadingHTTPServer + Server-Sent Events (SSE) backend.
Reads LLAMA_SWAP_API_KEY server-side and never exposes it to browsers.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any
import urllib.parse

# Allow running both as `python3 tools/local-dashboard/server.py` and via unittest imports.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from router_client import RouterClient, scrub_secrets  # noqa: E402
from telemetry import TelemetryCollector  # noqa: E402


STATIC_DIR = _THIS_DIR / "static"

EXTRA_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
    ".ico": "image/x-icon",
}


class DashboardServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying shared telemetry, router client, and SSE shutdown state."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        telemetry: TelemetryCollector,
        router: RouterClient,
        static_dir: Path = STATIC_DIR,
        sse_interval: float = 2.0,
        journal_runner: Any = None,
    ) -> None:
        self.telemetry = telemetry
        self.router = router
        self.static_dir = static_dir.resolve()
        self.sse_interval = max(0.1, float(sse_interval))
        self.journal_runner = journal_runner
        self.shutdown_event = threading.Event()
        super().__init__(server_address, DashboardRequestHandler)

    def shutdown(self) -> None:
        self.shutdown_event.set()
        super().shutdown()


class DashboardRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: DashboardServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy per-poll stdout logs; scrub any accidental secret in error messages.
        msg = format % args if args else format
        if any(code in msg for code in (" 500 ", " 502 ", " 504 ")):
            safe = scrub_secrets(msg, self.server.router._secrets_list())
            sys.stderr.write(f"[l3ms-dashboard] {self.address_string()} - {safe}\n")

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, status: int, content_type: str, data: bytes, cache: str = "no-cache") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError:
            return {}
        if length <= 0 or length > 65536:
            return {}
        body = self.rfile.read(length)
        try:
            parsed = json.loads(body.decode("utf-8", errors="replace"))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _serve_static(self, rel_path: str) -> None:
        # Reject raw traversal or backslash tricks before resolving
        clean = urllib.parse.unquote(rel_path).lstrip("/")
        if not clean or ".." in Path(clean).parts or "\\" in clean:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden path"})
            return

        candidate = (self.server.static_dir / clean).resolve()
        try:
            candidate.relative_to(self.server.static_dir)
        except ValueError:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden path"})
            return

        if not candidate.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "static file not found"})
            return

        ctype = EXTRA_MIME_TYPES.get(candidate.suffix.lower())
        if not ctype:
            guessed, _ = mimetypes.guess_type(str(candidate))
            ctype = guessed or "application/octet-stream"
        self._send_bytes(HTTPStatus.OK, ctype, candidate.read_bytes())

    def _handle_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        while not self.server.shutdown_event.is_set():
            try:
                telemetry = self.server.telemetry.snapshot()
                running_info = self.server.router.fetch_running(timeout=2.5)
                payload = {
                    "ts": time.time(),
                    "telemetry": telemetry,
                    "router": running_info,
                }
                frame = f"event: snapshot\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                break
            except Exception as exc:
                try:
                    err_payload = {"error": scrub_secrets(str(exc), self.server.router._secrets_list())}
                    self.wfile.write(f"event: error\ndata: {json.dumps(err_payload)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except OSError:
                    break

            if self.server.shutdown_event.wait(self.server.sse_interval):
                break

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in {"/", "/index.html"}:
            self._serve_static("index.html")
            return

        if path == "/favicon.svg":
            self._serve_static("favicon.svg")
            return

        if path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
            return

        if path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "l3ms-local-dashboard"})
            return

        if path == "/api/telemetry":
            self._send_json(HTTPStatus.OK, self.server.telemetry.snapshot())
            return

        if path == "/api/events":
            self._handle_sse()
            return

        if path == "/api/router":
            include_unlisted = qs.get("include_unlisted", ["0"])[0].lower() in {"1", "true", "yes"}
            self._send_json(HTTPStatus.OK, self.server.router.get_router_state(include_unlisted=include_unlisted))
            return

        if path == "/api/logs":
            source = qs.get("source", ["llama-swap"])[0]
            if source not in {"llama-swap", "engine"}:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "source must be 'llama-swap' or 'engine'"})
                return
            try:
                lines = int(qs.get("lines", ["100"])[0])
            except ValueError:
                lines = 100
            data = self.server.router.get_logs(
                source=source,
                lines=lines,
                journal_runner=self.server.journal_runner,
            )
            self._send_json(HTTPStatus.OK, data)
            return

        if path == "/api/benches":
            selected_file = qs.get("file", [None])[0] or None
            try:
                limit = int(qs.get("limit", ["60"])[0])
            except ValueError:
                limit = 60
            self._send_json(HTTPStatus.OK, self.server.router.get_benches(selected_file=selected_file, limit=limit))
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown endpoint: {path}"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/models/unload":
            body = self._read_json_body()
            model_id = body.get("model") if isinstance(body.get("model"), str) else None
            result = self.server.router.unload_models(model_id=model_id)
            status = HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_GATEWAY
            self._send_json(status, result)
            return

        # POST /api/models/<id>/wake or POST /api/models/<id>/smoke
        if path.startswith("/api/models/"):
            rest = path[len("/api/models/") :]
            if rest.endswith("/wake"):
                model_id = urllib.parse.unquote(rest[: -len("/wake")])
                result = self.server.router.wake_model(model_id)
                status = HTTPStatus.OK if result["ok"] else (HTTPStatus.BAD_REQUEST if result.get("error") == "invalid model id" else HTTPStatus.BAD_GATEWAY)
                self._send_json(status, result)
                return

            if rest.endswith("/smoke"):
                model_id = urllib.parse.unquote(rest[: -len("/smoke")])
                result = self.server.router.smoke_model(model_id)
                status = HTTPStatus.OK if result["ok"] else (HTTPStatus.BAD_REQUEST if result.get("error") == "invalid model id" else HTTPStatus.BAD_GATEWAY)
                self._send_json(status, result)
                return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown POST endpoint: {path}"})


def create_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8085,
    repo_root: Path | str = Path("/home/kchauhan/repos/l3ms"),
    swap_url: str = "http://127.0.0.1:8080",
    tailscale_host: str = "yeti-cachy.llama-alnair.ts.net",
    ui_port: int = 8089,
    sse_interval: float = 2.0,
    api_key: str | None = None,
    telemetry: TelemetryCollector | None = None,
    router: RouterClient | None = None,
    journal_runner: Any = None,
) -> DashboardServer:
    root_path = Path(repo_root).resolve()
    tel = telemetry if telemetry is not None else TelemetryCollector()
    rtr = (
        router
        if router is not None
        else RouterClient(
            repo_root=root_path,
            swap_url=swap_url,
            api_key=api_key,
            tailscale_host=tailscale_host,
            ui_port=ui_port,
        )
    )
    return DashboardServer(
        (host, port),
        telemetry=tel,
        router=rtr,
        sse_interval=sse_interval,
        journal_runner=journal_runner,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L3MS Local Live Control Dashboard (:8085)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8085, help="Bind port (default: 8085)")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent.parent.parent),
        help="Path to L3MS repository root",
    )
    parser.add_argument(
        "--swap-url",
        default="http://127.0.0.1:8080",
        help="Local llama-swap base URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--tailscale-host",
        default="yeti-cachy.llama-alnair.ts.net",
        help="Tailscale FQDN for deep links",
    )
    parser.add_argument("--ui-port", type=int, default=8089, help="Fixed Strata/llama.cpp UI port (default: 8089)")
    parser.add_argument("--sse-interval", type=float, default=2.0, help="SSE push interval in seconds (default: 2.0)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    server = create_server(
        host=args.host,
        port=args.port,
        repo_root=args.repo_root,
        swap_url=args.swap_url,
        tailscale_host=args.tailscale_host,
        ui_port=args.ui_port,
        sse_interval=args.sse_interval,
    )

    def _handle_sig(_signum: int, _frame: Any) -> None:
        server.shutdown_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    sys.stderr.write(
        f"[l3ms-dashboard] listening on http://{args.host}:{args.port} "
        f"(repo={Path(args.repo_root).resolve()}, swap={args.swap_url})\n"
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
