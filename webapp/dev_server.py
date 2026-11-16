"""Run the web app locally without Vercel:  python webapp/dev_server.py  then open http://localhost:8000

Serves the static page from this folder and routes POST /api/predict to the same handler
Vercel uses in production. Needs only the Python standard library.
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "api"))
import predict as backend  # noqa: E402


class Dev(SimpleHTTPRequestHandler):
    _send = backend.handler._send

    def do_POST(self):
        if self.path.split("?")[0] == "/api/predict":
            return backend.handler.do_POST(self)
        self.send_error(404)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/predict":
            return backend.handler.do_GET(self)
        super().do_GET()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), partial(Dev, directory=str(HERE))).serve_forever()
