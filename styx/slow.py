"""The same endpoint at a fraction of its speed: every byte of a reply is passed on `factor` times
as long after its request began as it arrived. Weights, hardware and replies stay what they are.

python -m styx.slow <port> <upstream> <factor>     python -m styx.slow 8002 http://127.0.0.1:8001 2
"""

import http.client
import queue
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, UPSTREAM, FACTOR = int(sys.argv[1]), urllib.parse.urlparse(sys.argv[2]), float(sys.argv[3])


class Slow(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def relay(self):
        start = time.time()
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        up = http.client.HTTPConnection(UPSTREAM.hostname, UPSTREAM.port, timeout=3600)
        up.request(self.command, self.path, body, {k: v for k, v in self.headers.items() if k.lower() != "host"})
        reply = up.getresponse()
        self.send_response(reply.status)
        for k, v in reply.getheaders():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        arrived = queue.Queue()  # read as it comes, so that holding one chunk back does not date the next

        def read():
            while chunk := reply.read1(65536):
                arrived.put((time.time(), chunk))
            arrived.put((time.time(), b""))

        threading.Thread(target=read, daemon=True).start()
        while True:
            at, chunk = arrived.get()
            time.sleep(max(0, start + FACTOR * (at - start) - time.time()))
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk) if chunk else b"0\r\n\r\n")
            self.wfile.flush()
            if not chunk:
                break

    do_GET = do_POST = relay

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Slow).serve_forever()
