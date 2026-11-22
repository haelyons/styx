"""The same endpoint at a fraction of its speed: every byte of a reply is passed on `factor` times
as long after its request began as it arrived. Weights, hardware and replies stay what they are.

python -m styx.slow <port> <upstream> <factor>     python -m styx.slow 8002 http://127.0.0.1:8001 2
"""

import http.client
import sys
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
        while chunk := reply.read1(65536):
            time.sleep(max(0, start + FACTOR * (time.time() - start) - time.time()))
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    do_GET = do_POST = relay

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Slow).serve_forever()
