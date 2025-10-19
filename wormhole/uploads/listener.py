# listener.py
import http.server
import socketserver

PORT = 5005


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        print(f"\n--- Received POST ---\n{body}\n---------------------\n")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


print(f"Listening on port {PORT}...")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
