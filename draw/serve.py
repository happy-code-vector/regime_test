#!/usr/bin/env python3
"""
Simple HTTP server to view the regime chart.
Run this script and open http://localhost:8000/regime_chart.html in your browser.
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        regime_url = f'http://localhost:{PORT}/regime_chart.html'
        close_url = f'http://localhost:{PORT}/close_chart.html'
        print(f"🚀 Server started at {regime_url}")
        print(f"   Also available: {close_url}")
        print("Press Ctrl+C to stop the server")

        # Open browser automatically
        webbrowser.open(regime_url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
            httpd.server_close()
