#!/usr/bin/env python3
import os
import json
import http.server
import socketserver
from urllib.parse import urlparse

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # API Endpoints
        if parsed_path.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Read channel stats
            stats = {}
            stats_path = os.path.join(DIRECTORY, "channel_stats.json")
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, "r") as f:
                        stats = json.load(f)
                except Exception as e:
                    stats = {"error": f"Failed to load stats: {str(e)}"}
            
            # Check pipeline active status
            # The pipeline runs on OCI and syncs logs/heartbeat here via rsync, so
            # a local `ps` check is unreliable. Instead trust the heartbeat file:
            # "running" is true only if it says running AND was freshened recently.
            import time
            is_running = False
            hb_path = os.path.join(DIRECTORY, "logs", "pipeline_heartbeat.json")
            if os.path.exists(hb_path):
                try:
                    with open(hb_path, "r") as f:
                        hb = json.load(f)
                    hb_ts = hb.get("ts", "")
                    if hb.get("running") and hb_ts:
                        hb_time = time.mktime(time.strptime(hb_ts, "%Y-%m-%dT%H:%M:%SZ"))
                        # Fresh heartbeat (< 60s old) => pipeline is live right now.
                        is_running = (time.time() - hb_time) < 60
                except Exception:
                    is_running = False
                
            response = {
                "is_running": is_running,
                "channel_stats": stats
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif parsed_path.path == "/api/published":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            published = []
            published_path = os.path.join(DIRECTORY, "published_topics.json")
            if os.path.exists(published_path):
                try:
                    with open(published_path, "r") as f:
                        published = json.load(f)
                except Exception as e:
                    published = [{"headline": f"Error loading published topics: {str(e)}"}]
                    
            self.wfile.write(json.dumps(published).encode())
            
        elif parsed_path.path == "/api/logs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Return last 100 lines of standard log
            log_lines = []
            log_path = os.path.join(DIRECTORY, "logs", "pipeline_run.log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", errors="ignore") as f:
                        log_lines = f.readlines()[-150:]
                except Exception as e:
                    log_lines = [f"Error reading logs: {str(e)}"]
            
            # Return last 50 structured logs if available
            structured_logs = []
            struct_log_path = os.path.join(DIRECTORY, "logs", "csvg_execution.log")
            if os.path.exists(struct_log_path):
                try:
                    with open(struct_log_path, "r", errors="ignore") as f:
                        lines = f.readlines()[-50:]
                        for line in lines:
                            if line.strip():
                                structured_logs.append(json.loads(line))
                except Exception:
                    pass

            response = {
                "raw_logs": "".join(log_lines),
                "structured_logs": structured_logs
            }
            self.wfile.write(json.dumps(response).encode())
            
        else:
            # Default behavior for serving static files
            super().do_GET()

if __name__ == "__main__":
    # Ensure logs folder exists
    os.makedirs(os.path.join(DIRECTORY, "logs"), exist_ok=True)
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"CSVG Pi 5 Verification Dashboard running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
