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
            # We can check if a process with main.py is running
            import subprocess
            is_running = False
            try:
                # Simple ps check
                ps_output = subprocess.check_output(["ps", "aux"]).decode()
                if "main.py --global" in ps_output or "main.py" in ps_output:
                    # Exclude the grep/check itself if it matches
                    is_running = True
            except Exception:
                pass
                
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
        print(f"🚀 CSVG Pi 5 Verification Dashboard running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
