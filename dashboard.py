"""
Autoresearch Dashboard — single-file Python web app.
Pulls results.tsv and codex.log from S3 every 30s, serves a live dashboard.

Usage: python3 dashboard.py
Then open http://localhost:9090
"""

import csv
import io
import json
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

SOURCES = {
    "1xA100 (Speed)": {
        "results": "s3://fuelos-autoresearch/latest-1xa100/experiments/004_kernel_speedrun/results.tsv",
        "log": "s3://fuelos-autoresearch/latest-1xa100/codex.log",
        "mode": "speed",
    },
    "8xH100 (BPB)": {
        "results": "s3://fuelos-autoresearch/latest-8xh100/experiments/002_autoresearch/results.tsv",
        "log": "s3://fuelos-autoresearch/latest-8xh100/codex.log",
        "mode": "bpb",
    },
}
AWS_PROFILE = "fuelos"
PORT = 9090

cache = {
    "sources": {name: {"results": [], "codex_log": "", "mode": cfg["mode"]} for name, cfg in SOURCES.items()},
    "last_sync": "",
}


def s3_cat(path):
    r = subprocess.run(
        ["aws", "s3", "cp", path, "-", "--profile", AWS_PROFILE],
        capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else ""


def sync():
    while True:
        for name, paths in SOURCES.items():
            tsv = s3_cat(paths["results"])
            if tsv:
                reader = csv.DictReader(io.StringIO(tsv), delimiter="\t")
                cache["sources"][name]["results"] = list(reader)

            log = s3_cat(paths["log"])
            if log:
                lines = log.strip().split("\n")
                cache["sources"][name]["codex_log"] = "\n".join(lines[-40:])

        cache["last_sync"] = time.strftime("%H:%M:%S")
        time.sleep(30)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autoresearch Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, system-ui, monospace; padding: 24px; }
  h1 { font-size: 20px; margin-bottom: 16px; color: #58a6ff; }
  .meta { font-size: 12px; color: #484f58; margin-bottom: 20px; }
  .tabs { display: flex; gap: 0; margin-bottom: 20px; }
  .tab { padding: 10px 24px; background: #161b22; border: 1px solid #30363d; cursor: pointer; font-size: 13px; color: #8b949e; }
  .tab:first-child { border-radius: 8px 0 0 8px; }
  .tab:last-child { border-radius: 0 8px 8px 0; }
  .tab.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
  .cards { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; min-width: 140px; }
  .card .label { font-size: 11px; color: #484f58; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 24px; font-weight: bold; margin-top: 4px; }
  .best { color: #3fb950; }
  .chart-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; height: 500px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px; }
  th { text-align: left; padding: 8px 10px; border-bottom: 2px solid #30363d; color: #484f58; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  td { padding: 8px 10px; border-bottom: 1px solid #21262d; }
  tr.keep { border-left: 3px solid #3fb950; }
  tr.crash { border-left: 3px solid #f85149; opacity: 0.6; }
  tr.discard { border-left: 3px solid #484f58; opacity: 0.6; }
  tr.best-row td { background: #0d2818; }
  .log-box { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; font-size: 12px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; color: #8b949e; }
  h2 { font-size: 14px; color: #58a6ff; margin-bottom: 8px; }
  .source-section { display: none; }
  .source-section.active { display: block; }
</style>
</head>
<body>
<h1>Autoresearch Dashboard</h1>
<div class="meta" id="meta">Loading...</div>
<div class="tabs" id="tabs"></div>
<div id="sections"></div>
<script>
const COLORS = { '1xA100': '#58a6ff', '8xH100': '#d2a8ff' };
let charts = {};
let activeTab = null;
let cachedData = null;

function switchTab(name) {
  activeTab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.name === name));
  document.querySelectorAll('.source-section').forEach(s => s.classList.toggle('active', s.dataset.name === name));
  renderSource(name, cachedData.sources[name]);
}

function renderSource(name, src) {
  const mode = src.mode;
  const results = src.results;
  const total = results.length;
  const keeps = results.filter(r => r.status === 'keep').length;
  const crashes = results.filter(r => r.status === 'crash').length;
  const color = COLORS[name] || '#58a6ff';

  if (mode === 'speed') {
    const valid = results.filter(r => r.status === 'keep' && parseFloat(r.ms_per_step) > 0);
    const bestMs = valid.length ? Math.min(...valid.map(r => parseFloat(r.ms_per_step))) : null;
    const bestMfu = valid.length ? Math.max(...valid.map(r => parseFloat(r.mfu_percent))) : null;

    document.getElementById('cards-' + name).innerHTML =
      '<div class="card"><div class="label">Best ms/step</div><div class="value best">' + (bestMs ? bestMs.toFixed(1) : '-') + '</div></div>' +
      '<div class="card"><div class="label">Best MFU%</div><div class="value" style="color:#d2a8ff">' + (bestMfu ? bestMfu.toFixed(1) + '%' : '-') + '</div></div>' +
      '<div class="card"><div class="label">Experiments</div><div class="value">' + total + '</div></div>' +
      '<div class="card"><div class="label">Kept</div><div class="value" style="color:#3fb950">' + keeps + '</div></div>' +
      '<div class="card"><div class="label">Crashes</div><div class="value" style="color:#f85149">' + crashes + '</div></div>';

    const chartData = valid.map((r, i) => ({ x: i + 1, y: parseFloat(r.ms_per_step) }));
    if (charts[name]) charts[name].destroy();
    charts[name] = new Chart(document.getElementById('chart-' + name), {
      type: 'line',
      data: { datasets: [{ label: 'ms/step', data: chartData, borderColor: color, backgroundColor: color + '1a', fill: true, tension: 0.3, pointRadius: 5, pointBackgroundColor: chartData.map(p => p.y === bestMs ? '#3fb950' : color) }] },
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 'Experiment #', color: '#8b949e', font: { size: 14 } }, ticks: { color: '#8b949e', font: { size: 13 }, stepSize: 1 }, grid: { color: '#21262d' } }, y: { title: { display: true, text: 'ms / step (lower = better)', color: '#8b949e', font: { size: 14 } }, ticks: { color: '#8b949e', font: { size: 13 }, callback: function(v) { return v.toFixed(1); } }, grid: { color: '#21262d' }, min: chartData.length ? Math.min(...chartData.map(p => p.y)) * 0.9 : undefined, max: chartData.length ? Math.max(...chartData.map(p => p.y)) * 1.1 : undefined } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: function(ctx) { return ctx.parsed.y.toFixed(2) + ' ms/step'; } } } } }
    });

    let html = '';
    results.forEach((r, i) => {
      const isBest = r.status === 'keep' && parseFloat(r.ms_per_step) === bestMs;
      const cls = r.status + (isBest ? ' best-row' : '');
      html += '<tr class="' + cls + '"><td>' + (i+1) + '</td><td>' + r.commit + '</td><td>' + (r.ms_per_step || '-') + '</td><td>' + (r.mfu_percent || '-') + '</td><td>' + (r.val_bpb || '-') + '</td><td>' + r.status + '</td><td>' + r.description + '</td></tr>';
    });
    document.getElementById('tbody-' + name).innerHTML = html;

  } else {
    const valid = results.filter(r => r.status === 'keep' && parseFloat(r.post_quant_bpb) > 0);
    const bestBpb = valid.length ? Math.min(...valid.map(r => parseFloat(r.post_quant_bpb))) : null;

    document.getElementById('cards-' + name).innerHTML =
      '<div class="card"><div class="label">Best BPB</div><div class="value best">' + (bestBpb ? bestBpb.toFixed(4) : '-') + '</div></div>' +
      '<div class="card"><div class="label">Experiments</div><div class="value">' + total + '</div></div>' +
      '<div class="card"><div class="label">Kept</div><div class="value" style="color:#3fb950">' + keeps + '</div></div>' +
      '<div class="card"><div class="label">Crashes</div><div class="value" style="color:#f85149">' + crashes + '</div></div>';

    const chartData = valid.map((r, i) => ({ x: i + 1, y: parseFloat(r.post_quant_bpb) }));
    if (charts[name]) charts[name].destroy();
    charts[name] = new Chart(document.getElementById('chart-' + name), {
      type: 'line',
      data: { datasets: [{ label: 'Post-Quant BPB', data: chartData, borderColor: color, backgroundColor: color + '1a', fill: true, tension: 0.3, pointRadius: 5, pointBackgroundColor: chartData.map(p => p.y === bestBpb ? '#3fb950' : color) }] },
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', title: { display: true, text: 'Experiment #', color: '#8b949e', font: { size: 14 } }, ticks: { color: '#8b949e', font: { size: 13 }, stepSize: 1 }, grid: { color: '#21262d' } }, y: { title: { display: true, text: 'Post-Quant BPB', color: '#8b949e', font: { size: 14 } }, ticks: { color: '#8b949e', font: { size: 13 }, callback: function(v) { return v.toFixed(4); } }, grid: { color: '#21262d' }, min: chartData.length ? Math.min(...chartData.map(p => p.y)) - 0.005 : undefined, max: chartData.length ? Math.max(...chartData.map(p => p.y)) + 0.005 : undefined } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: function(ctx) { return 'BPB: ' + ctx.parsed.y.toFixed(6); } } } } }
    });

    let html = '';
    results.forEach((r, i) => {
      const isBest = r.status === 'keep' && parseFloat(r.post_quant_bpb) === bestBpb;
      const cls = r.status + (isBest ? ' best-row' : '');
      const size = r.artifact_bytes > 0 ? (parseFloat(r.artifact_bytes) / 1e6).toFixed(1) + ' MB' : '-';
      html += '<tr class="' + cls + '"><td>' + (i+1) + '</td><td>' + r.commit + '</td><td>' + r.val_bpb + '</td><td>' + r.post_quant_bpb + '</td><td>' + size + '</td><td>' + r.status + '</td><td>' + r.description + '</td></tr>';
    });
    document.getElementById('tbody-' + name).innerHTML = html;
  }

  document.getElementById('log-' + name).textContent = src.codex_log || 'No log data yet';
}

async function refresh() {
  const r = await fetch('/api/data?' + Date.now());
  cachedData = await r.json();
  document.getElementById('meta').textContent = 'Last sync: ' + cachedData.last_sync + ' \\u00b7 Auto-refresh 30s';

  const names = Object.keys(cachedData.sources);

  // Build tabs + sections once
  if (!activeTab) {
    document.getElementById('tabs').innerHTML = names.map(n => '<div class="tab" data-name="' + n + '" onclick="switchTab(\\'' + n + '\\')">' + n + '</div>').join('');
    document.getElementById('sections').innerHTML = names.map(n => {
      const m = cachedData.sources[n].mode;
      const hdr = m === 'speed'
        ? '<th>#</th><th>Commit</th><th>ms/step</th><th>MFU%</th><th>Val BPB</th><th>Status</th><th>Description</th>'
        : '<th>#</th><th>Commit</th><th>Val BPB</th><th>Post-Quant BPB</th><th>Artifact</th><th>Status</th><th>Description</th>';
      return '<div class="source-section" data-name="' + n + '">' +
        '<div class="cards" id="cards-' + n + '"></div>' +
        '<div class="chart-wrap"><canvas id="chart-' + n + '"></canvas></div>' +
        '<h2>Experiments</h2>' +
        '<table><thead><tr>' + hdr + '</tr></thead><tbody id="tbody-' + n + '"></tbody></table>' +
        '<h2>Agent Log</h2>' +
        '<div class="log-box" id="log-' + n + '"></div>' +
        '</div>';
    }).join('');
    activeTab = names.length > 1 ? names[1] : names[0];
  }

  switchTab(activeTab);
}
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path.startswith("/api/data"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(cache).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=sync, daemon=True).start()
    print(f"Dashboard running at http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()
