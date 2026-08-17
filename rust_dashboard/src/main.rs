// CSVG live pipeline dashboard — Rust.
//
// A zero-dependency HTTP server (std only) that replaces the legacy Python
// `dashboard_server.py`. It reads the pipeline's runtime files under
// `CSVG_ROOT` (default: current working directory) and serves a self-contained
// SPA plus JSON endpoints:
//
//   GET /                -> dashboard SPA (web/index.html, embedded)
//   GET /api/status      -> heartbeat freshness, latest stage, channel stats
//   GET /api/logs        -> last N lines of pipeline_run.log / csvg_execution.log
//   GET /api/published   -> published_topics.json
//   GET /api/budget      -> per-run + monthly budget ledger (run_budget.json)
//   GET /api/runs        -> recent state checkpoints (logs/state_*.json)
//   GET /health          -> {"ok":true}
//
// Environment:
//   CSVG_ROOT             runtime root dir holding logs/ + *.json  (default ".")
//   CSVG_DASHBOARD_PORT   listen port                          (default "8080")
//   CSVG_BIND             bind address                         (default "0.0.0.0")
//   CSVG_HEARTBEAT_TTL    heartbeat freshness window in seconds (default 60)

mod json;

use json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const HTML: &str = include_str!("../web/index.html");
const MAX_REQUEST_BYTES: usize = 1 << 16;

fn main() {
    let root = env::var("CSVG_ROOT").unwrap_or_else(|_| ".".to_string());
    let port = env::var("CSVG_DASHBOARD_PORT").unwrap_or_else(|_| "8080".to_string());
    let bind = env::var("CSVG_BIND").unwrap_or_else(|_| "0.0.0.0".to_string());

    let listener = TcpListener::bind(format!("{bind}:{port}")).unwrap_or_else(|e| {
        eprintln!("[csvg-dashboard] bind {bind}:{port} failed: {e}");
        std::process::exit(1);
    });
    eprintln!("[csvg-dashboard] listening on http://{bind}:{port}  root={root}");

    for stream in listener.incoming() {
        match stream {
            Ok(conn) => {
                let root = root.clone();
                std::thread::spawn(move || handle(conn, &root));
            }
            Err(_) => {}
        }
    }
}

// ── HTTP plumbing ─────────────────────────────────────────────────────────

fn handle(mut stream: TcpStream, root: &str) {
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 2048];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() >= MAX_REQUEST_BYTES || ends_with_headers(&buf) {
                    break;
                }
            }
            Err(_) => return,
        }
    }
    let req = String::from_utf8_lossy(&buf);
    let first = req.lines().next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let (method, raw_path) = (parts.next().unwrap_or("GET"), parts.next().unwrap_or("/"));
    if method != "GET" && method != "POST" {
        respond(stream, 405, "text/plain; charset=utf-8", b"method not allowed".as_slice());
        return;
    }
    let path = raw_path.split('?').next().unwrap_or("/");

    let (code, ctype, body) = route(root, path, raw_path);
    respond(stream, code, ctype, &body);
}

fn ends_with_headers(buf: &[u8]) -> bool {
    buf.windows(4).any(|w| w == b"\r\n\r\n")
}

fn respond(mut stream: TcpStream, code: u16, ctype: &str, body: &[u8]) {
    let reason = match code {
        200 => "OK",
        204 => "No Content",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\nX-Powered-By: csvg-dashboard-rust\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

// ── routing ────────────────────────────────────────────────────────────────

fn route(root: &str, path: &str, raw_path: &str) -> (u16, &'static str, Vec<u8>) {
    match path {
        "/" | "/index.html" => (200, "text/html; charset=utf-8", HTML.as_bytes().to_vec()),
        "/favicon.ico" => (204, "text/plain", Vec::new()),
        "/health" => json_response(&obj(&[("ok", Value::Bool(true))])),
        "/api/status" => json_response(&api_status(root)),
        "/api/logs" => json_response(&api_logs(root)),
        "/api/published" => json_response(&api_published(root)),
        "/api/budget" => json_response(&api_budget(root)),
        "/api/usage" => json_response(&api_provider_usage(root)),
        "/api/analytics" => json_response(&api_analytics(root)),
        "/api/runs" => json_response(&api_runs(root)),
        "/api/seeding" => json_response(&api_seeding(root)),
        "/api/seeding/logs" => json_response(&api_seeding_logs(root)),
        "/api/seeding/trigger" => json_response(&api_seeding_trigger(root, raw_path)),
        "/api/cron" => {
            println!("[csvg-dashboard] GET /api/cron endpoint hit");
            json_response(&api_cron(root))
        }
        "/api/cron/update" => {
            println!("[csvg-dashboard] POST /api/cron/update endpoint hit");
            json_response(&api_cron_update(root, raw_path))
        }
        "/api/healthcheck/run" => json_response(&api_healthcheck_run(root)),
        "/api/cleanup/run" => json_response(&api_cleanup_run(root, raw_path)),
        "/api/usage/refresh" => json_response(&api_usage_refresh(root)),
        "/api/pipeline/start" | "/api/pipeline/resume" => json_response(&api_pipeline_control(root, path, raw_path)),
        "/api/pipeline/stop" => json_response(&api_pipeline_stop(root)),
        "/api/seeding/stop" => json_response(&api_seeding_stop(root)),
        _ => (404, "text/plain; charset=utf-8", b"not found".to_vec()),
    }
}

fn json_response(v: &Value) -> (u16, &'static str, Vec<u8>) {
    (200, "application/json", json::serialize(v).into_bytes())
}

// ── file helpers ───────────────────────────────────────────────────────────

fn read_text(root: &str, rel: &str) -> Option<String> {
    let p = PathBuf::from(root).join(rel);
    let p = fs::canonicalize(p).ok()?;
    fs::read(&p)
        .ok()
        .map(|b| String::from_utf8_lossy(&b).into_owned())
}

fn read_json(root: &str, rel: &str) -> Value {
    read_text(root, rel)
        .and_then(|t| json::parse(&t))
        .unwrap_or(Value::Null)
}

fn exists(root: &str, rel: &str) -> bool {
    Path::new(root).join(rel).exists()
}

fn now_epoch() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ── JSON object helpers ────────────────────────────────────────────────────

fn obj(fields: &[(&str, Value)]) -> Value {
    let mut m = BTreeMap::new();
    for (k, v) in fields {
        m.insert(k.to_string(), v.clone());
    }
    Value::Obj(m)
}

fn num(n: f64) -> Value {
    Value::Num(n)
}

fn s(x: &str) -> Value {
    Value::Str(x.to_string())
}

fn arr(items: Vec<Value>) -> Value {
    Value::Arr(items)
}

fn string_value(b: &Value) -> Option<String> {
    match b {
        Value::Str(s) => Some(s.clone()),
        Value::Num(n) => Some(json::fmt_num(*n)),
        Value::Bool(x) => Some(x.to_string()),
        _ => None,
    }
}

// Parse "YYYY-MM-DDTHH:MM:SS[.fff][Z|+00:00]" as a UTC epoch (seconds).
fn parse_iso_utc(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.len() < 20 {
        return None;
    }
    let y: i64 = s[0..4].parse().ok()?;
    let mo: i64 = s[5..7].parse().ok()?;
    let d: i64 = s[8..10].parse().ok()?;
    let h: i64 = s[11..13].parse().ok()?;
    let mi: i64 = s[14..16].parse().ok()?;
    let sec: i64 = s[17..19].parse().ok()?;

    // Only accept explicit UTC markers to avoid TZ-skewed freshness checks.
    let tz_ok = s.ends_with('Z') || s.ends_with("+00:00");
    if !tz_ok {
        return None;
    }
    let mut frac = 0.0;
    if s.chars().nth(19) == Some('.') {
        let body = &s[20..];
        let cut = body.find(['+', 'Z', '-']).unwrap_or(body.len());
        if cut > 0 {
            if let Ok(v) = body[..cut].parse::<f64>() {
                frac = v / 10f64.powi(cut as i32);
            }
        }
    }
    let days = days_from_civil(y, mo, d);
    Some(days as f64 * 86400.0 + (h * 3600 + mi * 60 + sec) as f64 + frac)
}

// Howard Hinnant's civil-from-days inverse.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

// ── /api/status ────────────────────────────────────────────────────────────

fn api_status(root: &str) -> Value {
    let ttl = env::var("CSVG_HEARTBEAT_TTL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(60.0);

    let mut is_running = false;
    let mut hb_ts = String::new();
    let mut hb_age: Option<f64> = None;

    if exists(root, "logs/pipeline_heartbeat.json") {
        let hb = read_json(root, "logs/pipeline_heartbeat.json");
        let running = matches!(&hb, Value::Obj(_))
            && hb.get_str("running").map(|v| v == "true").unwrap_or(false);
        if let Some(ts) = hb.get_str("ts") {
            hb_ts = ts.clone();
            if let Some(epoch) = parse_iso_utc(&ts) {
                let age = (now_epoch() - epoch).max(0.0);
                hb_age = Some(age);
                is_running = running && age < ttl;
            }
        }
    }

    let stats = if exists(root, "channel_stats.json") {
        read_json(root, "channel_stats.json")
    } else {
        Value::Null
    };

    let (latest_stage, _) = latest_state(root);

    let mut latest_run: Option<Value> = None;
    let agg = read_json(root, "logs/run_budget.json");
    if let Value::Obj(m) = &agg {
        let mut entries: Vec<(String, Value)> =
            m.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort_by(|a, b| b.1.get_str("started_at").cmp(&a.1.get_str("started_at")));
        latest_run = entries.first().map(|(_, v)| v.clone());
    }

    obj(&[
        ("is_running", Value::Bool(is_running)),
        ("heartbeat_ts", s(&hb_ts)),
        ("heartbeat_age_sec", hb_age.map(num).unwrap_or(Value::Null)),
        ("latest_stage", s(&latest_stage)),
        ("execution_stage", s(&latest_stage)),
        ("latest_run", latest_run.unwrap_or(Value::Null)),
        ("channel_stats", stats),
    ])
}

// newest state checkpoint -> (execution_stage, pipeline_id)
fn latest_state(root: &str) -> (String, String) {
    let dir = Path::new(root).join("logs");
    let Ok(rd) = fs::read_dir(&dir) else { return ("NOT_STARTED".into(), String::new()) };
    let mut states: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("state_") && name.ends_with(".json")
        })
        .collect();
    if states.is_empty() {
        return ("NOT_STARTED".into(), String::new());
    }
    states.sort_by(mtime_desc);
    let name = states[0].file_name().unwrap().to_str().unwrap().to_string();
    let v = read_json(root, &format!("logs/{name}"));
    let stage = v.get_str("execution_stage").unwrap_or_else(|| "UNKNOWN".into());
    let pid = v.get_str("pipeline_id").unwrap_or_default();
    (stage, pid)
}

fn last_failed_state(root: &str) -> Option<(String, String)> {
    let dir = Path::new(root).join("logs");
    let rd = fs::read_dir(&dir).ok()?;
    let mut states: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("state_") && name.ends_with(".json")
        })
        .collect();
    if states.is_empty() {
        return None;
    }
    states.sort_by(mtime_desc);
    for path in states {
        let name = path.file_name()?.to_str()?.to_string();
        let v = read_json(root, &format!("logs/{name}"));
        let stage = v.get_str("execution_stage").unwrap_or_else(|| "UNKNOWN".into());
        let pid = v.get_str("pipeline_id").unwrap_or_default();
        if stage != "PUBLISHED_SUCCESS" && stage != "DONE" && !pid.is_empty() {
            return Some((stage, pid));
        }
    }
    None
}

fn mtime_desc(a: &PathBuf, b: &PathBuf) -> std::cmp::Ordering {
    let ma = fs::metadata(a).and_then(|m| m.modified()).ok();
    let mb = fs::metadata(b).and_then(|m| m.modified()).ok();
    mb.cmp(&ma)
}

// ── /api/logs ──────────────────────────────────────────────────────────────

fn api_logs(root: &str) -> Value {
    let raw = read_text(root, "logs/pipeline_run.log")
        .map(|t| {
            let lines: Vec<&str> = t.lines().collect();
            let start = lines.len().saturating_sub(150);
            let mut tail = lines[start..].to_vec();
            tail.reverse();
            tail.join("\n")
        })
        .unwrap_or_default();

    let structured: Vec<Value> = read_text(root, "logs/csvg_execution.log")
        .map(|t| {
            t.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .rev()
                .take(50)
                .map(|l| s(l))
                .collect()
        })
        .unwrap_or_default();

    obj(&[
        ("raw_logs", s(&raw)),
        ("structured_logs", arr(structured)),
    ])
}

// ── /api/published ─────────────────────────────────────────────────────────

fn api_published(root: &str) -> Value {
    read_json(root, "published_topics.json")
}

// ── /api/seeding ───────────────────────────────────────────────────────────

fn api_seeding(root: &str) -> Value {
    let mut val = read_json(root, "logs/reddit_rotation_state.json");
    let is_running = is_seeding_running();
    match val {
        Value::Obj(ref mut map) => {
            map.insert("is_running".to_string(), Value::Bool(is_running));
            val
        }
        _ => obj(&[
            ("is_running", Value::Bool(is_running)),
            ("accounts", obj(&[])),
            ("subreddits", obj(&[])),
            ("posted_threads", arr(vec![])),
        ]),
    }
}

fn api_seeding_logs(root: &str) -> Value {
    let raw = read_text(root, "logs/seeding_execution.log")
        .or_else(|| read_text(root, "logs/reddit_link_seeder_cron.log"))
        .map(|t| {
            let lines: Vec<&str> = t.lines().collect();
            let start = lines.len().saturating_sub(150);
            lines[start..].join("\n")
        })
        .unwrap_or_default();

    obj(&[
        ("logs", s(&raw)),
        ("is_running", Value::Bool(is_seeding_running())),
    ])
}


fn api_cron(_root: &str) -> Value {
    let mut oci_lines = Vec::new();
    if let Ok(out) = std::process::Command::new("ssh")
        .args(&["-o", "StrictHostKeyChecking=no", "oci-prod", "crontab -l"])
        .output() {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let l = line.trim();
                if l.contains("cron_publish.sh") && !l.starts_with('#') {
                    let parts: Vec<&str> = l.split_whitespace().collect();
                    if parts.len() >= 5 {
                        oci_lines.push(parts[..5].join(" "));
                    }
                }
            }
        }
    }

    let mut seeder_lines = Vec::new();
    if let Ok(out) = std::process::Command::new("crontab")
        .arg("-l")
        .output() {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let l = line.trim();
                if l.contains("reddit_link_seeder.py") && !l.starts_with('#') {
                    let parts: Vec<&str> = l.split_whitespace().collect();
                    if parts.len() >= 5 {
                        seeder_lines.push(parts[..5].join(" "));
                    }
                }
            }
        }
    }

    obj(&[
        ("pipeline_cron", json::Value::Arr(oci_lines.into_iter().map(|l| s(&l)).collect())),
        ("seeder_cron", json::Value::Arr(seeder_lines.into_iter().map(|l| s(&l)).collect())),
    ])
}

fn api_cron_update(_root: &str, raw_path: &str) -> Value {
    let pipeline_param = get_query_param(raw_path, "pipeline").unwrap_or_default();
    let seeder_param = get_query_param(raw_path, "seeder").unwrap_or_default();

    let mut oci_success = true;
    let mut oci_err = String::new();
    let mut pi_success = true;
    let mut pi_err = String::new();

    // 1. Update OCI crontab
    if !pipeline_param.is_empty() {
        let mut current_crontab = String::new();
        if let Ok(out) = std::process::Command::new("ssh")
            .args(&["-o", "StrictHostKeyChecking=no", "oci-prod", "crontab -l"])
            .output() {
            if out.status.success() {
                current_crontab = String::from_utf8_lossy(&out.stdout).into_owned();
            }
        }

        let mut new_lines = Vec::new();
        let mut has_header = false;
        for line in current_crontab.lines() {
            let l = line.trim();
            if l.contains("cron_publish.sh") {
                continue;
            }
            if l.contains("region-optimized publication") {
                has_header = true;
                continue;
            }
            new_lines.push(line.to_string());
        }

        if !has_header {
            new_lines.push("# CSVG region-optimized publication (cron_publish.sh)".to_string());
            new_lines.push("MAILTO=\"\"".to_string());
        }
        for cron_expr in pipeline_param.split('|') {
            let expr = cron_expr.trim();
            if !expr.is_empty() {
                new_lines.push(format!("{} /home/ubuntu/buzzdropfeedv2/cron_publish.sh", expr));
            }
        }
        new_lines.push("".to_string());

        let new_crontab = new_lines.join("\n");
        let spawned = std::process::Command::new("ssh")
            .args(&["-o", "StrictHostKeyChecking=no", "oci-prod", "crontab -"])
            .stdin(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        match spawned {
            Ok(mut child) => {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(new_crontab.as_bytes());
                }
                if let Ok(output) = child.wait_with_output() {
                    if !output.status.success() {
                        oci_success = false;
                        oci_err = String::from_utf8_lossy(&output.stderr).into_owned();
                    }
                } else {
                    oci_success = false;
                    oci_err = "Failed to wait for remote crontab process".to_string();
                }
            }
            Err(e) => {
                oci_success = false;
                oci_err = e.to_string();
            }
        }
    }

    // 2. Update local crontab (Pi 5)
    if !seeder_param.is_empty() {
        let mut current_crontab = String::new();
        if let Ok(out) = std::process::Command::new("crontab")
            .arg("-l")
            .output() {
            if out.status.success() {
                current_crontab = String::from_utf8_lossy(&out.stdout).into_owned();
            }
        }

        let mut new_lines = Vec::new();
        for line in current_crontab.lines() {
            let l = line.trim();
            if l.contains("reddit_link_seeder.py") {
                continue;
            }
            new_lines.push(line.to_string());
        }

        new_lines.push(format!("{} cd /home/jeevanjoshi/buzzdropfeedv2 && flock -n /tmp/reddit_link.lock venv/bin/python reddit_link_seeder.py --max 1 >> /home/jeevanjoshi/buzzdropfeedv2/logs/reddit_link_seeder_cron.log 2>&1", seeder_param.trim()));
        new_lines.push("".to_string());

        let new_crontab = new_lines.join("\n");
        let spawned = std::process::Command::new("crontab")
            .arg("-")
            .stdin(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        match spawned {
            Ok(mut child) => {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(new_crontab.as_bytes());
                }
                if let Ok(output) = child.wait_with_output() {
                    if !output.status.success() {
                        pi_success = false;
                        pi_err = String::from_utf8_lossy(&output.stderr).into_owned();
                    }
                } else {
                    pi_success = false;
                    pi_err = "Failed to wait for local crontab process".to_string();
                }
            }
            Err(e) => {
                pi_success = false;
                pi_err = e.to_string();
            }
        }
    }

    let success = oci_success && pi_success;
    let mut err_parts = Vec::new();
    if !oci_success {
        err_parts.push(format!("OCI: {}", oci_err));
    }
    if !pi_success {
        err_parts.push(format!("Pi: {}", pi_err));
    }
    let err = err_parts.join("; ");

    obj(&[
        ("success", Value::Bool(success)),
        ("error", s(&err)),
    ])
}

// ── /api/analytics ──────────────────────────────────────────────────────────
// Analytics feedback loop: per-video growth metrics + niche signal written by
// src/engine/analytics_feedback.py (FactRetriever's "top growth drivers" bias).

fn api_analytics(root: &str) -> Value {
    let v = read_json(root, "logs/analytics_feedback.json");
    match v {
        Value::Null => obj(&[
            ("schema", s("analytics_feedback/v1")),
            ("videos", Value::Arr(Vec::new())),
            ("niche_signal", obj(&[])),
            ("captured_at", s("—")),
        ]),
        other => other,
    }
}

// ── /api/usage ──────────────────────────────────────────────────────────────
// Realtime provider API-usage (fal / Google / OpenRouter): live-capture ledger
// + authoritative provider pulls, written by src/engine/api_usage.py.

fn api_provider_usage(root: &str) -> Value {
    let v = read_json(root, "logs/provider_usage.json");
    match v {
        Value::Null => obj(&[
            ("schema", s("provider_usage/v1")),
            ("updated_at", s("—")),
            ("live", obj(&[])),
            ("daily", obj(&[])),
            ("provider_pull", obj(&[])),
            ("missing", Value::Bool(true)),
        ]),
        other => other,
    }
}

// ── /api/budget ────────────────────────────────────────────────────────────

fn api_budget(root: &str) -> Value {
    let agg = read_json(root, "logs/run_budget.json");
    let mut runs: Vec<Value> = Vec::new();
    let mut monthly: BTreeMap<String, BTreeMap<&'static str, f64>> = BTreeMap::new();

    if let Value::Obj(m) = &agg {
        let mut entries: Vec<(String, Value)> =
            m.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort_by(|a, b| b.1.get_str("started_at").cmp(&a.1.get_str("started_at")));

        for (_id, rec) in entries {
            if rec == Value::Null {
                continue;
            }
            let est_usd = rec
                .get_obj("totals")
                .and_then(|t| t.get("est_usd"))
                .and_then(|v| match v {
                    Value::Num(n) => Some(*n),
                    _ => None,
                })
                .unwrap_or(0.0);
            let yt_units = rec
                .get_obj("totals")
                .and_then(|t| t.get("yt_units"))
                .and_then(|v| match v {
                    Value::Num(n) => Some(*n),
                    _ => None,
                })
                .unwrap_or(0.0);

            runs.push(rec.clone());
            if let Some(mo) = rec.get_str("started_at").map(|t| t[..t.len().min(7)].to_string()) {
                let e = monthly.entry(mo).or_default();
                *e.entry("est_usd").or_insert(0.0) += est_usd;
                *e.entry("yt_units").or_insert(0.0) += yt_units;
                *e.entry("runs").or_insert(0.0) += 1.0;
            }
        }
    }

    let mut monthly_val = BTreeMap::new();
    for (mo, sums) in &monthly {
        monthly_val.insert(
            mo.clone(),
            obj(&[
                ("est_usd", num(sums.get("est_usd").copied().unwrap_or(0.0))),
                ("yt_units", num(sums.get("yt_units").copied().unwrap_or(0.0))),
                ("runs", num(sums.get("runs").copied().unwrap_or(0.0))),
            ]),
        );
    }

    let mut total_est = 0.0;
    let mut total_yt = 0.0;
    for sums in monthly.values() {
        total_est += sums.get("est_usd").copied().unwrap_or(0.0);
        total_yt += sums.get("yt_units").copied().unwrap_or(0.0);
    }

    obj(&[
        (
            "total",
            obj(&[
                ("runs", num(runs.len() as f64)),
                ("est_usd", num(total_est)),
                ("yt_units", num(total_yt)),
            ]),
        ),
        ("latest", runs.first().cloned().unwrap_or(Value::Null)),
        ("runs", arr(runs)),
        ("monthly", Value::Obj(monthly_val)),
    ])
}

// ── /api/runs ──────────────────────────────────────────────────────────────

fn api_runs(root: &str) -> Value {
    let dir = Path::new(root).join("logs");
    let Ok(rd) = fs::read_dir(&dir) else { return arr(Vec::new()) };
    let mut states: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("state_") && name.ends_with(".json")
        })
        .collect();
    states.sort_by(mtime_desc);

    let mut out = Vec::new();
    for p in states.iter().take(20) {
        let rel = format!("logs/{}", p.file_name().unwrap().to_str().unwrap());
        let v = read_json(root, &rel);
        if v == Value::Null {
            continue;
        }
        let topic = json::get_path(&v, &["selected_topic", "headline"])
            .and_then(string_value)
            .unwrap_or_default();
        let script = json::get_path(&v, &["script_data", "title"])
            .and_then(string_value)
            .unwrap_or_default();
        let video_id = json::get_path(&v, &["upload_metadata", "video_id"])
            .and_then(string_value)
            .unwrap_or_default();
        let niche = json::get_path(&v, &["selected_topic", "niche_category"])
            .and_then(string_value)
            .unwrap_or_default();
        let audience = json::get_path(&v, &["selected_topic", "audience_type"])
            .and_then(string_value)
            .unwrap_or_default();
        let topsis = json::get_path(&v, &["selected_topic", "topsis_score"])
            .and_then(string_value)
            .unwrap_or_default();
        let runtime = json::get_path(&v, &["script_data", "estimated_runtime_seconds"])
            .and_then(string_value)
            .unwrap_or_default();
        let shots = json::get_path(&v, &["script_data", "target_shots"])
            .and_then(string_value)
            .unwrap_or_default();
        let revenue = json::get_path(&v, &["revenue_forecast", "total_expected_revenue_usd"])
            .and_then(string_value)
            .unwrap_or_default();
        // Maturity-aware reporting: an unmonetized / brand-new channel is NOT
        // eligible for the forecast revenue (ads can't run yet). Surface the flag
        // and the unsclaled aspirational view count so the card shows context
        // instead of presenting a $0 / tiny forecast as if it were realized.
        let rev_eligible = json::get_path(&v, &["revenue_forecast", "monetization_eligible"])
            .and_then(string_value)
            .unwrap_or_else(|| "true".to_string());
        let projected_views = json::get_path(&v, &["revenue_forecast", "projected_views_at_scale"])
            .and_then(string_value)
            .unwrap_or_default();
        let region = v.get_str("region").unwrap_or_default();
        let fact_count = json::get_path(&v, &["verified_facts"])
            .map(|val| match val {
                Value::Arr(a) => a.len().to_string(),
                _ => "0".to_string(),
            })
            .unwrap_or_else(|| "0".to_string());
        let pinned_comment = json::get_path(&v, &["upload_metadata", "pinned_comment_text"])
            .and_then(string_value)
            .unwrap_or_default();
        let playlist_url = json::get_path(&v, &["upload_metadata", "playlist_url"])
            .and_then(string_value)
            .unwrap_or_default();

        out.push(obj(&[
            ("pipeline_id", s(&v.get_str("pipeline_id").unwrap_or_default())),
            ("timestamp", s(&v.get_str("timestamp").unwrap_or_default())),
            ("execution_stage", s(&v.get_str("execution_stage").unwrap_or_default())),
            ("headline", s(&topic)),
            ("script_title", s(&script)),
            ("video_id", s(&video_id)),
            ("niche", s(&niche)),
            ("audience", s(&audience)),
            ("topsis", s(&topsis)),
            ("runtime", s(&runtime)),
            ("shots", s(&shots)),
            ("revenue", s(&revenue)),
            ("rev_eligible", s(&rev_eligible)),
            ("projected_views", s(&projected_views)),
            ("region", s(&region)),
            ("fact_count", s(&fact_count)),
            ("pinned_comment", s(&pinned_comment)),
            ("playlist_url", s(&playlist_url)),
        ]));
    }
    arr(out)
}

fn url_decode(s: &str) -> String {
    let mut res = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            let mut hex = String::new();
            if let Some(h1) = chars.next() { hex.push(h1); }
            if let Some(h2) = chars.next() { hex.push(h2); }
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                res.push(byte as char);
            }
        } else if c == '+' {
            res.push(' ');
        } else {
            res.push(c);
        }
    }
    res
}

fn get_query_param(raw_path: &str, key: &str) -> Option<String> {
    let parts: Vec<&str> = raw_path.split('?').collect();
    if parts.len() < 2 {
        return None;
    }
    for pair in parts[1].split('&') {
        let kv: Vec<&str> = pair.split('=').collect();
        if kv.len() == 2 && kv[0] == key {
            return Some(url_decode(kv[1]));
        }
    }
    None
}

fn api_pipeline_control(root: &str, path: &str, raw_path: &str) -> Value {
    let is_pi = std::path::Path::new("/home/jeevanjoshi").exists();
    let root_dir = if is_pi { "/home/jeevanjoshi/buzzdropfeedv2" } else { root };

    if is_pipeline_running(root_dir) {
        return obj(&[
            ("success", Value::Bool(false)),
            ("command", s("")),
            ("error", s("A pipeline run is already in progress. Duplicate runs are not allowed.")),
        ]);
    }

    if is_budget_exceeded(root_dir) {
        let current_cost = get_current_month_cost(root_dir);
        let cap = env::var("CSVG_BUDGET_CAP_USD")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(50.0);
        return obj(&[
            ("success", Value::Bool(false)),
            ("command", s("")),
            ("error", s(&format!("Monthly budget limit of ${:.2} has been exceeded. Current usage: ${:.2}. Operations are locked.", cap, current_cost))),
        ]);
    }

    let mut args = Vec::new();
    
    // Auto-skip pre-flight interactive prompt when running headless from dashboard
    args.push("--skip-health-check".to_string());
    
    if let Some(reg) = get_query_param(raw_path, "region") {
        if reg == "global" {
            args.push("--global".to_string());
        } else if reg == "india" {
            args.push("--india".to_string());
        }
    }
    if let Some(rag) = get_query_param(raw_path, "rag") {
        args.push("--rag".to_string());
        args.push(rag);
    }
    if let Some(renderer) = get_query_param(raw_path, "renderer") {
        args.push("--renderer".to_string());
        args.push(renderer);
    }
    
    let is_resume = path.ends_with("/resume");
    if is_resume {
        args.push("--resume".to_string());
        if let Some(pid) = get_query_param(raw_path, "pipeline_id") {
            if !pid.is_empty() {
                args.push(pid);
            }
        } else {
            // Find last failed run dynamically
            if let Some((_, pid)) = last_failed_state(root_dir) {
                args.push(pid);
            } else {
                return obj(&[
                    ("success", Value::Bool(false)),
                    ("command", s("")),
                    ("error", s("No failed or incomplete runs found to resume.")),
                ]);
            }
        }
    }

    let flag_str = args.join(" ");
    let cmd_str = format!("cd {} && ./run_production.sh {}", root_dir, flag_str);
    
    let spawned = if is_pi {
        std::process::Command::new("ssh")
            .args(&[
                "-o", "StrictHostKeyChecking=no",
                "oci-prod",
                &cmd_str
            ])
            .spawn()
    } else {
        std::process::Command::new("bash")
            .args(&[
                "-c",
                &cmd_str
            ])
            .spawn()
    };

    let (success, err_msg) = match spawned {
        Ok(_) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };

    obj(&[
        ("success", Value::Bool(success)),
        ("command", s(&cmd_str)),
        ("error", s(&err_msg)),
    ])
}

fn api_seeding_trigger(root: &str, raw_path: &str) -> Value {
    if is_seeding_running() {
        return obj(&[
            ("success", Value::Bool(false)),
            ("command", s("")),
            ("error", s("A seeding campaign is already in progress. Duplicate runs are not allowed.")),
        ]);
    }

    let is_pi = std::path::Path::new("/home/jeevanjoshi").exists();
    let root_dir = if is_pi { "/home/jeevanjoshi/buzzdropfeedv2" } else { root };

    if is_budget_exceeded(root_dir) {
        let current_cost = get_current_month_cost(root_dir);
        let cap = env::var("CSVG_BUDGET_CAP_USD")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(50.0);
        return obj(&[
            ("success", Value::Bool(false)),
            ("command", s("")),
            ("error", s(&format!("Monthly budget limit of ${:.2} has been exceeded. Current usage: ${:.2}. Seeding operations are locked.", cap, current_cost))),
        ]);
    }

    let mut args = Vec::new();
    if let Some(vid) = get_query_param(raw_path, "video") {
        if !vid.is_empty() {
            args.push("--video".to_string());
            args.push(vid);
        }
    }
    if let Some(max_p) = get_query_param(raw_path, "max") {
        if !max_p.is_empty() {
            args.push("--max".to_string());
            args.push(max_p);
        }
    }

    // Reddit seeder runs locally on the Pi (residential IP); OCI cloud IP is blocked by Reddit.
    let python_bin = format!("{}/venv/bin/python", root_dir);
    let seeder_script = format!("{}/reddit_link_seeder.py", root_dir);
    let log_path = format!("{}/logs/reddit_link_seeder_cron.log", root_dir);

    let mut full_args = vec![seeder_script];
    full_args.extend(args.clone());

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path);

    let mut cmd = std::process::Command::new(&python_bin);
    cmd.args(&full_args).current_dir(root_dir);

    if let Ok(file) = log_file {
        if let Ok(dup_file) = file.try_clone() {
            cmd.stdout(std::process::Stdio::from(file));
            cmd.stderr(std::process::Stdio::from(dup_file));
        }
    }
    let spawned = cmd.spawn();

    let cmd_str = format!("{}/venv/bin/python {}/reddit_link_seeder.py {} >> {}/logs/reddit_link_seeder_cron.log 2>&1 &", root_dir, root_dir, args.join(" "), root_dir);

    let (success, err_msg) = match spawned {
        Ok(_) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };

    obj(&[
        ("success", Value::Bool(success)),
        ("command", s(&cmd_str)),
        ("error", s(&err_msg)),
    ])
}

fn api_pipeline_stop(root: &str) -> Value {
    let is_pi = std::path::Path::new("/home/jeevanjoshi").exists();
    
    // Command to kill main.py / run_production.sh / cron / ffmpeg child processes
    let kill_cmd = "pkill -9 -f run_production.sh 2>/dev/null; pkill -9 -f main.py 2>/dev/null; pkill -9 -f cron_publish.sh 2>/dev/null; pkill -9 -f healthcheck.py 2>/dev/null; pkill -9 -f ffmpeg 2>/dev/null; true";
    let hb_cmd = "echo '{\"running\":false,\"ts\":\"\"}' > logs/pipeline_heartbeat.json 2>/dev/null; true";

    if is_pi {
        // Kill on OCI cloud host via SSH host oci-prod AND update its heartbeat
        let _ = std::process::Command::new("ssh")
            .args(&["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5", "oci-prod",
                &format!("{}; {}", kill_cmd, hb_cmd)])
            .output();
    } else {
        // Kill locally and update heartbeat
        let _ = std::process::Command::new("bash")
            .args(&["-c", &format!("{}; {}", kill_cmd, hb_cmd)])
            .output();
    }

    // Also update local pipeline_heartbeat.json on the Pi to immediately reflect stopped
    let hb_path = Path::new(root).join("logs/pipeline_heartbeat.json");
    if hb_path.exists() {
        let _ = fs::write(&hb_path, b"{\"running\":false,\"ts\":\"\"}\n");
    }

    obj(&[
        ("success", Value::Bool(true)),
        ("error", s("")),
    ])
}

fn api_seeding_stop(_root: &str) -> Value {
    // Seeder runs on the Pi; kill all reddit seeding processes, warmups, and lockfiles
    let kill_cmd = "pkill -9 -f reddit_link_seeder.py; pkill -9 -f reddit_warmup.py; pkill -9 -f post_reddit_links.py; pkill -9 -f active_thread_seeder.py; rm -f /tmp/reddit_*.lock /tmp/reddit_link.lock /tmp/reddit_warmup.lock";
    let output = std::process::Command::new("bash")
        .args(&["-c", kill_cmd])
        .output();

    let (success, err_msg) = match output {
        Ok(_) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };

    obj(&[
        ("success", Value::Bool(success)),
        ("error", s(&err_msg)),
    ])
}

fn is_pipeline_running(root: &str) -> bool {
    let ttl = env::var("CSVG_HEARTBEAT_TTL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(60.0);

    if exists(root, "logs/pipeline_heartbeat.json") {
        let hb = read_json(root, "logs/pipeline_heartbeat.json");
        let running = hb.get_str("running").map(|v| v == "true").unwrap_or(false);
        if let Some(ts) = hb.get_str("ts") {
            if let Some(epoch) = parse_iso_utc(&ts) {
                let age = (now_epoch() - epoch).max(0.0);
                return running && age < ttl;
            }
        }
    }
    false
}

fn is_seeding_running() -> bool {
    // Seeder runs locally on the Pi (residential IP, not blocked by Reddit).
    // Specifically match python/python3 interpreter executing the scripts,
    // avoiding false positives from pgrep/bash/tailscale arguments.
    if let Ok(out) = std::process::Command::new("pgrep")
        .args(&["-f", "python.*reddit_link_seeder\\.py"])
        .output()
    {
        if out.status.success() && !out.stdout.is_empty() {
            return true;
        }
    }
    if let Ok(out) = std::process::Command::new("pgrep")
        .args(&["-f", "python.*reddit_warmup\\.py"])
        .output()
    {
        if out.status.success() && !out.stdout.is_empty() {
            return true;
        }
    }
    false
}

fn current_month_str() -> String {
    let epoch = now_epoch() as i64;
    let days = epoch / 86400;
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    let mut y = (yoe as i64) + era * 400;
    let doy = doe - (365*yoe + yoe/4 - yoe/100);
    let mp = (5*doy + 2)/153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    if m <= 2 {
        y += 1;
    }
    format!("{:04}-{:02}", y, m)
}

fn get_current_month_cost(root: &str) -> f64 {
    let current_mo = current_month_str();
    let agg = read_json(root, "logs/run_budget.json");
    let mut total = 0.0;
    if let Value::Obj(m) = &agg {
        for rec in m.values() {
            if let Some(started) = rec.get_str("started_at") {
                if started.starts_with(&current_mo) {
                    let est_usd = rec
                        .get_obj("totals")
                        .and_then(|t| t.get("est_usd"))
                        .and_then(|v| match v {
                            Value::Num(n) => Some(*n),
                            _ => None,
                        })
                        .unwrap_or(0.0);
                    total += est_usd;
                }
            }
        }
    }
    total
}

fn is_budget_exceeded(root: &str) -> bool {
    let limit = env::var("CSVG_BUDGET_CAP_USD")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(50.0);
    get_current_month_cost(root) >= limit
}

fn api_healthcheck_run(root: &str) -> Value {
    let is_pi = std::path::Path::new("/home/jeevanjoshi").exists();
    // Remote commands run on the OCI master (repo is /home/ubuntu/...), while
    // local execution runs against the dashboard host's own root.
    let remote_root = if is_pi { OCI_ROOT_DIR } else { root };
    let python_bin = format!("{}/venv/bin/python", remote_root);
    let health_script = format!("{}/healthcheck.py", remote_root);

    let cmd_str = format!("cd {} && ./venv/bin/python healthcheck.py", remote_root);
    let output = if is_pi {
        std::process::Command::new("ssh")
            .args(&[
                "-o", "StrictHostKeyChecking=no",
                "oci-prod",
                &cmd_str
            ])
            .output()
    } else {
        std::process::Command::new(python_bin)
            .arg(health_script)
            .current_dir(root)
            .output()
    };

    let (success, log_out) = match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let combined = format!("{}{}", stdout, stderr);
            (out.status.success(), combined)
        }
        Err(e) => (false, e.to_string()),
    };

    obj(&[
        ("success", Value::Bool(success)),
        ("output", s(&log_out)),
    ])
}

// ── /api/cleanup/run ─────────────────────────────────────────────────────────
// Runs cleanup.py on the OCI master (via ssh oci-prod when the dashboard lives
// on the Pi). `?force=1` tightens keep-counts (aggressive prune).

const OCI_ROOT_DIR: &str = "/home/ubuntu/buzzdropfeedv2";

fn run_remote_or_local(root: &str, cmd_str: &str) -> (bool, String) {
    let is_pi = std::path::Path::new("/home/jeevanjoshi").exists();
    let output = if is_pi {
        std::process::Command::new("ssh")
            .args(&["-o", "StrictHostKeyChecking=no", "oci-prod", cmd_str])
            .output()
    } else {
        std::process::Command::new("bash")
            .arg("-lc")
            .arg(cmd_str)
            .current_dir(root)
            .output()
    };
    match output {
        Ok(out) => {
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            (out.status.success(), combined)
        }
        Err(e) => (false, e.to_string()),
    }
}

fn api_cleanup_run(root: &str, raw_path: &str) -> Value {
    let is_pi = std::path::Path::new("/home/jeevanjoshi").exists();
    let target_root = if is_pi { OCI_ROOT_DIR } else { root };
    let mut cmd = format!("cd {} && ./venv/bin/python cleanup.py", target_root);
    if raw_path.contains("force") {
        cmd.push_str(" --force");
    }
    let (success, log_out) = run_remote_or_local(root, &cmd);
    obj(&[
        ("success", Value::Bool(success)),
        ("output", s(&log_out)),
    ])
}

// ── /api/usage/refresh ───────────────────────────────────────────────────────
// Re-runs get_api_usage.py on the OCI master (keys live there) to re-pull
// authoritative fal/OpenRouter/Google usage, then rsyncs the refreshed
// logs/provider_usage.json back to the Pi so the dashboard reflects it.

fn api_usage_refresh(root: &str) -> Value {
    let is_pi = std::path::Path::new("/home/jeevanjoshi").exists();
    let target_root = if is_pi { OCI_ROOT_DIR } else { root };
    let cmd = format!("cd {} && ./venv/bin/python get_api_usage.py", target_root);
    let (success, log_out) = run_remote_or_local(root, &cmd);
    if is_pi && success {
        let _ = std::process::Command::new("rsync")
            .args(&[
                "-az",
                &format!("oci-prod:{}/logs/provider_usage.json", OCI_ROOT_DIR),
                "/home/jeevanjoshi/buzzdropfeedv2/logs/",
            ])
            .output();
    }
    obj(&[
        ("success", Value::Bool(success)),
        ("output", s(&log_out)),
    ])
}