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
    if method != "GET" {
        respond(stream, 405, "text/plain; charset=utf-8", b"method not allowed".as_slice());
        return;
    }
    let path = raw_path.split('?').next().unwrap_or("/");

    let (code, ctype, body) = route(root, path);
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

fn route(root: &str, path: &str) -> (u16, &'static str, Vec<u8>) {
    match path {
        "/" | "/index.html" => (200, "text/html; charset=utf-8", HTML.as_bytes().to_vec()),
        "/favicon.ico" => (204, "text/plain", Vec::new()),
        "/health" => json_response(&obj(&[("ok", Value::Bool(true))])),
        "/api/status" => json_response(&api_status(root)),
        "/api/logs" => json_response(&api_logs(root)),
        "/api/published" => json_response(&api_published(root)),
        "/api/budget" => json_response(&api_budget(root)),
        "/api/runs" => json_response(&api_runs(root)),
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
        let region = v.get_str("region").unwrap_or_default();
        let fact_count = json::get_path(&v, &["verified_facts"])
            .map(|val| match val {
                Value::Arr(a) => a.len().to_string(),
                _ => "0".to_string(),
            })
            .unwrap_or_else(|| "0".to_string());

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
            ("region", s(&region)),
            ("fact_count", s(&fact_count)),
        ]));
    }
    arr(out)
}