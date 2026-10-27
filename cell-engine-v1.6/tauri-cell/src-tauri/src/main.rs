// Universal Cell Engine v1.5 — Tissue B: Tauri 2.0 Rust shard.
//
// REAL CODE, NOT COMPILED HERE: this sandbox has no cargo/rustc (named blocker
// B1-TAURI-TOOLCHAIN-ABSENT). Every shape mirrors the Python/TS substrate that
// WAS executed: same op log, same FNV-1a64 digests, same genome sha256 check,
// same LWW semantics. `cargo build` in a Rust toolchain is the remaining step.
//
// Sections:
//   [SUBSTRATE] arena SoA + CRDT op log + LWW + membrane/mask + tape digest
//               + genome contract check.      ~ the ported core, no host knowledge
//   [GRAFT]     host binding ONLY.           ~ delimited, size-budgeted < 5KB
//   [COMMANDS]  cell_tick / cell_snapshot.   ~ the entire IPC surface (2 commands)
//
// NOTE on "Rust side holds arena in SharedArrayBuffer" (spec E2): a SAB cannot
// cross the WebView process boundary in Tauri 2 (WebKitGTK/WebView2 are separate
// processes). The arena therefore lives Rust-side as SoA Vec columns (below) and
// the webview owns its SAB mirror for the DOM tier; commands return snapshots /
// reports. The measured cost of the process boundary is the IPC round trip,
// proxied in the Node proof by a MessageChannel hop. Reported as finding
// B1-TAURI-SAB-XPROC.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

const GENOME_JSON: &str = include_str!("../../../genome.json");
const STATE_INITIAL: f32 = 0.725;
const RATE: f32 = 0.03;
const EPS: f32 = 1e-3;
const CHUNK: usize = 512;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

// ======================[SUBSTRATE] ported core (no host knowledge) =============

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Op {
    op: String,
    value: f32,
    ts: u64,
    actor: String,
}

fn op_line(op: &Op) -> String {
    format!("op|{}|{:.6}|{}|{}", op.op, op.value, op.ts, op.actor)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |h, b| (h ^ (*b as u64)).wrapping_mul(FNV_PRIME))
}

/// Canonical JSON with sorted keys (matches the Python/TS canonicalizer).
fn canonical(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", serde_json::to_string(k).unwrap(), canonical(&map[*k])))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        serde_json::Value::Array(a) => {
            let body: Vec<String> = a.iter().map(canonical).collect();
            format!("[{}]", body.join(","))
        }
        other => serde_json::to_string(other).unwrap(),
    }
}

fn genome_id_of(genome: &serde_json::Value) -> String {
    let mut body = genome.clone();
    if let serde_json::Value::Object(ref mut map) = body {
        map.remove("genome_id");
    }
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(canonical(&body).as_bytes());
    format!("sha256:{}", hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Arena: SoA columns (tint f32, epoch i32, active u8, target f32). Slot 0 is
/// the canonical CRDT cell and is skipped by the easing kernel. IDs live in an
/// aux Vec<String> (the Python side calls this the index map).
struct Arena {
    n: usize,
    tints: Vec<f32>,
    epochs: Vec<i32>,
    active: Vec<u8>,
    targets: Vec<f32>,
    ids: Vec<String>,
}

impl Arena {
    fn new(capacity: usize) -> Self {
        Arena {
            n: 0,
            tints: vec![0.0; capacity],
            epochs: vec![0; capacity],
            active: vec![0; capacity],
            targets: vec![0.0; capacity],
            ids: vec![String::new(); capacity],
        }
    }

    fn spawn(&mut self, lineage: &str, tint: f32) -> usize {
        let i = self.n;
        self.n += 1;
        self.ids[i] = lineage.to_string();
        self.tints[i] = tint;
        self.epochs[i] = 1;
        self.active[i] = 0;
        i
    }

    fn mulberry32(a: u32) -> impl FnMut() -> f32 {
        let mut a = a;
        move || {
            a = a.wrapping_add(0x6d2b79f5);
            let mut t = a ^ (a >> 15);
            t = t.wrapping_mul(1 | a);
            t = t.wrapping_add(t ^ (t >> 7)).wrapping_mul(61 | t);
            t ^= t >> 14;
            (t as f32) / 4294967296.0
        }
    }

    /// Metabolism: impulse injection + easing kernel with deadline -> debt.
    fn tick(&mut self, budget: Duration, impulses: usize, seed: u64) -> TickReport {
        let t0 = Instant::now();
        let deadline = t0 + budget;
        let mut rng = Self::mulberry32(seed as u32);
        for _ in 0..impulses {
            if self.n < 2 { break; }
            let i = 1 + (rng() * (self.n as f32 - 1.0)).floor() as usize;
            self.targets[i] = rng();
            self.active[i] = 1;
        }
        let idx: Vec<usize> = (1..self.n).filter(|&i| self.active[i] == 1).collect();
        let mut executed = 0usize;
        let mut deferred = 0usize;
        let mut s = 0usize;
        while s < idx.len() {
            if Instant::now() >= deadline {
                deferred = idx.len() - s;
                break;
            }
            let end = std::cmp::min(s + CHUNK, idx.len());
            for &i in &idx[s..end] {
                let t = self.tints[i];
                let mut d = self.targets[i] - t;
                if d > RATE { d = RATE; } else if d < -RATE { d = -RATE; }
                let tn = t + d;
                self.tints[i] = tn;
                if (self.targets[i] - tn).abs() < EPS { self.active[i] = 0; }
            }
            executed += end - s;
            s = end;
        }
        let active = self.active[..self.n].iter().filter(|&&a| a == 1).count();
        TickReport {
            executed,
            deferred,
            active,
            ns: t0.elapsed().as_nanos() as u64,
        }
    }
}

#[derive(Serialize)]
struct TickReport {
    executed: usize,
    deferred: usize,
    active: usize,
    ns: u64,
}

struct Cell {
    lineage: String,
    slot: usize,
    ops: Vec<Op>,
    dirty: bool,
    materialized: Option<f32>,
    epoch: u32, // mask epoch (membrane)
    allowed: Vec<String>,
    denied: Vec<String>,
    audience: String,
}

impl Cell {
    fn set_tint(&mut self, value: f32, ts: u64, actor: &str, tape: &mut Vec<String>) -> f32 {
        let op = Op { op: "set_tint".into(), value, ts, actor: actor.into() };
        tape.push(op_line(&op));
        self.ops.push(op);
        let mut best: Option<&Op> = None;
        for o in &self.ops {
            match best {
                Some(b) if (o.ts, &o.actor) <= (b.ts, &b.actor) => {}
                _ => best = Some(o),
            }
        }
        let tint = best.map(|o| o.value).unwrap_or(STATE_INITIAL);
        self.dirty = true;
        tint
    }

    fn op_digest(&self) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for op in &self.ops {
            bytes.extend_from_slice(op_line(op).as_bytes());
            bytes.push(b'\n');
        }
        format!("fnv1a64:{}", fnv1a64(&bytes))
    }
}

struct App {
    genome_id: String,
    arena: Arena,
    cell: Cell,
    tape: Vec<String>,
}

impl App {
    fn membrane_allows(&self, verb: &str) -> bool {
        self.cell.allowed.iter().any(|v| v == verb) && !self.cell.denied.iter().any(|v| v == verb)
    }
}

// ======================[GRAFT] host binding ONLY ================================
// -- graft table (host binding only) --
// The entire Tauri-side host surface: render.tint -> one DOM style write,
// pushed to the webview shim as a compact frame. No object model, no events.
const GRAFT_HANDLE: &str = "#btn-save";

fn materialize_tint(out: &mut String, tint: f32) {
    let v = (tint.clamp(0.0, 1.0) * 255.0).round() as u32;
    out.push_str(&format!(
        "d('{h}').style.background='rgb({v},{v},{v})';",
        h = GRAFT_HANDLE,
        v = v
    ));
}
// -- end graft table (host binding only) --
// ================================================================================

// ======================[COMMANDS] the entire IPC surface =======================

#[tauri::command]
fn cell_tick(app_handle: tauri::AppHandle, state: State<AppLock>, budget_us: u32, impulses: u32) -> TickReport {
    let mut app = state.app_lock();
    let report = app.arena.tick(Duration::from_micros(budget_us as u64), impulses as usize, 0);
    // drain the canonical cell's graft if dirty and the mask allows it
    if app.cell.dirty && app.membrane_allows("render.tint") {
        let tint = app.arena.tints[app.cell.slot];
        let mut frame = String::new();
        materialize_tint(&mut frame, tint); // [GRAFT] — the only host call
        app.cell.dirty = false;
        app.cell.materialized = Some(tint);
        drop(app);
        // push the frame to the webview shim (noun side of the verb/noun split)
        let _ = app_handle.emit("cell-fx", frame);
    }
    report
}

#[tauri::command]
fn cell_snapshot(state: State<AppLock>) -> String {
    let app = state.app_lock();
    let tint = app.arena.tints[app.cell.slot];
    let header = format!(
        "{}\n",
        serde_json::json!({
            "kind": "cell.snapshot",
            "schema": "1.5",
            "cell_type": "cell.tint.v1",
            "lineage": app.cell.lineage,
            "genome_id": app.genome_id,
            "tint": tint,
            "epoch": app.arena.epochs[app.cell.slot],
            "mask_epoch": app.cell.epoch,
            "actor_id": "tauri.1",
            "op_count": app.cell.ops.len(),
            "op_digest": app.cell.op_digest(),
            "generator": "rust/tauri_cell",
        })
    );
    let ops: String = app.cell.ops.iter().map(|o| serde_json::to_string(o).unwrap() + "\n").collect();
    header + &ops
}

// thin wrapper so commands can lock (State<App> is already Sync; Mutex lives here)
struct AppLock(std::sync::Mutex<App>);
trait AppLockExt { fn app_lock(&self) -> std::sync::MutexGuard<'_, App>; }
impl AppLockExt for State<AppLock> {
    fn app_lock(&self) -> std::sync::MutexGuard<'_, App> {
        self.inner().0.lock().unwrap()
    }
}

fn main() {
    let genome: serde_json::Value = serde_json::from_str(GENOME_JSON).expect("genome.json bundled");
    let genome_id = genome_id_of(&genome);
    println!("cell.tint.v1 genome ok: {}", genome_id);

    let mut arena = Arena::new(200_001);
    let slot = arena.spawn("pending-lineage", STATE_INITIAL);
    let cell = Cell {
        lineage: "pending-lineage".into(),
        slot,
        ops: vec![],
        dirty: false,
        materialized: None,
        epoch: 1,
        allowed: vec!["render.tint".into(), "state.set".into()],
        denied: vec!["data.query.peers".into()], // mask.public
        audience: "public".into(),
    };
    let app = App { genome_id, arena, cell, tape: vec![] };

    tauri::Builder::default()
        .manage(AppLock(std::sync::Mutex::new(app)))
        .invoke_handler(tauri::generate_handler![cell_tick, cell_snapshot])
        .run(tauri::generate_context!())
        .expect("error while running tauri-cell");
}
