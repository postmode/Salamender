// v1.6 Phase B harness — Tauri tissue experiments (Node proof tier).
//
// REAL here : SharedArrayBuffer arena + worker_threads scheduler (same SAB/
//             postMessage/Atomics semantics as a Web Worker in the webview),
//             CRDT replay of the Python-exported snapshot (E3, cross-process,
//             cross-language), membrane flip, determinism digests, MessageChannel
//             round-trip as the IPC proxy cost.
// STANDIN-DOM: the graft writes to a plain object (no WebKit in this sandbox).
// NOT COMPILED HERE: the Rust shard (src-tauri/src/main.rs) — no cargo toolchain.
//
// Usage: node --experimental-strip-types test_tauri.ts
import { MessageChannel } from 'node:worker_threads';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import {
  ROOT, SNAPSHOT_PATH, SNAPSHOT_RETURN, LINEAGE, STATE_INITIAL,
  loadGenome, loadMask, canonical, genomeIdOf, fnv1a64, f32round,
  buildArena, makeElement, GraftTable, Tissue, WorkerScheduler,
} from './src/main.ts';

const OPS: Array<[number, number]> = [[0.75, 1], [0.788, 2], [0.812, 3], [0.85, 4], [0.884, 5]];
const N_PERF = 200_000;
const PY = JSON.parse(readFileSync(ROOT + 'perf/_python_digests.json', 'utf8'));
import { existsSync } from 'node:fs';
const RUST_BIN = ROOT + 'tauri-cell/src-tauri/target/release/tauri-cell';
const RUST_SHARD_LABEL = existsSync(RUST_BIN)
  ? 'COMPILED (cargo build --release, ELF at src-tauri/target/release/tauri-cell)'
  : 'NOT COMPILED (no cargo in sandbox) — real code at src-tauri/src/main.rs';

function pct(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.round((p / 100) * (sorted.length - 1)));
  return sorted[i];
}
function freshTissue(genome: Record<string, unknown>, name = 'tauri'): Tissue {
  const arena = buildArena(8);
  const graft = new GraftTable(makeElement('btn-save'));
  return new Tissue(genome, loadMask(ROOT + 'masks/public.json'), arena, graft, name);
}

// --------------------------------------------- E6+E5: the scripted scenario (mirror of Python)
function runScenario(): Record<string, unknown> {
  const genome = loadGenome();
  const tissue = freshTissue(genome);
  const sabRef = tissue.arena.sab;
  const slot = tissue.ensureCell(LINEAGE, STATE_INITIAL, 'blender.1');
  for (const [v, ts] of OPS) tissue.cellSetTint(v, ts, 'blender.1');
  tissue.cellTick(1);
  const drop = tissue.peerQuery(2);
  const flip = tissue.flipMask(loadMask(ROOT + 'masks/trusted.json'));
  const allow = tissue.peerQuery(3);
  tissue.cellTick(4);
  return {
    tapeDigest: tissue.tape.digest(),
    opDigest: tissue.oplog.digest(),
    slot,
    tint: tissue.arena.tint[tissue.slot],
    ops: tissue.oplog.ops.length,
    dropPublic: !drop,
    allowTrusted: allow,
    flip,
    sabIdentityPreserved: sabRef === tissue.arena.sab,
    tapeLines: tissue.tape.lines.slice(),
    telemetry: tissue.tape.telemetry.slice(),
  };
}

// --------------------------------------------- E3: cross-host regeneration (the falsifier)
function e3Regeneration(genome: Record<string, unknown>): Record<string, unknown> {
  if (!existsSync(SNAPSHOT_PATH)) throw new Error('python snapshot missing — run Phase A first');
  const [header, ops] = Tissue.readSnapshot(SNAPSHOT_PATH);
  const genomeMatches = header['genome_id'] === genome['genome_id'];
  if (!genomeMatches) throw new Error('genome contract mismatch across tissues');
  const tissue = freshTissue(genome, 'tauri-regen');
  const checks = tissue.regenerateFrom(SNAPSHOT_PATH);
  const asserts = {
    lineage_identical: checks['lineage'] === LINEAGE,
    tint_resumed_at_0_884: Math.abs(Number(checks['tint']) - 0.884) < 1e-6,
    tint_not_reset_to_0_725: Math.abs(Number(checks['tint']) - 0.725) > 1e-6,
    op_log_length_identical: checks['ops'] === 5 && ops.length === 5,
    causal_digest_identical: checks['digest'] === header['op_digest'] && checks['digest'] === PY.e3_export_digest,
    genome_identical: genomeMatches,
    header_ops: header['op_count'],
    header_digest: header['op_digest'],
  };
  // continuation: the cell keeps LIVING in tauri — one more op, then materialize
  const tint6 = tissue.cellSetTint(0.91, 6, 'tauri.1');
  tissue.cellTick(5);
  const headerOut = tissue.exportSnapshot(SNAPSHOT_RETURN, 'node/tauri_cell');
  return {
    asserts,
    all_passed: Object.values(asserts).every(Boolean),
    continued_tint: tint6,
    materialized_background: tissue.graft.element.style.background,
    return_header: headerOut,
  };
}

// --------------------------------------------- E4: frame budget @ 200k (SAB + worker)
async function e4Perf(): Promise<Record<string, unknown>> {
  const n = N_PERF + 1;
  const t0 = performance.now();
  const arena = buildArena(n);
  // bulk-spawn simulated perf cells: slot 0 reserved for the canonical cell.
  // Start from a steady-state-ish subset active (not all 200k) — the JS scalar
  // kernel is ~3.6x slower than numpy's vectorized one; a full initial wave
  // would take hundreds of ticks to drain and saturate the budget.
  arena.tint[0] = f32round(STATE_INITIAL);
  arena.index.set(LINEAGE, 0);
  for (let i = 1; i < n; i++) {
    arena.target[i] = Math.random();
    arena.active[i] = i < 12_000 ? 1 : 0;
  }
  const buildMs = performance.now() - t0;

  const sched = new WorkerScheduler(arena.sab, n);
  await sched.ready;
  const ticks = 240, budgetUs = 2000, impulses = 800;
  for (let k = 0; k < 120; k++) await sched.tick(budgetUs, impulses, 10_000 + k); // warm-up to steady state
  const ns: number[] = [];
  let active = 0, deferred = 0;
  for (let k = 0; k < ticks; k++) {
    const r = await sched.tick(budgetUs, impulses, k);
    ns.push(r.ns);
    active += r.active;
    deferred += r.deferred;
  }
  const s = ns.slice().sort((a, b) => a - b);
  // resting floor: everything asleep, no impulses
  arena.active.fill(0);
  const rest: number[] = [];
  for (let j = 0; j < 60; j++) rest.push((await sched.tick(2000, 0, 0)).ns);
  // worst case: all 200k active, generous budget -> true full cost
  for (let i = 1; i < n; i++) {
    arena.target[i] = Math.random();
    arena.active[i] = 1;
  }
  const worst = await sched.tick(500_000, 0, 999);
  // graft cost (STANDIN-DOM), measured separately
  const graft = new GraftTable(makeElement('btn-save'));
  const g0 = performance.now();
  const G = 20000;
  for (let i = 0; i < G; i++) graft.apply('render.tint', 0.5 + (i % 32) / 1024);
  const graftUsPerCall = ((performance.now() - g0) * 1000) / G;
  sched.terminate();
  return {
    cells: N_PERF,
    build_ms: buildMs,
    ticks, budget_us: budgetUs, impulses_per_tick: impulses,
    tick_p50_us: pct(s, 50) / 1000,
    tick_p99_us: pct(s, 99) / 1000,
    active_avg: Math.round(active / ticks),
    activity_rate_pct: Math.round((100 * active) / (ticks * N_PERF) * 100) / 100,
    deferred_total: deferred,
    resting_floor_us: pct(rest.slice().sort((a, b) => a - b), 50) / 1000,
    worst_case_all_active: { executed: worst.executed, deferred: worst.deferred, ns: worst.ns },
    worst_case_all_active_us: worst.ns / 1000,
    graft_us_per_call_standin_dom: graftUsPerCall,
    graft_projection_14k_us: graftUsPerCall * 14000,
  };
}

// --------------------------------------------- IPC proxy: MessageChannel round trip
function ipcProxyRtt(): Promise<Record<string, unknown>> {
  // ping on port1 -> echo on port2 -> pong back on port1. The structured-clone
  // hop is the honest proxy for the Tauri invoke boundary (labeled PROXY).
  const { port1, port2 } = new MessageChannel();
  const rtts: number[] = [];
  const N = 1000;
  return new Promise((resolve) => {
    let i = 0;
    let t0 = performance.now();
    port2.onmessage = (e) => port2.postMessage(e.data); // the "Rust side" echo
    port1.onmessage = () => {
      rtts.push(performance.now() - t0);
      i += 1;
      if (i < N) {
        t0 = performance.now();
        port1.postMessage({ cmd: 'cell_tick', budget_us: 2000, impulses: 800 });
      } else {
        port1.close();
        port2.close();
        rtts.sort((a, b) => a - b);
        resolve({ rtt_p50_us: pct(rtts, 50) * 1000, rtt_p99_us: pct(rtts, 99) * 1000, samples: N });
      }
    };
    port1.postMessage({ cmd: 'cell_tick', budget_us: 2000, impulses: 800 });
  });
}

// --------------------------------------------- main
async function main() {
  const genome = loadGenome();
  const genomeNode = genomeIdOf(genome);
  const genomeMatch = genomeNode === (genome as Record<string, unknown>)['genome_id'] && genomeNode === PY.genome_sha;

  const r1 = runScenario();
  const r2 = runScenario();
  const nodeDeterministic = r1['tapeDigest'] === r2['tapeDigest'];
  const crossRuntime = r1['tapeDigest'] === PY.scenario_digest_run1;

  const e3 = e3Regeneration(genome);
  const e4 = await e4Perf();
  const ipc = await ipcProxyRtt();

  const perf = {
    phase: 'B (node)',
    environment: {
      node: process.version,
      runtime: 'worker_threads + SharedArrayBuffer (webview-equivalent semantics)',
      labels: {
        arena_scheduler_crdt_membrane: 'REAL (SAB + worker, this process tree)',
        dom_materialize: 'STANDIN-DOM',
        rust_shard: RUST_SHARD_LABEL,
        tauri_ipc: 'PROXY (MessageChannel round trip; real invoke untested)',
      },
    },
    e2_tauri_tissue: {
      commands_declared: ['cell_tick', 'cell_snapshot'],
      ipc_proxy: ipc,
      genome_check: { node: genomeNode, python: PY.genome_sha, match: genomeMatch },
    },
    e3_regeneration: e3,
    e4_frame_budget_200k: e4,
    e5_membrane: {
      public_drop: r1['dropPublic'],
      trusted_allow: r1['allowTrusted'],
      flip: r1['flip'],
      sab_identity_preserved: r1['sabIdentityPreserved'],
      tape_digest: r1['tapeDigest'],
    },
    e6_determinism: {
      run1_tape_digest: r1['tapeDigest'],
      run2_tape_digest: r2['tapeDigest'],
      deterministic: nodeDeterministic,
      cross_runtime_with_python: crossRuntime,
    },
    tape_lines: r1['tapeLines'],
  };
  writeFileSync(ROOT + 'perf/tauri_perf.json', JSON.stringify(perf, null, 2), 'utf8');

  const D = `Universal Cell Engine v1.6 — determinism digest evidence
genome contract
  sha256 (python)    : ${PY.genome_sha}
  sha256 (node)      : ${genomeNode}
  match              : ${genomeMatch ? '[OK]' : '[FAIL]'}

scenario causal tape (cell.tint.v1, 9 canonical events — see tape lines below)
  python run #1 / #2 : ${PY.scenario_digest_run1} / ${PY.scenario_digest_run2}
  python deterministic: ${PY.scenario_deterministic ? '[OK]' : '[FAIL]'}
  node   run #1 / #2 : ${r1['tapeDigest']} / ${r2['tapeDigest']}
  node   deterministic: ${nodeDeterministic ? '[OK]' : '[FAIL]'}
  python == node     : ${crossRuntime ? '[OK] CROSS-RUNTIME DETERMINISTIC' : '[FAIL]'}

E3 salamander (python process -> node process, cross-language CRDT replay)
  exported op digest (python) : ${PY.e3_export_digest}
  recomputed in node          : ${e3['asserts'] !== undefined ? (e3 as Record<string, Record<string, unknown>>)['asserts']['causal_digest_identical'] : false}
  all E3 asserts passed       : ${(e3 as Record<string, unknown>)['all_passed'] ? '[OK]' : '[FAIL]'}
  return leg (node -> python) : asserted in perf/regeneration_roundtrip.json

tape lines (identical in python and node):
${(r1['tapeLines'] as string[]).map((l) => '  ' + l).join('\n')}
`;
  writeFileSync(ROOT + 'perf/determinism_digest.txt', D, 'utf8');

  console.log(D);
  console.log('E3 asserts:', JSON.stringify(e3, null, 2).slice(0, 1500));
}

await main();
