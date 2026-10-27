// Universal Cell Engine v1.6 — Tissue B: Tauri webview shim (framework-free).
//
// PRODUCTION (Tauri 2.0): the scheduler + CRDT live in Rust (src-tauri/src/main.rs);
// this shim calls invoke('cell_tick') / invoke('cell_snapshot') over IPC and drains
// render.tint effects to the DOM in one rAF pass.
//
// NODE PROOF (this sandbox: no cargo, no WebKit): the SAME shim runs unchanged with
// scheduler.worker.mjs as the compute thread over a REAL SharedArrayBuffer — the
// same SAB + postMessage + typed-view semantics a Web Worker uses in the webview.
// The arena SAB is owned by the webview side; Rust mirrors state via commands.
//
// STANDIN-DOM: `el.style.background` writes to a plain object here, not a real
// WebKit DOM node. Numbers touching it are labeled STANDIN-DOM in perf output.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';

export const ROOT = fileURLToPath(new URL('../../', import.meta.url));
export const GENOME_PATH = ROOT + 'genome.json';
export const MASK_PUBLIC = ROOT + 'masks/public.json';
export const MASK_TRUSTED = ROOT + 'masks/trusted.json';
export const SNAPSHOT_PATH = ROOT + 'regeneration/crdt_snapshot.jsonl';
export const SNAPSHOT_RETURN = ROOT + 'regeneration/crdt_snapshot_return.jsonl';

export const STATE_INITIAL = 0.725;
export const LINEAGE = '01a17d1d-e392-7d0b-9a87-07348c36a325';
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const U64 = (1n << 64n) - 1n;

// ---------------------------------------------------------------- contract utils
export function canonical(o: unknown): string {
  if (o === null || typeof o !== 'object') return JSON.stringify(o);
  if (Array.isArray(o)) return '[' + o.map(canonical).join(',') + ']';
  const obj = o as Record<string, unknown>;
  return '{' + Object.keys(obj).sort().map((k) => JSON.stringify(k) + ':' + canonical(obj[k])).join(',') + '}';
}

export function sha256Hex(s: string): string {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

export function genomeIdOf(genome: Record<string, unknown>): string {
  const rest: Record<string, unknown> = { ...genome };
  delete rest['genome_id'];
  return 'sha256:' + sha256Hex(canonical(rest));
}

export function loadGenome(path: string = GENOME_PATH): Record<string, unknown> {
  const g = JSON.parse(readFileSync(path, 'utf8'));
  if (genomeIdOf(g) !== (g as Record<string, unknown>)['genome_id']) {
    throw new Error('genome contract violated');
  }
  return g;
}

export function fnv1a64(s: string): string {
  const bytes = Buffer.from(s, 'utf8');
  let h = FNV_OFFSET;
  for (const b of bytes) {
    h ^= BigInt(b);
    h = (h * FNV_PRIME) & U64;
  }
  return 'fnv1a64:' + h.toString();
}

export function f32round(x: number): number {
  return Math.fround(x);
}

export function fmt6(x: number): string {
  return f32round(x).toFixed(6);
}

// ---------------------------------------------------------------- mask / membrane / tape
export interface MaskDoc {
  mask_id: string;
  epoch: number;
  audience: string;
  allowed_verbs: string[];
  denied_verbs: string[];
}

export class Mask {
  maskId: string;
  epoch: number;
  audience: string;
  allowed: Set<string>;
  denied: Set<string>;

  constructor(doc: MaskDoc) {
    this.maskId = doc.mask_id;
    this.epoch = doc.epoch;
    this.audience = doc.audience;
    this.allowed = new Set(doc.allowed_verbs);
    this.denied = new Set(doc.denied_verbs);
  }

  allows(verb: string): boolean {
    return this.allowed.has(verb) && !this.denied.has(verb);
  }
}

export function loadMask(path: string): Mask {
  return new Mask(JSON.parse(readFileSync(path, 'utf8')) as MaskDoc);
}

export interface TelemetryEvent {
  event: string;
  wall: number;
  [k: string]: unknown;
}

export class Tape {
  lines: string[] = [];
  telemetry: TelemetryEvent[] = [];

  add(line: string): void {
    this.lines.push(line);
  }

  event(name: string, kv: Record<string, unknown> = {}): void {
    this.telemetry.push({ event: name, ...kv, wall: Date.now() / 1000 });
  }

  digest(): string {
    return fnv1a64(this.lines.join('\n') + '\n');
  }
}

export class Membrane {
  tape: Tape;
  mask: Mask | null = null;

  constructor(tape: Tape) {
    this.tape = tape;
  }

  bake(mask: Mask): void {
    const prev = this.mask ? this.mask.epoch : null;
    this.mask = mask;
    if (prev !== null && prev !== mask.epoch) {
      this.tape.add(`mask.flip|${prev}|${mask.epoch}`);
      this.tape.event('mask.flip', { from: prev, to: mask.epoch });
    }
  }

  attempt(verb: string, tick: number): boolean {
    if (this.mask && this.mask.allows(verb)) {
      this.tape.add(`membrane.allow|${verb}|${this.mask.epoch}`);
      return true;
    }
    const epoch = this.mask ? this.mask.epoch : 0;
    this.tape.add(`membrane.drop|${verb}|${epoch}`);
    this.tape.event('membrane.drop', { verb, epoch, tick });
    return false;
  }
}

// ---------------------------------------------------------------- CRDT state
export interface Op {
  op: string;
  value: number;
  ts: number;
  actor: string;
}

export function opLine(op: Op): string {
  return `op|${op.op}|${op.value.toFixed(6)}|${op.ts}|${op.actor}`;
}

export class OpLog {
  ops: Op[] = [];

  append(value: number, ts: number, actor: string): Op {
    const op: Op = { op: 'set_tint', value: f32round(value), ts, actor };
    this.ops.push(op);
    return op;
  }

  lww(): number | null {
    let bestKey: [number, string] | null = null;
    let bestVal: number | null = null;
    for (const op of this.ops) {
      const key: [number, string] = [op.ts, op.actor];
      if (bestKey === null || key > bestKey) {
        bestKey = key;
        bestVal = op.value;
      }
    }
    return bestVal;
  }

  digest(): string {
    let h = FNV_OFFSET;
    for (const op of this.ops) {
      const s = opLine(op) + '\n';
      for (const b of Buffer.from(s, 'utf8')) {
        h ^= BigInt(b);
        h = (h * FNV_PRIME) & U64;
      }
    }
    return 'fnv1a64:' + h.toString();
  }
}

// ---------------------------------------------------------------- Layer 2: arena (SAB)
// Same column set as the Blender numpy dtype, laid out on one SharedArrayBuffer:
// tint f32[N] | epoch i32[N] | active u8[N] | target f32[N], plus a 16-word header.
// IDs live in a JS Map on the owning thread (U36 is not SAB-friendly), exactly the
// "Float32Array + Int32Array views + JS index map" layout from the spec.
export const HDR_WORDS = 16;

export interface SabArena {
  sab: SharedArrayBuffer;
  n: number;
  hdr: Int32Array;
  tint: Float32Array;
  epoch: Int32Array;
  active: Uint8Array;
  target: Float32Array;
  index: Map<string, number>;
}

export function buildArena(n: number): SabArena {
  const hdrBytes = HDR_WORDS * 4;
  const tintOff = hdrBytes;
  const epochOff = tintOff + n * 4;
  const activeOff = epochOff + n * 4;
  const targetOff = (activeOff + n + 3) & ~3;
  const total = targetOff + n * 4;
  const sab = new SharedArrayBuffer(total);
  return {
    sab,
    n,
    hdr: new Int32Array(sab, 0, HDR_WORDS),
    tint: new Float32Array(sab, tintOff, n),
    epoch: new Int32Array(sab, epochOff, n),
    active: new Uint8Array(sab, activeOff, n),
    target: new Float32Array(sab, targetOff, n),
    index: new Map(),
  };
}

// ---------------------------------------------------------------- Layer 4: graft table
export interface DomElement {
  id: string;
  style: { background: string };
}

export function makeElement(id: string): DomElement {
  return { id, style: { background: 'rgb(185,185,185)' } }; // STANDIN-DOM
}

export class GraftTable {
  tissue = 'tauri';
  table: Record<string, (tint: number) => void>;
  element: DomElement;

  constructor(element: DomElement) {
    this.element = element;
    this.table = {
      // the entire Tauri-side host binding: one style write per drain
      'render.tint': (tint: number) => {
        const v = Math.round(f32round(tint) * 255);
        this.element.style.background = `rgb(${v},${v},${v})`;
      },
    };
  }

  apply(verb: string, tint: number): boolean {
    const fn = this.table[verb];
    if (!fn) return false;
    fn(tint);
    return true;
  }
}

// ---------------------------------------------------------------- scheduler (worker)
export interface TickReport {
  executed: number;
  deferred: number;
  active: number;
  ns: number;
}

export class WorkerScheduler {
  worker: Worker;
  n: number;
  private pending: ((r: TickReport) => void) | null = null;
  ready: Promise<void>;
  private readyResolve!: () => void;

  constructor(sab: SharedArrayBuffer, n: number) {
    this.n = n;
    const workerPath = fileURLToPath(new URL('./scheduler.worker.mjs', import.meta.url));
    this.worker = new Worker(workerPath);
    this.ready = new Promise<void>((res) => {
      this.readyResolve = res;
    });
    this.worker.on('message', (m: { cmd: string } & TickReport) => {
      if (m.cmd === 'ready') this.readyResolve();
      else if (m.cmd === 'report') {
        const p = this.pending;
        this.pending = null;
        if (p) p(m);
      }
    });
    this.worker.postMessage({ cmd: 'init', sab, n });
  }

  tick(budgetUs: number, impulses: number, seed: number): Promise<TickReport> {
    return new Promise<TickReport>((resolve) => {
      this.pending = resolve;
      this.worker.postMessage({ cmd: 'tick', budgetUs, impulses, seed });
    });
  }

  terminate(): void {
    this.worker.terminate();
  }
}

// ---------------------------------------------------------------- tissue (canonical cell)
export class Tissue {
  name: string;
  genome: Record<string, unknown>;
  tape = new Tape();
  membrane: Membrane;
  arena: SabArena;
  scheduler: WorkerScheduler | null;
  graft: GraftTable;
  oplog = new OpLog();
  lineage = '';
  slot = -1;
  actor = 'tauri.1';
  dirty = false;
  materialized: number | null = null;

  constructor(
    genome: Record<string, unknown>,
    mask: Mask,
    arena: SabArena,
    graft: GraftTable,
    name = 'tauri',
    scheduler: WorkerScheduler | null = null,
  ) {
    this.name = name;
    this.genome = genome;
    this.membrane = new Membrane(this.tape);
    this.membrane.bake(mask);
    this.arena = arena;
    this.graft = graft;
    this.scheduler = scheduler;
  }

  ensureCell(lineage: string, tint: number = STATE_INITIAL, actor = 'tauri.1'): number {
    this.lineage = lineage;
    this.actor = actor;
    const i = this.arena.index.size;
    this.arena.tint[i] = f32round(tint);
    this.arena.epoch[i] = 1;
    this.arena.active[i] = 0;
    this.arena.index.set(lineage, i);
    this.slot = i;
    return i;
  }

  cellSetTint(value: number, ts: number, actor?: string): number {
    const op = this.oplog.append(value, ts, actor || this.actor);
    this.tape.add(opLine(op));
    const tint = f32round(this.oplog.lww() as number);
    this.arena.tint[this.slot] = tint;
    this.arena.epoch[this.slot] += 1;
    this.dirty = true;
    return tint;
  }

  cellTick(tickNo: number): boolean {
    if (!this.dirty) return false;
    if (!this.membrane.attempt('render.tint', tickNo)) return false;
    const tint = this.arena.tint[this.slot];
    const ok = this.graft.apply('render.tint', tint);
    this.materialized = tint;
    this.dirty = false;
    return ok;
  }

  peerQuery(tickNo: number): boolean {
    if (this.membrane.attempt('data.query.peers', tickNo)) {
      this.tape.event('peer.query.dispatched', { tick: tickNo, epoch: this.membrane.mask ? this.membrane.mask.epoch : 0 });
      return true;
    }
    return false;
  }

  flipMask(mask: Mask): Record<string, unknown> {
    const sabBefore = this.arena.sab;
    const slotBefore = this.slot;
    const lineageBefore = this.lineage;
    const opsBefore = this.oplog.ops.length;
    this.membrane.bake(mask);
    return {
      kernel_reloaded: false, // scheduler worker / arena untouched by a mask flip
      sab_identity_preserved: sabBefore === this.arena.sab,
      slot_changed: slotBefore !== this.slot,
      lineage_changed: lineageBefore !== this.lineage,
      ops_preserved: this.oplog.ops.length === opsBefore,
      epoch: this.membrane.mask ? this.membrane.mask.epoch : 0,
    };
  }

  exportSnapshot(path: string, generator: string): Record<string, unknown> {
    const tint = this.arena.tint[this.slot];
    const header = {
      kind: 'cell.snapshot',
      schema: 'v1.6',
      cell_type: 'cell.tint.v1',
      lineage: this.lineage,
      genome_id: this.genome['genome_id'],
      tint,
      epoch: this.arena.epoch[this.slot],
      mask_epoch: this.membrane.mask ? this.membrane.mask.epoch : 0,
      actor_id: this.actor,
      op_count: this.oplog.ops.length,
      op_digest: this.oplog.digest(),
      generator,
    };
    let out = JSON.stringify(header) + '\n';
    for (const op of this.oplog.ops) out += JSON.stringify(op) + '\n';
    writeFileSync(path, out, 'utf8');
    return header;
  }

  static readSnapshot(path: string): [Record<string, unknown>, Op[]] {
    const lines = readFileSync(path, 'utf8').split('\n').filter((s) => s.trim().length > 0);
    const header = JSON.parse(lines[0]) as Record<string, unknown>;
    const ops = lines.slice(1).map((s) => JSON.parse(s) as Op);
    return [header, ops];
  }

  regenerateFrom(path: string): Record<string, unknown> {
    const [header, ops] = Tissue.readSnapshot(path);
    if (header['genome_id'] !== this.genome['genome_id']) throw new Error('snapshot genome mismatch');
    this.lineage = String(header['lineage']);
    this.actor = String(header['actor_id'] || 'tauri.1');
    this.ensureCell(this.lineage, STATE_INITIAL, this.actor);
    this.arena.epoch[this.slot] = Number(header['epoch'] || 1);
    this.oplog.ops = ops.slice();
    const tint = f32round(this.oplog.lww() as number);
    this.arena.tint[this.slot] = tint;
    this.dirty = true;
    return {
      lineage: this.lineage,
      tint,
      ops: this.oplog.ops.length,
      digest: this.oplog.digest(),
      digest_matches_header: this.oplog.digest() === header['op_digest'],
      epoch: this.arena.epoch[this.slot],
      header_mask_epoch: header['mask_epoch'],
    };
  }
}
