// Universal Cell Engine v1.5 — Tissue B: Tauri webview shim (framework-free).
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
export function canonical(o) {
    if (o === null || typeof o !== 'object')
        return JSON.stringify(o);
    if (Array.isArray(o))
        return '[' + o.map(canonical).join(',') + ']';
    const obj = o;
    return '{' + Object.keys(obj).sort().map((k) => JSON.stringify(k) + ':' + canonical(obj[k])).join(',') + '}';
}
export function sha256Hex(s) {
    return createHash('sha256').update(s, 'utf8').digest('hex');
}
export function genomeIdOf(genome) {
    const rest = { ...genome };
    delete rest['genome_id'];
    return 'sha256:' + sha256Hex(canonical(rest));
}
export function loadGenome(path = GENOME_PATH) {
    const g = JSON.parse(readFileSync(path, 'utf8'));
    if (genomeIdOf(g) !== g['genome_id']) {
        throw new Error('genome contract violated');
    }
    return g;
}
export function fnv1a64(s) {
    const bytes = Buffer.from(s, 'utf8');
    let h = FNV_OFFSET;
    for (const b of bytes) {
        h ^= BigInt(b);
        h = (h * FNV_PRIME) & U64;
    }
    return 'fnv1a64:' + h.toString();
}
export function f32round(x) {
    return Math.fround(x);
}
export function fmt6(x) {
    return f32round(x).toFixed(6);
}
export class Mask {
    maskId;
    epoch;
    audience;
    allowed;
    denied;
    constructor(doc) {
        this.maskId = doc.mask_id;
        this.epoch = doc.epoch;
        this.audience = doc.audience;
        this.allowed = new Set(doc.allowed_verbs);
        this.denied = new Set(doc.denied_verbs);
    }
    allows(verb) {
        return this.allowed.has(verb) && !this.denied.has(verb);
    }
}
export function loadMask(path) {
    return new Mask(JSON.parse(readFileSync(path, 'utf8')));
}
export class Tape {
    lines = [];
    telemetry = [];
    add(line) {
        this.lines.push(line);
    }
    event(name, kv = {}) {
        this.telemetry.push({ event: name, ...kv, wall: Date.now() / 1000 });
    }
    digest() {
        return fnv1a64(this.lines.join('\n') + '\n');
    }
}
export class Membrane {
    tape;
    mask = null;
    constructor(tape) {
        this.tape = tape;
    }
    bake(mask) {
        const prev = this.mask ? this.mask.epoch : null;
        this.mask = mask;
        if (prev !== null && prev !== mask.epoch) {
            this.tape.add(`mask.flip|${prev}|${mask.epoch}`);
            this.tape.event('mask.flip', { from: prev, to: mask.epoch });
        }
    }
    attempt(verb, tick) {
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
export function opLine(op) {
    return `op|${op.op}|${op.value.toFixed(6)}|${op.ts}|${op.actor}`;
}
export class OpLog {
    ops = [];
    append(value, ts, actor) {
        const op = { op: 'set_tint', value: f32round(value), ts, actor };
        this.ops.push(op);
        return op;
    }
    lww() {
        let bestKey = null;
        let bestVal = null;
        for (const op of this.ops) {
            const key = [op.ts, op.actor];
            if (bestKey === null || key > bestKey) {
                bestKey = key;
                bestVal = op.value;
            }
        }
        return bestVal;
    }
    digest() {
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
export function buildArena(n) {
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
export function makeElement(id) {
    return { id, style: { background: 'rgb(185,185,185)' } }; // STANDIN-DOM
}
export class GraftTable {
    tissue = 'tauri';
    table;
    element;
    constructor(element) {
        this.element = element;
        this.table = {
            // the entire Tauri-side host binding: one style write per drain
            'render.tint': (tint) => {
                const v = Math.round(f32round(tint) * 255);
                this.element.style.background = `rgb(${v},${v},${v})`;
            },
        };
    }
    apply(verb, tint) {
        const fn = this.table[verb];
        if (!fn)
            return false;
        fn(tint);
        return true;
    }
}
export class WorkerScheduler {
    worker;
    n;
    pending = null;
    ready;
    readyResolve;
    constructor(sab, n) {
        this.n = n;
        const workerPath = fileURLToPath(new URL('./scheduler.worker.mjs', import.meta.url));
        this.worker = new Worker(workerPath);
        this.ready = new Promise((res) => {
            this.readyResolve = res;
        });
        this.worker.on('message', (m) => {
            if (m.cmd === 'ready')
                this.readyResolve();
            else if (m.cmd === 'report') {
                const p = this.pending;
                this.pending = null;
                if (p)
                    p(m);
            }
        });
        this.worker.postMessage({ cmd: 'init', sab, n });
    }
    tick(budgetUs, impulses, seed) {
        return new Promise((resolve) => {
            this.pending = resolve;
            this.worker.postMessage({ cmd: 'tick', budgetUs, impulses, seed });
        });
    }
    terminate() {
        this.worker.terminate();
    }
}
// ---------------------------------------------------------------- tissue (canonical cell)
export class Tissue {
    name;
    genome;
    tape = new Tape();
    membrane;
    arena;
    scheduler;
    graft;
    oplog = new OpLog();
    lineage = '';
    slot = -1;
    actor = 'tauri.1';
    dirty = false;
    materialized = null;
    constructor(genome, mask, arena, graft, name = 'tauri', scheduler = null) {
        this.name = name;
        this.genome = genome;
        this.membrane = new Membrane(this.tape);
        this.membrane.bake(mask);
        this.arena = arena;
        this.graft = graft;
        this.scheduler = scheduler;
    }
    ensureCell(lineage, tint = STATE_INITIAL, actor = 'tauri.1') {
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
    cellSetTint(value, ts, actor) {
        const op = this.oplog.append(value, ts, actor || this.actor);
        this.tape.add(opLine(op));
        const tint = f32round(this.oplog.lww());
        this.arena.tint[this.slot] = tint;
        this.arena.epoch[this.slot] += 1;
        this.dirty = true;
        return tint;
    }
    cellTick(tickNo) {
        if (!this.dirty)
            return false;
        if (!this.membrane.attempt('render.tint', tickNo))
            return false;
        const tint = this.arena.tint[this.slot];
        const ok = this.graft.apply('render.tint', tint);
        this.materialized = tint;
        this.dirty = false;
        return ok;
    }
    peerQuery(tickNo) {
        if (this.membrane.attempt('data.query.peers', tickNo)) {
            this.tape.event('peer.query.dispatched', { tick: tickNo, epoch: this.membrane.mask ? this.membrane.mask.epoch : 0 });
            return true;
        }
        return false;
    }
    flipMask(mask) {
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
    exportSnapshot(path, generator) {
        const tint = this.arena.tint[this.slot];
        const header = {
            kind: 'cell.snapshot',
            schema: '1.5',
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
        for (const op of this.oplog.ops)
            out += JSON.stringify(op) + '\n';
        writeFileSync(path, out, 'utf8');
        return header;
    }
    static readSnapshot(path) {
        const lines = readFileSync(path, 'utf8').split('\n').filter((s) => s.trim().length > 0);
        const header = JSON.parse(lines[0]);
        const ops = lines.slice(1).map((s) => JSON.parse(s));
        return [header, ops];
    }
    regenerateFrom(path) {
        const [header, ops] = Tissue.readSnapshot(path);
        if (header['genome_id'] !== this.genome['genome_id'])
            throw new Error('snapshot genome mismatch');
        this.lineage = String(header['lineage']);
        this.actor = String(header['actor_id'] || 'tauri.1');
        this.ensureCell(this.lineage, STATE_INITIAL, this.actor);
        this.arena.epoch[this.slot] = Number(header['epoch'] || 1);
        this.oplog.ops = ops.slice();
        const tint = f32round(this.oplog.lww());
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
