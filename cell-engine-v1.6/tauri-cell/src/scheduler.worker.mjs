// Universal Cell Engine v1.5 — the compute thread for the Tauri webview tissue.
//
// NODE PROOF: this runs as a worker_threads Worker over a real SharedArrayBuffer
// (identical SAB/Atomics/postMessage semantics to a Web Worker in the webview).
// PRODUCTION webview: swap the two transport lines below (parentPort <->
// self.onmessage/self.postMessage); the SAB math is byte-for-byte identical.
//
// The worker owns NOTHING but typed-array views: no ids, no objects, no host
// knowledge. It never sees a lineage id — cells are rows. Slot 0 is reserved
// for the canonical CRDT cell and is skipped by the easing kernel.
import { parentPort } from 'node:worker_threads';

const RATE = 0.03;
const EPS = 1e-3;
const CHUNK = 512;
const HDR_WORDS = 16;

let sab = null;
let n = 0;
let tint = null;
let active = null;
let target = null;
let hdr = null;

function views(s, count) {
  const hdrBytes = HDR_WORDS * 4;
  const tintOff = hdrBytes;
  const epochOff = tintOff + count * 4;
  const activeOff = epochOff + count * 4;
  const targetOff = (activeOff + count + 3) & ~3;
  sab = s;
  n = count;
  hdr = new Int32Array(sab, 0, HDR_WORDS);
  tint = new Float32Array(sab, tintOff, count);
  active = new Uint8Array(sab, activeOff, count);
  target = new Float32Array(sab, targetOff, count);
}

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function tick({ budgetUs, impulses, seed }) {
  const t0 = performance.now();
  const deadline = t0 + budgetUs / 1000;
  const rng = mulberry32(seed | 0);
  // morphogen impulses: wake cells (indices 1..n-1) with fresh targets
  for (let j = 0; j < impulses; j++) {
    const i = 1 + Math.floor(rng() * (n - 1));
    target[i] = rng();
    active[i] = 1;
  }
  const idx = [];
  for (let i = 1; i < n; i++) if (active[i]) idx.push(i);
  let executed = 0;
  let deferred = 0;
  for (let s = 0; s < idx.length; s += CHUNK) {
    if (performance.now() >= deadline) {
      deferred = idx.length - s;
      break;
    }
    const end = Math.min(s + CHUNK, idx.length);
    for (let k = s; k < end; k++) {
      const i = idx[k];
      const t = tint[i];
      let d = target[i] - t;
      if (d > RATE) d = RATE;
      else if (d < -RATE) d = -RATE;
      const tn = Math.fround(t + d);
      tint[i] = tn;
      if (Math.abs(target[i] - tn) < EPS) active[i] = 0; // settled: metabolic sleep
    }
    executed += end - s;
  }
  let activeCount = 0;
  for (let i = 0; i < n; i++) if (active[i]) activeCount++;
  const ns = Math.round((performance.now() - t0) * 1e6); // ms diff -> ns
  Atomics.store(hdr, 0, executed);
  Atomics.store(hdr, 1, deferred);
  Atomics.store(hdr, 2, activeCount);
  return { executed, deferred, active: activeCount, ns };
}

parentPort.on('message', (m) => {
  if (m.cmd === 'init') {
    views(m.sab, m.n);
    parentPort.postMessage({ cmd: 'ready' });
  } else if (m.cmd === 'tick') {
    parentPort.postMessage({ cmd: 'report', ...tick(m) });
  } else if (m.cmd === 'rest') {
    // resting-floor measurement: no impulses, pure scan cost
    const t0 = performance.now();
    let count = 0;
    for (let i = 1; i < n; i++) if (active[i]) count++;
    parentPort.postMessage({ cmd: 'report', executed: 0, deferred: 0, active: count, ns: Math.round((performance.now() - t0) * 1e6) });
  }
});
