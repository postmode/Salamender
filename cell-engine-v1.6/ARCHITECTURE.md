# Universal Cell Engine v1.5 — Real Tissue Proof: Architecture & Evidence

> Falsifier round. One universal cell (`cell.tint.v1`, genome `sha256:fa267e9c…778ca`)
> had to survive destruction of its Blender-side tissue and regenerate inside a
> Tauri-side tissue with the same lineage, same genome hash, zero state loss.
> **It did.** Everything below is either measured in this repo (path given) or
> explicitly labeled STANDIN / NOT COMPILED. Reproduce everything:
> `bash scripts/run_v15.sh` (phases A → B → C).

## 0. Verdict

**CONDITIONALLY VIABLE** — every acceptance criterion that could be executed in this
sandbox passed, including the full cross-process, cross-language falsifier (E3) and
its return leg. The conditions are two *environmental* blockers (this sandbox has no
Blender binary and no cargo/rustc), not architectural failures:

| Blocker | Name | What is missing |
|---|---|---|
| Blender tissue | `B1-BLENDER-BINARY-ABSENT` | viewport/timer/depsgraph numbers; addon logic, lifecycle, graft path all exercised headless with a labeled `_BpyStandin` |
| Tauri tissue | `B1-TAURI-TOOLCHAIN-ABSENT` | `cargo build` + `tauri dev` run of the real Rust shard; SAB/worker/IPC-proxy tier fully executed in Node |

One *architectural* finding confirmed the sharding fix you anticipated (§4 F2):
per-cell Python grafting does not scale; the arena must sit in a faster tier with
bpy materializing only bound verbs.

## 1. The four layers as built (no collapsed actor system)

| Layer | Blender tissue | Tauri tissue |
|---|---|---|
| 1 Genome | `genome.json` — content-addressed, sha256 verified at load in **both** tissues; zero host knowledge | same file, same hash |
| 2 Arena | numpy structured array, spec dtype `[('id','U36'),('tint','f4'),('epoch','i4'),('active','bool')]` + contiguous hot-column mirrors (§4 F1) | `SharedArrayBuffer`: `Float32Array` tint + `Int32Array` epoch + `Uint8Array` active + JS `Map` id index — the spec layout |
| 3 Scheduler | vectorized metabolism, deadline in µs, over-budget remainder = **metabolic debt** | worker thread over the SAB (same semantics as a Web Worker in the webview) |
| 4 Graft | `{'render.tint': ob.material_slots[0].material.diffuse_color = (t,t,t,1.0)}` — the only host code in the addon | `{'render.tint': el.style.background = 'rgb(…)'}` — 523 bytes of Rust (budget 5KB) |

Addons/shards stay tiny: Blender addon **535 LOC** (budget 600), Rust graft section
**523 bytes** (budget 5000), webview shim 464 LOC TS + 103 LOC worker.

## 2. Experiments and measured results

### E3 — Salamander (the falsifier): PASS, fully real, cross-process + cross-language

Python process: 5 ops 0.725→0.884, exports `regeneration/crdt_snapshot.jsonl`,
process exits ("Blender destroyed"). Node process (different language, different
runtime, different process): re-imports, replays the CRDT log, asserts.

- lineage identical: `01a17d1d-e392-7d0b-9a87-07348c36a325` ✓
- tint resumed at **0.884** (not 0.725) ✓
- op log length identical (5) ✓
- causal digest identical: `fnv1a64:11520714716851241111` (python-computed == node-recomputed) ✓
- continuation in Tauri: op 6 → 0.910, materialized `rgb(232,232,232)` ✓
- return leg (node→python): `perf/regeneration_roundtrip.json` — 6 ops, tint 0.910, digest match ✓

### E6 — Determinism: PASS, including cross-runtime

`perf/determinism_digest.txt`: identical tape digest
`fnv1a64:9575468746670681964` for python run1/run2, node run1/run2, **and python == node**
(9 canonical tape lines, byte-identical in both languages). Genome sha256 recomputed
identically in both tissues.

### E5 — Membrane flip without kernel change: PASS (both tissues)

Public: `data.query.peers` dropped, `membrane.drop` telemetry, no exception.
Flip to trusted (epoch 1→2, logged with wall ts): same call allowed. Proven
unchanged across the flip: kernel function identity (python), SAB object identity,
slot index, lineage, op count. No restart, no reload.

### E1 — Blender addon lifecycle: PASS (logic real, event loop STANDIN)

register → ops → unregister (snapshot exported) → register → tint restored **0.884**
with same lineage and 5 ops; `load_post` path re-asserts. Enable ≈ 0.09–0.16 ms in-process
(module import 184 ms cold, dominated by numpy). `bpy` materialize path exercised against
the labeled standin; real viewport untested (blocker above).

### E4 — Frame budget @ 200k cells + 1 real graft: PASS

| Measure | Blender tier (numpy) | Tauri tier (SAB+worker) |
|---|---|---|
| steady tick p50 / p99 | **866 µs / 951 µs** (budget 2 ms) | **740 µs / 786 µs** (budget 2 ms) |
| deferred (debt) in steady state | 0 | 0 |
| active cells avg | 8,384 (4.19%) | 9,827 (4.91%) |
| resting floor (200k asleep) | 359 µs | 453 µs |
| worst case: all 200k active | 2.63 ms | 5.50 ms |
| graft cost per call (separate) | 0.63 µs (STANDIN-bpy) | 0.12 µs (STANDIN-DOM) |
| 14k-graft projection | **8.85 ms — over budget** | 1.63 ms |
| IPC proxy (MessageChannel RTT) | — | p50 6.5 µs, p99 156 µs |

Debt mechanism verified separately: over-budget ticks defer and report; the host
frame is never blocked.

## 3. Where numbers are real vs labeled

| Artifact | Status |
|---|---|
| arena, scheduler, CRDT, membrane, tape/digests, lifecycle round-trips, E3 both legs | **REAL, executed** |
| `bpy` materialize path | **STANDIN-bpy** (`_BpyStandin` mirrors the exact attribute path) |
| DOM materialize | **STANDIN-DOM** (plain object; real WebKit style recalc unmeasured) |
| Tauri IPC | **PROXY** (MessageChannel structured-clone hop) |
| Rust shard (`src-tauri/src/main.rs`, 332 LOC) | **REAL CODE, NOT COMPILED** (no cargo here) |
| Blender 60 FPS frame with addon active | **UNTESTED** |

## 4. Findings (numbers, not claims)

- **F1 — spec dtype is cache-hostile.** The literal structured dtype interleaves
  columns at a 153-byte row stride: a full 200k `flatnonzero` scan costs
  **908 µs strided vs 359 µs contiguous** (2.5×), and a strided-gather kernel
  measured ~2.8 ms/tick vs 866 µs — over budget on scan alone. Fix as built: keep
  the spec dtype as body-of-record, run metabolism on contiguous mirror columns,
  sync at boundaries. Same columns, same semantics, one measured layout decision.
- **F2 — the Python graft wall (the sharding fix is mandatory).** 0.63 µs per
  STANDIN-bpy materialize × 14k grafts = **8.85 ms > 2 ms budget**. One real Cube
  graft costs ~0.63 µs/frame — fine; thousands do not. Fix (as you proposed): the
  arena lives in a faster tier (numpy hot columns here, Rust in production), and
  bpy materializes **only bound verbs** — graft count is bounded by host nouns
  (scene objects), never by cell count. The Node/SAB tier already shows the shape:
  14k graft projection 1.63 ms, scheduler off the UI thread.
- **F3 — scalar JS kernel is 2.1× slower than vectorized numpy** (5.50 ms vs
  2.63 ms worst case). The Rust tier (AOT, SoA, no GC) is expected to close and
  invert this; the ratios, not the absolutes, are the signal.
- **F4 — `B1-TAURI-SAB-XPROC`:** a SharedArrayBuffer cannot cross the
  WebView process boundary in Tauri 2. The arena SAB is webview-owned (scheduler
  in a Web Worker / worker_threads — as executed here); Rust mirrors state via the
  two commands. The boundary cost is the IPC hop: 6.5 µs p50.

## 5. Regulatory contract (exact schema)

`genome.json` / `manifest.json` / `masks/{public,trusted}.json` — byte-exact to the
v1.5 spec, with `genome_id = "sha256:" + sha256(canonical-json of the genome minus
genome_id)`, recomputed and asserted in Python, Node, and (in the Rust port) at
startup. Snapshot format: JSONL header + op lines
(`regeneration/crdt_snapshot*.jsonl`). Digests: FNV-1a 64 over canonical lines;
tape lines carry only deterministic fields (wall clocks live in telemetry).

## 6. Residual walls after v1.5

1. Real Blender: viewport frame cost with addon at 60 FPS, depsgraph interplay,
   real `diffuse_color` RNA cost (expect 2–6 µs vs 0.63 µs standin).
2. Real Tauri: `cargo build` + `tauri dev`, real WebKit style recalc, real invoke
   RTT (vs 6.5 µs proxy).
3. WASM membrane tier (enforced, not cooperative) — untouched by design in v1.5.
4. Cross-host *live* sync (this proof is snapshot-based regeneration; live op-log
   delta streaming is v2).
5. Multi-tissue scheduler / organism-level supervision beyond one canonical cell.

## 7. Layout & reproduction

```
blender_addon/cell_engine_blender/__init__.py   535 LOC  the addon (real bpy + labeled headless guard)
blender_addon/blender_manifest.toml                     Blender 4.2+ extension manifest
blender_addon/test_blender.py                  275 LOC  Phase A/C harness
tauri-cell/src/main.ts                         464 LOC  webview shim (runs in Node proof)
tauri-cell/src/scheduler.worker.mjs            103 LOC  SAB compute thread
tauri-cell/test_tauri.ts                       264 LOC  Phase B harness
tauri-cell/src-tauri/                           332 LOC  real Rust shard (NOT COMPILED here)
regeneration/crdt_snapshot.jsonl                        python -> node export (E3)
regeneration/crdt_snapshot_return.jsonl                 node -> python return leg
perf/{blender,tauri}_perf.json                          all measured numbers + labels
perf/determinism_digest.txt                             digest evidence sheet
```

```bash
bash /home/z/my-project/scripts/run_v15.sh     # phases A -> B -> C, from scratch
```
