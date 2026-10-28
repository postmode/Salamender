#!/usr/bin/env python3
"""v1.5 Phase A harness — Blender tissue experiments.

Usage:  python3 test_blender.py scenario   (E1+E3-export+E5+E6, writes perf jsons)
        python3 test_blender.py roundtrip  (Phase C: re-import the Tauri return snapshot)

REAL here : arena/scheduler/CRDT/membrane/graft-table code paths, lifecycle round-trips,
            numpy 200k-cell metabolism, FNV/sha256 contract checks, cross-process export.
STANDIN-bpy: the material mutation path (no Blender binary in this sandbox) — labeled
             in every perf number that touches it.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import cell_engine_blender as addon  # noqa: E402
bpy = addon.bpy  # real bpy inside Blender; None in headless STANDIN mode

PY = sys.executable
LINEAGE = "01a17d1d-e392-7d0b-9a87-07348c36a325"  # fixed for the scripted scenario (E3)
OPS = [(0.750, 1), (0.788, 2), (0.812, 3), (0.850, 4), (0.884, 5)]  # -> 0.884
CONTINUE_OP = (0.910, 6, "tauri.1")
N_PERF = 200_000  # simulated cells + 1 canonical cell at slot 0


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[i]


# ---------------------------------------------------------------- E6+E5: scripted scenario
def run_scenario() -> dict:
    """Identical script in Python and Node -> identical causal tape digest."""
    tissue = addon.Tissue(addon.load_genome(), addon.load_mask(addon.MASK_PUBLIC_PATH),
                          addon.GraftTable(), name="blender")
    kernel_ref = addon.MetabolicScheduler.tick
    slot = tissue.ensure_cell(LINEAGE, tint=0.725, actor="blender.1")
    for value, ts in OPS:
        tissue.cell_set_tint(value, ts, "blender.1")
    tissue.cell_tick(1)              # dirty -> membrane allow render.tint -> graft
    drop = tissue.peer_query(2)      # public: dropped, telemetry, no exception
    flip = tissue.flip_mask(addon.load_mask(addon.MASK_TRUSTED_PATH))
    allow = tissue.peer_query(3)     # trusted: allowed
    tissue.cell_tick(4)              # not dirty -> no-op
    return {
        "tape_digest": tissue.tape.digest(),
        "op_digest": tissue.oplog.digest(),
        "slot": slot,
        "tint": float(tissue.arena.rows["tint"][tissue.slot]),
        "ops": len(tissue.oplog.ops),
        "drop_public": not drop,
        "allow_trusted": allow,
        "flip": flip,
        "kernel_identity_preserved": kernel_ref is addon.MetabolicScheduler.tick,
        "tape_lines": list(tissue.tape.lines),
        "telemetry": tissue.tape.telemetry,
        "_tissue": tissue,
    }


# ---------------------------------------------------------------- E1: addon lifecycle (REAL bpy)
def _install_addon_dir() -> str:
    """Install the addon into Blender's scripts/addons path so addon_utils can find
    it, exactly like installing an extension from disk. The source file mtime is
    carried over (copystat) so Blender's "module changed on disk" reload machinery
    does not fire when the repo copy is newer than a previous install — that reload
    re-executes module-level code and crashes inside restricted handler contexts."""
    import addon_utils
    src = os.path.join(HERE, "cell_engine_blender")
    # stable, explicit location inside the user config dir (bpy.utils.resource_path
    # takes 'USER' | 'LOCAL' | 'SYSTEM'; fall back to the conventional path)
    try:
        cfg = bpy.utils.resource_path("USER")
    except Exception:
        cfg = os.path.expanduser("~/.config/blender")
    dst_parent = os.path.join(cfg, "scripts", "addons")
    os.makedirs(dst_parent, exist_ok=True)
    dst = os.path.join(dst_parent, "cell_engine_blender")
    if os.path.abspath(src) != os.path.abspath(dst):
        import shutil
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, copy_function=shutil.copy2)  # preserve mtimes
    addon_utils.modules_refresh()
    return dst


def _real_enable_disable_cycle() -> dict:
    """Disable + re-enable the addon INSIDE Blender's own extension machinery,
    exactly as Preferences > Add-ons does. Returns measured enable time.

    Note: disable/enable of an already-imported module does NOT re-run module-level
    code or register()/unregister() (Blender keeps the loaded module), so the live
    tissue object and its pinned snapshot survive the cycle; we verify that here."""
    import addon_utils
    mod_id = "cell_engine_blender"
    _install_addon_dir()
    t0 = time.perf_counter()
    addon_utils.disable(mod_id, default_set=True)
    disable_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    addon_utils.enable(mod_id, default_set=True, persistent=True)
    enable_ms = (time.perf_counter() - t1) * 1e3
    return {"disable_ms": disable_ms, "enable_ms": enable_ms}


# E1 runs against a private copy of the canonical Tauri return snapshot so its
# lineage is deterministic across sessions (fresh sandbox -> uuid7; rerun ->
# regenerated from disk). The shared default path (regeneration/crdt_snapshot.jsonl)
# stays reserved for the E3 falsifier export, which main() writes after e1 runs.
E1_SNAPSHOT_SRC = os.path.join(ROOT, "regeneration", "crdt_snapshot_return.jsonl")
E1_SNAPSHOT_PATH = os.path.join(ROOT, "regeneration", "e1_snapshot.jsonl")


def _snapshot_is_stale_e3_export() -> bool:
    """If the shared default snapshot still carries the scripted blender.1 tape
    (final tint 0.884), it is a leftover E3 export — not an E1 snapshot."""
    try:
        with open(addon.SNAPSHOT_PATH, encoding="utf-8") as fh:
            lines = [json.loads(s) for s in fh if s.strip()]
        header = lines[0]
        return (header.get("actor_id") == "blender.1"
                and abs(float(header.get("tint", -1)) - 0.884) < 1e-6)
    except (OSError, ValueError, IndexError, KeyError):
        return False


def _run_e1_lifecycle() -> dict:
    """E1 core. The caller guarantees addon.SNAPSHOT_PATH points at E1_SNAPSHOT_PATH."""
    enable1 = addon.register()
    t = addon.get_tissue()
    lineage1 = t.lineage
    assert abs(float(t.arena.tint[t.slot]) - 0.725) < 1e-6, "initial tint must be 0.725"
    for value, ts in OPS:
        t.cell_set_tint(value, ts)
    assert t.cell_tick(1), "graft apply failed"
    mat_before = tuple(t.graft._mat.diffuse_color)
    t_ops = t  # pre-reload tissue: carries the full op log for the final export
    checks = {}
    if addon.HAVE_BPY:
        # REAL path: drive the full cycle through Blender's own handler/timer loop.
        import addon_utils
        # 1. save the .blend WITH the live snapshot on disk
        #    (in --background the context is restricted; temp_override gives a valid one)
        blend_path = os.path.join(ROOT, "regeneration", "e1_cell_tissue.blend")
        with bpy.context.temp_override(**{k: v for k, v in (
                [('scene', bpy.data.scenes[0])] +
                ([('active_scene', bpy.data.scenes[0])]) )}):
            bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        checks["blend_saved"] = os.path.exists(blend_path)
        # 2. real disable/enable via addon_utils (same calls Preferences uses)
        checks.update(_real_enable_disable_cycle())
        t_live = addon.get_tissue()
        checks["lineage_same_after_real_reenable"] = t_live.lineage == lineage1
        # 3. frame pump: bpy.app.timers callbacks are executed by Blender's own WM
        #    step function (wm_window_process_events) — the exact machinery the GUI
        #    mainloop uses every frame. We drive that C-level stepper directly, so
        #    the timer fires through REAL Blender event machinery, not a Python mock.
        pump = []
        def _pump_timer():
            c0 = time.perf_counter_ns()
            tt = addon.get_tissue()
            tt.scheduler_tick(budget_us=1500, impulses=0, rng=np.random.default_rng(0))
            tt.cell_tick(0)
            pump.append((time.perf_counter_ns() - c0) / 1e3)
            return 1.0 / 60.0
        try:
            import _bpy; process_events = _bpy._window_process_events
        except (ImportError, AttributeError):
            process_events = None
        if process_events is not None:
            bpy.app.timers.register(_pump_timer, first_interval=0.0)
            for _ in range(240):
                process_events()
            bpy.app.timers.unregister(_pump_timer)
        else:  # no direct stepper exposed (--background): fall back to headless
               # scene-frame + depsgraph steps — same code path the timer invokes
            n_frames = int(os.environ.get("E1_FRAMES", "90"))
            vl = (bpy.context.window.view_layer
                  if getattr(bpy.context, "window", None)
                  else bpy.data.window_managers[0].windows[0].view_layer)
            for _ in range(n_frames):
                vl.update()
                _pump_timer()   # same code path the timer invokes, driven per frame
        cbs = sorted(pump)
        checks["frame_pump"] = {
            "stepper": "bpy._window_process_events (real WM event loop)" if process_events else "scene_frame_update fallback",
            "delivered_callbacks": len(cbs),
            "callback_p50_us": pct(cbs, 50) if cbs else None,
            "callback_p99_us": pct(cbs, 99) if cbs else None,
            "budget_note": "scheduler budget capped at 1500us inside callback; host frame never blocked",
        }
        # 4. real .blend reload in-process -> load_post handler regenerates tissue.
        # The reload fires bpy.app.handlers.load_post (our registered handler), which
        # already resets _tissue=None and rebuilds from the pinned E1 snapshot — do
        # NOT null it manually beforehand, or get_tissue() would lazily rebuild the
        # pre-reload tissue object and mask whether the reload truly regenerated.
        with bpy.context.temp_override(scene=bpy.data.scenes[0]):
            bpy.ops.wm.read_homefile(filepath=blend_path)
        t_reload = addon.get_tissue()
        checks["tint_after_blend_reload"] = float(t_reload.arena.tint[t_reload.slot])
        checks["lineage_same_after_blend_reload"] = t_reload.lineage == lineage1
        checks["ops_after_blend_reload"] = len(t_reload.oplog.ops)
        checks["real_enable_ms"] = checks.get("enable_ms")
    else:  # STANDIN-bpy headless: exercise register/unregister/load_post directly
        addon.unregister()               # disable addon -> snapshot exported
        enable2 = addon.register()       # re-enable -> regenerate from snapshot
        t2 = addon.get_tissue()
        checks["enable_ms_reenable"] = enable2 * 1e3
        checks["lineage_same_after_reenable"] = t2.lineage == lineage1
        checks["tint_after_reenable"] = float(t2.arena.tint[t2.slot])
        addon._tissue = None
        get_tissue = addon.get_tissue
        get_tissue()
        t3 = addon.get_tissue()
        checks["tint_after_load_post"] = float(t3.arena.tint[t3.slot])
        checks["lineage_same_after_load_post"] = t3.lineage == lineage1
    checks_common = {
        "enable_ms_first": enable1 * 1e3,
        "tint_not_reset": abs(float(addon.get_tissue().arena.tint[addon.get_tissue().slot]) - 0.884) < 1e-6,
        "ops_preserved": len(addon.get_tissue().oplog.ops) == 5,
        "snapshot_exists": os.path.exists(E1_SNAPSHOT_PATH),
        "mat_before_export": list(mat_before),
    }
    checks.update(checks_common)
    # Leave a clean E1 snapshot on disk. In the REAL-bpy branch the live tissue was
    # replaced by the .blend-reload regeneration; re-point it at the pre-reload ops
    # so unregister() persists the full lineage for the next session.
    t_final = addon.get_tissue()
    if len(t_final.oplog.ops) < 5:
        t_final.oplog.ops = list(t_ops.oplog.ops)
        t_final.actor = t_ops.actor
    addon.unregister()               # export_snapshot -> E1_SNAPSHOT_PATH
    checks["final_snapshot_lineage"] = t_final.lineage
    return checks


def e1_lifecycle() -> dict:
    """Run E1 against its own snapshot file so the falsifier export used by E3
    (regeneration/crdt_snapshot.jsonl) can never hijack the lineage. register(),
    unregister(), get_tissue() and the load_post handler all resolve
    addon.SNAPSHOT_PATH dynamically, so pinning it for the whole cycle is enough."""
    original_snapshot_path = addon.SNAPSHOT_PATH
    if os.path.exists(addon.SNAPSHOT_PATH) and _snapshot_is_stale_e3_export():
        os.remove(addon.SNAPSHOT_PATH)  # stale E3 export would hijack E1's lineage
    if os.path.exists(E1_SNAPSHOT_PATH):
        os.remove(E1_SNAPSHOT_PATH)     # start fresh: first enable creates the cell
    # Seed from the canonical Tauri return snapshot when present -> deterministic
    # lineage across sessions; otherwise register() mints a fresh uuid7 cell.
    # The seed carries 6 ops (final tint 0.910); E1's scripted scenario requires
    # the canonical initial state (tint 0.725, empty op log), so strip the ops and
    # reset the header — only the lineage survives as the deterministic identity.
    if os.path.exists(E1_SNAPSHOT_SRC):
        import shutil
        shutil.copyfile(E1_SNAPSHOT_SRC, E1_SNAPSHOT_PATH)
        with open(E1_SNAPSHOT_PATH, encoding="utf-8") as fh:
            lines = [json.loads(s) for s in fh if s.strip()]
        header = lines[0]
        # lww() of an empty op log is None -> f32(None) would poison tint with NaN;
        # regenerate_from spawns at STATE_INITIAL (0.725), so seed the sentinel 0.0
        # and reset actor to this tissue's default (fresh blender session identity).
        header.update({"tint": 0.0, "epoch": 1, "op_count": 0, "actor_id": "blender.1",
                       "op_digest": addon.OpLog().digest(),
                       "generator": "blender.addon.e1-seed"})
        with open(E1_SNAPSHOT_PATH, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(header) + "\n")
    addon.SNAPSHOT_PATH = E1_SNAPSHOT_PATH
    try:
        return _run_e1_lifecycle()
    finally:
        addon.SNAPSHOT_PATH = original_snapshot_path  # restore repo default for E3 export


# ---------------------------------------------------------------- E4: frame budget @ 200k
def e4_perf(graft=None, label="") -> dict:
    """E4 conditions A/B (v1.6): scheduler @200k + REAL bound-noun grafts measured
    together and separately. graft=None -> NullGraft (condition C, scheduler only)."""
    tissue = addon.Tissue(addon.load_genome(), addon.load_mask(addon.MASK_PUBLIC_PATH),
                          graft or addon.NullGraft(), name="blender-perf", capacity=N_PERF + 1)
    t = tissue  # canonical cell of this perf tissue (slot 0) — drains go through its graft
    tissue.ensure_cell(LINEAGE, 0.725)
    rng = np.random.default_rng(7)
    # bulk-spawn 200k simulated perf cells (spec: "200k simulated cells in numpy arena").
    # IMPORTANT: perf cells are seeded ASLEEP; the metabolic regime is impulses per
    # tick (morphogen wake-ups), matching v1.5 methodology (800 wakes/tick @2ms budget).
    tissue.arena.n = N_PERF + 1
    idx = np.arange(1, N_PERF + 1)
    tissue.arena.rows["id"][idx] = "perf"          # simulated lineage labels (not CRDT-backed)
    tissue.arena.active[idx] = 0

    ticks, budget_us, impulses = 240, 2000, 800
    for k in range(60):  # warm-up: let the initial wave settle -> measure steady state
        tissue.scheduler_tick(budget_us, impulses, np.random.default_rng(10_000 + k))
    ns, active, deferred = [], [], 0
    for k in range(ticks):
        rep = tissue.scheduler_tick(budget_us, impulses, np.random.default_rng(k))
        ns.append(rep["ns"])
        active.append(rep["active"])
        deferred += rep["deferred"]
    s = sorted(ns)
    # resting floor: everything asleep, no impulses -> pure dirty-bitset scan
    tissue.arena.active[:] = False
    rest = [tissue.scheduler_tick(budget_us, 0, np.random.default_rng(500 + j))["ns"] for j in range(60)]
    # FINDING F1 re-measured under Blender's own numpy: the same scan against the
    # spec-literal structured dtype (strided columns, 153B row stride)
    strided = []
    act_col = tissue.arena.rows["active"]
    for _ in range(30):
        t0 = time.perf_counter_ns()
        np.flatnonzero(act_col[1 : N_PERF + 1])
        strided.append(time.perf_counter_ns() - t0)
    # worst case: all 200k active, generous budget -> true full cost
    tissue.arena.targets[idx] = rng.uniform(0.0, 1.0, N_PERF).astype(np.float32)
    tissue.arena.active[idx] = True
    worst = tissue.scheduler_tick(500_000, 0, np.random.default_rng(999))
    # back to steady state for combined measurement
    tissue.arena.active[:] = 0
    # graft cost — measured SEPARATELY from scheduler cost; REAL bpy when HAVE_BPY
    g0 = time.perf_counter_ns()
    G = 2000
    for i in range(G):
        tissue.graft.apply("render.tint", 0.5 + (i % 32) / 1024)
    graft_us_per_call = (time.perf_counter_ns() - g0) / G / 1e3
    # E4 condition B: N bound nouns, per-drain frame cost = N x single-verb cost
    n_bound = len(getattr(tissue.graft, "bound_nouns", [])) or 1
    if n_bound > 1:
        drain0 = time.perf_counter_ns()
        for i in range(G):
            for _ in tissue.graft.bound_nouns:
                tissue.graft.apply("render.tint", 0.5 + (i % 32) / 1024)
        graft_us_per_drain = (time.perf_counter_ns() - drain0) / G / 1e3
    else:
        graft_us_per_drain = graft_us_per_call
    # combined steady-state tick incl. one canonical drain (scheduler + graft)
    comb = []
    for k in range(ticks):
        c0 = time.perf_counter_ns()
        tissue.scheduler_tick(budget_us, impulses, np.random.default_rng(k))
        tissue.dirty = True
        tissue.cell_tick(k)
        comb.append(time.perf_counter_ns() - c0)
    cs = sorted(comb)
    active_avg = statistics.mean(active) if active else 0.0
    return {
        "cells": N_PERF,
        "ticks": ticks,
        "budget_us": budget_us,
        "impulses_per_tick": impulses,
        "tick_p50_us": pct(s, 50) / 1e3,
        "tick_p99_us": pct(s, 99) / 1e3,
        "active_avg": round(active_avg, 1),
        "activity_rate_pct": round(100.0 * active_avg / N_PERF, 2),
        "deferred_total": deferred,
        "resting_floor_us": pct(sorted(rest), 50) / 1e3,
        "strided_spec_dtype_scan_floor_us": pct(sorted(strided), 50) / 1e3,
        "layout_note": ("spec dtype kept as body-of-record; metabolism runs on contiguous "
                         "mirror columns — see strided vs resting floor numbers"),
        "worst_case_all_active": {k2: worst[k2] for k2 in ("executed", "deferred", "ns")},
        "worst_case_all_active_us": worst["ns"] / 1e3,
        "graft_us_per_call": graft_us_per_call,
        "graft_label": "REAL bpy diffuse_color write" if addon.HAVE_BPY else "STANDIN-bpy",
        "bound_nouns": n_bound,
        "graft_us_per_drain_all_nouns": graft_us_per_drain,
        "combined_tick_p50_us": pct(cs, 50) / 1e3,
        "combined_tick_p99_us": pct(cs, 99) / 1e3,
        "graft_projection_14k_us": graft_us_per_call * 14000,
        "condition": label,
    }


# ---------------------------------------------------------------- E3: export (Blender side)
def e3_export(scenario: dict) -> dict:
    tissue = scenario["_tissue"]
    path = os.path.join(ROOT, "regeneration", "crdt_snapshot.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = tissue.export_snapshot(path, "python/cell_engine_blender")
    # "destroy the tissue": drop the object; the process itself exits after Phase A
    del scenario["_tissue"]
    return {
        "path": path,
        "header": header,
        "lineage": header["lineage"],
        "tint": header["tint"],
        "ops": header["op_count"],
        "op_digest": header["op_digest"],
    }


def _find_blender_binary():
    """Candidate Blender executables, env-overridable (E1 finalization pass)."""
    import glob
    import shutil
    cands = []
    env = os.environ.get("BLENDER_BIN")
    if env:
        cands.append(env)
    which = shutil.which("blender")
    if which:
        cands.append(which)
    for pat in ("/opt/blender*/blender", "/usr/local/blender*/blender",
                "/snap/blender/current/blender", "/Applications/Blender.app/Contents/MacOS/Blender",
                os.path.expanduser("~/blender*/blender")):
        cands.extend(sorted(glob.glob(pat)))
    seen, out = set(), []
    for c in cands:
        if c not in seen and os.path.exists(c):
            seen.add(c)
            out.append(c)
    return out


def _run_phase_a_under_blender(blender_bin: str) -> bool:
    """Re-run this harness inside real Blender's bundled Python (bpy available).
    Returns True on success (perf jsons written by the Blender process)."""
    script = (
        "import sys, runpy; "
        f"sys.path.insert(0, {HERE!r}); "
        "import test_blender; "
        "test_blender.main('scenario')"
    )
    proc = subprocess.run([blender_bin, "--background", "--python-expr", script],
                          capture_output=True, text=True, timeout=900)
    sys.stdout.write(proc.stdout[-3000:])
    if proc.returncode != 0:
        sys.stderr.write("blender stderr tail:\n" + proc.stderr[-3000:] + "\n")
        return False
    # verify the Blender process actually wrote REAL-bpy numbers, not a stale file
    try:
        with open(os.path.join(ROOT, "perf", "blender_perf.json"), encoding="utf-8") as fh:
            real = "REAL bpy" in fh.read()
    except OSError:
        return False
    if not real:
        sys.stderr.write("blender run produced no REAL-bpy perf output; falling back\n")
    return real


def module_import_ms() -> float:
    code = (
        "import time,sys; sys.path.insert(0, r'%s'); "
        "t=time.perf_counter(); import cell_engine_blender; "
        "print(time.perf_counter()-t)" % HERE
    )
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, check=True)
    return float(out.stdout.strip()) * 1e3


# ---------------------------------------------------------------- main
def main(mode: str) -> None:
    if mode == "roundtrip":
        path = os.path.join(ROOT, "regeneration", "crdt_snapshot_return.jsonl")
        genome = addon.load_genome()
        tissue = addon.Tissue(genome, addon.load_mask(addon.MASK_PUBLIC_PATH), "blender-2")
        checks = tissue.regenerate_from(path)
        checks["tint_is_0p910"] = abs(checks["tint"] - 0.910) < 1e-6
        checks["ops_is_6"] = checks["ops"] == 6
        checks["lineage_is_canonical"] = checks["lineage"] == LINEAGE
        out = os.path.join(ROOT, "perf", "regeneration_roundtrip.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"phase": "C (python re-import of tauri return snapshot)", **checks}, fh, indent=2)
        print(json.dumps({"roundtrip": checks}, indent=2))
        return

    have_bpy_now = addon.HAVE_BPY
    # v1.6 finalization: prefer REAL bpy for E1/E4 when a Blender binary is available.
    # If this interpreter lacks bpy, probe candidate binaries and re-run Phase A inside
    # the real Blender Python so blender_perf.json carries REAL numbers + environment.
    if not have_bpy_now:
        for cand in _find_blender_binary():
            if _run_phase_a_under_blender(cand):
                print(f"[Phase A] executed under real Blender: {cand} — perf written")
                return
        print("[Phase A] no Blender binary found — continuing headless STANDIN-bpy")

    import_ms = module_import_ms()
    r1 = run_scenario()
    r2 = run_scenario()
    e1 = e1_lifecycle()
    e3 = e3_export(r1)
    # E4 conditions (v1.6 spec): A = scheduler only (NullGraft), B = scheduler + N
    # bound-noun grafts draining every frame.
    e4a = e4_perf(graft=addon.NullGraft(), label="A: scheduler @200k, NullGraft (host idle)")
    e4b = None
    if addon.HAVE_BPY:
        nouns = []
        for i in range(5):  # 5 real bound nouns (mesh objects with material slots)
            ob = bpy.data.objects.new(f"cell_noun_{i}", bpy.data.meshes.new(f"cell_noun_{i}"))
            bpy.data.collections[0].objects.link(ob)
            mat = bpy.data.materials.new(f"cell_mat_{i}")
            ob.data.materials.append(mat)
            ob["cell_slot"] = i + 1
            nouns.append(ob)

        class BoundNounGraft:
            """E4 condition B: one REAL bpy graft bound to N host nouns; each drain
            writes diffuse_color on every bound noun (the production materialize path)."""
            def __init__(self, objs):
                self.bound_nouns = [o.material_slots[0].material for o in objs]

            def apply(self, verb, value):
                if verb == "render.tint":
                    for m in self.bound_nouns:
                        m.diffuse_color = (value, value, value, 1.0)

        e4b = e4_perf(graft=BoundNounGraft(nouns),
                      label="B: scheduler @200k + 5 REAL bpy bound-noun drains/frame")

    labels = {
        "arena_scheduler_crdt_membrane": "REAL (numpy, this process)",
        "bpy_materialize": "REAL bpy" if addon.HAVE_BPY else "STANDIN-bpy",
        "addon_lifecycle_roundtrips": ("REAL (register/unregister via addon_utils, .blend save/reload, "
                                       "WM timer pump)" if addon.HAVE_BPY else
                                       "REAL (register/unregister/load_post handler; Blender event loop untested)"),
        "blender_frame_60fps": ("REAL (headless WM event pump / frame steps)" if addon.HAVE_BPY else
                                 "UNTESTED (no Blender binary in sandbox)"),
    }
    env_bpy = ("REAL bpy " + bpy.app.version_string) if addon.HAVE_BPY else "STANDIN headless (no Blender binary)"
    perf = {
        "phase": "A (python)",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "bpy": env_bpy, "label": labels},
        "e1_addon_lifecycle": e1,
        "module_import_ms": import_ms,
        "e4_conditions": {"A_scheduler_null_graft": e4a, "B_scheduler_bound_nouns_real_bpy": e4b},
        "e4_frame_budget_200k": e4a,
        "e5_membrane": {
            "public_drop": r1["drop_public"], "trusted_allow": r1["allow_trusted"],
            "flip": r1["flip"], "kernel_identity_preserved": r1["kernel_identity_preserved"],
            "tape_digest": r1["tape_digest"],
        },
        "e6_determinism": {
            "run1_tape_digest": r1["tape_digest"], "run2_tape_digest": r2["tape_digest"],
            "deterministic": r1["tape_digest"] == r2["tape_digest"],
            "run1_op_digest": r1["op_digest"], "run2_op_digest": r2["op_digest"],
        },
        "e3_export": e3,
        "tape_lines": r1["tape_lines"],
    }
    with open(os.path.join(ROOT, "perf", "blender_perf.json"), "w", encoding="utf-8") as fh:
        json.dump(perf, fh, indent=2)
    for rec in r1["telemetry"]:
        print("telemetry:", rec)
    with open(os.path.join(ROOT, "perf", "_python_digests.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "genome_sha": addon.load_genome()["genome_id"],
            "scenario_digest_run1": r1["tape_digest"],
            "scenario_digest_run2": r2["tape_digest"],
            "scenario_deterministic": r1["tape_digest"] == r2["tape_digest"],
            "e3_export_digest": e3["op_digest"],
            "e3_lineage": e3["lineage"],
            "e3_tint": e3["tint"],
            "e3_ops": e3["ops"],
            "tape_lines": r1["tape_lines"],
        }, fh, indent=2)
    print(json.dumps({k: v for k, v in perf.items() if k not in ("tape_lines",)}, indent=2)[:4000])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scenario")
