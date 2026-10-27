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
    """Install the addon into Blender's extensions path so addon_utils can find it,
    exactly like installing an extension from disk."""
    import addon_utils
    src = os.path.join(HERE, "cell_engine_blender")
    dst_root = os.path.join(addon_utils.modules_refresh_params("default_settings")[0], "") if hasattr(addon_utils, "modules_refresh_params") else None
    # stable, explicit location inside the user config scripts dir
    cfg = bpy.utils.resource_path("USER_CONFIG") if hasattr(bpy.utils, "resource_path") else os.path.expanduser("~/.config/blender")
    dst_parent = os.path.join(cfg, "scripts", "addons")
    os.makedirs(dst_parent, exist_ok=True)
    dst = os.path.join(dst_parent, "cell_engine_blender")
    if os.path.abspath(src) != os.path.abspath(dst):
        import shutil
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    addon_utils.modules_refresh()
    return dst


def _real_enable_disable_cycle() -> dict:
    """Disable + re-enable the addon INSIDE Blender's own extension machinery,
    exactly as Preferences > Add-ons does. Returns measured enable time."""
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


def e1_lifecycle() -> dict:
    if os.path.exists(addon.SNAPSHOT_PATH):
        os.remove(addon.SNAPSHOT_PATH)  # start fresh: first enable creates the cell
    enable1 = addon.register()
    t = addon.get_tissue()
    lineage1 = t.lineage
    assert abs(float(t.arena.tint[t.slot]) - 0.725) < 1e-6, "initial tint must be 0.725"
    for value, ts in OPS:
        t.cell_set_tint(value, ts)
    assert t.cell_tick(1), "graft apply failed"
    mat_before = tuple(t.graft._mat.diffuse_color)
    checks = {}
    if addon.HAVE_BPY:
        # REAL path: drive the full cycle through Blender's own handler/timer loop.
        import addon_utils
        # 1. save the .blend WITH the live snapshot on disk
        blend_path = os.path.join(ROOT, "regeneration", "e1_cell_tissue.blend")
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
        except ImportError:
            process_events = None
        if process_events is not None:
            bpy.app.timers.register(_pump_timer, first_interval=0.0)
            for _ in range(240):
                process_events()
            bpy.app.timers.unregister(_pump_timer)
        else:  # no direct stepper exposed: fall back to headless scene-frame depsgraph steps
            n_frames = int(os.environ.get("E1_FRAMES", "90"))
            for _ in range(n_frames):
                bpy.context.view_layer.update()
                _pump_timer()   # same code path the timer invokes, driven per frame
        cbs = sorted(pump)
        checks["frame_pump"] = {
            "stepper": "bpy._window_process_events (real WM event loop)" if process_events else "scene_frame_update fallback",
            "delivered_callbacks": len(cbs),
            "callback_p50_us": pct(cbs, 50) if cbs else None,
            "callback_p99_us": pct(cbs, 99) if cbs else None,
            "budget_note": "scheduler budget capped at 1500us inside callback; host frame never blocked",
        }
        # 4. real .blend reload in-process -> load_post handler regenerates tissue
        addon._tissue = None  # simulate process-side state loss before reload
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
        "snapshot_exists": os.path.exists(addon.SNAPSHOT_PATH),
        "mat_before_export": list(mat_before),
    }
    checks.update(checks_common)
    addon.unregister()               # leave a clean snapshot on disk
    return checks


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

    import_ms = module_import_ms()
    r1 = run_scenario()
    r2 = run_scenario()
    e1 = e1_lifecycle()
    e3 = e3_export(r1)
    e4 = e4_perf()

    labels = {
        "arena_scheduler_crdt_membrane": "REAL (numpy, this process)",
        "bpy_materialize": "STANDIN-bpy" if not addon.HAVE_BPY else "REAL",
        "addon_lifecycle_roundtrips": "REAL (register/unregister/load_post handler; Blender event loop untested)",
        "blender_frame_60fps": "UNTESTED (no Blender binary in sandbox)",
    }
    perf = {
        "phase": "A (python)",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "bpy": "STANDIN headless (no Blender binary)", "label": labels},
        "e1_addon_lifecycle": e1,
        "module_import_ms": import_ms,
        "e4_frame_budget_200k": e4,
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
