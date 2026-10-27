"""Universal Cell Engine v1.6 — Tissue A: Blender addon (Layer 4 graft + wiring).

Layering (v1.6 sharded layout):
  Layer 1 Genome    — engine.py (immutable, content-addressed; zero host code)
  Layer 2 Arena     — engine.py (spec dtype body-of-record + contiguous hot mirror) [numpy]
  Layer 3 Scheduler — engine.py (metabolism over the hot mirror, deadline -> debt)
  Layer 4 Graft     — THIS FILE: verb -> native bpy call, bound nouns only (F2 fix:
                      graft cost is O(bound nouns), never O(cell count)).

REAL in any environment: everything above plus snapshot lifecycle.
If `import bpy` fails (no Blender binary), HAVE_BPY=False and the graft falls back
to a labeled _BpyStandin; perf numbers touching it are labeled STANDIN-bpy.
In v1.6 the primary proof runs INSIDE real Blender 4.2 (`blender --background
--python ...`), so the standin path is not used for the headline numbers.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .engine import (  # noqa: F401  (re-export the whole substrate API)
    ARENA_DTYPE, GENOME_PATH, MASK_PUBLIC_PATH, MASK_TRUSTED_PATH, SNAPSHOT_PATH,
    STATE_INITIAL, Arena, Mask, Membrane, MetabolicScheduler, OpLog, Tape, Tissue,
    canonical_json, f32, fmt6, fnv1a64, genome_id_of, load_genome, load_mask, op_line,
    uuid7,
)

try:
    import bpy  # real Blender

    HAVE_BPY = True
except Exception:
    HAVE_BPY = False

    class _Mat:  # STANDIN-bpy: mirrors the exact attribute path the real addon drives
        def __init__(self):
            self.diffuse_color = (0.8, 0.8, 0.8, 1.0)

    class _Slot:
        def __init__(self):
            self.material = _Mat()

    class _Obj:
        def __init__(self, name):
            self.name = name
            self.material_slots = [_Slot()]

    class _Data:
        def __init__(self):
            self.objects = {"Cube": _Obj("Cube")}

    class _Bpy:
        def __init__(self):
            self.data = _Data()

    bpy = _Bpy()  # type: ignore[assignment]


# ---------------------------------------------------------------- Layer 4: graft table
class GraftTable:
    """The ONLY host-specific code. verb -> native call, resolved once per bound noun.
    graft_table_blender = { "render.tint": Cube material diffuse_color write }
    Total graft cost is O(bound nouns) — here: one Cube. Never iterated per cell.
    """

    def __init__(self, obj_name: str = "Cube"):
        self.tissue = "blender"
        self.table = {}
        self.bound_nouns = []
        self._obj = None
        self._mat = None
        if HAVE_BPY:
            scene_objs = bpy.context.scene.objects if hasattr(bpy, "context") else {}
            self._obj = scene_objs.get(obj_name) or bpy.data.objects.get(obj_name)
            if self._obj is not None:
                self._mat = self._ensure_material(self._obj)
                self.table["render.tint"] = self._apply_tint
                self.bound_nouns.append(self._obj.name)
        else:  # STANDIN-bpy headless mode
            self._obj = bpy.data.objects.get(obj_name)
            if self._obj is not None:
                self._mat = self._obj.material_slots[0].material
                self.table["render.tint"] = self._apply_tint
                self.bound_nouns.append(obj_name)

    @staticmethod
    def _ensure_material(obj):
        if obj.material_slots and obj.material_slots[0].material:
            return obj.material_slots[0].material
        mat = bpy.data.materials.new("CellTint")
        try:
            mat.use_nodes = True
        except Exception:
            pass
        obj.data.materials.append(mat)
        return mat

    def _apply_tint(self, tint: float) -> None:
        v = f32(tint)
        self._mat.diffuse_color = (v, v, v, 1.0)  # the exact bpy mutation path

    def apply(self, verb: str, *args) -> bool:
        fn = self.table.get(verb)
        if fn is None:
            return False
        fn(*args)
        return True


class NullGraft:
    """Perf arena tissue: scheduler-only, zero bound nouns (graft cost measured separately)."""

    tissue = "none"
    table = {}
    bound_nouns = []

    def apply(self, verb: str, *args) -> bool:
        return False


# ---------------------------------------------------------------- tissue (runtime glue)
_tissue: Tissue | None = None


def get_tissue() -> Tissue:
    global _tissue
    if _tissue is None:
        genome = load_genome()
        mask = load_mask(MASK_PUBLIC_PATH)
        _tissue = Tissue(genome, mask, GraftTable(), name="blender")
        if os.path.exists(SNAPSHOT_PATH):
            _tissue.regenerate_from(SNAPSHOT_PATH)  # re-enable / .blend reload path
        else:
            _tissue.ensure_cell(uuid7())  # fresh lineage, persisted on unregister
    return _tissue


if HAVE_BPY:

    class CELL_PT_tissue(bpy.types.Panel):
        bl_idname = "CELL_PT_tissue"
        bl_label = "Cell Tissue"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Cell Engine"

        def draw(self, context):
            t = get_tissue()
            col = self.layout.column()
            col.label(text=f"genome  {t.genome['genome_id'][:22]}…")
            col.label(text=f"lineage {t.lineage}")
            col.label(text=f"tint    {float(t.arena.tint[t.slot]):.3f}")
            col.label(text=f"mask    epoch={t.membrane.mask.epoch} {t.membrane.mask.audience}")
            col.label(text=f"ops     {len(t.oplog.ops)}")
            r = t.last_report
            col.label(text=f"tick    exec={r.get('executed', 0)} debt={r.get('deferred', 0)}")

    def _frame() -> float:  # bpy.app.timers callback, 60 Hz — host frame never blocks
        t = get_tissue()
        rng = np.random.default_rng()
        t.scheduler_tick(budget_us=1500, impulses=0)
        t.cell_tick(0)
        return 1.0 / 60.0

    def _on_load_post(_dummy=None):  # .blend reload -> regenerate from snapshot
        global _tissue
        _tissue = None
        get_tissue()


def register() -> float:
    t0 = time.perf_counter()
    get_tissue()
    if HAVE_BPY:
        try:
            bpy.utils.register_class(CELL_PT_tissue)
        except ValueError:
            pass  # already registered (addon auto-refetch after manual disable/enable)
        if _on_load_post not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(_on_load_post)
        if not bpy.app.timers.is_registered(_frame):
            bpy.app.timers.register(_frame, first_interval=0.0)
    return time.perf_counter() - t0


def unregister() -> None:
    global _tissue
    if _tissue is not None:
        _tissue.export_snapshot(SNAPSHOT_PATH, "blender.addon")
    if HAVE_BPY:
        if bpy.app.timers.is_registered(_frame):
            bpy.app.timers.unregister(_frame)
        if _on_load_post in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_on_load_post)
        try:
            bpy.utils.unregister_class(CELL_PT_tissue)
        except Exception:
            pass
    _tissue = None
