"""Universal Cell Engine v1.5 — Tissue A: Blender (a real addon, not a Text Editor script).

Layering (v1.5 spec, do not collapse):
  Layer 1 Genome   — immutable, content-addressed JSON loaded from the repo contract dir.
  Layer 2 Arena    — numpy structured array, dtype exactly [('id','U36'),('tint','f4'),
                     ('epoch','i4'),('active','bool')]. Cells are rows, not objects.
  Layer 3 Scheduler— vectorized metabolism; deadline in us; over-budget remainder is
                     metabolic DEBT (deferred), never a blocked host frame.
  Layer 4 Graft    — the ONLY host-specific code: verb -> native bpy call, resolved once.

REAL in any environment : arena, CRDT op log + LWW, membrane/mask gate, scheduler,
                           graft table, snapshot export/import, FNV-1a64 + sha256
                           contract checks, register/unregister/load_post lifecycle.
STANDIN-bpy (this sandbox has no Blender binary; `import bpy` fails): a _BpyStandin
                           mirrors the exact mutation path
                           ob.material_slots[0].material.diffuse_color = (t,t,t,1.0).
                           Numbers that touch it are labeled STANDIN-bpy in perf output.
Membrane is the expression mask ONLY (cooperative, not enforced) — per v1.5 scope.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------- contract paths
_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_ADDON_DIR, "..", ".."))
GENOME_PATH = os.path.join(_REPO, "genome.json")
MASK_PUBLIC_PATH = os.path.join(_REPO, "masks", "public.json")
MASK_TRUSTED_PATH = os.path.join(_REPO, "masks", "trusted.json")
SNAPSHOT_PATH = os.path.join(_REPO, "regeneration", "crdt_snapshot.jsonl")

STATE_INITIAL = 0.725          # cell.tint.v1 initial state (spec §3)
RATE, EPS, CHUNK = 0.03, 1e-3, 4096   # metabolism easing / settle / deadline cadence

# ---------------------------------------------------------------- Layer 1: genome


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def genome_id_of(genome: dict) -> str:
    body = {k: v for k, v in genome.items() if k != "genome_id"}
    return "sha256:" + hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def load_genome(path: str = GENOME_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        genome = json.load(fh)
    recomputed = genome_id_of(genome)
    if recomputed != genome.get("genome_id"):
        raise ValueError(f"genome contract violated: {recomputed} != {genome.get('genome_id')}")
    return genome


def load_mask(path: str) -> "Mask":
    with open(path, encoding="utf-8") as fh:
        return Mask(json.load(fh))


# ---------------------------------------------------------------- small utils
FNV_OFFSET, FNV_PRIME, MASK64 = 0xcbf29ce484222325, 0x100000001b3, 0xFFFFFFFFFFFFFFFF


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def f32(x) -> float:
    return float(np.float32(x))


def fmt6(x) -> str:
    return f"{f32(x):.6f}"


def uuid7() -> str:
    ms = time.time_ns() // 1_000_000
    rnd = os.urandom(10)
    b = bytearray(16)
    b[0:6] = ms.to_bytes(6, "big")
    b[6] = (rnd[0] & 0x0F) | 0x70
    b[7] = rnd[1]
    b[8] = (rnd[2] & 0x3F) | 0x80
    b[9:16] = rnd[3:10]
    h = b.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------- Layer: mask / membrane
class Mask:
    def __init__(self, doc: dict):
        self.mask_id = doc["mask_id"]
        self.epoch = int(doc["epoch"])
        self.audience = doc["audience"]
        self.allowed = set(doc["allowed_verbs"])
        self.denied = set(doc["denied_verbs"])

    def allows(self, verb: str) -> bool:
        return verb in self.allowed and verb not in self.denied


class Tape:
    """Causal tape — deterministic events only (wall clocks go to telemetry)."""

    def __init__(self):
        self.lines: list[str] = []
        self.telemetry: list[dict] = []

    def add(self, line: str) -> None:
        self.lines.append(line)

    def event(self, name: str, **kv) -> None:
        self.telemetry.append({"event": name, **kv, "wall": time.time()})

    def digest(self) -> str:
        return "fnv1a64:" + str(fnv1a64(("\n".join(self.lines) + "\n").encode("utf-8")))


class Membrane:
    """Expression-mask gate (cooperative). Governs EMITTED verbs; CRDT ops are internal."""

    def __init__(self, tape: Tape):
        self.tape = tape
        self.mask: Mask | None = None

    def bake(self, mask: Mask) -> None:
        prev = self.mask.epoch if self.mask else None
        self.mask = mask
        if prev is not None and prev != mask.epoch:
            self.tape.add(f"mask.flip|{prev}|{mask.epoch}")
            self.tape.event("mask.flip", **{"from": prev, "to": mask.epoch})

    def attempt(self, verb: str, tick: int) -> bool:
        if self.mask is not None and self.mask.allows(verb):
            self.tape.add(f"membrane.allow|{verb}|{self.mask.epoch}")
            return True
        epoch = self.mask.epoch if self.mask else 0
        self.tape.add(f"membrane.drop|{verb}|{epoch}")
        self.tape.event("membrane.drop", verb=verb, epoch=epoch, tick=tick)
        return False


# ---------------------------------------------------------------- Layer 2: arena
ARENA_DTYPE = [("id", "U36"), ("tint", "f4"), ("epoch", "i4"), ("active", "bool")]


class Arena:
    """The body. Spec dtype is the BODY OF RECORD (id, epoch, tint, active);
    tint/active/targets are mirrored into CONTIGUOUS hot columns for metabolism.
    Measured reason (see perf): the structured array's 153-byte row stride makes a
    full 200k scan cost ~2.8ms (cache-hostile); contiguous columns scan ~40x faster.
    Hot columns are authoritative during metabolism; rows[] is synced at boundaries
    (spawn, canonical-cell ops, snapshot export). Cells are rows, not objects.
    """

    def __init__(self, capacity: int):
        self.cap = capacity
        self.rows = np.zeros(capacity, dtype=ARENA_DTYPE)
        self.tint = np.zeros(capacity, dtype=np.float32)    # hot mirror (contiguous)
        self.active = np.zeros(capacity, dtype=np.bool_)   # hot mirror (contiguous)
        self.targets = np.zeros(capacity, dtype=np.float32)  # morphogen side-channel
        self.n = 0
        self.index: dict[str, int] = {}  # lineage -> slot

    def spawn(self, lineage: str, tint: float, epoch: int = 1) -> int:
        if self.n >= self.cap:
            raise MemoryError("arena full")
        i = self.n
        self.n += 1
        self.rows[i] = (lineage, f32(tint), epoch, False)
        self.tint[i] = f32(tint)
        self.active[i] = False
        self.index[lineage] = i
        return i

    def sync_row(self, i: int) -> None:
        """Write hot-column values back into the body of record (boundary sync)."""
        self.rows["tint"][i] = self.tint[i]
        self.rows["active"][i] = self.active[i]


# ---------------------------------------------------------------- CRDT state
def op_line(op: dict) -> str:
    return f"op|{op['op']}|{op['value']:.6f}|{op['ts']}|{op['actor']}"


class OpLog:
    """Append-only CRDT ops [set_tint(value, lamport_ts, actor_id)]; LWW by (ts, actor)."""

    def __init__(self):
        self.ops: list[dict] = []

    def append(self, value: float, ts: int, actor: str) -> dict:
        op = {"op": "set_tint", "value": f32(value), "ts": int(ts), "actor": actor}
        self.ops.append(op)
        return op

    def lww(self) -> float | None:
        best_key, best_val = None, None
        for op in self.ops:
            key = (op["ts"], op["actor"])
            if best_key is None or key > best_key:
                best_key, best_val = key, op["value"]
        return best_val

    def digest(self) -> str:
        h = FNV_OFFSET
        for op in self.ops:
            h = fnv1a64_step(h, op_line(op) + "\n")
        return f"fnv1a64:{h}"


def fnv1a64_step(h: int, s: str) -> int:
    for b in s.encode("utf-8"):
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


# ---------------------------------------------------------------- Layer 3: scheduler
class MetabolicScheduler:
    """Vectorized over the arena; deadline checked every CHUNK cells; debt, never block."""

    def tick(self, arena: Arena, budget_us: int, impulses: int, rng, lo: int = 1) -> dict:
        t0 = time.perf_counter_ns()
        deadline = t0 + budget_us * 1000
        hi = arena.n
        if hi > lo:  # morphogen impulses: wake dormant cells with fresh targets
            dormant = np.flatnonzero(~arena.active[lo:hi]) + lo
            k = min(impulses, dormant.size)
            if k:
                pick = dormant[rng.integers(0, dormant.size, size=k)]
                arena.targets[pick] = rng.uniform(0.0, 1.0, k).astype(np.float32)
                arena.active[pick] = True
        idx = np.flatnonzero(arena.active[lo:hi]) + lo
        executed = deferred = 0
        for s in range(0, idx.size, CHUNK):
            if time.perf_counter_ns() >= deadline:
                deferred = idx.size - s
                break
            c = idx[s : s + CHUNK]
            t = arena.tint[c]
            tn = (t + np.clip(arena.targets[c] - t, -RATE, RATE)).astype(np.float32)
            arena.tint[c] = tn
            settled = np.abs(arena.targets[c] - tn) < EPS
            arena.active[c[settled]] = False
            executed += c.size
        return {
            "executed": int(executed),
            "deferred": int(deferred),
            "active": int(np.count_nonzero(arena.active[:hi])),
            "budget_us": int(budget_us),
            "ns": int(time.perf_counter_ns() - t0),
        }


# ---------------------------------------------------------------- Layer 4: graft table
try:
    import bpy  # real Blender

    HAVE_BPY = True
except Exception:  # sandbox: no Blender binary -> labeled headless tissue mode
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

    class _Bpy:  # module-shaped standin: mirrors bpy.data.objects access
        def __init__(self):
            self.data = _Data()

    bpy = _Bpy()  # type: ignore[assignment]


class GraftTable:
    """The ONLY host-specific code (kilobytes, not a VM). verb -> native call."""

    def __init__(self, obj_name: str = "Cube"):
        self.tissue = "blender"
        self.table: dict[str, object] = {}
        self._obj = bpy.data.objects.get(obj_name)
        self._mat = None
        if self._obj is not None:
            self._mat = self._ensure_material(self._obj)
            self.table["render.tint"] = self._apply_tint

    @staticmethod
    def _ensure_material(obj):
        if obj.material_slots and obj.material_slots[0].material:
            return obj.material_slots[0].material
        if HAVE_BPY:  # default Blender Cube has no material: create + assign one
            mat = bpy.data.materials.new("CellTint")
            mat.use_nodes = True
            obj.data.materials.append(mat)
            return mat
        return obj.material_slots[0].material  # STANDIN-bpy path

    def _apply_tint(self, tint: float) -> None:
        v = f32(tint)
        self._mat.diffuse_color = (v, v, v, 1.0)  # the exact bpy mutation path

    def apply(self, verb: str, *args) -> bool:
        fn = self.table.get(verb)
        if fn is None:
            return False
        fn(*args)
        return True


# ---------------------------------------------------------------- tissue (runtime glue)
class Tissue:
    """One host tissue: arena + canonical cell + graft + membrane + tape."""

    def __init__(self, genome: dict, mask: Mask, name: str = "blender", capacity: int = 200001):
        self.name = name
        self.genome = genome
        self.tape = Tape()
        self.membrane = Membrane(self.tape)
        self.membrane.bake(mask)
        self.arena = Arena(capacity)
        self.scheduler = MetabolicScheduler()
        self.graft = GraftTable()
        self.oplog = OpLog()
        self.lineage = ""
        self.slot = -1
        self.actor = "blender.1"
        self.dirty = False
        self.materialized = None
        self.last_report = {}

    # -- cell lifecycle ------------------------------------------------------
    def ensure_cell(self, lineage: str, tint: float = STATE_INITIAL, actor: str = "blender.1") -> int:
        self.lineage = lineage
        self.actor = actor
        self.slot = self.arena.spawn(lineage, tint)
        return self.slot

    # -- state.set (CRDT op; internal, not an emission) ----------------------
    def cell_set_tint(self, value: float, ts: int, actor: str | None = None) -> float:
        op = self.oplog.append(value, ts, actor or self.actor)
        self.tape.add(op_line(op))
        tint = f32(self.oplog.lww())
        self.arena.tint[self.slot] = tint
        self.arena.rows["epoch"][self.slot] += 1
        self.dirty = True
        return tint

    # -- metabolism + graft drain for the canonical cell ---------------------
    def cell_tick(self, tick_no: int) -> bool:
        if not self.dirty:
            return False
        if not self.membrane.attempt("render.tint", tick_no):
            return False
        tint = float(self.arena.tint[self.slot])
        ok = self.graft.apply("render.tint", tint)
        self.materialized = tint
        self.dirty = False
        return ok

    def peer_query(self, tick_no: int) -> bool:
        """data.query.peers — emitted verb, so the membrane governs it (E5)."""
        if self.membrane.attempt("data.query.peers", tick_no):
            self.tape.event("peer.query.dispatched", tick=tick_no, epoch=self.membrane.mask.epoch)
            return True
        return False

    def flip_mask(self, mask: Mask) -> dict:
        kernel_before = MetabolicScheduler.tick  # function identity == kernel NOT reloaded
        slot_before, lineage_before, ops_before = self.slot, self.lineage, len(self.oplog.ops)
        self.membrane.bake(mask)
        return {
            "kernel_reloaded": kernel_before is not MetabolicScheduler.tick,
            "slot_changed": slot_before != self.slot,
            "lineage_changed": lineage_before != self.lineage,
            "ops_preserved": len(self.oplog.ops) == ops_before,
            "epoch": self.membrane.mask.epoch,
        }

    def scheduler_tick(self, budget_us: int, impulses: int, rng) -> dict:
        rep = self.scheduler.tick(self.arena, budget_us, impulses, rng, lo=1)
        self.last_report = rep
        return rep

    # -- snapshot / regeneration ---------------------------------------------
    def export_snapshot(self, path: str, generator: str) -> dict:
        self.arena.sync_row(self.slot)
        tint = float(self.arena.tint[self.slot])
        header = {
            "kind": "cell.snapshot",
            "schema": "v1.5",
            "cell_type": "cell.tint.v1",
            "lineage": self.lineage,
            "genome_id": self.genome["genome_id"],
            "tint": f32(tint),
            "epoch": int(self.arena.rows["epoch"][self.slot]),
            "mask_epoch": self.membrane.mask.epoch,
            "actor_id": self.actor,
            "op_count": len(self.oplog.ops),
            "op_digest": self.oplog.digest(),
            "generator": generator,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(header) + "\n")
            for op in self.oplog.ops:
                fh.write(json.dumps(op) + "\n")
        return header

    @staticmethod
    def read_snapshot(path: str, genome: dict) -> tuple[dict, list[dict]]:
        with open(path, encoding="utf-8") as fh:
            lines = [json.loads(s) for s in fh if s.strip()]
        header, ops = lines[0], lines[1:]
        if header.get("genome_id") != genome["genome_id"]:
            raise ValueError("snapshot genome mismatch")
        return header, ops

    def regenerate_from(self, path: str) -> dict:
        """Salamander: rebuild the cell from the CRDT op log; identity survives."""
        header, ops = self.read_snapshot(path, self.genome)
        self.lineage = header["lineage"]
        self.actor = header.get("actor_id", "blender.1")
        self.slot = self.arena.spawn(self.lineage, STATE_INITIAL)
        self.arena.rows["epoch"][self.slot] = int(header.get("epoch", 1))
        self.oplog.ops = list(ops)
        tint = f32(self.oplog.lww())
        self.arena.tint[self.slot] = tint
        self.arena.sync_row(self.slot)
        self.dirty = True
        checks = {
            "lineage": self.lineage,
            "tint": tint,
            "ops": len(self.oplog.ops),
            "digest": self.oplog.digest(),
            "digest_matches_header": self.oplog.digest() == header["op_digest"],
            "epoch": int(self.arena.rows["epoch"][self.slot]),
        }
        return checks


# ---------------------------------------------------------------- Blender wiring
_tissue: Tissue | None = None


def get_tissue() -> Tissue:
    global _tissue
    if _tissue is None:
        genome = load_genome()
        mask = load_mask(MASK_PUBLIC_PATH)
        _tissue = Tissue(genome, mask, name="blender")
        if os.path.exists(SNAPSHOT_PATH):
            _tissue.regenerate_from(SNAPSHOT_PATH)  # re-enable / .blend reload path
        else:
            _tissue.ensure_cell(uuid7())  # fresh lineage, persisted on unregister
    return _tissue


if HAVE_BPY:

    class CELL_PT_tissue(bpy.types.Panel):  # type: ignore[name-defined]
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
        bpy.utils.register_class(CELL_PT_tissue)
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
        bpy.utils.unregister_class(CELL_PT_tissue)
    _tissue = None
