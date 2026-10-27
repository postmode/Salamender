"""Universal Cell Engine v1.6 — core substrate (Layer 1-3 + CRDT + membrane).

No host code here. `__init__.py` adds Layer 4 (graft table) and Blender wiring.

v1.6 changes vs v1.5:
  F1 hot/cold split is CANONICAL: spec dtype = body of record; metabolism runs on
     contiguous mirror columns (Float32 tint / Int32 epoch / Uint8 active).
  E3 falsifier: deterministic UUIDv7 generator so a scripted scenario carries ONE
     lineage across Blender -> Tauri -> Blender tissues (real processes).
  Snapshot schema tag: "v1.6". Digests: FNV-1a 64 over canonical lines.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

_ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_ADDON_DIR, "..", ".."))
GENOME_PATH = os.path.join(_REPO, "genome.json")
MASK_PUBLIC_PATH = os.path.join(_REPO, "masks", "public.json")
MASK_TRUSTED_PATH = os.path.join(_REPO, "masks", "trusted.json")
SNAPSHOT_PATH = os.path.join(_REPO, "regeneration", "crdt_snapshot.jsonl")

STATE_INITIAL = 0.725
RATE, EPS, CHUNK = 0.03, 1e-3, 4096


# ---------------------------------------------------------------- contract utils
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


FNV_OFFSET, FNV_PRIME, MASK64 = 0xcbf29ce484222325, 0x100000001b3, 0xFFFFFFFFFFFFFFFF


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def fnv1a64_step(h: int, s: str) -> int:
    for b in s.encode("utf-8"):
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def f32(x) -> float:
    return float(np.float32(x))


def fmt6(x) -> str:
    return f"{f32(x):.6f}"


def uuid7(ms: int | None = None, rand: bytes | None = None) -> str:
    """UUIDv7. Deterministic when ms+rand given (cross-language reproducible)."""
    if ms is None:
        ms = time.time_ns() // 1_000_000
    if rand is None:
        rand = os.urandom(10)
    b = bytearray(16)
    b[0:6] = int(ms).to_bytes(6, "big")
    b[6] = (rand[0] & 0x0F) | 0x70
    b[7] = rand[1]
    b[8] = (rand[2] & 0x3F) | 0x80
    b[9:16] = rand[3:10]
    h = b.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------- mask / membrane / tape
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
    """Causal tape — deterministic events only (wall clocks live in telemetry)."""

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


# ---------------------------------------------------------------- Layer 2: sharded arena
ARENA_DTYPE = [("id", "U36"), ("tint", "f4"), ("epoch", "i4"), ("active", "bool")]


class Arena:
    """Body of record = spec-dtype structured array (identity/audit).
    Hot mirror = CONTIGUOUS columns (tint f32 / epoch i32 / active u8) — what the
    scheduler iterates (canonical fix for finding F1: 153B row stride is cache hostile).
    Cells are rows, not objects.
    """

    def __init__(self, capacity: int):
        self.cap = capacity
        self.rows = np.zeros(capacity, dtype=ARENA_DTYPE)   # cold: body of record
        self.tint = np.zeros(capacity, dtype=np.float32)    # hot mirror
        self.epoch = np.zeros(capacity, dtype=np.int32)     # hot mirror
        self.active = np.zeros(capacity, dtype=np.uint8)    # hot mirror
        self.targets = np.zeros(capacity, dtype=np.float32)  # morphogen side-channel
        self.n = 0
        self.index: dict[str, int] = {}

    def spawn(self, lineage: str, tint: float, epoch: int = 1) -> int:
        if self.n >= self.cap:
            raise MemoryError("arena full")
        i = self.n
        self.n += 1
        self.rows[i] = (lineage, f32(tint), epoch, False)
        self.tint[i] = f32(tint)
        self.epoch[i] = epoch
        self.active[i] = 0
        self.index[lineage] = i
        return i

    def sync_row(self, i: int) -> None:
        """Boundary sync: hot columns -> body of record."""
        self.rows["tint"][i] = self.tint[i]
        self.rows["epoch"][i] = self.epoch[i]
        self.rows["active"][i] = bool(self.active[i])


# ---------------------------------------------------------------- CRDT state
def op_line(op: dict) -> str:
    return f"op|{op['op']}|{op['value']:.6f}|{op['ts']}|{op['actor']}"


class OpLog:
    """Append-only CRDT ops [set_tint(value, lamport_ts, actor)]; LWW by (ts, actor)."""

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


# ---------------------------------------------------------------- Layer 3: scheduler
class MetabolicScheduler:
    """Iterates the CONTIGUOUS hot mirror in index (topological) order.
    Deadline checked every CHUNK cells; over-budget remainder = metabolic DEBT.
    The host frame is never blocked.
    """

    def tick(self, arena: Arena, budget_us: int, impulses: int, rng, lo: int = 1) -> dict:
        t0 = time.perf_counter_ns()
        deadline = t0 + budget_us * 1000
        hi = arena.n
        if hi > lo:
            dormant = np.flatnonzero(arena.active[lo:hi] == 0) + lo
            k = min(impulses, dormant.size)
            if k:
                pick = dormant[rng.integers(0, dormant.size, size=k)]
                arena.targets[pick] = rng.uniform(0.0, 1.0, k).astype(np.float32)
                arena.active[pick] = 1
        idx = np.flatnonzero(arena.active[lo:hi] != 0) + lo
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
            arena.active[c[settled]] = 0
            executed += c.size
        return {
            "executed": int(executed),
            "deferred": int(deferred),
            "active": int(np.count_nonzero(arena.active[:hi])),
            "budget_us": int(budget_us),
            "ns": int(time.perf_counter_ns() - t0),
        }


# ---------------------------------------------------------------- tissue (host-agnostic glue)
class Tissue:
    """One tissue: arena + canonical cell + membrane + tape. The graft table is
    injected by the host layer (Layer 4); this class has zero host knowledge."""

    def __init__(self, genome: dict, mask: Mask, graft, name: str = "python",
                 capacity: int = 200001):
        self.name = name
        self.genome = genome
        self.tape = Tape()
        self.membrane = Membrane(self.tape)
        self.membrane.bake(mask)
        self.arena = Arena(capacity)
        self.scheduler = MetabolicScheduler()
        self.graft = graft          # Layer 4, injected (host-specific)
        self.oplog = OpLog()
        self.lineage = ""
        self.slot = -1
        self.actor = "python.1"
        self.dirty = False
        self.materialized = None
        self.last_report: dict = {}

    def ensure_cell(self, lineage: str, tint: float = STATE_INITIAL, actor: str | None = None) -> int:
        self.lineage = lineage
        if actor:
            self.actor = actor
        self.slot = self.arena.spawn(lineage, tint)
        return self.slot

    def cell_set_tint(self, value: float, ts: int, actor: str | None = None) -> float:
        op = self.oplog.append(value, ts, actor or self.actor)
        self.tape.add(op_line(op))
        tint = f32(self.oplog.lww())
        self.arena.tint[self.slot] = tint
        self.arena.epoch[self.slot] += 1
        self.dirty = True
        return tint

    def cell_tick(self, tick_no: int) -> bool:
        """Drain the canonical cell's bound verb through membrane -> graft."""
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
        if self.membrane.attempt("data.query.peers", tick_no):
            self.tape.event("peer.query.dispatched", tick=tick_no, epoch=self.membrane.mask.epoch)
            return True
        return False

    def flip_mask(self, mask: Mask) -> dict:
        kernel_before = MetabolicScheduler.tick
        arena_before = self.arena
        slot_before, lineage_before = self.slot, self.lineage
        ops_before = len(self.oplog.ops)
        self.membrane.bake(mask)
        return {
            "kernel_reloaded": kernel_before is not MetabolicScheduler.tick,
            "arena_identity_preserved": arena_before is self.arena,
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
        header = {
            "kind": "cell.snapshot",
            "schema": "v1.6",
            "cell_type": "cell.tint.v1",
            "lineage": self.lineage,
            "genome_id": self.genome["genome_id"],
            "tint": f32(float(self.arena.tint[self.slot])),
            "epoch": int(self.arena.epoch[self.slot]),
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
        self.actor = header.get("actor_id", self.actor)
        self.slot = self.arena.spawn(self.lineage, STATE_INITIAL)
        self.arena.epoch[self.slot] = int(header.get("epoch", 1))
        self.oplog.ops = list(ops)
        tint = f32(self.oplog.lww())
        self.arena.tint[self.slot] = tint
        self.arena.sync_row(self.slot)
        self.dirty = True
        return {
            "lineage": self.lineage,
            "tint": tint,
            "ops": len(self.oplog.ops),
            "digest": self.oplog.digest(),
            "digest_matches_header": self.oplog.digest() == header["op_digest"],
            "epoch": int(self.arena.epoch[self.slot]),
        }
