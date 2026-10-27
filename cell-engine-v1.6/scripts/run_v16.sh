#!/usr/bin/env bash
# Universal Cell Engine v1.6 — full real-tissue proof: Phase A -> B -> C.
#   Phase A : E1/E3-export/E5/E6 + E4 condition C, inside REAL Blender 4.2 LTS
#             (bundled numpy) and/or system python3+numpy (headless STANDIN-bpy tier).
#   Phase B : Node/Tauri-tier harness (E3 replay, E4 SAB+worker, E5, E6, IPC proxy).
#   Rust    : cargo build --release of the src-tauri shard (E2), if a toolchain exists.
#   Phase C : python re-import of the node return snapshot (roundtrip leg).
# Optional env: BLENDER=/path/to/blender PY=python3 NPM_TEST=0 SKIP_RUST=1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER="${BLENDER:-blender}"
PY="${PY:-python3}"

have() { command -v "$1" >/dev/null 2>&1; }

echo "== preflight ==========================================================="
if have "$BLENDER"; then
  echo "blender : $("$BLENDER" --version | head -1)"
else
  echo "blender : ABSENT -> Phase A runs headless (STANDIN-bpy labels)"
fi
have "$PY"        && echo "python  : $("$PY" --version)"            || { echo "python3 required"; exit 1; }
"$PY" -c "import numpy" && echo "numpy   : $("$PY" -c 'import numpy;print(numpy.__version__)')"
have node         && echo "node    : $(node --version)"             || echo "node    : ABSENT -> skipping Phase B"
have cargo        && echo "cargo   : $(cargo --version)"            || echo "cargo   : ABSENT -> skipping Rust build (E2)"

echo "== phase A (Blender tissue: E1+E3 export+E4 cond C+E5+E6) =============="
if have "$BLENDER"; then
  "$BLENDER" --background --factory-startup --python "$HERE/blender_addon/test_blender.py" -- scenario
else
  ( cd "$HERE/blender_addon" && "$PY" test_blender.py scenario )
fi

echo "== phase B (Tauri/webview tier in Node: E2-proxy,E3,E4,E5,E6) =========="
if have node; then
  ( cd "$HERE/tauri-cell" && node --experimental-strip-types test_tauri.ts )
else
  echo "SKIPPED (no node)"
fi

echo "== rust shard build (E2: cargo build --release) ========================"
if have cargo && [ "${SKIP_RUST:-0}" != "1" ]; then
  ( cd "$HERE/tauri-cell/src-tauri" && cargo build --release )
  echo "rust shard: COMPILED -> $(ls -la "$HERE/tauri-cell/src-tauri/target/release/"tauri-cell* 2>/dev/null | head -1 || true)"
else
  echo "SKIPPED (no cargo or SKIP_RUST=1)"
fi

echo "== phase C (return leg: python re-import of tauri snapshot) ============"
if have "$BLENDER"; then
  "$BLENDER" --background --factory-startup --python "$HERE/blender_addon/test_blender.py" -- roundtrip
else
  ( cd "$HERE/blender_addon" && "$PY" test_blender.py roundtrip )
fi

echo "== done. evidence: perf/blender_perf.json perf/tauri_perf.json \\"
echo "           perf/regeneration_roundtrip.json perf/determinism_digest.txt"
