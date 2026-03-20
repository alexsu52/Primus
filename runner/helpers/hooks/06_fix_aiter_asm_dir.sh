#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: fix amd-aiter AITER_ASM_DIR missing gfx prefix.
#
# Problem:
#   amd-aiter 0.0.0's core.py sets:
#       AITER_ASM_DIR = f"{AITER_META_DIR}/hsa/"        (no gfx)
#   but the C++ codegen reads this env var and appends "fmha_v3_fwd/MI300/",
#   producing a path like .../hsa/fmha_v3_fwd/MI300/ which does not exist.
#   The correct layout is .../hsa/{gfx}/fmha_v3_fwd/MI300/.
#
#   Newer versions (workspace aiter 0.1.6+) already include {gfx} in
#   AITER_ASM_DIR and are not affected.
#
# Fix:
#   Patch core.py in-place so AITER_ASM_DIR includes the hardware-detected
#   gfx prefix (via rocminfo), making it immune to GPU_ARCHS ordering issues.
#
###############################################################################

set -euo pipefail

if [[ "${FIX_AITER_ASM_DIR:-0}" != "1" ]]; then
    exit 0
fi

python3 << 'PYEOF'
import importlib.util, os, re, subprocess, sys

spec = importlib.util.find_spec("aiter")
if not spec or not spec.submodule_search_locations:
    sys.exit(0)

patched = False
for base in spec.submodule_search_locations:
    core_path = os.path.join(base, "jit", "core.py")
    if not os.path.isfile(core_path):
        continue

    with open(core_path) as f:
        content = f.read()

    old = 'AITER_ASM_DIR = f"{AITER_META_DIR}/hsa/"'
    if old not in content:
        continue

    # Detect actual GPU arch from hardware — immune to GPU_ARCHS ordering
    try:
        out = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"\b(gfx\w+)\b", out, re.IGNORECASE)
        gfx = m.group(1).lower() if m else None
    except Exception:
        gfx = None

    if not gfx:
        print("[fix_aiter_asm_dir] Cannot detect GPU arch from rocminfo, skipping", file=sys.stderr)
        continue

    new = f'AITER_ASM_DIR = f"{{AITER_META_DIR}}/hsa/{gfx}/"'

    content = content.replace(old, new)

    try:
        with open(core_path, "w") as f:
            f.write(content)
        patched = True
        print(f"[fix_aiter_asm_dir] Patched {core_path}: AITER_ASM_DIR -> .../hsa/{gfx}/")
    except PermissionError:
        print(f"[fix_aiter_asm_dir] Cannot write {core_path} (read-only)", file=sys.stderr)

if not patched:
    print("[fix_aiter_asm_dir] No patch needed (core.py already correct or not found)")
PYEOF
