"""Check that this package, in THIS environment, reproduces the vectors it was shipped with.

Run this first, before you build anything on top of the encoder:

    python verify_install.py

It loads the model, encodes the fixture molecules, and compares against the values recorded
when the package was built. It is not a smoke test -- it is the one check that separates
"installed" from "installed correctly", and there is a specific failure it exists to catch.

minimol's own loader calls `load_state_dict(..., strict=False)` and discards the result, so a
version skew that renames a single parameter yields a **partly randomly initialised trunk with
no error and no warning**. That model runs, returns plausible-looking vectors of the right
shape, and is worthless. `encoder.py` loads with `strict=True` to prevent it; this script is
the independent confirmation, because it compares numbers rather than key names.

WHY THE TOLERANCE IS 1e-4 AND NOT ZERO
--------------------------------------
The fixture was generated on CPU, where the computation is deterministic and repeatable
exactly. Yours may not be the same hardware. Measured in the source project:

    same CPU, twice                    0.000e+00
    CPU vs the reference it matches    0.000e+00
    CUDA vs the same CPU reference     ~7e-06        float arithmetic, not a wrong model
    a wrongly-loaded model             O(1)          i.e. ~5 orders of magnitude larger

So 1e-4 sits far above the hardware floor and far below any real reconstruction error. An
exact-match test would fail on a GPU for no reason; a loose one gives up no sensitivity,
because a broken load is not slightly wrong, it is random.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TOL = 1e-4


def main(argv=None):
    device = (argv or sys.argv[1:] or [None])[0]

    from minimol_ampc import MiniMolAmpcEncoder

    fixture = np.load(HERE / "fixtures" / "fixture.npz")
    info = json.loads((HERE / "fixtures" / "fixture.json").read_text())
    smiles = list(info["smiles"])
    print(f"fixture: {len(smiles)} molecules, generated on {info['generated_on']} "
          f"(torch {info['versions']['torch']}, device {info['device']})")

    enc = MiniMolAmpcEncoder.load(HERE / "model" / "final.pt", device=device)
    print(f"loaded {HERE / 'model' / 'final.pt'} on {enc.device} "
          f"| pooled512 -> 512-d, z -> {enc.embed_dim}-d")

    got = enc.encode_all(smiles)

    checks, worst = [], 0.0
    for key, expected in (("pooled512", fixture["pooled512"]),
                          ("z", fixture["z"]),
                          ("pprop", fixture["pprop"]),
                          ("logit", fixture["logit"])):
        actual = getattr(got, key)
        if actual.shape != expected.shape:
            print(f"[FAIL] {key}: shape {actual.shape}, expected {expected.shape}")
            checks.append(False)
            continue
        delta = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max())
        worst = max(worst, delta)
        ok = delta < TOL
        checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {key:<10} {str(expected.shape):<14} "
              f"max|Δ| = {delta:.3e}  (tol {TOL:.0e})")

    # A shape-and-tolerance check would still pass on a constant output, so pin the thing a
    # collapsed or zeroed model could not fake: the vectors have to actually vary.
    spread = float(got.pooled512.std())
    ok = spread > 1e-3
    checks.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] pooled512 varies   std = {spread:.4f} (want > 1e-3)")

    passed = all(checks)
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'} "
          f"({sum(checks)}/{len(checks)} checks, worst max|Δ| = {worst:.3e})")
    if not passed:
        print("\nDo not use this install. The most likely causes, in order:\n"
              "  1. a different graphium/minimol version -- this package needs graphium "
              "2.4.7 and minimol 1.3.4 exactly\n"
              "  2. a torch that does not match the pinned 2.6.0+cu124, which orphans the "
              "compiled torch-scatter/sparse/cluster extensions\n"
              "See README.md, 'Installing'.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
