"""The 20-line version: SMILES in, 512-d vectors and predicted pProp out.

    python examples/quickstart.py
"""

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from minimol_ampc import MiniMolAmpcEncoder     # noqa: E402

SMILES = [
    "CCO",                                       # ethanol
    "c1ccccc1",                                  # benzene
    "CC(=O)Oc1ccccc1C(=O)O",                     # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",              # caffeine
]


def main():
    # device=None picks cuda when it is available, cpu otherwise. Both work; the vectors
    # agree to ~7e-6, which is float arithmetic rather than a difference in the model.
    enc = MiniMolAmpcEncoder.load(PKG / "model" / "final.pt", device=None)
    print(f"loaded on {enc.device}\n")

    # THE EXPORT. One row per molecule, 512 columns, float32.
    emb = enc.encode(SMILES)
    print(f"enc.encode(...) -> {emb.shape} {emb.dtype}   <- the 512-d pooled embedding\n")

    # Everything else the model produces, from the same forward pass.
    out = enc.encode_all(SMILES)
    print(f"{'molecule':<32} {'pProp':>7} {'P(>=3.5)':>9}   first 3 dims of pooled512")
    for i, smi in enumerate(SMILES):
        p = 1.0 / (1.0 + pow(2.718281828459045, -float(out.logit[i])))
        dims = "  ".join(f"{v:+.4f}" for v in out.pooled512[i, :3])
        print(f"{smi:<32} {out.pprop[i]:>7.3f} {p:>9.4f}   {dims}")

    print(f"\nz (the head's shared layer) is also available: {out.z.shape} -- but see "
          "README.md,\n'Which vector to use'. pooled512 is the one to hand a GP kernel.")
    print("\npProp: higher = more potent, capped at 7.0. The classifier's threshold is "
          f"pProp >= {enc.pprop_edge}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
