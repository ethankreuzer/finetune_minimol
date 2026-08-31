"""A frozen molecular encoder: SMILES -> 512-d vector, fine-tuned on AmpC docking scores.

    from minimol_ampc import MiniMolAmpcEncoder

    enc = MiniMolAmpcEncoder.load("model/final.pt")
    emb = enc.encode(["CCO", "c1ccccc1"])        # (2, 512) float32

See README.md for installation and the active-learning recipe, and MODEL_CARD.md for what
this model was trained on and what its numbers are worth.

A note on imports, because this package does something unusual and you should not have to
discover it the hard way
------------------------------------------------------------------------------------------
`trunk.py`, `head.py`, `model.py` and `normalization.py` are vendored **byte-identically**
from the source repository -- `meta.json` records their sha256 sums, so you can diff them
against that repository and see nothing. Byte-identity is worth having: it is what lets you
prove no one edited the model's definition on the way out. The price is that they import each
other **flatly** (`model.py` does `from head import MLPHead`), which needs this directory on
`sys.path` and needs bare names like `model` in `sys.modules`.

`model` and `featurize` are generic enough that your project may well have its own. So this
module borrows those names and gives them back:

  1. anything you already have under one of those names is set aside,
  2. this directory goes on `sys.path` and the vendored modules are imported,
  3. `sys.path` is restored, the bare names are released, and your modules are put back.

The imported modules keep working afterwards -- they hold direct references to each other,
not name lookups. So **import order does not matter**, nothing of yours is shadowed, and
`import model` after this still gets yours. The modules remain reachable as
`minimol_ampc.trunk`, `minimol_ampc.model`, and so on.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# The vendored flat-import modules, plus this package's own two. One list, because the
# borrow and the give-back must cover exactly the same set.
_FLAT_MODULES = ("trunk", "head", "model", "normalization", "featurize", "encoder")

# Step 1: set aside anything of the importer's that already holds one of these names.
_displaced = {n: sys.modules.pop(n) for n in _FLAT_MODULES if n in sys.modules}

# Step 2: borrow the path and import. `encoder` pulls in the other five.
sys.path.insert(0, str(_HERE))
try:
    import encoder as _encoder
    import featurize as _featurize

    # Step 3: give everything back. Taken in `finally` so a failed import cannot leave the
    # importer's `sys.path` and `sys.modules` in a state this package changed -- a partially
    # applied import is confusing enough without also being sticky.
finally:
    # EVERY occurrence, not one. `encoder.py` and `featurize.py` each insert this directory
    # themselves so they remain usable when imported directly, so by now there are three
    # copies -- and removing only the first leaves this directory at the front of `sys.path`,
    # where `minimol_ampc/model.py` shadows any `model.py` of the importer's. That is the
    # exact silent collision the borrow-and-return exists to prevent.
    _here = str(_HERE)
    sys.path[:] = [entry for entry in sys.path if entry != _here]
    for _name in _FLAT_MODULES:
        _ours = sys.modules.pop(_name, None)
        if _ours is not None:
            # Still reachable under the package, which is how the docs spell it.
            sys.modules[f"{__name__}.{_name}"] = _ours
    sys.modules.update(_displaced)

featurize = _featurize
encoder = _encoder
EncodeResult = _encoder.EncodeResult
MiniMolAmpcEncoder = _encoder.MiniMolAmpcEncoder

__all__ = ["MiniMolAmpcEncoder", "EncodeResult", "encoder", "featurize"]
