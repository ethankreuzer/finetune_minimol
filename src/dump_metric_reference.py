"""Record pProp_MLP's metric outputs on a fixed input, so the port can be checked against them.

Run this **under pProp_MLP's interpreter**, not this repo's -- that is the entire point. The
two projects pin incompatible stacks (graphium holds this one to scipy<1.14 and torch
2.6.0+cu124), so the only way to compare implementations is to run each under its own
environment and compare the numbers on disk.

    /home/ethan2/pProp_MLP/.venv/bin/python src/dump_metric_reference.py

Writes `data/reference/pprop_mlp_metrics_v1/{reference.json,meta.json}`, mirroring
`src/dump_reference_embeddings.py` -> `verify_trunk.py`. `src/verify_metrics.py` reads it.

The input is synthetic and seeded rather than a slice of the real CSV: it has to exercise
both classes and a wide pProp range in a few thousand rows, and it must not make the check
depend on a 331k-row file that the split guard already hashes elsewhere.
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

SOURCE_SRC = Path("/home/ethan2/pProp_MLP/src")

# Mirrors this repo's binary task. `groups=[0, 1]` makes pProp_MLP's grouped weighting the
# identity, which is what makes its `weighted_*` outputs directly comparable to our
# `*_balanced` ones.
CLASS_NAMES = ["lt3.5", "ge3.5"]
WEIGHT_GROUPS = [0, 1]
EDGE = 3.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", type=Path,
                   default=Path("data/reference/pprop_mlp_metrics_v1"))
    p.add_argument("-n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def make_inputs(n, seed):
    """A seeded target/prediction pair. Gamma-shaped to resemble the real pProp skew."""
    rng = np.random.default_rng(seed)
    y = np.clip(rng.gamma(1.4, 0.9, n), 0, 7.0)
    pred = y * 0.7 + rng.normal(0, 0.5, n)
    return y, pred


def main(argv=None):
    args = parse_args(argv)
    if not SOURCE_SRC.is_dir():
        raise SystemExit(f"pProp_MLP source not found at {SOURCE_SRC}")
    sys.path.insert(0, str(SOURCE_SRC))
    from metrics import (compute_class_mae_metrics,  # noqa: E402
                         compute_class_pearson_metrics)

    y, pred = make_inputs(args.n, args.seed)
    y_class = (y >= EDGE).astype(np.int64)

    mae = compute_class_mae_metrics(y, pred, y_class, CLASS_NAMES, groups=WEIGHT_GROUPS)
    pear = compute_class_pearson_metrics(y, pred, y_class, CLASS_NAMES, groups=WEIGHT_GROUPS)

    reference = {
        "mae": mae["mae"],
        "weighted_mae": mae["weighted_mae"],
        "mae_skill": mae["mae_skill"],
        "weighted_mae_skill": mae["weighted_mae_skill"],
        "pearson": pear["pearson"],
        "pearson_weighted": pear["pearson_weighted"],
        "n_positive": int(y_class.sum()),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=SOURCE_SRC.parent).stdout.strip() or None
    except Exception:
        sha = None
    (args.out / "meta.json").write_text(json.dumps({
        "script": "src/dump_metric_reference.py",
        "argv": sys.argv,
        "source_repo": str(SOURCE_SRC.parent),
        "source_git_sha": sha,
        "n": args.n, "seed": args.seed, "edge": EDGE,
        "class_names": CLASS_NAMES, "weight_groups": WEIGHT_GROUPS,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }, indent=2) + "\n")

    print(f"wrote {args.out}")
    for k, v in reference.items():
        print(f"  {k:22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
