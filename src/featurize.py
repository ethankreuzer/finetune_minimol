"""Featurize the whole subset once and cache it, in CSV row order.

SMILES -> PyG graph is a pure function of the string: it does not depend on model weights,
fold, seed, or epoch, so the same answer is recomputed every epoch of every run unless it
is cached. Measured on this box: featurization runs at ~775 mol/s (graphium's featurizer
sets `featurization_n_jobs = -1`, so it already uses all cores), while a training epoch
over 265k rows takes 1.7 min. Re-featurizing per epoch would therefore spend ~85% of
training recomputing an unchanging answer. Hence NOTES §7 Phase 2: featurize once.

Two decisions worth stating, both measured rather than assumed (see `meta.json`):

*Format* -- the graphs are stored PyG-collated, as the `(data, slices)` pair that
`InMemoryDataset.collate` produces, not as a list of `Data` objects. Both were benchmarked
on 10,000 molecules and extrapolated:

    list-of-Data            5.44 GB   load 81.7 s
    collated (data,slices)  4.81 GB   load  1.0 s      <- this

An 80x faster load matters because the 5-fold x 2-seed grid is 10 processes, each paying
the load once. `collate` -> `separate` was verified to round-trip every key of every graph
exactly, and to yield bit-identical embeddings.

*Contents* -- graphs only, no target. `score`, `pprop` and `ipw` are read from the CSV at
training time by row index. Keeping the target out means the choice between them (NOTES §1,
still open) does not invalidate this cache.

The cache is written in **exact CSV row order**. That is what lets `splits.py` indices --
which are positions into that row order -- index the cache directly, with no mapping layer.
`meta.json` records the CSV's sha256 and `features.py` re-hashes on load and refuses a
mismatch, the same guard `splits.py` applies for the same reason: `subset.py` takes a
`--seed`, so a regenerated CSV is a different 331k set and silently invalidates this cache.

Usage:
    python src/featurize.py
    python src/featurize.py --limit 5000 -o data/features/smoke
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import _sha256_file  # noqa: E402  (same repo, reused rather than duplicated)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("-o", "--out", type=Path, default=Path("data/features/minimol_v1"),
                   help="output directory; graphs.pt and meta.json are written into it")
    p.add_argument("--chunk", type=int, default=25000,
                   help="molecules per featurization call; bounds peak memory and gives "
                        "progress. Does not affect the result.")
    p.add_argument("--limit", type=int, default=None,
                   help="featurize only the first N rows (smoke test). Recorded in "
                        "meta.json, which makes the partial cache self-identifying.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    started_all = time.time()
    print(f"hashing {args.input} ...", flush=True)
    input_sha = _sha256_file(args.input)

    df = pd.read_csv(args.input, usecols=["SMILES"])
    smiles = df["SMILES"].tolist()
    n_rows_csv = len(smiles)
    if args.limit is not None:
        smiles = smiles[:args.limit]
    print(f"{len(smiles):,} molecules to featurize (CSV has {n_rows_csv:,} rows)")

    # Imported here, not at module scope: constructing the trunk pulls in graphium and
    # hydra, which is slow and pointless if the arguments are wrong.
    from torch_geometric.data import InMemoryDataset
    from trunk import MiniMolTrunk

    started = time.time()
    trunk = MiniMolTrunk()
    print(f"constructed MiniMolTrunk in {time.time() - started:.1f}s "
          f"(featurization_n_jobs = {getattr(trunk.datamodule, 'featurization_n_jobs', '?')})")

    graphs = []
    started = time.time()
    for lo in range(0, len(smiles), args.chunk):
        chunk = smiles[lo:lo + args.chunk]
        # featurize_raw raises on a SMILES that fails -- graphium returns the error *string*
        # in place of the graph, which would otherwise become a silent hole in the cache and
        # shift every subsequent row out of alignment with the CSV.
        graphs.extend(trunk.featurize_raw(chunk))
        done = len(graphs)
        rate = done / (time.time() - started)
        eta = (len(smiles) - done) / rate if rate else float("nan")
        print(f"  {done:>7,} / {len(smiles):,}  ({rate:,.0f} mol/s, eta {eta/60:4.1f} min)",
              flush=True)

    elapsed = time.time() - started
    print(f"featurized {len(graphs):,} graphs in {elapsed/60:.1f} min "
          f"({len(graphs)/elapsed:,.0f} mol/s)")

    # The alignment invariant this whole cache rests on. A short cache would silently
    # mis-index every fold, so assert rather than trust the loop above.
    expected = len(smiles)
    if len(graphs) != expected:
        raise SystemExit(f"featurized {len(graphs)} graphs for {expected} SMILES -- "
                         "row alignment with the CSV is broken, refusing to write")

    args.out.mkdir(parents=True, exist_ok=True)
    graphs_path = args.out / "graphs.pt"

    started = time.time()
    data, slices = InMemoryDataset.collate(graphs)
    torch.save((data, slices), graphs_path)
    print(f"wrote {graphs_path} "
          f"({graphs_path.stat().st_size / 1e9:.2f} GB) in {time.time() - started:.1f}s")

    # Record the schema, so a loader can detect a cache built by a different featurizer
    # version without unpickling and inspecting it by hand.
    schema = {k: (list(v.shape[1:]) if torch.is_tensor(v) and v.dim() > 1 else "scalar")
              for k, v in sorted(graphs[0].to_dict().items())}

    import graphium
    import minimol
    import rdkit
    meta = {
        "script": "src/featurize.py",
        "argv": sys.argv,
        "input": str(args.input),
        "input_abspath": str(args.input.resolve()),
        "input_sha256": input_sha,
        "input_rows": n_rows_csv,
        "limit": args.limit,
        "n_graphs": len(graphs),
        "row_order": "exact CSV row order; splits.py indices apply directly",
        "graphs_file": graphs_path.name,
        "graphs_sha256": _sha256_file(graphs_path),
        "graphs_bytes": graphs_path.stat().st_size,
        "feature_schema": schema,
        "featurize_seconds": round(elapsed, 1),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "graphium": graphium.__version__,
            "minimol": getattr(minimol, "__version__", "1.3.4"),
            "rdkit": rdkit.__version__,
        },
        "host": platform.node(),
    }
    meta_path = args.out / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}")
    print(f"total {(time.time() - started_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
