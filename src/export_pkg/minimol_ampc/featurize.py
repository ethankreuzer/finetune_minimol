"""SMILES -> PyG graphs, cached to disk. The step before the model, and it is not optional.

MiniMol does not consume SMILES. It consumes a molecular graph with precomputed node, edge and
positional-encoding features, built by graphium's featurizer. `MiniMolAmpcEncoder.encode` does
that conversion for you every call -- which is the right thing for a one-shot script and the
wrong thing for an active-learning loop, because SMILES -> graph is a pure function of the
string. It does not depend on the weights, so re-deriving it every round is pure waste.

Measured in the source project: featurization runs at ~775-1,745 mol/s, while a training epoch
over 265k graphs takes ~1.7 min. Featurizing per epoch would have spent ~85% of training
recomputing an unchanging answer. The same arithmetic applies to a scoring loop.

    from minimol_ampc import featurize

    featurize.build_cache(pool_smiles, "cache/round0")     # once, CPU-bound
    graphs, meta = featurize.load_cache("cache/round0")    # ~1 s thereafter
    emb = enc.encode_graphs(graphs)                        # every round

Or from the command line:

    python -m minimol_ampc.featurize --smiles pool.txt --out cache/round0

Two decisions carried over from the source project, both measured rather than assumed:

*Format.* Graphs are stored PyG-**collated** -- the `(data, slices)` pair, a few large
concatenated tensors plus a slice index -- not as a list of `Data`. Benchmarked on 10,000
molecules and extrapolated to the 331k set:

    list-of-Data            5.44 GB   load 81.7 s
    collated (data, slices) 4.81 GB   load  1.0 s     <- this

*Parallelism.* Do **not** wrap this in a process pool. graphium sets
`featurization_n_jobs = -1` and already uses every core; a second pool oversubscribes the
machine and runs slower than the single call.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data.separate import separate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trunk import MiniMolTrunk, cast_features    # noqa: E402

GRAPHS_FILE = "graphs.pt"
SMILES_FILE = "smiles.txt"
META_FILE = "meta.json"


class GraphCache(Dataset):
    """Random-access view over the collated cache; `ds[i]` rebuilds graph `i` on demand.

    `separate` was verified in the source project to round-trip every key of every graph
    exactly and to yield bit-identical embeddings, so the 80x load saving costs nothing.

    `__getitem__` returns a FRESH `Data` each call, which is load-bearing: the trunk's forward
    pass rewrites `feat` on the collated *batch* (the positional encoders concatenate into
    it), so a batch is single-use. The source graphs are not mutated -- `Batch.from_data_list`
    copies -- so one cache can be served for every round of a loop with no defensive copying.
    """

    def __init__(self, data, slices, smiles=None, meta=None):
        self._data, self._slices = data, slices
        self.smiles, self.meta = smiles, meta or {}
        self._len = len(next(iter(slices.values()))) - 1

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._len
        if not 0 <= idx < self._len:
            raise IndexError(f"index {idx} out of range for {self._len} graphs")
        return separate(cls=self._data.__class__, batch=self._data, idx=idx,
                        slice_dict=self._slices, decrement=False)


def _sha256_lines(lines):
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()


def build_cache(smiles, out_dir, chunk=25000, on_invalid="raise", trunk=None, quiet=False):
    """Featurize `smiles` once and write a reloadable cache. Returns the meta dict.

    Row order is the order you passed in, and is asserted rather than assumed -- a dropped
    molecule would shift every subsequent row out of alignment with your labels, which is a
    silent, total corruption of anything you compute afterwards.

    `on_invalid="skip"` drops unfeaturizable SMILES and records them in `meta.json` under
    `invalid`; the cache then holds `n_graphs` rows against your `n_input` inputs, and
    `load_cache` hands back the kept SMILES so you can realign. `"raise"` (the default)
    refuses to write a cache at all.

    Pass `trunk` to reuse an encoder's already-constructed trunk
    (`enc.model.trunk`) and skip ~30 s of graphium construction.
    """
    if on_invalid not in ("raise", "skip"):
        raise ValueError(f"on_invalid must be 'raise' or 'skip', got {on_invalid!r}")
    smiles = [smiles] if isinstance(smiles, str) else list(smiles)
    out_dir = Path(out_dir)
    say = (lambda *a, **k: None) if quiet else print

    own_trunk = trunk is None
    if own_trunk:
        started = time.time()
        trunk = MiniMolTrunk()
        say(f"constructed MiniMolTrunk in {time.time() - started:.1f}s "
            f"(featurization_n_jobs = "
            f"{getattr(trunk.datamodule, 'featurization_n_jobs', '?')}; do not add a "
            f"second process pool)")

    import os
    from contextlib import redirect_stderr, redirect_stdout

    graphs, kept_smiles, invalid = [], [], []
    started = time.time()
    for lo in range(0, len(smiles), chunk):
        block = smiles[lo:lo + chunk]
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            feats, _ = trunk.datamodule._featurize_molecules(block)
        for s, f in zip(block, feats):
            # graphium hands back the error STRING rather than raising, so this branch is
            # the only thing standing between a bad SMILES and a silent hole in the cache.
            if isinstance(f, str):
                invalid.append(s)
                continue
            graphs.append(f)
            kept_smiles.append(s)
        done = lo + len(block)
        rate = done / max(time.time() - started, 1e-9)
        say(f"  {done:>9,} / {len(smiles):,}  ({rate:,.0f} mol/s, "
            f"eta {(len(smiles) - done) / max(rate, 1e-9) / 60:5.1f} min)", flush=True)

    if invalid and on_invalid == "raise":
        raise ValueError(
            f"{len(invalid)} of {len(smiles)} SMILES could not be featurized, e.g. "
            f"{invalid[:3]}. Pass on_invalid='skip' to drop them; the cache then records "
            "which ones and load_cache returns the SMILES that survived.")

    graphs = cast_features(graphs)
    if len(graphs) != len(kept_smiles):
        raise RuntimeError(f"{len(graphs)} graphs for {len(kept_smiles)} SMILES -- row "
                           "alignment is broken, refusing to write")

    elapsed = time.time() - started
    say(f"featurized {len(graphs):,} graphs in {elapsed / 60:.1f} min "
        f"({len(graphs) / max(elapsed, 1e-9):,.0f} mol/s)")

    from torch_geometric.data import InMemoryDataset
    out_dir.mkdir(parents=True, exist_ok=True)
    data, slices = InMemoryDataset.collate(graphs)
    torch.save((data, slices), out_dir / GRAPHS_FILE)
    (out_dir / SMILES_FILE).write_text("\n".join(kept_smiles) + "\n")

    import graphium
    meta = {
        "n_input": len(smiles), "n_graphs": len(graphs),
        "invalid": invalid,
        "smiles_sha256": _sha256_lines(kept_smiles),
        "graphs_file": GRAPHS_FILE, "smiles_file": SMILES_FILE,
        "row_order": "the order build_cache was called with, minus any skipped SMILES",
        "featurize_seconds": round(elapsed, 1),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "graphium": graphium.__version__},
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    say(f"wrote {out_dir} ({(out_dir / GRAPHS_FILE).stat().st_size / 1e9:.2f} GB)")
    return meta


def load_cache(out_dir):
    """`(GraphCache, meta)`. The cache's own SMILES are on `cache.smiles`, in row order.

    The recorded sha256 of those SMILES is re-checked, for the same reason the source project
    re-hashes its CSV on load: a cache silently paired with the wrong molecule list attaches
    every graph to the wrong label, and every number downstream is then plausible and wrong.
    """
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / META_FILE).read_text())
    smiles = (out_dir / meta["smiles_file"]).read_text().splitlines()

    actual = _sha256_lines(smiles)
    if actual != meta["smiles_sha256"]:
        raise ValueError(f"{out_dir / meta['smiles_file']} has changed since the cache was "
                         f"built ({actual[:12]}... != {meta['smiles_sha256'][:12]}...). The "
                         "graphs are aligned to the old order and are now wrong.")

    data, slices = torch.load(out_dir / meta["graphs_file"], weights_only=False)
    cache = GraphCache(data, slices, smiles=smiles, meta=meta)
    if len(cache) != meta["n_graphs"] or len(smiles) != meta["n_graphs"]:
        raise ValueError(f"{out_dir} holds {len(cache)} graphs and {len(smiles)} SMILES but "
                         f"meta.json claims {meta['n_graphs']} -- the cache is corrupt")
    return cache, meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", type=Path, required=True,
                   help="a text file with one SMILES per line")
    p.add_argument("--out", type=Path, required=True, help="cache directory to write")
    p.add_argument("--chunk", type=int, default=25000,
                   help="molecules per featurization call; bounds peak memory and gives "
                        "progress. Does not change the result.")
    p.add_argument("--skip-invalid", action="store_true",
                   help="drop SMILES that fail to featurize instead of refusing the batch")
    args = p.parse_args(argv)

    smiles = [s.strip() for s in args.smiles.read_text().splitlines() if s.strip()]
    print(f"{len(smiles):,} SMILES from {args.smiles}")
    meta = build_cache(smiles, args.out, chunk=args.chunk,
                       on_invalid="skip" if args.skip_invalid else "raise")
    if meta["invalid"]:
        print(f"skipped {len(meta['invalid'])} unfeaturizable SMILES "
              f"(recorded in {args.out / META_FILE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
