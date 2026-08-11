"""Load the cached graphs written by `featurize.py`. Use this, not `torch.load` directly.

Mirrors `splits.py`: the loader owns the provenance guard, so every consumer gets it for
free and nobody has to remember to check.

    from features import load_features
    from splits import load_fold

    ds = load_features("data/features/minimol_v1")
    train_idx, val_idx = load_fold("data/splits/cluster_kfold_v1", fold=0)

    loader = DataLoader(Subset(ds, train_idx), batch_size=256, shuffle=True,
                        collate_fn=trunk.collate)

`ds[i]` is the graph for row `i` of `ampc_subset_331k.csv`, so split indices index it
directly -- there is no mapping layer to get wrong.

Two things the guard exists to catch:

* `subset.py` takes a `--seed`. Re-running it produces a *different* 331,480-molecule set
  under the same filename, which would leave this cache silently mis-aligned -- every graph
  attached to the wrong target. The CSV sha256 in `meta.json` is re-checked on load.
* A cache built with `--limit` is shorter than the CSV. Indexing it with full-set fold
  indices would raise deep inside a DataLoader worker, so it is rejected up front unless
  the caller passes `allow_partial=True`.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data.separate import separate

from splits import _sha256_file


class GraphCache(Dataset):
    """Random-access view over the PyG-collated cache.

    `featurize.py` stores the graphs collated -- a few large concatenated tensors plus a
    slice index -- because that loads ~80x faster than a list of `Data` objects. `separate`
    reconstructs one graph on demand, which was verified to round-trip every key exactly.

    `__getitem__` returns a fresh `Data` each call. That matters: the trunk's forward pass
    rewrites `feat` on the *collated batch* (the positional encoders concatenate into it),
    so a collated batch is single-use. Measured: the source graphs themselves are not
    mutated -- `Batch.from_data_list` copies -- so this cache is safe to serve from for
    every epoch of every run without defensive copying.
    """

    def __init__(self, data, slices, meta):
        self._data, self._slices, self.meta = data, slices, meta
        first = next(iter(slices.values()))
        self._len = len(first) - 1

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._len
        if not 0 <= idx < self._len:
            raise IndexError(f"index {idx} out of range for {self._len} graphs")
        return separate(cls=self._data.__class__, batch=self._data, idx=idx,
                        slice_dict=self._slices, decrement=False)


def load_features(root, check_input=True, allow_partial=False):
    """Load a feature cache, refusing one that does not match its source CSV.

    `check_input=False` skips only the (slow) CSV re-hash -- the length and schema checks
    always run. Use it in a DataLoader worker that has already been validated once, not to
    get past a failure.
    """
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())

    csv = Path(meta["input_abspath"])
    if check_input:
        # Resolved from meta.json, not from the caller's CWD, so this works from a SLURM
        # working directory -- same reason splits.py stores an absolute path.
        if not csv.exists():
            raise FileNotFoundError(
                f"{csv} is recorded as the source of {root} but does not exist")
        actual = _sha256_file(csv)
        if actual != meta["input_sha256"]:
            raise ValueError(
                f"{csv} has changed since {root} was built "
                f"(sha256 {actual[:12]}... != {meta['input_sha256'][:12]}...). "
                "The cached graphs are aligned to the old row order and are now wrong. "
                "Re-run: python src/featurize.py")

    if meta.get("limit") is not None and not allow_partial:
        raise ValueError(
            f"{root} was built with --limit {meta['limit']} and covers only "
            f"{meta['n_graphs']:,} of {meta['input_rows']:,} rows, so full-set fold indices "
            "would run off the end. Re-run featurize.py without --limit, or pass "
            "allow_partial=True if a partial cache is genuinely what you want.")

    data, slices = torch.load(root / meta["graphs_file"], weights_only=False)
    ds = GraphCache(data, slices, meta)

    if len(ds) != meta["n_graphs"]:
        raise ValueError(f"{root} holds {len(ds)} graphs but meta.json claims "
                         f"{meta['n_graphs']} -- the cache is corrupt")
    if not allow_partial and len(ds) != meta["input_rows"]:
        raise ValueError(f"{root} holds {len(ds)} graphs for a {meta['input_rows']}-row CSV")
    return ds
