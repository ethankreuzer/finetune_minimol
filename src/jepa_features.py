"""Load the Mol-JEPA embedding cache written by `jepa_embed.py`. Use this, not `np.load`.

Mirrors `features.py`, which mirrors `splits.py`: the loader owns the provenance guard, so
every consumer gets it for free and nobody has to remember to check.

    from jepa_features import load_embeddings
    from splits import load_fold

    X = load_embeddings("data/embeddings/moljepa_v1", readout="cls")   # [331480, 512]
    train_idx, val_idx = load_fold("data/splits/cluster_kfold_v1", fold=0)
    X[train_idx]                       # split indices index it directly

`X[i]` is the encoding of row `i` of `ampc_subset_331k.csv` -- there is no mapping layer to
get wrong.

THE READOUT GRAMMAR

Which part of Mol-JEPA is *the* encoder is an open question, so the readout is a small
expression rather than a fixed choice. The cache holds two column-aligned `[N, 13, 512]`
arrays sharing one name list (`token_names`, measured off the model by `jepa_embed.py`):

    embeddings.npy   the raw transformer output
    projected.npy    the per-token linear heads -- row 0 is the model's `cls` output,
                     rows 1.. are its `predictions`

A readout is a `+`-joined list of:

    cls              the model's headline CLS output   (sugar for proj:cls)
    emb:<name>       one raw token         e.g. emb:cls, emb:graph, emb:ecfp
    proj:<name>      one projected token   e.g. proj:cls, proj:boltz, proj:tdc_targets
    emb:*            all 13 raw tokens, flattened
    proj:*           all 13 projected tokens, flattened

so `cls`, `emb:graph+emb:ecfp`, and `emb:cls+proj:*` are all expressible without a code
change. `describe()` prints what a given cache actually offers.

Note `emb:cls` and `proj:cls` are different vectors -- the second is a `nn.Linear` applied
to the first. `cls` resolves to `proj:cls` because that is what `model(...).cls` returns.

Memory: the arrays are opened memory-mapped, so a single-token readout touches 679 MB of the
17.6 GB on disk. A `*` term materialises its whole array (8.8 GB), which is deliberate -- it
is the caller asking for it.
"""

import json
from pathlib import Path

import numpy as np

from splits import _sha256_file


def _meta(root):
    return json.loads((Path(root) / "meta.json").read_text())


def load_meta(root, check_input=True, allow_partial=False):
    """Validate a cache and return its meta.json, without reading the big arrays.

    Split out so a caller that only needs the width or the token names -- `train_jepa.py`
    sizing the head, `run_config.py` stamping provenance -- pays none of the I/O.
    """
    root = Path(root)
    meta = _meta(root)

    csv = Path(meta["input_abspath"])
    if check_input:
        # Resolved from meta.json, not the caller's CWD, so this works from a SLURM working
        # directory -- same reason splits.py stores an absolute path.
        if not csv.exists():
            raise FileNotFoundError(
                f"{csv} is recorded as the source of {root} but does not exist")
        actual = _sha256_file(csv)
        if actual != meta["input_sha256"]:
            raise ValueError(
                f"{csv} has changed since {root} was built "
                f"(sha256 {actual[:12]}... != {meta['input_sha256'][:12]}...). "
                "The cached embeddings are aligned to the old row order and are now wrong. "
                "Re-run: python src/jepa_embed.py")

    if meta.get("limit") is not None and not allow_partial:
        raise ValueError(
            f"{root} was built with --limit {meta['limit']} and covers only "
            f"{meta['n_molecules']:,} of {meta['input_rows']:,} rows, so full-set fold "
            "indices would run off the end. Re-run jepa_embed.py without --limit, or pass "
            "allow_partial=True if a partial cache is genuinely what you want.")
    if not allow_partial and meta["n_molecules"] != meta["input_rows"]:
        raise ValueError(f"{root} holds {meta['n_molecules']} rows for a "
                         f"{meta['input_rows']}-row CSV")
    return meta


def _open(root, meta, which):
    spec = meta["arrays"][which]
    path = Path(root) / spec["file"]
    arr = np.load(path, mmap_mode="r")
    if list(arr.shape) != list(spec["shape"]):
        raise ValueError(f"{path} has shape {arr.shape} but meta.json claims "
                         f"{tuple(spec['shape'])} -- the cache is corrupt")
    return arr


def readout_terms(readout):
    return [t.strip() for t in str(readout).split("+") if t.strip()]


PREFIXES = {"emb": "embeddings", "proj": "projected"}


def _resolve(term, meta):
    """(which_array, index_or_None) for one readout term, or raise with the valid names."""
    names = meta["token_names"]                       # shared by both arrays, same order
    if term == "cls":
        # What `model(...).cls` returns is the projected column 0, not the raw one.
        term = "proj:cls"
    if ":" not in term:
        raise ValueError(f"readout term {term!r} is not understood. Expected 'cls', "
                         f"'emb:<name>', 'proj:<name>', 'emb:*' or 'proj:*'.")
    kind, name = term.split(":", 1)
    which = PREFIXES.get(kind)
    if which is None:
        raise ValueError(f"readout term {term!r} has unknown prefix {kind!r}; "
                         f"expected one of {sorted(PREFIXES)}")
    if which not in meta["arrays"]:
        raise ValueError(f"readout term {term!r} needs array {which!r}, which this cache "
                         f"does not have. Present: {sorted(meta['arrays'])}")
    if name == "*":
        return which, None
    if name not in names:
        raise ValueError(f"readout term {term!r}: {name!r} is not a token name in this "
                         f"cache. Available: {names}")
    return which, names.index(name)


def readout_dim(meta, readout):
    """Width of a readout, without touching the arrays."""
    dim, total = meta["embed_dim"], 0
    for term in readout_terms(readout):
        which, idx = _resolve(term, meta)
        n_rows = meta["arrays"][which]["shape"][1]          # 13 for both arrays
        total += dim if idx is not None else n_rows * dim
    if total == 0:
        raise ValueError(f"readout {readout!r} selects nothing")
    return total


def load_embeddings(root, readout="cls", check_input=True, allow_partial=False):
    """Return `[N, D]` float32 for `readout`, in exact CSV row order."""
    root = Path(root)
    meta = load_meta(root, check_input=check_input, allow_partial=allow_partial)

    opened, parts = {}, []
    for term in readout_terms(readout):
        which, idx = _resolve(term, meta)
        arr = opened.setdefault(which, _open(root, meta, which))
        parts.append(np.asarray(arr[:, idx, :]) if idx is not None
                     else np.asarray(arr).reshape(arr.shape[0], -1))
    if not parts:
        raise ValueError(f"readout {readout!r} selects nothing")

    # float32 explicitly: a float64 array here would silently double every downstream tensor
    # and change the loss's numerics rather than failing.
    #
    # Note a deliberate asymmetry. A single-token term goes through fancy indexing, which
    # copies, so it comes back as ~679 MB of real memory. A `*` term reshapes the memmap,
    # which is already contiguous float32, so `ascontiguousarray` is a no-op and the result
    # stays memory-mapped -- 8.8 GB read lazily per batch rather than resident. Both index
    # identically; the `*` case just trades RAM for I/O, which is the right way round.
    X = (parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1))
    return np.ascontiguousarray(X, dtype=np.float32)


def describe(root, check_input=False):
    meta = load_meta(root, check_input=check_input, allow_partial=True)
    lines = [
        f"{root}",
        f"  {meta['n_molecules']:,} molecules of {meta['input_rows']:,} CSV rows"
        + (f"   (--limit {meta['limit']})" if meta.get("limit") else ""),
        f"  encoder {meta['encoder']}  {meta['hf_repo']}@{meta.get('hf_revision')}",
        f"  embed_dim {meta['embed_dim']}",
        f"  tokens {meta['token_names']}",
        "  readouts: cls (= proj:cls), emb:<token>, proj:<token>, emb:*, proj:*",
        "  arrays:",
    ] + [
        f"    {k:12s} {tuple(v['shape'])}  {v.get('what', '')}"
        for k, v in sorted(meta["arrays"].items())
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(describe(sys.argv[1] if len(sys.argv) > 1 else "data/embeddings/moljepa_v1"))
