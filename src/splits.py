"""Load a precomputed split produced by `src/split.py`.

The point of this module is the provenance check. `subset.py` takes `--seed`, so rerunning
it with different arguments silently produces a different 331k set — and every split built
against the old one becomes wrong without anything failing. Hashing the input CSV at load
time turns that from a silent wrong result into a loud error at startup.

Usage:
    from splits import load_assignments, load_fold, load_fingerprints

    df = load_assignments("data/splits/cluster_kfold_v1")
    train_idx, val_idx = load_fold("data/splits/cluster_kfold_v1", fold=0)

Indices are positions into the input CSV's row order, so `df.iloc[train_idx]` and a
DataFrame read straight from `data/ampc_subset_331k.csv` line up.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(split_dir):
    return json.loads((Path(split_dir) / "meta.json").read_text())


def resolve_input(meta, input_csv=None):
    """Locate the source CSV a split was built from.

    meta["input"] is the path as typed on the command line and so is CWD-relative — fine
    from the repo root, wrong from an sbatch job running out of /home/ethan2/logs/. Hence
    the recorded absolute path is tried first, and an explicit override wins over both.
    """
    for candidate in (input_csv, meta.get("input_abspath"), meta.get("input")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        f"source CSV not found. Tried {input_csv!r}, {meta.get('input_abspath')!r}, "
        f"{meta.get('input')!r} (the last is relative to CWD {Path.cwd()}). "
        "Pass input_csv= explicitly if the data has moved."
    )


def verify(split_dir, input_csv=None):
    """Raise unless the source CSV still hashes to what the split was built from."""
    split_dir = Path(split_dir)
    meta = load_meta(split_dir)
    path = resolve_input(meta, input_csv)

    actual = _sha256_file(path)
    if actual != meta["input_sha256"]:
        raise ValueError(
            f"{path} has changed since {split_dir} was generated "
            f"(sha256 {actual[:12]} != {meta['input_sha256'][:12]}). "
            "Regenerate the split with src/split.py, or point at the original CSV."
        )
    return meta


def load_assignments(split_dir, check=True):
    """Return the assignments table, one row per molecule in input CSV order."""
    split_dir = Path(split_dir)
    if check:
        verify(split_dir)
    df = pd.read_csv(split_dir / "assignments.csv")
    if not (df["row_idx"].to_numpy() == np.arange(len(df))).all():
        raise ValueError(f"{split_dir}/assignments.csv is not in input row order")
    return df


def load_fold(split_dir, fold, check=True):
    """Return (train_idx, val_idx) for one fold, as int arrays of CSV row positions."""
    meta = load_meta(split_dir)
    if not 0 <= fold < meta["k"]:
        raise ValueError(f"fold {fold} out of range for k={meta['k']}")
    f = load_assignments(split_dir, check=check)["fold"].to_numpy()
    return np.flatnonzero(f != fold), np.flatnonzero(f == fold)


def load_fingerprints(split_dir, unpack=False):
    """Packed ECFP bits as uint8[n, fp_size/8], or unpacked bits if `unpack`."""
    packed = np.load(Path(split_dir) / "fingerprints.npy")
    return np.unpackbits(packed, axis=1) if unpack else packed
