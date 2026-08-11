"""Precompute a cluster-based K-fold split of the AmpC subset.

A random split of this data would be dishonest. It is drawn from a make-on-demand library
with heavy analog density, so a random validation fold has near-duplicates of nearly every
molecule sitting in the training set, and the resulting metric measures interpolation
rather than generalisation to new chemotypes.

So molecules are clustered first, and whole clusters are dealt to folds:

    ECFP4 -> sphere exclusion (LeaderPicker) -> nearest-centroid assignment
          -> stratified LPT packing of clusters into K folds

Stratified, because the potent tail is thin: ~3,150 molecules at pProp >= 3.5 and ~100 at
>= 5.0 out of 331k. Packing clusters by size alone would let a single cluster carrying a
large share of that tail land in one fold, so the tail strata are packed by tail count
instead.

The output is written once and reused by every training run, so that fold-to-fold variance
(data) stays separable from seed-to-seed variance (optimisation), and so that a rerun
months later reproduces the same partition instead of drawing a fresh one.

That reproducibility claim rests on LeaderPicker being order-stable under threading, which
is not documented anywhere. Checked on the full 331,480-molecule set at distance 0.65: three
runs at `numThreads=64` and two at `numThreads=1` all returned byte-identical centroid index
lists (32,254 centroids; 8s threaded vs 169s single). Hence the threaded default. Everything
downstream is deterministic by construction — `Pool.map` preserves input order and
`np.argmax`/`np.argmin` break ties toward the lowest index.

The 0.65 default was swept, not assumed: 0.70/0.75/0.80 buy essentially no extra separation
(median nearest-neighbour Tanimoto stays at 0.508 across all of them) while concentrating
the potent tail into ever fewer clusters -- the largest single cluster's share of the 100
molecules at pProp>=5.0 goes 9% -> 24% -> 32% -> 45%. See NOTES.md §11.2.

Usage:
    python src/split.py
    python src/split.py --dist-thresh 0.70 --outdir data/splits/cluster_kfold_v2
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers

RDLogger.DisableLog("rdApp.*")

# Tail thresholds used for stratification and reporting. These match the pProp levels
# subset.py already reports on, so the two files talk about the same tail.
TAIL_HI = 5.0
TAIL_LO = 3.5

# Module-level state shared with Pool workers. On Linux the default start method is fork,
# so anything assigned here before a Pool is created is inherited copy-on-write rather
# than pickled to every worker -- which matters when the payload is the 265k-molecule
# training fingerprint list used by the nearest-neighbour diagnostic.
_G = {}


# --------------------------------------------------------------------------- input


def read_subset(path):
    """Return (smiles, score, pprop, bin_label, ipw) as parallel lists/arrays.

    Split from the right, as subset.py does: a comma inside a SMILES would corrupt a
    naive parse, but the trailing four fields are always score, pprop, bin, ipw.
    """
    smiles, score, pprop, bins, ipw = [], [], [], [], []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split(",")
        expected = ["SMILES", "score", "pprop", "bin", "ipw"]
        if header != expected:
            raise SystemExit(f"unexpected header in {path}: {header}")
        for line in fh:
            s, sc, pp, b, w = line.rstrip("\n").rsplit(",", 4)
            smiles.append(s)
            score.append(float(sc))
            pprop.append(float(pp))
            bins.append(b)
            ipw.append(float(w))
    return (smiles, np.array(score), np.array(pprop), bins, np.array(ipw))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


# ------------------------------------------------------------------- fingerprints


def _init_fp(radius, fp_size):
    _G["gen"] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)


def _fp_worker(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Binary text, not the object: ExplicitBitVect does not survive the pickling that
    # Pool uses to return results.
    return DataStructs.BitVectToBinaryText(_G["gen"].GetFingerprint(mol))


def fingerprint(smiles, radius, fp_size, procs):
    with Pool(procs, initializer=_init_fp, initargs=(radius, fp_size)) as p:
        raw = p.map(_fp_worker, smiles, chunksize=500)
    bad = [i for i, r in enumerate(raw) if r is None]
    if bad:
        raise SystemExit(f"{len(bad)} SMILES failed to parse, first at row {bad[0]}")
    return raw


def _scaffold_worker(smiles):
    mol = Chem.MolFromSmiles(smiles)
    generic = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
    return Chem.MolToSmiles(generic)


def generic_scaffold_ids(smiles, procs):
    """Integer id per molecule for its generic (framework) Murcko scaffold.

    Not used by the split itself -- it costs a few seconds and makes a scaffold-split
    comparison available later without regenerating anything.
    """
    with Pool(procs) as p:
        scaffolds = p.map(_scaffold_worker, smiles, chunksize=500)
    order = {}
    ids = np.empty(len(scaffolds), dtype=np.int32)
    for i, s in enumerate(scaffolds):
        ids[i] = order.setdefault(s, len(order))
    return ids, len(order)


# ---------------------------------------------------------------------- clustering


def _init_assign(centroid_bins):
    _G["cent"] = [DataStructs.CreateFromBinaryText(b) for b in centroid_bins]


def _assign_worker(fp_bin):
    sims = DataStructs.BulkTanimotoSimilarity(
        DataStructs.CreateFromBinaryText(fp_bin), _G["cent"]
    )
    j = int(np.argmax(sims))
    return j, 1.0 - sims[j]


def cluster(fp_bins, dist_thresh, pick_threads, procs):
    """Sphere-exclusion clustering.

    LeaderPicker chooses centroids no two of which are within `dist_thresh` Tanimoto
    distance. Every molecule is then assigned to its *nearest* centroid, which guarantees
    a complete partition with no unassigned leftovers -- at the cost of placing outliers
    in clusters whose centroid is farther away than the threshold. `dist_to_centroid` is
    returned so that looseness stays visible instead of hidden.
    """
    fps = [DataStructs.CreateFromBinaryText(b) for b in fp_bins]
    picker = rdSimDivPickers.LeaderPicker()
    picks = list(
        picker.LazyBitVectorPick(fps, len(fps), dist_thresh, numThreads=pick_threads)
    )

    centroid_bins = [fp_bins[i] for i in picks]
    with Pool(procs, initializer=_init_assign, initargs=(centroid_bins,)) as p:
        out = p.map(_assign_worker, fp_bins, chunksize=200)

    cluster_id = np.array([o[0] for o in out], dtype=np.int32)
    dist = np.array([o[1] for o in out], dtype=np.float32)
    # Centroids must be their own cluster; if not, something is wrong with the picker.
    for rank, idx in enumerate(picks):
        if cluster_id[idx] != rank:
            raise SystemExit(f"centroid {idx} assigned to cluster {cluster_id[idx]}")
    return cluster_id, dist, np.array(picks, dtype=np.int64)


# -------------------------------------------------------------- fold assignment


def assign_folds(cluster_id, pprop, k):
    """Deal whole clusters into K folds by stratified LPT (longest-processing-time).

    Three strata, most-potent first. Each is packed by the quantity it exists to balance:
    the pProp>=5.0 stratum by its count of such molecules, likewise >=3.5, and the bulk by
    cluster size. Balancing the tail strata by *size* would be the bug -- a cluster holding
    a large share of the 100 potent molecules may itself be small, and would then be packed
    as if it were unimportant.

    Total fold sizes still come out near-equal because the running size load carries
    across strata and the bulk stratum, which dominates, is packed against it.
    """
    n_clusters = int(cluster_id.max()) + 1
    size = np.bincount(cluster_id, minlength=n_clusters)
    n_hi = np.bincount(cluster_id, weights=pprop >= TAIL_HI, minlength=n_clusters)
    n_lo = np.bincount(cluster_id, weights=pprop >= TAIL_LO, minlength=n_clusters)

    fold_of_cluster = np.full(n_clusters, -1, dtype=np.int8)
    size_load = np.zeros(k)

    strata = [
        (np.flatnonzero(n_hi > 0), n_hi, np.zeros(k)),
        (np.flatnonzero((n_hi == 0) & (n_lo > 0)), n_lo, np.zeros(k)),
        (np.flatnonzero((n_hi == 0) & (n_lo == 0)), size, size_load),
    ]
    for members, key, load in strata:
        # Ties broken by cluster id, and argmin ties to the lowest fold index, so the
        # whole assignment is reproducible and auditable.
        for c in sorted(members, key=lambda c: (-key[c], c)):
            f = int(np.argmin(load))
            fold_of_cluster[c] = f
            load[f] += key[c]
            if load is not size_load:
                size_load[f] += size[c]

    if (fold_of_cluster < 0).any():
        raise SystemExit("some clusters were never assigned to a fold")
    return fold_of_cluster[cluster_id], fold_of_cluster


# ----------------------------------------------------------------- diagnostics


def _nn_worker(item):
    self_pos, fp_bin = item
    sims = DataStructs.BulkTanimotoSimilarity(
        DataStructs.CreateFromBinaryText(fp_bin), _G["ref_fps"]
    )
    if self_pos >= 0:
        sims[self_pos] = -1.0  # a molecule is its own nearest neighbour; skip it
    return max(sims)


def nn_max(fp_bins, query_idx, ref_idx, procs, exclude_self=False):
    """Max Tanimoto from each query molecule to any molecule in the reference set.

    The reference fingerprints are staged in a module global *before* the Pool is created,
    so fork inherits them copy-on-write instead of pickling ~68 MB to every worker.
    """
    _G["ref_fps"] = [DataStructs.CreateFromBinaryText(fp_bins[i]) for i in ref_idx]
    pos = {g: j for j, g in enumerate(ref_idx)} if exclude_self else {}
    items = [(pos.get(i, -1), fp_bins[i]) for i in query_idx]
    try:
        with Pool(procs) as p:
            sims = p.map(_nn_worker, items, chunksize=8)
    finally:
        del _G["ref_fps"]
    return np.array(sims)


def sample_fold(fold, target_fold, n_sample, seed):
    val = np.flatnonzero(fold == target_fold)
    rng = np.random.default_rng(seed)
    return rng.choice(val, size=min(n_sample, len(val)), replace=False)


def random_folds(fold, k, rng):
    """A random partition with exactly the same fold sizes, as the baseline to beat."""
    sizes = np.bincount(fold, minlength=k)
    out = np.repeat(np.arange(k, dtype=np.int8), sizes)
    rng.shuffle(out)
    return out


def _pct(x, q):
    return float(np.percentile(x, q))


def diagnostics(args, smiles, pprop, ipw, cluster_id, dist, fold, fold_of_cluster,
                fp_bins, n_scaffolds):
    rng = np.random.default_rng(args.seed)
    n = len(smiles)
    L = []
    w = L.append

    w(f"# Split diagnostics — `{args.outdir.name}`\n")
    w(f"Generated by `src/split.py` from `{args.input}` "
      f"(rdkit {rdkit.__version__}).\n")
    w(f"ECFP{2 * args.radius}/{args.fp_size}, sphere exclusion at Tanimoto distance "
      f"{args.dist_thresh}, K={args.k}.\n")

    # --- 1. separation, against a same-sized random split
    w("\n## 1. Between-fold chemical separation\n")
    w(f"For each fold, max Tanimoto from {args.diag_sample:,} sampled validation molecules "
      "to any *training* molecule. The random split has identical fold sizes and is the "
      "baseline the cluster split has to beat.\n")
    rand_fold = random_folds(fold, args.k, rng)
    stats = {}
    w("\n| fold | split | median NN | 90th pct | frac NN >= 0.7 | frac NN >= 0.5 |")
    w("|---|---|---|---|---|---|")
    for f in range(args.k):
        for name, fv in (("cluster", fold), ("random", rand_fold)):
            t = time.time()
            sample = sample_fold(fv, f, args.diag_sample, args.seed)
            s = nn_max(fp_bins, sample, np.flatnonzero(fv != f), args.procs)
            stats[(name, f)] = s
            print(f"  nn-similarity fold {f} ({name}): {time.time() - t:.0f}s",
                  file=sys.stderr)
            w(f"| {f} | {name} | {np.median(s):.3f} | {_pct(s, 90):.3f} | "
              f"{(s >= 0.7).mean():.2%} | {(s >= 0.5).mean():.2%} |")
    cl = np.concatenate([stats[("cluster", f)] for f in range(args.k)])
    rd = np.concatenate([stats[("random", f)] for f in range(args.k)])
    w(f"\nPooled over folds, median shift vs random **{np.median(rd) - np.median(cl):+.3f}**, "
      f"and molecules with a training neighbour >= 0.7 fall from {(rd >= 0.7).mean():.2%} "
      f"to {(cl >= 0.7).mean():.2%}.\n")

    # --- 1b. within-fold, which is what says *why* the cross-fold number sits where it does
    w("\n### 1b. Within-fold neighbours — is there anything to separate?\n")
    w("Two worlds produce the same cross-fold number: (a) tight analog series exist and "
      "clustering kept them together, so what is left across folds is background "
      "similarity between unrelated molecules; (b) the library has no tight analogs at "
      "all, so a cluster split removes nothing. To tell them apart, look at each "
      "molecule's nearest neighbour *inside its own fold* (self excluded).\n")
    w("\nThe comparison must be pool-size matched: a within-fold search covers ~1/K as "
      "many molecules as a cross-fold one, which lowers the max on its own. So the "
      "random split is measured within-fold too, and each column is compared only to the "
      "same column.\n")
    w("\n| fold | cluster within | random within | cluster cross | random cross |")
    w("|---|---|---|---|---|")
    within = {}
    for name, fv in (("cluster", fold), ("random", rand_fold)):
        for f in range(args.k):
            t = time.time()
            sample = sample_fold(fv, f, args.diag_sample, args.seed)
            within[(name, f)] = nn_max(fp_bins, sample, np.flatnonzero(fv == f),
                                       args.procs, exclude_self=True)
            print(f"  within-fold nn {f} ({name}): {time.time() - t:.0f}s",
                  file=sys.stderr)
    for f in range(args.k):
        cells = [within[("cluster", f)], within[("random", f)],
                 stats[("cluster", f)], stats[("random", f)]]
        w(f"| {f} | " + " | ".join(f"{(s >= 0.7).mean():.2%}" for s in cells) + " |")
    pooled = {k: np.concatenate([within[(k, f)] for f in range(args.k)])
              for k in ("cluster", "random")}
    w("\n(cells are the fraction of sampled molecules with a neighbour at Tanimoto >= 0.7; "
      "*within* searches ~66k molecules, *cross* searches ~265k)\n")
    cw, rw = (pooled["cluster"] >= 0.7).mean(), (pooled["random"] >= 0.7).mean()
    cc, rc = (cl >= 0.7).mean(), (rd >= 0.7).mean()
    w(f"\nPooled: within-fold **{cw:.2%}** (cluster) vs **{rw:.2%}** (random) — same pool "
      f"size, so clustering concentrates close neighbours inside folds by "
      f"{cw / rw:.1f}x. Cross-fold **{cc:.2%}** vs **{rc:.2%}** — the same pairs removed "
      f"from across the split, a {rc / cc:.1f}x reduction.")
    w("\nBoth effects pointing the same way means close analogs genuinely exist in this "
      "library and the split moved them to one side. If instead the two within-fold "
      "numbers were equal, there would have been no analog structure to separate and the "
      "cluster split would be doing little.")
    w(f"\nMedians move much less (within {np.median(pooled['cluster']):.3f} vs cross "
      f"{np.median(cl):.3f}), because the median molecule's nearest neighbour is an "
      "unrelated molecule at background similarity either way. The split acts on the "
      "close-neighbour tail, not on the bulk.\n")

    # --- 2. cluster tightness
    w("\n## 2. Cluster tightness\n")
    w("Distance from each molecule to its assigned centroid. Nearest-centroid assignment "
      "guarantees a full partition, but places outliers beyond the picking threshold; a "
      "large fraction past it means loose clusters and degraded separation.\n")
    w(f"\n- median {np.median(dist):.3f}, 90th pct {_pct(dist, 90):.3f}, "
      f"max {dist.max():.3f}")
    w(f"- beyond the {args.dist_thresh} threshold: "
      f"{(dist > args.dist_thresh).mean():.2%} of molecules\n")

    # --- 3. cluster sizes
    sizes = np.bincount(cluster_id)
    w("\n## 3. Cluster size distribution\n")
    w(f"\n- {len(sizes):,} clusters over {n:,} molecules (mean {sizes.mean():.1f})")
    w(f"- singletons: {(sizes == 1).sum():,} clusters "
      f"({(sizes == 1).sum() / len(sizes):.1%}), holding "
      f"{(sizes == 1).sum() / n:.1%} of molecules")
    w(f"- largest clusters: {', '.join(str(x) for x in sorted(sizes)[-5:][::-1])}\n")
    w("\n| size | clusters | molecules |")
    w("|---|---|---|")
    edges = [1, 2, 3, 5, 10, 25, 100, 1_000, 10 ** 9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sizes >= lo) & (sizes < hi)
        if m.any():
            label = f"{lo}" if hi == lo + 1 else f"{lo}–{hi - 1}"
            w(f"| {label} | {m.sum():,} | {sizes[m].sum():,} |")

    # --- 4. tail concentration
    w("\n## 4. Tail concentration\n")
    w("How the thin potent tail is distributed over clusters. If it lives in very few "
      "clusters, per-fold tail counts are skewed no matter how they are packed — which "
      "is the argument for pooling out-of-fold predictions before computing tail "
      "metrics.\n")
    w("\n| threshold | molecules | clusters holding them | largest single-cluster share |")
    w("|---|---|---|---|")
    for thr in (TAIL_LO, TAIL_HI):
        m = pprop >= thr
        counts = np.bincount(cluster_id[m])
        counts = counts[counts > 0]
        w(f"| pProp >= {thr} | {m.sum():,} | {len(counts):,} | "
          f"{counts.max()} ({counts.max() / m.sum():.1%}) |")

    # --- 5. per-fold table
    w("\n## 5. Per-fold composition\n")
    w("\n| fold | n | clusters | mean pProp | sd | n >= 3.5 | n >= 5.0 | sum ipw |")
    w("|---|---|---|---|---|---|---|---|")
    for f in range(args.k):
        m = fold == f
        w(f"| {f} | {m.sum():,} | {(fold_of_cluster == f).sum():,} | "
          f"{pprop[m].mean():.3f} | {pprop[m].std():.3f} | "
          f"{(pprop[m] >= TAIL_LO).sum():,} | {(pprop[m] >= TAIL_HI).sum():,} | "
          f"{ipw[m].sum():,.0f} |")

    w("\n## 6. Notes and caveats\n")
    w(f"\n- Generic Murcko scaffolds: {n_scaffolds:,} distinct, stored per molecule but "
      "**not** used by this split — available for a scaffold-split comparison later.")
    w("- This is a plain K-fold: each fold is used for both early stopping and reporting, "
      "so the reported CV score is mildly optimistic. Fixing the epoch budget instead of "
      "early-stopping removes that bias.")
    w("- The tail is thin enough that per-fold tail metrics are noise. Report tail "
      "metrics on pooled out-of-fold predictions — the K validation folds cover every "
      "row exactly once.")
    w("- Splits are fixed across model seeds. The model seed governs initialisation, "
      "dropout and shuffling only; it never touches the partition.")
    w(f"- `{args.input.name}` is sorted by `(-pprop, smiles)`, i.e. sorted by the target. "
      "Shuffle explicitly in the DataLoader; do not rely on file order.\n")
    return "\n".join(L)


# ---------------------------------------------------------------------- output


def check_partition(fold, cluster_id, k, n):
    if len(fold) != n or (fold < 0).any() or (fold >= k).any():
        raise SystemExit("fold vector is malformed")
    if np.bincount(fold, minlength=k).sum() != n:
        raise SystemExit("folds do not cover every row exactly once")
    # Every cluster must lie wholly inside one fold -- the entire point of the exercise.
    order = np.argsort(cluster_id, kind="stable")
    c, f = cluster_id[order], fold[order]
    boundary = np.flatnonzero(np.diff(c) == 0)
    if (f[boundary] != f[boundary + 1]).any():
        raise SystemExit("a cluster is split across folds")


def write_outputs(args, smiles, pprop, bins, cluster_id, dist, scaffold_id, fold,
                  fp_bins, n_scaffolds, input_sha, elapsed):
    args.outdir.mkdir(parents=True, exist_ok=True)

    assignments = args.outdir / "assignments.csv"
    with open(assignments, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["row_idx", "smiles", "pprop", "bin", "cluster_id",
                     "dist_to_centroid", "generic_scaffold_id", "fold"])
        for i in range(len(smiles)):
            wr.writerow([i, smiles[i], pprop[i], bins[i], cluster_id[i],
                         f"{dist[i]:.4f}", scaffold_id[i], fold[i]])

    # uint8[n, fp_size/8] packed bits, so any later split variant reuses identical
    # features rather than recomputing them under a possibly different rdkit.
    packed = np.frombuffer(b"".join(fp_bins), dtype=np.uint8).reshape(
        len(fp_bins), args.fp_size // 8
    )
    np.save(args.outdir / "fingerprints.npy", packed)

    # Content hash over what actually defines the split. Byte-identity of the CSV is the
    # wrong criterion (float formatting, ordering); this is the right one.
    h = hashlib.sha256()
    for i in range(len(smiles)):
        h.update(f"{i},{cluster_id[i]},{fold[i]}\n".encode())
    split_sha = h.hexdigest()

    meta = {
        "input": str(args.input),
        # Absolute path too: `input` is as-typed and therefore CWD-relative, which breaks
        # for any consumer with a different CWD -- e.g. an sbatch job running from
        # /home/ethan2/logs/.
        "input_abspath": str(args.input.resolve()),
        "input_sha256": input_sha,
        "input_bytes": args.input.stat().st_size,
        "rows": len(smiles),
        "split_sha256": split_sha,
        "k": args.k,
        "fold_sizes": np.bincount(fold, minlength=args.k).tolist(),
        "clustering": {
            "method": "sphere exclusion (rdkit LeaderPicker) + nearest centroid",
            "dist_thresh": args.dist_thresh,
            "n_clusters": int(cluster_id.max()) + 1,
            "pick_threads": args.pick_threads,
        },
        "fingerprint": {
            "type": "morgan",
            "radius": args.radius,
            "fp_size": args.fp_size,
        },
        "fold_assignment": {
            "method": "stratified LPT on whole clusters",
            "strata": [f"pprop >= {TAIL_HI}", f"pprop >= {TAIL_LO}", "remainder"],
            "seed": args.seed,
        },
        "n_generic_scaffolds": int(n_scaffolds),
        "rdkit_version": rdkit.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "git_commit": git_commit(),
        "argv": sys.argv,
        "elapsed_sec": round(elapsed, 1),
    }
    (args.outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return assignments, split_sha


# ------------------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path,
                   default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("-o", "--outdir", type=Path,
                   default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--k", type=int, default=5, help="number of folds")
    p.add_argument("--dist-thresh", type=float, default=0.65,
                   help="Tanimoto distance threshold for sphere exclusion")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--fp-size", type=int, default=2048)
    p.add_argument("--procs", type=int, default=64)
    p.add_argument("--pick-threads", type=int, default=64,
                   help="threads for LeaderPicker (verified order-stable; see module docs)")
    p.add_argument("--diag-sample", type=int, default=5000,
                   help="validation molecules sampled for the NN-similarity diagnostic")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for diagnostic sampling and the random-split baseline")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    started = time.time()

    def step(msg):
        print(f"[{time.time() - started:6.1f}s] {msg}", file=sys.stderr)

    input_sha = sha256_file(args.input)
    smiles, score, pprop, bins, ipw = read_subset(args.input)
    if len(set(smiles)) != len(smiles):
        raise SystemExit("input SMILES are not unique; subset.py should have deduplicated")
    step(f"read {len(smiles):,} rows  sha256 {input_sha[:12]}")

    fp_bins = fingerprint(smiles, args.radius, args.fp_size, args.procs)
    step(f"fingerprinted (ECFP{2 * args.radius}/{args.fp_size})")

    cluster_id, dist, picks = cluster(fp_bins, args.dist_thresh, args.pick_threads,
                                      args.procs)
    step(f"clustered into {len(picks):,} clusters at distance {args.dist_thresh}")

    scaffold_id, n_scaffolds = generic_scaffold_ids(smiles, args.procs)
    step(f"generic scaffolds: {n_scaffolds:,} distinct")

    fold, fold_of_cluster = assign_folds(cluster_id, pprop, args.k)
    check_partition(fold, cluster_id, args.k, len(smiles))
    step(f"folds assigned, sizes {np.bincount(fold, minlength=args.k).tolist()}")

    report = diagnostics(args, smiles, pprop, ipw, cluster_id, dist, fold,
                         fold_of_cluster, fp_bins, n_scaffolds)
    step("diagnostics computed")

    path, split_sha = write_outputs(args, smiles, pprop, bins, cluster_id, dist,
                                    scaffold_id, fold, fp_bins, n_scaffolds, input_sha,
                                    time.time() - started)
    (args.outdir / "diagnostics.md").write_text(report + "\n")
    step(f"wrote {args.outdir}/  split_sha256 {split_sha[:12]}")
    print(report)


if __name__ == "__main__":
    main()
