"""Build an enriched training subset from the AmpC 10M docking file.

pProp = -log10(rank_max / 1e7) is a quantile, so the bulk of the file sits at low pProp
and the molecules of interest are a thin tail. This script keeps the whole tail and
subsamples the bulk:

    pProp >= 2.5          take everything
    each 0.5 bin below    take N (default 60,000), or all of it if smaller

Every row carries its 0.5-wide bin and an inverse-probability weight, so metrics computed
on the subset can be corrected back to natural frequency later.

Usage:
    python src/subset.py
    python src/subset.py --seed 1 --per-bin 60000 -o data/other_subset.csv
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

BIN_WIDTH = 0.5
TAKE_ALL_ABOVE = 2.5
# pProp maxes out at -log10(1/1e7) = 7.0, so the last bin is [6.5, 7.0].
N_BINS = 14


def bin_index(pprop):
    """0.5-wide bin, clamped so pProp == 7.0 lands in the top bin rather than past it."""
    return min(int(pprop / BIN_WIDTH), N_BINS - 1)


def bin_label(idx):
    return f"{idx * BIN_WIDTH:.1f}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path,
                   default=Path("data/ampc_unif_random_10M_pprop10M.csv"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--per-bin", type=int, default=60_000,
                   help="rows to keep from each 0.5 bin below the take-all threshold")
    p.add_argument("--take-all-above", type=float, default=TAKE_ALL_ABOVE,
                   help="keep every molecule at or above this pProp")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def stream_rows(path):
    """Yield (smiles, score, pprop) from the CSV.

    Split from the right so a comma inside a SMILES would not corrupt the parse; the
    trailing two fields are always score and pprop.
    """
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split(",")
        if header[:3] != ["SMILES", "score", "pprop"]:
            raise SystemExit(f"unexpected header in {path}: {header}")
        for line in fh:
            smiles, score, pprop = line.rstrip("\n").rsplit(",", 2)
            yield smiles, score, float(pprop)


def canonicalize_and_dedupe(rows):
    """Rewrite SMILES in RDKit canonical form, then collapse repeats.

    The source file is not fully canonical (~2.6% of rows differ, almost all in how
    stereocentres are written) and contains a small number of duplicate molecules. Where a
    molecule appears more than once with different docking scores, keep the best (most
    negative) score — docking reports the best pose found, so the better score is the
    better-converged result.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    best = {}
    stats = {"rewritten": 0, "unparseable": 0,
             "dropped_identical_score": 0, "dropped_differing_score": 0}
    for smiles, score, pprop, b in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            stats["unparseable"] += 1
            continue
        canon = Chem.MolToSmiles(mol)
        if canon != smiles:
            stats["rewritten"] += 1

        prev = best.get(canon)
        if prev is None:
            best[canon] = (canon, score, pprop, b)
            continue
        if float(score) == float(prev[1]):
            stats["dropped_identical_score"] += 1
        else:
            stats["dropped_differing_score"] += 1
            if float(score) < float(prev[1]):    # more negative docks better
                best[canon] = (canon, score, pprop, b)

    return list(best.values()), stats


def build(args):
    rng = random.Random(args.seed)
    cutoff_bin = int(args.take_all_above / BIN_WIDTH)

    population = [0] * N_BINS          # rows seen per bin, over the whole file
    reservoirs = [[] for _ in range(N_BINS)]   # sampled rows for bins below the cutoff
    tail = []                          # every row at or above the cutoff

    started = time.time()
    for n, (smiles, score, pprop) in enumerate(stream_rows(args.input), 1):
        b = bin_index(pprop)
        population[b] += 1

        if b >= cutoff_bin:
            tail.append((smiles, score, pprop, b))
        else:
            # Algorithm R: fill the reservoir, then replace with probability k/seen.
            res = reservoirs[b]
            if len(res) < args.per_bin:
                res.append((smiles, score, pprop, b))
            else:
                j = rng.randrange(population[b])
                if j < args.per_bin:
                    res[j] = (smiles, score, pprop, b)

        if n % 1_000_000 == 0:
            print(f"  {n:>10,} rows  ({time.time() - started:5.1f}s)", file=sys.stderr)

    total_seen = sum(population)
    print(f"  {total_seen:>10,} rows  ({time.time() - started:5.1f}s)  done reading",
          file=sys.stderr)

    selected = tail + [row for res in reservoirs for row in res]
    selected, canon_stats = canonicalize_and_dedupe(selected)

    # Inverse-probability weight: how many library molecules each kept row stands for.
    # Computed after dedup so the weights still reconstruct the source row distribution.
    taken = [0] * N_BINS
    for row in selected:
        taken[row[3]] += 1
    ipw = [population[b] / taken[b] if taken[b] else 0.0 for b in range(N_BINS)]

    # Deterministic order so a rerun at the same seed is byte-identical.
    selected.sort(key=lambda r: (-r[2], r[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["SMILES", "score", "pprop", "bin", "ipw"])
        for smiles, score, pprop, b in selected:
            w.writerow([smiles, score, pprop, bin_label(b), f"{ipw[b]:.6f}"])

    return population, taken, ipw, selected, total_seen, canon_stats


def report(args, population, taken, ipw, selected, total_seen, canon_stats):
    print(f"\n{'bin':>10}  {'population':>12}  {'taken':>9}  {'fraction':>9}  {'ipw':>10}")
    print("-" * 58)
    for b in range(N_BINS):
        if not population[b] and not taken[b]:
            continue
        hi = (b + 1) * BIN_WIDTH
        label = f"{b * BIN_WIDTH:.1f}-{hi:.1f}"
        frac = taken[b] / population[b] if population[b] else 0.0
        print(f"{label:>10}  {population[b]:>12,}  {taken[b]:>9,}  "
              f"{frac:>8.3%}  {ipw[b]:>10.3f}")
    print("-" * 58)
    print(f"{'total':>10}  {total_seen:>12,}  {len(selected):>9,}  "
          f"{len(selected) / total_seen:>8.3%}")

    n35 = sum(1 for r in selected if r[2] >= 3.5)
    n50 = sum(1 for r in selected if r[2] >= 5.0)
    print(f"\npProp >= 3.5: {n35:,}   pProp >= 5.0: {n50:,}")
    print(f"\nSMILES rewritten to canonical form : {canon_stats['rewritten']:,}")
    print(f"duplicates dropped, same score     : {canon_stats['dropped_identical_score']:,}")
    print(f"duplicates dropped, best score kept: {canon_stats['dropped_differing_score']:,}")
    if canon_stats["unparseable"]:
        print(f"UNPARSEABLE SMILES DROPPED         : {canon_stats['unparseable']:,}")
    print(f"\nwrote {args.output}")

    meta = {
        "input": str(args.input),
        "input_bytes": os.path.getsize(args.input),
        "input_mtime": os.path.getmtime(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "per_bin": args.per_bin,
        "take_all_above": args.take_all_above,
        "bin_width": BIN_WIDTH,
        "rows_in": total_seen,
        "rows_out": len(selected),
        "smiles": "rdkit canonical (isomeric), deduplicated",
        "canonicalisation": canon_stats,
        "n_pprop_ge_3.5": n35,
        "n_pprop_ge_5.0": n50,
        "bins": [
            {"bin": bin_label(b), "population": population[b],
             "taken": taken[b], "ipw": ipw[b]}
            for b in range(N_BINS) if population[b] or taken[b]
        ],
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}")


def main(argv=None):
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    report(args, *build(args))


if __name__ == "__main__":
    main()
