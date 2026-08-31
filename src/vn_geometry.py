"""Score each MiniMol embedding by its GEOMETRY, not by how well it predicts pProp.

`vn_probe.py` already asked the prediction question and answered it: `pooled512` beats every
virtual node tap decisively -- 0.861 Spearman against 0.738, 0.241 average precision against
0.059 -- and the depth trend rises monotonically VN3 -> VN15. That result is not surprising and
it is not the whole question. **The trunk was fine-tuned end to end on pProp, so `pooled512` is
one linear layer from the prediction.** Any task that *is* pProp prediction favours it by
construction rather than by richness.

The encoder's consumer is a deep-kernel-learning GP inside an active-learning loop, and it uses
one property of this space that `goal_metric` never scores: *distance between molecules*. This
module measures that property, which is where a virtual node tap can plausibly win.

WHAT IS MEASURED, AND WHY EACH COLUMN EARNS ITS PLACE
-----------------------------------------------------
Every number here comes from `emb_readout.py` -- `FoldReference`, `structural_columns`,
`embedding_metrics` -- reused rather than reimplemented. That module's CLI reads one
`val_embeddings.npy` per run directory; this driver feeds it the six tensors out of
`vn_embeddings.npz` instead. That difference, and only that, is why this is a separate file.

  tanimoto_spearman   rho(embedding distance, 1 - ECFP4 Tanimoto). Does distance in this space
                      track structural dissimilarity, against a fingerprint the model never saw?
  scalarness          rho(embedding distance, |delta predicted pProp|). THE FAILURE MODE. At
                      ~1.0 the GP's notion of "far apart" IS "different predicted pProp", so
                      posterior variance stops tracking ignorance and active learning has
                      nothing to steer on.
  tanimoto_partial    the first with the second partialled out. A model that predicts pProp
                      well gets some structural correlation for free, because similar molecules
                      dock similarly; this is what survives removing that.
  knn20_jaccard       overlap between each molecule's 20 nearest by ECFP and its 20 nearest by
                      embedding. What a kernel with a length scale actually sees -- global rho
                      can be carried by the far tail while local neighbourhoods disagree.
  emb_effective_rank  how many dimensions are doing work, on the same scale as the width.

THE PRETRAINED ROW IS THE CONTROL
---------------------------------
`<root>/pretrained/frozen` holds the same six tensors from an un-fine-tuned MiniMol
(`extract_embeddings.py --pretrained`). Without it, a VN win here has two explanations that the
30 fine-tuned models cannot separate: the virtual node is intrinsically richer, or fine-tuning
simply never reached it. With it, every tap is read as a displacement from its own starting
point.

It has no head, so there is no `val_predictions.npy` and the three pred-dependent columns do not
exist for it. **They are written as nan by an explicit branch, not by nan arithmetic** --
`structural_columns` builds `pred_rank` unconditionally, and handing it a constant would give
`corrcoef` a zero-variance input, a RuntimeWarning, and a `denom` that quietly poisons
`tanimoto_partial`.

Usage:
    python src/vn_geometry.py --root outputs/vn_analysis/lyc0lh2d
    python src/vn_geometry.py --root outputs/vn_analysis/lyc0lh2d --limit 1   # the smoke gate
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emb_readout import (KNN_K, SAMPLE_SEED, FoldReference, euclidean_matrix,  # noqa: E402
                         knn_jaccard, knn_sets, structural_columns, upper)
from metrics import embedding_metrics                                      # noqa: E402
from scipy.stats import rankdata                                           # noqa: E402
from vn_probe import discover                                              # noqa: E402
from vn_taps import DISPLAY_NAMES, EMBEDDING_ORDER                         # noqa: E402

# The three columns that need the model's own predicted pProp. Named once, so the pretrained
# branch and the schema cannot drift apart.
PRED_COLUMNS = ("pred_tanimoto_spearman", "scalarness", "tanimoto_partial")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="outputs/vn_analysis/<sweep_id>")
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--fold", type=int, default=0,
                   help="the fold these runs validated on; the ECFP reference is drawn from it")
    p.add_argument("--n-sample", type=int, default=5000,
                   help="molecules in the structural reference. FIXED across every embedding "
                        "and every model -- knn20_jaccard is sample-size dependent, so two "
                        "sizes are two different measurements. Pairs grow as n^2")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N models -- the smoke gate")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="default: figures/<sweep_id>/geometry_metrics.csv")
    return p.parse_args(argv)


def structural_no_pred(ref, z_sample):
    """`structural_columns` minus everything that needs a prediction.

    The pretrained trunk has no head, so `|delta predicted pProp|` does not exist for it and
    the three columns built from it are nan. Computing them from a stand-in -- zeros, or the
    true pProp -- would produce numbers that look like the fine-tuned ones and mean something
    else entirely, which is worse than a gap.
    """
    emb_dist = euclidean_matrix(z_sample)
    emb_rank = rankdata(upper(emb_dist))
    emb_knn = knn_sets(emb_dist, KNN_K, larger_is_closer=False)
    return {"tanimoto_spearman": float(np.corrcoef(emb_rank, ref.dis_rank)[0, 1]),
            **{c: np.nan for c in PRED_COLUMNS},
            "knn20_jaccard": knn_jaccard(ref.tan_knn, emb_knn, KNN_K),
            "n_readout": len(ref.rows)}


def main(argv=None):
    args = parse_args(argv)
    sweep_id = args.root.name
    runs = discover(args.root)
    if args.limit:
        runs = runs[:args.limit]

    # Built ONCE. `rankdata` over 12.5M pairs is the expensive step here and it does not depend
    # on the model, so the cost is paid a single time across all 186 scorings.
    t0 = time.time()
    ref = FoldReference(str(args.splits), args.fold, args.n_sample)
    print(f"sweep {sweep_id} | {len(runs)} models x {len(EMBEDDING_ORDER)} embeddings = "
          f"{len(runs) * len(EMBEDDING_ORDER)} scorings")
    print(f"structural reference: {len(ref.rows):,} molecules from fold {args.fold} "
          f"({time.time() - t0:.0f}s to build)\n")

    records = []
    for i, r in enumerate(runs, 1):
        d = np.load(r["path"])
        # Sampling is by dataset ROW ID against the frozen split, never by position, and
        # `positions_in` raises if this run's rows do not contain the reference sample. That is
        # the guard that turns the `--subset` footgun into an error: a subset run holds only a
        # fraction of the fold and would otherwise join against almost nothing and return nan.
        pos = ref.positions_in(d["row_indices"])

        pred_path = r["path"].parent / "val_predictions.npy"
        pred = np.load(pred_path)[pos] if pred_path.exists() else None
        if pred is None and r["config"] != "pretrained":
            raise SystemExit(f"{pred_path} is missing, but {r['config']}/{r['bootstrap']} is "
                             "not the pretrained baseline -- a fine-tuned run without its "
                             "predictions cannot be scored on scalarness")

        # `z` -- the head export -- is scored beside the six trunk tensors even though it is
        # not in the npz. It is the CURRENT deliverable, so leaving it out would compare the
        # candidates against each other and never against the thing they would replace.
        extra = {}
        emb_path = r["path"].parent / "val_embeddings.npy"
        if emb_path.exists():
            extra["z"] = np.load(emb_path)

        line = f"[{i:>2}/{len(runs)}] {r['config']}/{r['bootstrap']}"
        for key in list(EMBEDDING_ORDER) + list(extra):
            src = extra[key] if key in extra else d[key]
            z = src[pos].astype(np.float64)
            cols = (structural_columns(ref, z, pred) if pred is not None
                    else structural_no_pred(ref, z))
            records.append({
                "sweep_id": sweep_id, "config": r["config"], "bootstrap": r["bootstrap"],
                "embedding": key,
                "embedding_label": DISPLAY_NAMES.get(key, "Head export (z)"),
                "fold": args.fold, "input_dim": int(src.shape[1]),
                "n_rows": int(len(d["row_indices"])),
                **cols, **embedding_metrics(z),
            })
            line += f" {key}:{cols['tanimoto_partial']:.3f}" if pred is not None else \
                    f" {key}:{cols['tanimoto_spearman']:.3f}*"
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    out = args.out or Path("figures") / sweep_id / "geometry_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".meta.json").write_text(json.dumps({
        "script": "src/vn_geometry.py", "argv": sys.argv, "sweep_id": sweep_id,
        "n_models": len(runs), "embeddings": list(EMBEDDING_ORDER) + ["z"],
        "fold": args.fold, "n_sample": len(ref.rows), "sample_seed": SAMPLE_SEED,
        "knn_k": KNN_K,
        "pred_columns_nan_for": "pretrained (no head, so no val_predictions.npy)",
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))

    print(f"\n{len(df)} rows -> {out}  ({(time.time() - t0) / 60:.1f} min)")
    cols = ["tanimoto_spearman", "scalarness", "tanimoto_partial", "knn20_jaccard",
            "emb_effective_rank"]
    ft = df[df["config"] != "pretrained"]
    print("\nmean over the fine-tuned models, by embedding:")
    # Grouped on `embedding`, not on the display label, and NOT reindexed onto
    # EMBEDDING_ORDER -- that constant knows only the six trunk tensors, so reindexing it would
    # silently drop `z`, which is the row a reader most wants to see.
    print(ft.groupby("embedding", sort=False)[cols].mean().round(4).to_string())
    pre = df[df["config"] == "pretrained"]
    if len(pre):
        print("\npretrained (un-fine-tuned) MiniMol, n = 1:")
        print(pre.set_index("embedding")[cols].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
