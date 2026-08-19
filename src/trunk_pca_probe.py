"""Does the learned 32-d bottleneck earn its place over a PCA of the fine-tuned trunk?

    .venv/bin/python src/trunk_pca_probe.py --runs outputs/ckpt_v4 -o reports/trunk_pca.csv

THE QUESTION. The collapse lives entirely in `head.shared`'s learned 32-d bottleneck: it is
trained to predict pProp, pProp needs ~2-4 directions, and S2.7 showed nothing ever asks the
other 30 to do anything (collapse by neglect, not compression). A PCA of the trunk's 512-d
output sidesteps that by not learning the projection at all -- the 32 directions are orthogonal
by construction and each carries the most variance remaining, so no dimension can be dead or a
duplicate of another.

WHAT PCA DOES AND DOES NOT GUARANTEE. It cannot produce a dead or duplicate direction. It does
NOT guarantee a high `emb_effective_rank`, which measures how EVENLY variance is spread: if the
trunk's own output is variance-concentrated, the top-32 spectrum stays skewed. `pca32_white`
divides each component by its standard deviation, which forces effective rank to exactly 32 in
closed form -- what the `vic` variance hinge spends a loss term approximating. The cost is the
classical one: whitening amplifies the lowest-variance directions, which are the noisiest.

THE BASIS IS FITTED ON TRAINING-FOLD MOLECULES ONLY. Fitting it on the validation fold would
leak the probe's test covariance into the featurization -- small, but exactly the kind of leak
that makes a comparison climb for the wrong reason. Training rows are also what a deployment
would actually have.

EACH CONFIG IS COMPARED AGAINST PCA OF *ITS OWN* TRUNK. `--w-vic` reaches the trunk through the
shared optimizer, so `A_base`'s trunk and `D_w3`'s trunk are different objects and borrowing one
for the other's comparison would confound the question.

Scored by the same held-out-cluster probe as S2.5, so the numbers drop straight into that table.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utility import load_reference, score_featurizations               # noqa: E402

PCA_KS = (2, 8, 32)


def trunk_from_checkpoint(ckpt_path):
    """Rebuild the FINE-TUNED trunk. Strict load: a silent partial load is the footgun."""
    import trunk as trunk_mod                                                  # noqa: E402
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state"]
    pref = "trunk."
    tsd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    if not tsd:
        raise ValueError(f"{ckpt_path}: no 'trunk.' keys in model_state")
    tk = trunk_mod.MiniMolTrunk()
    missing, unexpected = tk.load_state_dict(tsd, strict=False)
    # `minimol/model.py` uses strict=False elsewhere and silently tolerates a partly-random
    # trunk; this asserts instead of trusting it (CLAUDE.md, Footguns).
    if missing or unexpected:
        raise ValueError(f"{ckpt_path}: missing={list(missing)[:4]} "
                         f"unexpected={list(unexpected)[:4]}")
    tk.eval()
    return tk, ck["config"]


def trunk_forward(tk, rows, features_dir, batch_size=512):
    from features import load_features                                        # noqa: E402
    ds = load_features(str(features_dir))
    loader = DataLoader(Subset(ds, list(rows)), batch_size=batch_size, shuffle=False,
                        collate_fn=tk.collate)
    out = []
    with torch.no_grad():
        for batch in loader:
            out.append(tk(tk.to_device(batch)).detach().cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def pca_basis(x_fit):
    """Centre, then the right singular vectors. Returns (mu, components, per-dim std)."""
    mu = x_fit.mean(0)
    _, s, vt = np.linalg.svd(x_fit - mu, full_matrices=False)
    return mu, vt, s / np.sqrt(len(x_fit) - 1)


def effective_rank(z):
    ev = np.linalg.eigvalsh(np.cov(z, rowvar=False))
    ev = np.clip(ev, 0, None)
    p = ev / ev.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, default=[Path("outputs/ckpt_v4")])
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--csv-path", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--features", type=Path, default=Path("data/features/minimol_v1"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-per-half", type=int, default=10000)
    p.add_argument("--n-pca-fit", type=int, default=20000,
                   help="training-fold molecules used to fit the PCA basis")
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("-o", "--out", type=Path, default=Path("reports/trunk_pca.csv"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import splits as splits_mod                                               # noqa: E402
    train_idx, _ = splits_mod.load_fold(str(args.splits), fold=args.fold)
    _, halves, y = load_reference(args.splits, args.csv_path, args.fold,
                                  args.n_per_half, args.seed)
    rng = np.random.default_rng(args.seed)
    pca_rows = np.sort(rng.choice(np.asarray(train_idx), args.n_pca_fit, replace=False))
    print(f"fold {args.fold}: halves {len(halves[0])}/{len(halves[1])}; "
          f"PCA basis from {len(pca_rows)} TRAINING rows")

    runs = [d for root in args.runs for f in sorted(Path(root).rglob("final.pt"))
            for d in [f.parent]]
    if not runs:
        raise SystemExit(f"no checkpoints under {args.runs}")

    rows = []
    for d in runs:
        tk, cfg = trunk_from_checkpoint(d / "final.pt")
        cell, seed = d.parent.name, int(cfg["seed"])
        x_pca = trunk_forward(tk, pca_rows, args.features)
        mu, comps, sds = pca_basis(x_pca)
        h512 = [trunk_forward(tk, h, args.features) for h in halves]

        val_indices = np.load(d / "val_indices.npy")
        z32 = np.load(d / "val_embeddings.npy").astype(np.float64)
        pred = np.load(d / "val_predictions.npy").astype(np.float64)
        order = np.argsort(val_indices)
        srt = val_indices[order]

        per_dir = []
        for direction in (0, 1):
            hf, ht = (halves[0], halves[1]) if direction == 0 else (halves[1], halves[0])
            xf, xt = (h512[0], h512[1]) if direction == 0 else (h512[1], h512[0])
            pf = order[np.searchsorted(srt, hf)]
            pt = order[np.searchsorted(srt, ht)]
            feats = {"emb32": (z32[pf], z32[pt]),
                     "pred1": (pred[pf].reshape(-1, 1), pred[pt].reshape(-1, 1)),
                     "trunk512_ft": (xf, xt)}
            for k in PCA_KS:
                p = comps[:k]
                feats[f"trunkpca{k}"] = ((xf - mu) @ p.T, (xt - mu) @ p.T)
            pw = comps[:32]
            feats["trunkpca32_white"] = (((xf - mu) @ pw.T) / sds[:32],
                                         ((xt - mu) @ pw.T) / sds[:32])
            per_dir += score_featurizations(feats, y[hf], y[ht])

        agg = pd.DataFrame(per_dir).groupby(["feat", "probe"]).mean().reset_index()
        # Geometry of the featurizations themselves, on one half, for the rank comparison.
        geo = {"emb32": effective_rank(z32),
               "trunkpca32": effective_rank((h512[0] - mu) @ comps[:32].T),
               "trunkpca32_white": effective_rank(((h512[0] - mu) @ comps[:32].T) / sds[:32])}
        for _, r in agg.iterrows():
            rows.append({"cell": cell, "seed": seed, "run": str(d),
                         "w_vic": cfg.get("w_vic"),
                         "eff_rank": geo.get(r["feat"], np.nan), **r.to_dict()})
        kn = agg[agg.probe == "knn"].set_index("feat")["spearman"]
        print(f"{cell:>8} seed{seed}  knn: emb32={kn['emb32']:.4f} "
              f"pca32={kn['trunkpca32']:.4f} white={kn['trunkpca32_white']:.4f} "
              f"trunk512={kn['trunk512_ft']:.4f} | rank emb32={geo['emb32']:.1f} "
              f"pca32={geo['trunkpca32']:.1f} white={geo['trunkpca32_white']:.1f}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.out.with_suffix(".meta.json").write_text(json.dumps(
        {"script": "src/trunk_pca_probe.py", "argv": sys.argv[1:], "fold": args.fold,
         "n_per_half": int(len(halves[0])), "n_pca_fit": args.n_pca_fit,
         "seed": args.seed, "n_runs": int(df["run"].nunique())}, indent=2))
    print(f"\nwrote {args.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
