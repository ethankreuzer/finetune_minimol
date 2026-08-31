"""Assemble a self-contained, zippable handover package around one trained checkpoint.

This closes the export gap named in `CLAUDE.md`: `src/export.py` did not exist, and nothing on
disk was shippable. The head is trivially portable; the **trunk is not** -- reconstructing it
needs graphium 2.4.7, the minimol 1.3.4 wheel and `trunk.py`'s exact construction ordering, so
a bare `final.pt` is not a deliverable. This script produces the directory that is.

    python src/export.py --checkpoint handover/_train/cfg1_all/final.pt \
                         --out handover/minimol_ampc_encoder_v1

Written as a script rather than assembled by hand for the usual reason: the next configuration
to ship should cost one argument, not an afternoon, and `meta.json` should record what was
built rather than what someone remembers building.

Three things it does that are not copying:

* **Vendors byte-identically.** `trunk.py`, `head.py`, `model.py` and `normalization.py` go
  across unmodified, with their sha256 sums recorded, so a diff against this repository proves
  nothing was quietly edited in transit. That is why the package tolerates flat imports rather
  than rewriting them to relative ones.
* **Generates the fixture on CPU.** `verify_install.py` compares numbers rather than key names,
  because minimol's `load_state_dict(..., strict=False)` turns a version skew into a partly
  random trunk with no error. CPU because that is where the computation is exactly repeatable.
* **Interpolates the docs from the run.** Metrics, sizes, hyperparameters and hashes are read
  out of the checkpoint and its `meta.json`, never typed into the templates, so the card cannot
  drift from the model it describes.

The package writes to `handover/`, deliberately NOT under `outputs/` -- both feature probes
discover runs by `rglob` over a root and neither gates on the provenance triple, so a stray
`val_embeddings.npy` there would be silently pooled into an analysis.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

TEMPLATE_DIR = Path(__file__).resolve().parent / "export_pkg"

# Copied verbatim into the package. These four are what a consumer needs to reconstruct the
# model, and they are exactly the files `encoder.py` imports.
VENDORED = ("trunk.py", "head.py", "model.py", "normalization.py")

# The fine-tuned encoder's own performance, measured on HELD-OUT data by ten identically
# configured sibling models (`reports/sweep_lyc0lh2d.md`). Kept here as a constant because the
# shipped checkpoint is a refit on everything and therefore has no held-out number of its own
# -- the model card says so in as many words. Regenerate with the snippet in that report.
HELD_OUT_SOURCE = "outputs/vn_analysis/lyc0lh2d/cfg1"

FIXTURE_N = 64


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="final.pt from a `train.py --train-all` run")
    p.add_argument("--out", type=Path, required=True, help="package directory to write")
    p.add_argument("--csv", type=Path, default=REPO / "data/ampc_subset_331k.csv")
    p.add_argument("--features", type=Path, default=REPO / "data/features/minimol_v1")
    p.add_argument("--siblings", type=Path, default=REPO / HELD_OUT_SOURCE,
                   help="directory of held-out sibling runs whose metrics the card quotes")
    p.add_argument("--stats-batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="device for the pooled512 statistics pass; the fixture is always CPU")
    p.add_argument("--allow-partial-train", action="store_true",
                   help="ship a checkpoint that was NOT trained with --train-all. Off by "
                        "default: the generated model card claims the full dataset, and a "
                        "bootstrap replicate under that card would be a false document.")
    p.add_argument("--skip-stats", action="store_true",
                   help="skip the pooled512 mean/std pass (a full forward pass over the CSV)")
    return p.parse_args(argv)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def held_out_metrics(sibling_dir):
    """Mean and sd of every final-epoch `val/*` metric over the sibling runs.

    Read from the runs' own `meta.json` files rather than transcribed, so the card's table and
    the runs cannot disagree. Returns `{}` if the directory is absent -- the card then says so
    rather than the export failing, since the siblings are evidence about a configuration and
    not a dependency of the artifact.
    """
    metas = sorted(Path(sibling_dir).glob("*/meta.json"))
    if not metas:
        return {}
    runs = [json.loads(p.read_text()) for p in metas]

    # The card presents these as "N models at THIS configuration", so they have to actually be
    # one configuration. The glob does not know that: pointed one level higher it would happily
    # average across three different ones and report the mean as a property of this checkpoint.
    # Hyperparameters only -- the whole point is that the models differ in their data draw.
    varies = {"seed", "bootstrap_seed", "out", "fold", "num_workers", "outputs_root",
              "wandb_group", "wandb_tags", "sweep_id", "config_id"}
    signatures = {json.dumps({k: str(v) for k, v in sorted(r["config"].items())
                              if k not in varies}, sort_keys=True) for r in runs}
    if len(signatures) > 1:
        raise SystemExit(
            f"{sibling_dir} holds {len(signatures)} distinct hyperparameter configurations "
            f"across {len(runs)} runs. The model card would average them and call the result "
            "this configuration's held-out performance. Point --siblings at one configuration.")

    finals = [r["history"][-1] for r in runs]
    keys = [k for k in finals[0] if k.startswith("val/")]
    out = {"n_models": len(finals)}
    for k in keys:
        vals = [f[k] for f in finals if isinstance(f.get(k), (int, float))]
        if len(vals) == len(finals):
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def build_model(ckpt, device):
    """Reconstruct the trained model. Same route as `extract_embeddings.rebuild`."""
    from head import DualHead
    from model import MiniMolRegressor
    from trunk import MiniMolTrunk

    cfg = ckpt["config"]
    model = MiniMolRegressor(
        MiniMolTrunk(accelerator="gpu" if str(device).startswith("cuda") else "cpu"),
        DualHead(hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
                 embed_dim=cfg["embed_dim"],
                 cls_hidden_dim=cfg["cls_hidden_dim"], cls_n_layers=cfg["cls_n_layers"],
                 reg_hidden_dim=cfg["reg_hidden_dim"], reg_n_layers=cfg["reg_n_layers"],
                 dropout=cfg["dropout"], norm=cfg["head_norm"],
                 bottleneck_norm=cfg["bottleneck_norm"]),
    )
    incompatible = model.load_state_dict(ckpt["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint did not load cleanly: "
                           f"{incompatible.missing_keys[:3]} / "
                           f"{incompatible.unexpected_keys[:3]}")
    return model.to(device).eval()


def pick_fixture_rows(csv, n):
    """`n` row indices spanning the pProp range, deterministically.

    Spread across the range rather than sampled uniformly: the CSV is sorted by `-pprop`, and a
    uniform sample of a 331,480-row file with 3,153 positives would almost certainly contain
    none -- so a fixture drawn that way would never exercise the potent end, which is the part
    of the space the model was built for and the part most likely to move if something is wrong.
    """
    df = pd.read_csv(csv, usecols=["SMILES", "pprop"])
    order = np.argsort(-df["pprop"].to_numpy(dtype=np.float64), kind="stable")
    picks = np.linspace(0, len(order) - 1, n).round().astype(int)
    rows = order[picks]
    return df.iloc[rows]


def write_fixture(model, norm_stats, csv, out_dir, n=FIXTURE_N):
    """Encode `n` molecules on CPU and record the answer, for `verify_install.py`.

    CPU, always, regardless of `--device`: that is where the arithmetic is deterministic and
    exactly repeatable, so the recorded values are a fixed point rather than one machine's
    rounding. `verify_install.py` compares to 1e-4, far above the ~7e-6 CPU-vs-GPU floor.
    """
    from normalization import denormalize_pprop

    frame = pick_fixture_rows(csv, n)
    smiles = frame["SMILES"].tolist()

    cpu_model = model.to("cpu").eval()
    batch = cpu_model.trunk.featurize(smiles, to_device=False)
    with torch.no_grad():
        pooled = cpu_model.trunk(batch)
        logit, pred, z = cpu_model.head.forward_with_embedding(pooled)

    pprop = np.asarray(denormalize_pprop(pred.double().numpy(), norm_stats), dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "fixture.npz",
             pooled512=pooled.float().numpy(), z=z.float().numpy(),
             pprop=pprop, logit=logit.float().numpy(),
             true_pprop=frame["pprop"].to_numpy(dtype=np.float32))
    (out_dir / "fixture.json").write_text(json.dumps({
        "smiles": smiles,
        "n": len(smiles),
        "selection": "row indices spread evenly across the pProp-sorted CSV, so the potent "
                     "tail is represented; deterministic given the CSV",
        "device": "cpu",
        "generated_on": platform.node(),
        "tolerance": 1e-4,
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }, indent=2) + "\n")
    return smiles, pooled.float().numpy()


def _collate_cpu(graphs):
    """Mirror of `MiniMolTrunk.collate(to_device=False)` that carries no model reference."""
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(list(graphs))
    return {"features": batch, "batch_indices": batch.batch}


def pooled512_stats(model, features, device, batch_size, out_path):
    """Per-dimension mean and std of `pooled512` over the whole cached dataset.

    `pooled512` is a max-pool output, so its dimensions are neither centred nor commensurate.
    An RBF/Matern kernel weights every dimension equally, which means feeding it raw is a
    silent decision to weight dimensions by whatever scale they happen to have. Shipping the
    statistics makes standardizing a one-liner instead of a task the consumer has to think of.

    Computed in float64 by accumulating sums rather than stacking 331,480x512 floats.
    """
    from features import load_features
    from torch.utils.data import DataLoader

    cache = load_features(features)
    trunk = model.trunk
    n, dim = 0, None
    s1 = s2 = None

    # `_collate_cpu`, not `trunk.collate` and not a lambda: this runs inside forked
    # DataLoader workers, where a lambda cannot be pickled and binding the collate to a
    # CUDA-resident module would drag the model into every child. Same reason
    # `train.collate_batch` exists. The move to device happens in the loop below.
    loader = DataLoader(cache, batch_size=batch_size, shuffle=False, num_workers=8,
                        collate_fn=_collate_cpu)
    started = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pooled = trunk(trunk.to_device(batch, device)).double()
            if s1 is None:
                dim = pooled.shape[1]
                s1 = torch.zeros(dim, dtype=torch.float64, device=pooled.device)
                s2 = torch.zeros(dim, dtype=torch.float64, device=pooled.device)
            s1 += pooled.sum(0)
            s2 += (pooled * pooled).sum(0)
            n += pooled.shape[0]
            if i % 100 == 0:
                rate = n / max(time.time() - started, 1e-9)
                print(f"  {n:>9,} / {len(cache):,}  ({rate:,.0f} mol/s)", flush=True)

    mean = (s1 / n).cpu().numpy()
    var = (s2 / n).cpu().numpy() - mean ** 2
    std = np.sqrt(np.clip(var, 0.0, None))
    elapsed = time.time() - started
    # A zero-variance dimension would make the documented `(x - mean) / std` divide by zero in
    # the consumer's code rather than here. Floor it, and say how many were floored.
    dead = int((std < 1e-8).sum())
    std = np.maximum(std, 1e-8)
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32),
             n=np.int64(n))
    print(f"  pooled512 stats over {n:,} molecules in {elapsed/60:.1f} min "
          f"({n/max(elapsed,1e-9):,.0f} mol/s) | {dead} constant dims floored to 1e-8")
    return {"n": n, "dim": int(dim), "constant_dims": dead,
            "seconds": round(elapsed, 1), "mol_per_s": round(n / max(elapsed, 1e-9), 1)}


def render(template, fields):
    """Fill a `.tmpl`, failing loudly on a placeholder nobody supplied.

    `str.format` would raise `KeyError` with just the key name; this says which template, which
    is the difference between a five-second fix and a hunt.
    """
    try:
        return template.format(**fields)
    except KeyError as e:
        raise SystemExit(f"template placeholder {e} has no value; supplied: "
                         f"{sorted(fields)}") from None


def main(argv=None):
    args = parse_args(argv)
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    run_dir = ckpt_path.parent
    out = args.out.resolve()

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]

    # The refit-on-everything assertion. The generated MODEL_CARD.md states that this
    # checkpoint saw the whole dataset; shipping a fold-trained or bootstrap replicate under
    # that card would make the document false, and nothing downstream could tell.
    if not cfg.get("train_all") and not args.allow_partial_train:
        raise SystemExit(
            f"{ckpt_path} was not trained with --train-all (config['train_all'] = "
            f"{cfg.get('train_all')!r}), but the model card this script generates claims the "
            "full dataset. Retrain with --train-all, or pass --allow-partial-train and edit "
            "the card yourself.")

    run_meta = json.loads((run_dir / "meta.json").read_text())
    print(f"packaging {ckpt_path}\n  trained on {run_meta['n_train']:,} rows | "
          f"objective {run_meta['objective_version']} | git {run_meta['git_sha'][:8]}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(exist_ok=True)
    (out / "minimol_ampc").mkdir(exist_ok=True)
    (out / "examples").mkdir(exist_ok=True)

    # -- 1. vendored source, byte for byte -------------------------------------------
    vendored = {}
    for name in VENDORED:
        src = REPO / "src" / name
        dst = out / "minimol_ampc" / name
        shutil.copyfile(src, dst)
        vendored[name] = sha256_file(dst)
        if vendored[name] != sha256_file(src):
            raise SystemExit(f"{name} changed in transit -- refusing to ship")
    print(f"  vendored {len(vendored)} modules byte-identically")

    # -- 2. the package's own code, docs and env spec ---------------------------------
    for rel in ("minimol_ampc/__init__.py", "minimol_ampc/encoder.py",
                "minimol_ampc/featurize.py", "verify_install.py",
                "examples/quickstart.py", "examples/active_learning.py",
                "requirements.txt"):
        shutil.copyfile(TEMPLATE_DIR / rel, out / rel)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(REPO / name, out / name)

    # -- 3. the weights and their provenance ------------------------------------------
    shutil.copyfile(ckpt_path, out / "model" / "final.pt")
    ckpt_sha = sha256_file(out / "model" / "final.pt")
    (out / "model" / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copyfile(run_dir / "meta.json", out / "model" / "meta.json")

    # -- 4. the fixture (CPU) ----------------------------------------------------------
    device = args.device
    model = build_model(ckpt, "cpu")
    smiles, _ = write_fixture(model, ckpt["norm_stats"], args.csv, out / "fixtures")
    print(f"  fixture: {len(smiles)} molecules encoded on CPU")

    # -- 5. pooled512 statistics -------------------------------------------------------
    stats = None
    if not args.skip_stats:
        model = model.to(device)
        stats = pooled512_stats(model, args.features, device, args.stats_batch_size,
                                out / "model" / "pooled512_stats.npz")

    # -- 6. the two documents ----------------------------------------------------------
    sib = held_out_metrics(args.siblings)
    checks = run_meta.get("schedule_checks", {})
    fields = {
        "built_date": date.today().isoformat(),
        "git_sha": git_sha()[:12],
        "ckpt_sha": ckpt_sha[:16],
        "ckpt_mb": (out / "model" / "final.pt").stat().st_size / 1e6,
        "cpu_verified": date.today().isoformat(),
        "embed_dim": cfg["embed_dim"],
        "pprop_edge": ckpt.get("pprop_edge", 3.5),
        "n_train": run_meta["n_train"],
        "trunk_params": run_meta["params"]["trunk"],
        "head_params": run_meta["params"]["head"],
        "freeze_epochs": cfg["freeze_epochs"],
        "unfrozen_epochs": cfg["unfrozen_epochs"],
        "head_lr": cfg["head_lr"],
        "head_lr_unfrozen": cfg["head_lr_unfrozen"] or cfg["head_lr"],
        "trunk_lr": cfg["trunk_lr"],
        "batch_size": cfg["batch_size"],
        "pprop_norm": cfg["pprop_norm"],
        "objective_version": run_meta["objective_version"],
        "split_sha256": run_meta["split_sha256"][:16] + "...",
        "input_sha256": run_meta["input_sha256"][:16] + "...",
        "gpu_name": run_meta["hardware"].get("gpu_name", "unknown"),
        "total_minutes": run_meta["total_minutes"],
        "versions": ", ".join(f"{k} {v}" for k, v in run_meta["versions"].items()),
        "frozen_trunk_delta": checks.get("frozen_phase", {}).get("trunk_max_delta", float("nan")),
        "frozen_head_delta": checks.get("frozen_phase", {}).get("head_max_delta", float("nan")),
        "unfrozen_trunk_delta": checks.get("after_unfreeze", {}).get("trunk_max_delta", float("nan")),
        "stats_n": stats["n"] if stats else 0,
        "z_eff_rank": sib.get("val/emb_effective_rank", (float("nan"), 0))[0],
    }
    # Every held-out number in the card comes from the sibling runs' own meta.json, formatted
    # here rather than typed into the template -- so the table cannot drift from the runs it
    # claims to summarise, which is the same reason `n_train` is read from the run.
    def pm(key, places=4):
        mean, sd = sib.get(key, (float("nan"), float("nan")))
        return f"{mean:.{places}f} +/- {sd:.{places}f}"

    fields.update({
        "n_siblings": sib.get("n_models", 0),
        "sib_n_val": int(sib.get("val/n", (0, 0))[0]),
        "sib_n_pos": int(sib.get("val/n_positive", (0, 0))[0]),
        "sib_base_rate": sib.get("val/base_rate_uniform", (float("nan"), 0))[0],
        "m_pearson": pm("val/pearson_uniform"),
        "m_spearman": pm("val/spearman_uniform"),
        "m_r2": pm("val/r2_uniform"),
        "m_mae": pm("val/mae_uniform"),
        "m_ap": pm("val/ap_uniform"),
        "m_ef1": pm("val/ef_p3.5_top0.01", 1),
        "m_ef01": pm("val/ef_p3.5_top0.001", 1),
        "m_goal": pm("val/goal_metric"),
        "m_pearson_lt": f"{sib.get('val/pearson_group_lt', (float('nan'),))[0]:.3f}",
        "m_pearson_ge": f"{sib.get('val/pearson_group_ge', (float('nan'),))[0]:.3f}",
        "m_spearman_ge": f"{sib.get('val/spearman_group_ge', (float('nan'),))[0]:.3f}",
    })
    if not sib:
        raise SystemExit(
            f"no sibling runs under {args.siblings}, so the model card would have no held-out "
            "numbers at all -- and this checkpoint has none of its own. Point --siblings at a "
            "directory of runs that held a fold out.")
    for name in ("README.md", "MODEL_CARD.md"):
        text = render((TEMPLATE_DIR / f"{name}.tmpl").read_text(), fields)
        (out / name).write_text(text)
    print(f"  rendered README.md and MODEL_CARD.md")

    # -- 7. the package's own provenance ------------------------------------------------
    meta = {
        "script": "src/export.py",
        "argv": sys.argv,
        "built": date.today().isoformat(),
        "repo_git_sha": git_sha(),
        "checkpoint_source": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "vendored_sha256": vendored,
        "vendored_note": "byte-identical copies of src/*.py from the source repository at "
                         "repo_git_sha; diff them to confirm",
        "run_meta": {k: run_meta[k] for k in
                     ("n_train", "n_val", "objective_version", "split_sha256",
                      "input_sha256", "total_minutes", "val_is_in_sample")
                     if k in run_meta},
        "held_out_siblings": {"source": str(args.siblings),
                              "n_models": sib.get("n_models", 0)},
        "pooled512_stats": stats,
        "fixture": {"n": len(smiles), "device": "cpu", "tolerance": 1e-4},
        "versions": run_meta["versions"],
        "host": platform.node(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Running anything inside the package leaves `__pycache__` behind, and a rebuild would
    # otherwise ship one machine's bytecode -- compiled by a different interpreter than the
    # recipient's, and pure noise in a directory whose whole job is to be auditable.
    for cache_dir in out.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nwrote {out} ({total / 1e6:.0f} MB)\n"
          f"next: cd {out} && python verify_install.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
