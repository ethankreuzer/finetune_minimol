"""Aggregate the grid's per-run meta.json into a timing table for the report.

    python src/collect_runs.py
    python src/collect_runs.py --runs outputs --out reports/

WHY FIRST-EPOCH IS SEPARATED FROM STEADY STATE
-----------------------------------------------
The first epoch of a run pays costs no later epoch does: CUDA context creation, cuDNN
autotuning, DataLoader worker startup, and page-cache warm-up on a 4.8 GB feature file. The
first epoch *after unfreezing* pays some of them again, because the backward path through
the trunk is being built for the first time.

Averaging those into a per-epoch figure would inflate the cost model that the whole TamIA
projection rests on. With only 2 epochs per phase in this grid, epoch 1 of each phase is the
warm-up sample and epoch 2 is the steady-state one -- so the distinction is not a refinement,
it is half the data. Both are reported; the cost model uses steady state and says so.

Model metrics are carried through untouched. Analysing them is the separate results report.
"""

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--expect", type=int, default=10, help="runs expected; warn if fewer")
    return p.parse_args(argv)


def load_runs(root):
    runs = []
    # rglob, not glob: `run_config.py` nests a grid one level deeper, under the hyperparameter
    # configuration it belongs to (outputs/<sweep_id>/<config_id>/<run_id>/meta.json).
    for meta_path in sorted(root.rglob("meta.json")):
        meta = json.loads(meta_path.read_text())
        cfg, hw = meta.get("config", {}), meta.get("hardware", {})
        for h in meta["history"]:
            runs.append({
                "run": meta_path.parent.name,
                "fold": cfg.get("fold"), "seed": cfg.get("seed"),
                "epoch": h["epoch"], "phase": h["phase"],
                # first epoch overall, and the first after unfreezing, are both warm-up
                "warmup": h["epoch"] == 1 or h["epoch"] == (cfg.get("freeze_epochs", 0) + 1),
                "train_s": h.get("train_seconds"), "val_s": h.get("val_seconds"),
                "total_s": h.get("seconds"), "peak_gpu_gb": h.get("peak_gpu_gb"),
                "val_pearson": h.get("val/pearson"), "val_mse": h.get("val/mse"),
                "gpu": hw.get("gpu_name"), "batch_size": cfg.get("batch_size"),
                "workers": cfg.get("num_workers"),
                "n_train": meta.get("n_train"), "n_val": meta.get("n_val"),
                "run_minutes": meta.get("total_minutes"),
                "slurm_task": hw.get("slurm_array_task_id"),
            })
    return pd.DataFrame(runs)


def summarise(df):
    """Cost model inputs: steady-state medians per phase, with warm-up quantified."""
    out = {}
    for phase in sorted(df["phase"].unique()):
        sub = df[df["phase"] == phase]
        steady, warm = sub[~sub["warmup"]], sub[sub["warmup"]]
        entry = {
            "n_steady_epochs": int(len(steady)),
            "n_warmup_epochs": int(len(warm)),
            "steady_train_s_median": float(steady["train_s"].median()),
            "steady_train_s_std": float(steady["train_s"].std()) if len(steady) > 1 else 0.0,
            "steady_val_s_median": float(steady["val_s"].median()),
            "steady_epoch_s_median": float(steady["total_s"].median()),
            "warmup_epoch_s_median": float(warm["total_s"].median()) if len(warm) else None,
            "peak_gpu_gb_max": float(sub["peak_gpu_gb"].max()),
        }
        if entry["warmup_epoch_s_median"]:
            entry["warmup_overhead_pct"] = round(
                (entry["warmup_epoch_s_median"] / entry["steady_epoch_s_median"] - 1) * 100, 1)
        n_train = int(df["n_train"].iloc[0])
        entry["steady_train_mol_per_s"] = n_train / entry["steady_train_s_median"]
        out[phase] = entry
    return out


def main(argv=None):
    args = parse_args(argv)
    df = load_runs(args.runs)
    if df.empty:
        raise SystemExit(f"no runs found under {args.runs}")

    n_runs = df["run"].nunique()
    if n_runs != args.expect:
        print(f"WARNING: found {n_runs} runs, expected {args.expect}")

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "grid_timings.csv", index=False)

    phases = summarise(df)
    per_run = df.groupby("run")["total_s"].sum().div(60)
    grid = {
        "n_runs": int(n_runs),
        "gpus": sorted(df["gpu"].dropna().unique().tolist()),
        "epochs_per_run": int(df.groupby("run")["epoch"].max().median()),
        "run_minutes_median": float(per_run.median()),
        "run_minutes_min": float(per_run.min()),
        "run_minutes_max": float(per_run.max()),
        "total_gpu_hours": float(df["total_s"].sum() / 3600),
        "n_train": int(df["n_train"].iloc[0]), "n_val": int(df["n_val"].iloc[0]),
        "batch_size": int(df["batch_size"].iloc[0]),
        "num_workers": int(df["workers"].iloc[0]),
    }
    summary = {"grid": grid, "phases": phases,
               "note": ("Timings are end-to-end wall clock per epoch, including the CPU "
                        "data pipeline overlapped with the GPU. They are NOT gpu_only "
                        "figures -- see src/benchmark.py for those.")}
    (args.out / "grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{n_runs} runs, {grid['epochs_per_run']} epochs each, "
          f"{grid['total_gpu_hours']:.2f} GPU-hours total")
    print(f"per-run wall: median {grid['run_minutes_median']:.1f} min "
          f"(range {grid['run_minutes_min']:.1f}-{grid['run_minutes_max']:.1f})")
    print(f"\n{'phase':10s} {'steady epoch':>13s} {'train':>8s} {'val':>7s} "
          f"{'warmup':>8s} {'ovh':>6s} {'peak GB':>8s} {'mol/s':>9s}")
    for phase, e in phases.items():
        print(f"{phase:10s} {e['steady_epoch_s_median']:12.1f}s "
              f"{e['steady_train_s_median']:7.1f}s {e['steady_val_s_median']:6.1f}s "
              f"{(e['warmup_epoch_s_median'] or 0):7.1f}s "
              f"{e.get('warmup_overhead_pct', 0):5.1f}% "
              f"{e['peak_gpu_gb_max']:7.2f} {e['steady_train_mol_per_s']:9.0f}")
    print(f"\nwrote {args.out/'grid_timings.csv'} and {args.out/'grid_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
