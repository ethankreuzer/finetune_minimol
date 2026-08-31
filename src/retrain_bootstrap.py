"""Retrain a sweep's best configurations on bootstrap resamples, one worker per GPU.

The sweep saved no weights -- `run_config.py:238` forces `--no-save-checkpoint` on every trial,
because 34 MB x 10 models x a few hundred trials is ~85 GB that nothing reads. The internal
feature analysis is post-hoc on a trained model, so the winners have to be trained again, this
time keeping the checkpoints.

The design being executed: hold out one CV fold for validation, and train each configuration on
**10 bootstrap resamples** of the other four folds -- each resample drawn with replacement to the
full size of the training side. Three configurations x ten resamples is thirty models, and the
spread across the ten is what puts error bars on everything measured downstream. The model seed
moves with the bootstrap index, so that spread is data variance and optimisation variance
together rather than data variance alone.

Each model is a **subprocess**, not an in-process `train.main()` call. `run_config.py` calls
in-process and is right to -- it trains ten models one after another. Here three run at once on
three GPUs, and three CUDA contexts in one interpreter would be a new failure surface for no
gain. A subprocess also means one model dying takes only itself down, which matters across a
multi-hour run.

Usage:
    python src/retrain_bootstrap.py --configs outputs/vn_analysis/lyc0lh2d/top_configs.json
    python src/retrain_bootstrap.py --configs ... --limit 1          # the gate: one model
    python src/retrain_bootstrap.py --configs ... --dry-run
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv" / "bin" / "python"

# The model seed is offset from the bootstrap index rather than equal to it. Both are meant to
# vary together, but `train.py` seeds numpy globally from --seed while the resample draws from
# `default_rng(--bootstrap-seed)`; handing both the same integer invites the question of whether
# the two draws are independent. The offset costs nothing and removes the question.
SEED_OFFSET = 1000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", type=Path, required=True,
                   help="top_configs.json written by src/sweep_top.py")
    p.add_argument("--root", type=Path, default=None,
                   help="output root; default: the directory holding --configs")
    p.add_argument("--fold", type=int, default=0,
                   help="the held-out validation fold; the other four are the bootstrap source")
    p.add_argument("--n-bootstrap", type=int, default=10)
    p.add_argument("--n-configs", type=int, default=None,
                   help="how many ranks to retrain; default: all in the file")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-workers", type=int, default=12,
                   help="DataLoader workers PER model; multiplied by the number of GPUs")
    p.add_argument("--features", type=Path, default=Path("data/features/minimol_v1"))
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N jobs -- the gate before committing to all of them")
    p.add_argument("--force", action="store_true", help="retrain runs that already finished")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def done(d):
    """A run is complete when it has weights and provenance. Checkpoints are the point here."""
    return (d / "final.pt").exists() and (d / "meta.json").exists()


def build_jobs(spec, args, root):
    ranks = spec["ranks"][:args.n_configs] if args.n_configs else spec["ranks"]
    jobs = []
    for entry in ranks:
        cfg_dir = root / f"cfg{entry['rank']}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps(entry, indent=2))
        for b in range(args.n_bootstrap):
            out = cfg_dir / f"boot{b:02d}"
            argv = list(entry["argv"]) + [
                "--fold", str(args.fold),
                "--seed", str(SEED_OFFSET + b),
                "--bootstrap-seed", str(b),
                "--out", str(out),
                "--no-wandb",
                "--features", str(args.features),
                "--splits", str(args.splits),
                "--csv", str(args.csv),
                "--num-workers", str(args.num_workers),
            ]
            jobs.append({"rank": entry["rank"], "boot": b, "out": out,
                         "cmd": [str(PYTHON), str(REPO / "src" / "train.py")] + argv})
    return jobs


def worker(gpu, jobs_q, results, lock):
    while True:
        try:
            job = jobs_q.get_nowait()
        except queue.Empty:
            return
        job["out"].mkdir(parents=True, exist_ok=True)
        log = job["out"] / "train.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(job["cmd"], cwd=REPO, env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        ok = rc == 0 and done(job["out"])
        with lock:
            results.append({"rank": job["rank"], "boot": job["boot"], "gpu": gpu,
                            "returncode": rc, "ok": ok, "minutes": round(mins, 1),
                            "out": str(job["out"])})
            n = len(results)
            tag = "ok " if ok else "FAIL"
            print(f"[{n:>3}] {tag} gpu{gpu} cfg{job['rank']}/boot{job['boot']:02d} "
                  f"{mins:.1f} min" + ("" if ok else f" (rc={rc}, see {log})"), flush=True)
        jobs_q.task_done()


def main(argv=None):
    args = parse_args(argv)
    spec = json.loads(args.configs.read_text())
    root = args.root or args.configs.parent
    root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(spec, args, root)
    pending = [j for j in jobs if args.force or not done(j["out"])]
    if args.limit:
        pending = pending[:args.limit]

    print(f"sweep {spec.get('sweep_id')} | fold {args.fold} held out | "
          f"{len(jobs)} models total, {len(jobs) - len(pending)} already done, "
          f"{len(pending)} to run on GPUs {args.gpus}")
    if args.dry_run or not pending:
        for j in pending:
            print(" ", " ".join(j["cmd"]))
        return 0

    jobs_q = queue.Queue()
    for j in pending:
        jobs_q.put(j)
    results, lock = [], threading.Lock()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(g, jobs_q, results, lock), daemon=True)
               for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [r for r in results if not r["ok"]]
    report = root / "retrain_summary.json"
    report.write_text(json.dumps({"sweep": spec.get("sweep"), "fold": args.fold,
                                  "n_bootstrap": args.n_bootstrap, "seed_offset": SEED_OFFSET,
                                  "gpus": args.gpus, "results": sorted(
                                      results, key=lambda r: (r["rank"], r["boot"]))}, indent=2))
    print(f"\n{len(results) - len(failed)}/{len(results)} succeeded in "
          f"{(time.time() - t0) / 60:.1f} min wall | summary -> {report}")
    if failed:
        print("failed: " + ", ".join(f"cfg{r['rank']}/boot{r['boot']:02d}" for r in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
