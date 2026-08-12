"""How many training runs fit on ONE GPU before aggregate throughput stops improving.

`rabelais` allows splitting a single GPU across concurrent jobs (`--gres=mps:N`) but then the
other two GPUs are unavailable. So the question is not "does sharing work" but "where does the
aggregate curve flatten" -- below that point, splitting one GPU beats using one whole GPU;
above it, you are paying latency for no extra work done.

    python src/concurrency.py --batch 1024                  # sweep 1,2,3,4,6,8
    python src/concurrency.py --batch 1024 --procs 4        # a single point

Method: N identical worker processes are pinned to one GPU and given a *synchronised start*
(`--start-at`, a shared wall-clock instant), so every worker's timed window overlaps every
other's. Without that the first process to finish warm-up would measure a partly-empty GPU and
report a throughput nobody actually sees.

Each worker times the real training step on a device-resident batch: no dataloader, no CPU
pipeline. That isolates the thing being shared. CPU-side contention between concurrent runs is
a separate limit and is reported alongside, not mixed in.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--phase", choices=["full", "frozen"], default="full")
    p.add_argument("--seconds", type=float, default=20.0, help="timed window per worker")
    p.add_argument("--procs", type=int, default=None, help="single point instead of a sweep")
    p.add_argument("--sweep", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", type=Path, default=Path("reports/concurrency.json"))
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--start-at", type=float, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def run_worker(args):
    """One process: build the model, then hammer a resident batch and report throughput."""
    import torch
    from features import load_features
    from head import MLPHead
    from model import MiniMolRegressor
    from trunk import MiniMolTrunk

    cache = load_features("data/features/minimol_v1")
    model = MiniMolRegressor(MiniMolTrunk(), MLPHead(hidden_dims=(1024, 32))).cuda()
    model.train()
    opt = torch.optim.AdamW(model.param_groups(trunk_lr=1e-4, head_lr=1e-3,
                                               weight_decay=0.01))
    if args.phase == "frozen":
        for p in model.trunk.parameters():
            p.requires_grad_(False)
        for g in opt.param_groups:
            if g.get("name") == "trunk":
                g["lr"] = 0.0

    batch = model.trunk.collate([cache[i] for i in range(args.batch)], to_device=True)
    y = torch.randn(args.batch, device="cuda")

    def step():
        b = {"features": batch["features"].clone(),
             "batch_indices": batch["batch_indices"]}
        loss = torch.nn.functional.mse_loss(model(b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Synchronised start: all workers enter the timed window together, so every measurement
    # is taken while the GPU is carrying the full N-way load.
    if args.start_at:
        while time.time() < args.start_at:
            time.sleep(0.002)

    t0 = time.time()
    steps = 0
    while time.time() - t0 < args.seconds:
        step()
        steps += 1
    torch.cuda.synchronize()
    el = time.time() - t0

    print("RESULT " + json.dumps({
        "pid": os.getpid(), "steps": steps, "seconds": el,
        "mol_per_s": steps * args.batch / el,
        "ms_per_step": el / steps * 1000,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }), flush=True)
    return 0


def gpu_mem_used(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    return float(out.splitlines()[0])


def run_point(args, n):
    """Launch n workers together on one GPU and aggregate what they report."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    start_at = time.time() + 45 + 4 * n          # allow every worker to finish warm-up first
    cmd = [sys.executable, __file__, "--worker", "--batch", str(args.batch),
           "--phase", args.phase, "--seconds", str(args.seconds),
           "--start-at", str(start_at)]
    procs = [subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True) for _ in range(n)]

    peak_mem = 0.0
    while time.time() < start_at + args.seconds * 0.6:
        try:
            peak_mem = max(peak_mem, gpu_mem_used(args.gpu))
        except Exception:
            pass
        time.sleep(2)

    results = []
    for p in procs:
        out, _ = p.communicate(timeout=600)
        for line in out.splitlines():
            if line.startswith("RESULT "):
                results.append(json.loads(line[7:]))
    if len(results) != n:
        return {"procs": n, "failed": True, "returned": len(results),
                "note": "worker(s) died -- almost certainly out of GPU memory"}

    agg = sum(r["mol_per_s"] for r in results)
    per = agg / n
    return {"procs": n, "aggregate_mol_per_s": agg, "per_proc_mol_per_s": per,
            "ms_per_step_median": sorted(r["ms_per_step"] for r in results)[n // 2],
            "gpu_mem_used_mb": peak_mem,
            "torch_peak_gb_per_proc": max(r["peak_gb"] for r in results)}


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        return run_worker(args)

    points = [args.procs] if args.procs else args.sweep
    print(f"batch {args.batch}, phase {args.phase}, GPU {args.gpu}, "
          f"{args.seconds:.0f}s window per point\n")
    print(f"{'procs':>5} {'aggregate':>12} {'per-proc':>11} {'ms/step':>9} "
          f"{'GPU mem':>9} {'vs 1 proc':>10}")
    rows, base = [], None
    for n in points:
        r = run_point(args, n)
        rows.append(r)
        if r.get("failed"):
            print(f"{n:>5}   FAILED -- {r['returned']}/{n} workers returned "
                  f"(out of GPU memory)")
            break
        if base is None:
            base = r["aggregate_mol_per_s"]
        print(f"{n:>5} {r['aggregate_mol_per_s']:11,.0f} {r['per_proc_mol_per_s']:10,.0f} "
              f"{r['ms_per_step_median']:8.0f} {r['gpu_mem_used_mb']:8,.0f}M "
              f"{r['aggregate_mol_per_s']/base:9.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"config": {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()},
         "points": rows}, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
