"""Characterise what this workload actually costs, and what limits it.

The evidence base for `reports/compute_profile.md`. The question it exists to answer is not
"how fast is this" but "*what* is it waiting on" -- because that is what decides whether
H100/H200 hardware would help, and by how much. A projection built on a FLOPS ratio is
worthless if the step is launch-bound.

MUST RUN ON AN IDLE GPU. Anything else measures contention. Check with `nvidia-smi` first;
`--require-idle` refuses to start if another process is resident.

    python src/benchmark.py                       # everything, ~15 min
    python src/benchmark.py --only step,roofline  # a subset

THE MEASUREMENT ERROR THIS SCRIPT IS BUILT TO AVOID
---------------------------------------------------
An earlier probe timed `deepcopy + collate + forward + backward` in one loop and reported
"mol/s". That number conflates the CPU pipeline with the GPU step, and is not comparable to
the training loop's throughput, where DataLoader workers overlap the CPU half. It made the
GPU look ~4x slower than it is.

So every timing here is labelled as one of:
  gpu_only  -- batch pre-collated and resident on the device; synchronize; time N steps
  cpu_only  -- dataset __getitem__ + collate, no GPU
  e2e       -- the real loop, with workers, as training actually runs
Mixing them is how a compute report ends up recommending the wrong hardware.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import load_features          # noqa: E402
from head import MLPHead                    # noqa: E402
from model import MiniMolRegressor          # noqa: E402
from trunk import MiniMolTrunk              # noqa: E402

ALL_SECTIONS = ["step", "batch", "roofline", "kernels", "precision", "loader"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/features/minimol_v1"))
    p.add_argument("--out", type=Path, default=Path("reports/benchmark.json"))
    p.add_argument("--batch-size", type=int, default=256, help="the reference batch size")
    p.add_argument("--batch-sweep", type=int, nargs="*",
                   default=[64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--worker-sweep", type=int, nargs="*", default=[0, 4, 8, 16, 32])
    p.add_argument("--steps", type=int, default=20, help="timed steps per measurement")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3,
                   help="independent repeats of the reference step timing, for variance")
    p.add_argument("--only", default=None,
                   help=f"comma-separated subset of {ALL_SECTIONS}")
    p.add_argument("--allow-busy", action="store_true",
                   help="run even if the GPU already has resident processes (not advised)")
    return p.parse_args(argv)


# -- helpers ------------------------------------------------------------------------

def gpu_busy():
    """Other processes resident on this GPU, which would contaminate every timing."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return [l for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def timed(fn, steps, warmup):
    """Median and spread of `fn`, with the GPU synchronised around the timed region.

    Median rather than mean: a single scheduler hiccup or clock excursion skews a mean and
    would then propagate into the projection. Spread is reported so a noisy measurement is
    visible rather than averaged away.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        per.append(time.perf_counter() - t0)
    per = np.array(per)
    return {"ms_median": float(np.median(per) * 1000),
            "ms_p10": float(np.percentile(per, 10) * 1000),
            "ms_p90": float(np.percentile(per, 90) * 1000),
            "ms_min": float(per.min() * 1000)}


def build_model(hidden=(1024, 32)):
    m = MiniMolRegressor(MiniMolTrunk(), MLPHead(hidden_dims=hidden)).cuda()
    m.train()
    return m


def make_opt(model, trunk_lr=1e-4, head_lr=1e-3):
    return torch.optim.AdamW(model.param_groups(trunk_lr=trunk_lr, head_lr=head_lr,
                                                weight_decay=0.01))


def set_frozen(model, optimizer, frozen):
    for p in model.trunk.parameters():
        p.requires_grad_(not frozen)
    for g in optimizer.param_groups:
        if g.get("name") == "trunk":
            g["lr"] = 0.0 if frozen else 1e-4


def resident_batch(model, cache, n):
    """A collated batch already on the GPU, so timing excludes all CPU work."""
    batch = model.trunk.collate([cache[i] for i in range(n)], to_device=True)
    y = torch.randn(n, device="cuda")
    return batch, y


def step_fn(model, optimizer, batch, y):
    """One training step on an already-resident batch.

    NOTE the trunk mutates the batch it is given (the positional encoders concatenate into
    `feat`), so the same collated batch cannot be reused across steps. Re-collating would
    reintroduce CPU work into the timed region, so instead the batch is rebuilt on-device
    from a pristine copy -- a device-to-device clone, which stays off the CPU.
    """
    def run():
        b = {"features": batch["features"].clone(),
             "batch_indices": batch["batch_indices"]}
        loss = torch.nn.functional.mse_loss(model(b), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return run


# -- sections -----------------------------------------------------------------------

def bench_step(model, optimizer, cache, args):
    """Reference GPU-only step time, both phases, repeated for variance."""
    out = {}
    for phase in ("frozen", "full"):
        set_frozen(model, optimizer, phase == "frozen")
        batch, y = resident_batch(model, cache, args.batch_size)
        torch.cuda.reset_peak_memory_stats()
        reps = [timed(step_fn(model, optimizer, batch, y), args.steps, args.warmup)
                for _ in range(args.repeats)]
        med = float(np.median([r["ms_median"] for r in reps]))
        spread = float(np.std([r["ms_median"] for r in reps]))
        out[phase] = {"kind": "gpu_only", "batch_size": args.batch_size,
                      "ms_median": med, "ms_std_across_repeats": spread,
                      "mol_per_s": args.batch_size / (med / 1000),
                      "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                      "repeats": reps}
        print(f"  step [{phase:6s}] {med:6.1f} ms +-{spread:.1f}  "
              f"{args.batch_size/(med/1000):8.0f} mol/s (gpu_only)  "
              f"peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return out


def bench_batch(model, optimizer, cache, args):
    """Throughput and memory vs batch size: does it saturate below the memory limit?"""
    out = {}
    for phase in ("frozen", "full"):
        set_frozen(model, optimizer, phase == "frozen")
        rows = []
        for B in args.batch_sweep:
            try:
                batch, y = resident_batch(model, cache, B)
                torch.cuda.reset_peak_memory_stats()
                t = timed(step_fn(model, optimizer, batch, y), max(5, args.steps // 2),
                          args.warmup)
                rows.append({"batch_size": B, "ms_median": t["ms_median"],
                             "mol_per_s": B / (t["ms_median"] / 1000),
                             "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
                print(f"  batch [{phase:6s}] {B:>5} {t['ms_median']:7.1f} ms "
                      f"{B/(t['ms_median']/1000):8.0f} mol/s "
                      f"{torch.cuda.max_memory_allocated()/1e9:6.2f} GB")
            except torch.cuda.OutOfMemoryError:
                print(f"  batch [{phase:6s}] {B:>5} OOM")
                rows.append({"batch_size": B, "oom": True})
                torch.cuda.empty_cache()
                break
            finally:
                del batch, y
                torch.cuda.empty_cache()
        out[phase] = rows
    return out


def bench_roofline(args):
    """Achieved compute and bandwidth ceilings on THIS GPU.

    The projection to H100/H200 scales each part of the step by the appropriate hardware
    ratio. Using vendor peak numbers would inflate it -- no real kernel reaches peak. These
    two microbenchmarks give the ceilings an actual kernel can hit here, so the ratio is
    achieved-to-achieved.
    """
    out = {}
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)
    flops = 2 * n ** 3
    for label, tf32, dtype in (("fp32", False, torch.float32),
                               ("tf32", True, torch.float32),
                               ("bf16", False, torch.bfloat16)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        x, w = a.to(dtype), b.to(dtype)
        t = timed(lambda: torch.mm(x, w), 20, 5)
        tflops = flops / (t["ms_median"] / 1000) / 1e12
        out[f"gemm_{label}_tflops"] = tflops
        print(f"  roofline gemm {label}: {tflops:7.1f} TFLOP/s")
        del x, w
    torch.backends.cuda.matmul.allow_tf32 = False

    # Bandwidth: a large elementwise add reads 2 arrays and writes 1.
    n_el = 1 << 28
    x = torch.randn(n_el, device="cuda")
    y = torch.randn(n_el, device="cuda")
    t = timed(lambda: torch.add(x, y), 20, 5)
    gbs = 3 * n_el * 4 / (t["ms_median"] / 1000) / 1e9
    out["bandwidth_gb_s"] = gbs
    print(f"  roofline bandwidth: {gbs:7.0f} GB/s")
    del a, b, x, y
    torch.cuda.empty_cache()
    return out


def bench_kernels(model, optimizer, cache, args):
    """How much of the step is the GPU actually busy, and across how many kernels?

    The GPU-idle fraction is the launch/CPU-bound share -- the part that does NOT improve on
    faster hardware. It is the single most important number for the projection, because it
    sets a hard ceiling on any speedup.
    """
    from torch.profiler import ProfilerActivity, profile
    out = {}
    for phase in ("frozen", "full"):
        set_frozen(model, optimizer, phase == "frozen")
        batch, y = resident_batch(model, cache, args.batch_size)
        run = step_fn(model, optimizer, batch, y)
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()

        n = 10
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                run()
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / n

        events = prof.key_averages()
        gpu_us = sum(getattr(e, "self_device_time_total", 0) or 0 for e in events)
        n_kernels = sum(e.count for e in events
                        if (getattr(e, "self_device_time_total", 0) or 0) > 0)
        gpu_s = gpu_us / 1e6 / n
        top = sorted(events, key=lambda e: -(getattr(e, "self_device_time_total", 0) or 0))
        out[phase] = {
            "wall_ms_per_step": wall * 1000,
            "gpu_busy_ms_per_step": gpu_s * 1000,
            "gpu_busy_fraction": gpu_s / wall if wall else float("nan"),
            "kernel_launches_per_step": n_kernels / n,
            "top_kernels": [
                {"name": e.key,
                 "gpu_ms_per_step": (getattr(e, "self_device_time_total", 0) or 0) / 1e3 / n,
                 "calls_per_step": e.count / n}
                for e in top[:10]
                if (getattr(e, "self_device_time_total", 0) or 0) > 0],
        }
        print(f"  kernels [{phase:6s}] wall {wall*1000:6.1f} ms, gpu busy "
              f"{gpu_s*1000:6.1f} ms ({gpu_s/wall*100:4.1f}%), "
              f"{n_kernels/n:,.0f} kernels/step")
        del batch, y
        torch.cuda.empty_cache()
    return out


def bench_precision(model, optimizer, cache, args):
    """What bf16/TF32 buy here. Ampere gains less than Hopper, which the report must note."""
    out = {}
    set_frozen(model, optimizer, False)
    batch, y = resident_batch(model, cache, args.batch_size)
    for label in ("fp32", "tf32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = (label == "tf32")
        torch.backends.cudnn.allow_tf32 = (label == "tf32")
        base = step_fn(model, optimizer, batch, y)
        if label == "bf16":
            def run():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    base()
        else:
            run = base
        t = timed(run, args.steps, args.warmup)
        out[label] = {"ms_median": t["ms_median"],
                      "mol_per_s": args.batch_size / (t["ms_median"] / 1000)}
        print(f"  precision {label}: {t['ms_median']:6.1f} ms "
              f"{args.batch_size/(t['ms_median']/1000):8.0f} mol/s")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    del batch, y
    torch.cuda.empty_cache()
    return out


def bench_loader(cache, args):
    """The CPU pipeline's ceiling -- how fast data can be delivered, GPU excluded.

    Matters for the projection via Amdahl: against a 98 ms GPU step an 8 ms CPU step is
    invisible, but against a hypothetical 20 ms step it is a third of the time. If the GPU
    gets several times faster, the CPU pipeline is what the workload runs into next.
    """
    from torch.utils.data import DataLoader, Subset
    from train import RowDataset, collate_batch
    y = np.zeros(len(cache))
    w = np.ones(len(cache))
    idx = np.arange(min(20480, len(cache)))
    out = {}
    for nw in args.worker_sweep:
        dl = DataLoader(RowDataset(cache, y, w, idx), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_batch, num_workers=nw,
                        persistent_workers=nw > 0)
        it = iter(dl)
        next(it)                      # pay worker startup outside the timing
        t0 = time.perf_counter()
        n = 0
        for _ in it:
            n += args.batch_size
        el = time.perf_counter() - t0
        out[str(nw)] = {"kind": "cpu_only", "mol_per_s": n / el,
                        "ms_per_batch": el / (n / args.batch_size) * 1000}
        print(f"  loader workers={nw:>2}: {n/el:8.0f} mol/s "
              f"({el/(n/args.batch_size)*1000:5.1f} ms/batch, cpu_only)")
        del dl
    return out


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    busy = gpu_busy()
    if busy and not args.allow_busy:
        raise SystemExit(f"GPU already has resident processes: {busy}. Every timing would "
                         "measure contention. Wait, or pass --allow-busy to override.")

    sections = ALL_SECTIONS if not args.only else [s.strip() for s in args.only.split(",")]
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory/1e9:.0f} GB, "
          f"sm_{props.major}{props.minor})\n")

    cache = load_features(args.features)
    model = build_model()
    optimizer = make_opt(model)

    results, started = {}, time.time()
    if "step" in sections:
        print("[step] GPU-only reference step");      results["step"] = bench_step(model, optimizer, cache, args)
    if "batch" in sections:
        print("[batch] batch-size sweep");            results["batch"] = bench_batch(model, optimizer, cache, args)
    if "roofline" in sections:
        print("[roofline] achieved ceilings");        results["roofline"] = bench_roofline(args)
    if "kernels" in sections:
        print("[kernels] profile");                   results["kernels"] = bench_kernels(model, optimizer, cache, args)
    if "precision" in sections:
        print("[precision] fp32/tf32/bf16");          results["precision"] = bench_precision(model, optimizer, cache, args)
    if "loader" in sections:
        print("[loader] CPU pipeline ceiling");       results["loader"] = bench_loader(cache, args)

    import os
    meta = {
        "script": "src/benchmark.py",
        "argv": sys.argv,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "gpu": {"name": props.name, "total_gb": round(props.total_memory / 1e9, 1),
                "capability": f"sm_{props.major}{props.minor}",
                "multi_processor_count": props.multi_processor_count},
        "cpu_count": os.cpu_count(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda},
        "host": platform.node(),
        "minutes": round((time.time() - started) / 60, 2),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {args.out} ({meta['minutes']:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
