"""Reload a trained model and export MiniMol's internal features over its validation fold.

`train.py` writes `val_embeddings.npy` -- the head's `z`, one tensor, downstream of MiniMol's
final readout. This script goes back into the trunk and exports six more per run: the virtual
node state at five depths (`vn03 vn06 vn09 vn12 vn15`, 336-d each) and the trunk's own pooled
512-d output. Those are the tensors the feature analysis is about; none has been probed before.

The work here is reconstruction, not computation. `final.pt` holds a bare `state_dict`, so the
model has to be rebuilt from the `config` saved beside it and the weights loaded back in. Two
things make that riskier than it sounds:

  - minimol's own loader uses `load_state_dict(..., strict=False)`, which tolerates missing
    keys and yields a partly-random trunk with no error (CLAUDE.md footguns). Every load here
    is `strict=True` with the result asserted empty.
  - `MiniMolTrunk.__init__` ends with `self.train()`, and train mode applies random Laplacian
    sign-flip augmentation -- measured at max|d| ~ 2.8 on the virtual node between two
    otherwise identical passes (`verify_vn_taps.py`). Extraction without `eval()` would emit
    silently irreproducible embeddings.

So the run is checked against itself: the head's `z` is recomputed here and compared to the
`val_embeddings.npy` the training process already wrote. Agreement proves the reconstruction,
the state dict load, the eval mode and the row order all at once -- one assertion covering
every way this could quietly be wrong. It is a comparison worth making only because
`train.predict()` calls `model.eval()` too (`train.py:481`); if that ever changes, this check
becomes a false alarm.

`--pretrained` adds one more export from an **un-fine-tuned** trunk over the same molecules.
Every tensor above comes from a model trained on pProp, so none of them says how far
fine-tuning moved each tap; this is the "before" they are read against. It has no head, so the
`z` round-trip cannot be done and `check_against_reference` stands in for it.

Usage:
    python src/extract_embeddings.py --root outputs/vn_analysis/lyc0lh2d
    python src/extract_embeddings.py --root outputs/vn_analysis/lyc0lh2d --pretrained
    python src/extract_embeddings.py --root outputs/vn_analysis/_smoke --batch-size 256
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import load_features                          # noqa: E402
from head import DualHead                                   # noqa: E402
from model import MiniMolRegressor                          # noqa: E402
from train import RowDataset, collate_batch                 # noqa: E402
from trunk import MiniMolTrunk                              # noqa: E402
from vn_taps import POOLED_KEY, TAP_NAMES, VN_TAPS, capture_vn   # noqa: E402

OUT_NAME = "vn_embeddings.npz"

# The head's z is recomputed and compared against what training wrote. Not exact: the two
# passes use different batch sizes, and CUDA scatter reductions accumulate in nondeterministic
# order -- measured floor ~1.6e-6 on the 512-d output for two *identical* untapped GPU passes
# (`verify_vn_taps.py::check_cuda_floor`). 1e-4 is far above that floor and far below any real
# reconstruction error, which would be O(1) since a wrong trunk is a random trunk.
Z_TOL = 1e-4


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="directory to walk for run directories containing final.pt")
    p.add_argument("--features", type=Path, default=Path("data/features/minimol_v1"))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--force", action="store_true",
                   help="re-extract runs that already have vn_embeddings.npz")
    p.add_argument("--pretrained", action="store_true",
                   help="also export the same six tensors from an UN-fine-tuned MiniMol over "
                        "the same molecules, to <root>/pretrained/frozen/. The 'before' every "
                        "fine-tuned tap is read against")
    p.add_argument("--reference", type=Path,
                   default=Path("data/reference/minimol_v1_ref64"),
                   help="stem of the frozen reference embeddings; the pretrained pass asserts "
                        "against it, standing in for the z round-trip it cannot do")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def rebuild(ckpt, device):
    """`final.pt` -> a loaded `MiniMolRegressor` in eval mode.

    The head geometry is read back out of the saved config rather than assumed, because it is
    swept: `--n-layers`, `--embed-dim` and the two task-head shapes have all moved during this
    project's life, and a default-shaped head would fail to load rather than load wrongly --
    but only because of the strict=True below.
    """
    cfg = ckpt["config"]
    model = MiniMolRegressor(
        MiniMolTrunk(),
        DualHead(hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
                 embed_dim=cfg["embed_dim"],
                 cls_hidden_dim=cfg["cls_hidden_dim"], cls_n_layers=cfg["cls_n_layers"],
                 reg_hidden_dim=cfg["reg_hidden_dim"], reg_n_layers=cfg["reg_n_layers"],
                 dropout=cfg["dropout"], norm=cfg["head_norm"],
                 bottleneck_norm=cfg["bottleneck_norm"]),
    )
    incompatible = model.load_state_dict(ckpt["model_state"], strict=True)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    if missing or unexpected:
        raise RuntimeError(f"state dict did not load cleanly: {len(missing)} missing "
                           f"{missing[:3]}, {len(unexpected)} unexpected {unexpected[:3]}")
    return model.to(device).eval()


def extract(trunk, head, cache, rows, device, batch_size, num_workers):
    """Forward `rows` through the trunk, returning every tensor under study.

    Row order is the caller's order and is preserved: the loader does not shuffle, and each
    batch's own row indices come back so the ordering can be asserted rather than trusted.

    `head` may be None -- the pretrained pass has no head at all. The trunk-side work is
    identical either way, which is the whole reason that pass is a branch here rather than a
    second script: `capture_vn` and the row-order assertion are exactly what must not fork.
    """
    n = len(cache)
    dummy = np.zeros(n, dtype=np.float64)
    loader = DataLoader(RowDataset(cache, dummy, dummy, dummy, rows), shuffle=False,
                        batch_size=batch_size, collate_fn=collate_batch,
                        num_workers=num_workers, persistent_workers=num_workers > 0)

    names = [TAP_NAMES[i] for i in VN_TAPS] + [POOLED_KEY] + (["z"] if head is not None else [])
    parts = {name: [] for name in names}
    seen = []
    with torch.no_grad(), capture_vn(trunk) as cap:
        for batch, _, _, _, idx in loader:
            cap.clear()
            batch = trunk.to_device(batch, device)
            pooled = trunk(batch)
            cap.assert_fired_once()

            for i in VN_TAPS:
                parts[TAP_NAMES[i]].append(cap.tensors[i].float().cpu().numpy())
            parts[POOLED_KEY].append(pooled.float().cpu().numpy())
            if head is not None:
                _, _, z = head.forward_with_embedding(pooled)
                parts["z"].append(z.float().cpu().numpy())
            seen.append(idx.numpy())

    return ({k: np.concatenate(v).astype(np.float32) for k, v in parts.items()},
            np.concatenate(seen))


def run_one(run_dir, cache, args):
    ckpt = torch.load(run_dir / "final.pt", weights_only=False, map_location="cpu")
    model = rebuild(ckpt, args.device)

    # The run's OWN val row order, not a re-derivation of it. train.py may have subsetted the
    # validation fold (`--subset`, seeded by `--seed`), and reproducing that arithmetic here
    # would be a second implementation free to drift from the first. The file is authoritative.
    rows = np.load(run_dir / "val_indices.npy")

    t0 = time.time()
    tensors, seen = extract(model.trunk, model.head, cache, rows, args.device,
                            args.batch_size, args.num_workers)
    if not np.array_equal(seen, rows):
        raise RuntimeError(f"{run_dir}: emitted row order does not match val_indices.npy "
                           f"({int((seen != rows).sum())} of {len(rows)} positions differ)")

    z_ref = np.load(run_dir / "val_embeddings.npy")
    if z_ref.shape != tensors["z"].shape:
        raise RuntimeError(f"{run_dir}: recomputed z is {tensors['z'].shape}, but the run "
                           f"wrote {z_ref.shape}")
    z_delta = float(np.abs(z_ref - tensors["z"]).max())
    if not z_delta < Z_TOL:
        raise RuntimeError(f"{run_dir}: recomputed head embedding disagrees with the run's own "
                           f"val_embeddings.npy by max|d| = {z_delta:.3e} (tol {Z_TOL:.0e}). "
                           "The reloaded model is not the trained model.")

    payload = {k: v for k, v in tensors.items() if k != "z"}
    payload["row_indices"] = rows.astype(np.int64)
    out = run_dir / OUT_NAME
    np.savez(out, **payload)

    mb = out.stat().st_size / 1e6
    shapes = " ".join(f"{k}{v.shape}" for k, v in tensors.items() if k != "z")
    print(f"  {shapes} | z round-trip max|d| = {z_delta:.3e} | "
          f"{mb:.0f} MB in {time.time() - t0:.0f}s")
    return {"run": str(run_dir), "n_rows": int(len(rows)), "z_delta": z_delta,
            "mb": round(mb, 1),
            "shapes": {k: list(v.shape) for k, v in tensors.items() if k != "z"}}


PRETRAINED_DIR = ("pretrained", "frozen")

# The reference tolerance is chosen by DEVICE, because the reference was dumped on the CPU and
# that is where it is reproducible exactly. Measured on this box over the 64x512 reference:
#
#   CPU, twice           0.000e+00      deterministic
#   CPU vs reference     0.000e+00      the mode the reference was dumped in
#   CUDA vs reference    7.2e-06        pure CPU-vs-GPU float arithmetic, not a wrong tensor
#
# So the check runs on the CPU and demands EXACTNESS. A wrong trunk is a random trunk -- the
# error would be O(1), not O(1e-6) -- so an exact test costs nothing in false alarms and gives
# up nothing in sensitivity, whereas 1e-5 against an observed 7.2e-6 is 25% of headroom and one
# GPU generation from failing for no reason. `verify_trunk.py`'s 1e-5 stands as the fallback
# for a trunk that is already on an accelerator when it gets here.
REF_TOL_EXACT = 0.0
REF_TOL_ACCEL = 1e-5


def check_against_reference(trunk, ref_stem):
    """Assert this trunk IS stock MiniMol, against the 64 frozen reference embeddings.

    The fine-tuned path checks itself by recomputing the head's `z` and comparing it to the
    `val_embeddings.npy` the training run already wrote -- one assertion covering the
    reconstruction, the state-dict load, `eval()` mode and the row order at once. The
    pretrained path has no head and no training run, so that check does not exist and
    something has to take its place: a wrong trunk is a RANDOM trunk, and every number
    downstream of it would be plausible and meaningless.

    `data/reference/minimol_v1_ref64.npy` is what `verify_trunk.py` already compares against.
    Reusing it keeps one definition of "this is stock MiniMol" rather than inventing a second.

    Call this BEFORE moving the trunk to an accelerator -- see the tolerance constants.
    """
    ref_npy, ref_pt = ref_stem.with_suffix(".npy"), ref_stem.with_suffix(".features.pt")
    for p in (ref_npy, ref_pt):
        if not p.exists():
            raise SystemExit(f"missing {p}. Run: python src/dump_reference_embeddings.py")

    ref_emb = np.load(ref_npy)
    feats = torch.load(ref_pt, weights_only=False)
    on_cpu = trunk.device.type == "cpu"
    tol = REF_TOL_EXACT if on_cpu else REF_TOL_ACCEL
    with torch.no_grad():
        emb = trunk(trunk.collate(feats)).double().cpu().numpy()   # collate() follows the trunk

    if emb.shape != ref_emb.shape:
        raise RuntimeError(f"reference check: got {emb.shape}, expected {ref_emb.shape}")
    delta = float(np.abs(emb - ref_emb).max())
    if delta > tol:
        raise RuntimeError(
            f"pretrained trunk does not reproduce {ref_npy}: max|d| = {delta:.3e} on "
            f"{trunk.device} (tol {tol:.1e}). This is not stock MiniMol, so nothing extracted "
            "from it would mean what it claims.")
    print(f"  reference check: max|d| = {delta:.3e} over {ref_emb.shape[0]}x"
          f"{ref_emb.shape[1]} vs {ref_npy.name} on {trunk.device} "
          f"({'exact' if on_cpu else f'tol {tol:.0e}'})")
    return delta


def run_pretrained(root, cache, args):
    """Export the same six tensors from an UN-fine-tuned MiniMol, over the same molecules.

    Every tensor in the fine-tuned exports is a post-fine-tuning tensor, so nothing in them
    says how far fine-tuning actually moved each tap. This is the "before" they are read
    against: if the virtual node scores well on the structural probes and also sits near this
    baseline while `pooled512` has moved far from its own, the finding is that fine-tuning
    distorted the readout and left the virtual node's chemistry intact -- a mechanism rather
    than a number. It is also the frozen-MiniMol featurizer, i.e. the workflow this repo
    exists to beat, scored on identical probes.

    Rows are INHERITED from an already-extracted run rather than re-derived from the split.
    `train.py --subset` draws validation rows with an RNG, so re-deriving would be a second
    implementation free to disagree with the first; and `check_alignment` then covers this
    export like any other.
    """
    out_dir = root.joinpath(*PRETRAINED_DIR)
    out = out_dir / OUT_NAME
    if out.exists() and not args.force:
        print(f"{out.relative_to(root)} exists; skipping (--force to redo)")
        return None

    donors = sorted(p for p in root.rglob("val_indices.npy")
                    if p.parent != out_dir and (p.parent / "final.pt").exists())
    if not donors:
        raise SystemExit(f"--pretrained needs an existing run under {root} to take its "
                         "validation rows from; none has val_indices.npy beside final.pt")
    rows = np.load(donors[0])
    print(f"pretrained/frozen: {len(rows):,} rows inherited from "
          f"{donors[0].parent.relative_to(root)}")

    # eval() is not optional. Train mode applies random Laplacian sign-flip augmentation --
    # measured at max|d| ~ 2.8 on the virtual node between two otherwise identical passes --
    # and MiniMolTrunk.__init__ ends with self.train(), so extraction without this emits
    # silently irreproducible embeddings. The reference check below would catch it, which is
    # the second reason that check is here.
    trunk = MiniMolTrunk().eval()
    t0 = time.time()
    # Checked on the CPU, where the reference is reproducible EXACTLY, and only then moved to
    # the accelerator for the 66k-molecule pass. Both orderings verify the same trunk; this one
    # verifies it against a tolerance of zero.
    ref_delta = check_against_reference(trunk, args.reference)
    trunk.to(args.device)

    tensors, seen = extract(trunk, None, cache, rows, args.device,
                            args.batch_size, args.num_workers)
    if not np.array_equal(seen, rows):
        raise RuntimeError(f"pretrained: emitted row order does not match the donor's "
                           f"({int((seen != rows).sum())} of {len(rows)} positions differ)")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, row_indices=rows.astype(np.int64), **tensors)
    mb = out.stat().st_size / 1e6
    shapes = " ".join(f"{k}{v.shape}" for k, v in tensors.items())
    print(f"  {shapes} | {mb:.0f} MB in {time.time() - t0:.0f}s")
    return {"run": str(out_dir), "n_rows": int(len(rows)), "pretrained": True,
            "reference_max_delta": ref_delta, "donor": str(donors[0].parent),
            "mb": round(mb, 1),
            "shapes": {k: list(v.shape) for k, v in tensors.items()}}


def check_alignment(runs, root):
    """Every extracted run under one root must describe the same molecules, in the same order.

    The analysis stacks 30 models scored on one validation fold, so a spread across models is
    only a spread across models if the rows line up. Nothing upstream guarantees that:
    `train.py --subset` draws its validation rows with an RNG seeded by `--seed`, which this
    experiment varies per bootstrap, so subset runs in one directory legitimately disagree.

    Checked on every invocation, not only when something was extracted -- a re-run over an
    already-complete tree is exactly when someone is about to trust it.
    """
    # Discovered by walking for the artifact itself rather than from the caller's run list,
    # so the pretrained export -- which has no final.pt and so is not a "run" -- is covered
    # by the same guarantee as everything else it will be stacked against.
    have = sorted({p.parent for p in Path(root).rglob(OUT_NAME)})
    if len(have) < 2:
        return
    ref_dir, ref = have[0], np.load(have[0] / OUT_NAME)["row_indices"]
    bad = [str(d) for d in have[1:]
           if not np.array_equal(np.load(d / OUT_NAME)["row_indices"], ref)]
    if bad:
        raise SystemExit(
            f"row order differs across runs under {root}: {len(bad)} of {len(have)} disagree "
            f"with {ref_dir.name} ({len(ref):,} rows), e.g. {bad[:2]}. These embeddings are "
            "not row-aligned and must not be stacked. A --subset run is the usual cause.")
    print(f"row alignment: all {len(have)} runs share the same {len(ref):,} rows")


def main(argv=None):
    args = parse_args(argv)
    runs = sorted(p.parent for p in args.root.rglob("final.pt"))
    if not runs:
        raise SystemExit(f"no run directories with final.pt under {args.root}")

    todo = [d for d in runs if args.force or not (d / OUT_NAME).exists()]
    pretrained_todo = args.pretrained and (
        args.force or not args.root.joinpath(*PRETRAINED_DIR, OUT_NAME).exists())
    print(f"{len(runs)} run(s) under {args.root}; {len(todo)} to extract "
          f"({len(runs) - len(todo)} already done)"
          + (" | + the pretrained baseline" if pretrained_todo else ""))
    if not todo and not pretrained_todo:
        check_alignment(runs, args.root)
        return 0

    cache = load_features(args.features)
    print(f"feature cache: {len(cache):,} graphs | device {args.device}\n")

    results = []
    for d in todo:
        print(f"{d.relative_to(args.root)}")
        results.append(run_one(d, cache, args))
    if args.pretrained:
        entry = run_pretrained(args.root, cache, args)
        if entry is not None:
            results.append(entry)

    check_alignment(runs, args.root)

    # Merged into whatever is already recorded, keyed by run directory, rather than
    # overwritten. A second invocation -- adding only the pretrained baseline to a tree whose
    # 30 runs are already done -- would otherwise replace the provenance of all 30 with one
    # entry, and the file that records what was extracted is the wrong thing to lose.
    summary = args.root / "extraction_summary.json"
    merged = {}
    if summary.exists():
        merged = {r["run"]: r for r in json.loads(summary.read_text()).get("runs", [])}
    merged.update({r["run"]: r for r in results})
    ordered = [merged[k] for k in sorted(merged)]
    summary.write_text(json.dumps(
        {"root": str(args.root), "taps": [TAP_NAMES[i] for i in VN_TAPS],
         "vn_indices": list(VN_TAPS), "z_tol": Z_TOL, "reference_tol_cpu": REF_TOL_EXACT,
         "reference_tol_accel": REF_TOL_ACCEL,
         "runs": ordered}, indent=2))

    deltas = [r["z_delta"] for r in results if "z_delta" in r]
    worst = (f"worst z round-trip max|d| = {max(deltas):.3e} (tol {Z_TOL:.0e})" if deltas
             else "no z round-trip to check (pretrained only)")
    print(f"\n{len(results)} run(s) extracted | {worst} | summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
