"""Prove the Mol-JEPA embedding cache is reproducible, correctly aligned, and correctly named.

On the MiniMol side the load-bearing check is that `trunk.py` reproduces stock MiniMol
exactly. Here the trunk never trains, so the equivalent question is narrower and the stakes
are higher: the cache is computed **once**, and every number this branch ever reports is a
function of it. A cache that is subtly misaligned, silently truncated, or built from a
different Hub revision does not fail -- it produces plausible results.

Four things can go wrong, none of which announces itself:

  1. *The model moved.* The checkpoint is fetched with `trust_remote_code=True`; its code and
     its weights can both change. Checked against a frozen fixture, not against itself.
  2. *The rows are shifted.* If any molecule were dropped, every later row would be paired
     with the wrong pProp. Checked at six scattered indices, including both ends.
  3. *The columns are misnamed.* `cls` is a linear projection of `embeddings[:, 0]`, not a
     slice of it -- an easy and completely silent thing to get backwards. Re-derived here.
  4. *It is not reproducible.* On GPU without `use_deterministic_algorithms` the model
     differs from itself by ~2e-06 on a plain repeat, so rebuilding the cache would move
     every downstream number slightly.

Run `src/dump_jepa_reference.py` first -- it produces the frozen fixture in its own process.

Usage:
    python src/verify_jepa_embed.py
    python src/verify_jepa_embed.py --embeddings /tmp/jepa_smoke --partial
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Before torch -- cuBLAS reads it at init. See DETERMINISM in src/jepa_embed.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jepa_embed import HF_REPO, HF_REVISION, check_layout                  # noqa: E402
from jepa_features import load_embeddings, load_meta                       # noqa: E402

# Both ends and four interior rows, including 3152 (the last of the pProp >= 3.5 tail, since
# the CSV is sorted by -pprop) -- the same indices featurize.py's alignment was checked at.
ALIGNMENT_ROWS = [0, 1, 3152, 165740, 331478, 331479]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", type=Path, default=Path("data/embeddings/moljepa_v1"))
    p.add_argument("--ref", type=Path, default=Path("data/reference/moljepa_v1_ref64"),
                   help="stem written by dump_jepa_reference.py")
    p.add_argument("--splits", type=Path, default=Path("data/splits/cluster_kfold_v1"))
    p.add_argument("-o", "--out", type=Path, default=Path("verification_jepa.md"))
    p.add_argument("--partial", action="store_true",
                   help="the cache under test was built with --limit; skips the checks that "
                        "need full coverage rather than failing them")
    p.add_argument("--device", default="cpu",
                   help="device for the live-model checks. cpu is bit-exact unconditionally.")
    return p.parse_args(argv)


class Report:
    """Collects check results so a failure part-way still produces a full report."""

    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail, note=None):
        self.rows.append({"name": name, "passed": bool(passed),
                          "detail": detail, "note": note})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        if note:
            print(f"       {note}")
        return passed

    def skip(self, name, why):
        self.rows.append({"name": name, "passed": True, "detail": f"SKIPPED -- {why}",
                          "note": None})
        print(f"[SKIP] {name}: {why}")

    @property
    def ok(self):
        return all(r["passed"] for r in self.rows)


def load_model(device):
    import torch
    from transformers import AutoModel
    torch.use_deterministic_algorithms(True, warn_only=True)
    m = AutoModel.from_pretrained(HF_REPO, revision=HF_REVISION,
                                  trust_remote_code=True).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def embed(model, smiles, batch):
    """The same two tensors the cache stores, for a list of SMILES."""
    import torch
    embs, projs = [], []
    for lo in range(0, len(smiles), batch):
        with torch.no_grad():
            o = model(smiles[lo:lo + batch])
        embs.append(o.embeddings.float().cpu().numpy())
        projs.append(torch.cat([o.cls.unsqueeze(1), o.predictions], 1).float().cpu().numpy())
    return np.concatenate(embs, 0), np.concatenate(projs, 0)


# --------------------------------------------------------------------------- checks

def check_reference(report, model, ref_npz, ref_meta, batch=64):
    """The model still computes what it computed when the fixture was frozen.

    Bit-exact, not `allclose`: the fixture was produced on cpu, this runs on cpu, and cpu was
    measured deterministic across repeats, batch sizes and ordering. A tolerance here would
    hide exactly the drift this check exists to catch.
    """
    emb, proj = embed(model, ref_meta["smiles"], batch)
    d_emb = float(np.abs(emb - ref_npz["embeddings"]).max())
    d_prj = float(np.abs(proj - ref_npz["projected"]).max())
    ok = d_emb == 0.0 and d_prj == 0.0
    return report.add(
        "model matches the frozen reference", ok,
        f"max|delta| embeddings {d_emb:.3e}, projected {d_prj:.3e} over "
        f"{len(ref_meta['smiles'])} molecules",
        f"fixture from {ref_meta['hf_repo']}@{(ref_meta.get('hf_revision') or '?')[:12]}; "
        "catches a changed Hub revision, which trust_remote_code makes possible")


def check_eval_mode(report, model):
    """Dropout is 0.1 on all three expert encoders, so eval mode is not cosmetic."""
    import torch
    live = [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Dropout) and m.training]
    total = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Dropout))
    return report.add("all Dropout in eval mode", not live,
                      f"{total - len(live)}/{total} Dropout modules with training=False"
                      + (f"; live: {live[:3]}" if live else ""),
                      "a single live Dropout would make every cached row a random draw")


def check_determinism(report, model, smiles, device):
    """Re-running the extraction must give the same cache back.

    Three gating measurements, all of which must be bit-identical:

      * **repeat** -- the same call twice. The weakest possible form, and the one that fails
        first on GPU without `use_deterministic_algorithms` (2.1e-06, measured).
      * **re-chunking** -- the same molecule list at batch 8/16/256 instead of 64.
        `jepa_embed.py` walks the CSV in contiguous `--batch` slices, so this is exactly the
        freedom a rebuild has, and it is the one that matters for the cache.
      * **reversed order** -- the list embedded backwards and un-reversed.

    What this deliberately does NOT claim is invariance to batch *composition*. Measured, on
    six scattered rows against their natural chunks: **cpu 3.3e-07, cuda 7.5e-07**. Embedding
    a molecule alongside different neighbours is a different computation at the 1e-07 level
    on either device -- `to_dense_batch` pads the graph batch to its widest member and the
    reductions reassociate. Reported here rather than asserted, because it is a property of
    the model. Its practical consequence is recorded in CLAUDE.md: a rebuilt cache is
    bit-identical only at the same `--batch`, on the same device.
    """
    ref = embed(model, smiles, 64)
    measurements, worst = [], 0.0
    for label, emb2 in [
        ("repeat @64", embed(model, smiles, 64)),
        ("batch 8", embed(model, smiles, 8)),
        ("batch 16", embed(model, smiles, 16)),
        ("batch 256", embed(model, smiles, 256)),
        ("reversed", tuple(a[::-1] for a in embed(model, smiles[::-1], 64))),
    ]:
        d = max(float(np.abs(a - b).max()) for a, b in zip(emb2, ref))
        worst = max(worst, d)
        measurements.append(f"{label} {d:.1e}")

    # Non-gating, reported: a subset re-embedded alone rather than in its chunk.
    sub = [0, 1, 17, 40, 63][:len(smiles)]
    alone = embed(model, [smiles[i] for i in sub], 64)[0]
    d_comp = float(np.abs(alone - ref[0][sub]).max())

    return report.add(
        f"reproducible on {device}", worst == 0.0,
        "max|delta| " + ", ".join(measurements)
        + f"; batch COMPOSITION (reported, not asserted) {d_comp:.1e}",
        "re-chunking is the freedom a rebuild actually has, and it is bit-exact. Composition "
        "is not invariant on either device, so a rebuilt cache matches bit-for-bit only at "
        "the same --batch on the same device")


def check_layout_matches_meta(report, model, meta, smiles, device):
    """The names in meta.json are the ones the model actually produces, in that order."""
    import torch
    with torch.no_grad():
        out = model(smiles[:8])
    try:
        names = check_layout(model, out)
    except SystemExit as exc:
        return report.add("token layout re-derives", False, str(exc)[:160])
    ok = names == meta["token_names"] and meta.get("cls_index") == 0
    return report.add("token layout re-derives", ok,
                      f"{len(names)} tokens, cls at index {meta.get('cls_index')}: "
                      f"{'matches meta.json' if names == meta['token_names'] else f'MISMATCH {names}'}",
                      "cls is modality_pred[0](embeddings[:,0]), not a slice of it -- "
                      "check_layout asserts that relationship bit-exactly")


def check_alignment(report, model, root, meta, csv_smiles, rows, verify_device):
    """Cached row i is the encoding of CSV row i, at both ends and four interior points.

    Row alignment is the invariant the whole cache rests on, and a shift is silent: every
    molecule would simply be paired with a neighbouring molecule's pProp.

    **Each row is re-embedded inside its own natural chunk**, on the device the cache
    records -- i.e. the same `--batch`-aligned slice of the CSV that `jepa_embed.py` fed the
    model. That reproduces the cache's own arithmetic exactly, so bit-exactness is the right
    criterion. Measured, and the reason this is not simply `model([smiles])`:

        batch of 6 scattered rows vs their natural 256-chunks   cpu 3.3e-07, cuda 7.5e-07

    The model is **not invariant to batch composition** at the 1e-07 level, on either device
    -- `to_dense_batch` pads the graph batch to its widest member, and the reductions
    reassociate. That is a property of the model, not a defect in the cache, but it means a
    naive re-embed would fail this check for a reason that has nothing to do with alignment.
    The scattered figure is still reported, as context and as the scale that separates
    float noise from a genuine shift.

    Paired with a deliberate off-by-one, for the same reason `check_grad_flow` is
    two-sided: a tolerant check that has never been seen to reject a shifted comparison is
    indistinguishable from one that cannot.
    """
    rows = [r for r in rows if r < meta["n_molecules"]]
    batch = meta.get("batch", 256)
    cache_device = meta.get("device", verify_device)
    model.to(cache_device)

    emb_c = np.load(root / meta["arrays"]["embeddings"]["file"], mmap_mode="r")
    prj_c = np.load(root / meta["arrays"]["projected"]["file"], mmap_mode="r")

    fresh_emb, fresh_prj, shifted = [], [], []
    for r in rows:
        lo = (r // batch) * batch
        e, pr = embed(model, csv_smiles[lo:lo + batch], batch)
        fresh_emb.append(e[r - lo])
        fresh_prj.append(pr[r - lo])
        # The same fresh chunk, read one row along: what an off-by-one cache would look like.
        shifted.append(e[min(r - lo + 1, len(e) - 1)])
    fresh_emb, fresh_prj = np.stack(fresh_emb), np.stack(fresh_prj)

    d_emb = float(np.abs(np.asarray(emb_c[rows]) - fresh_emb).max())
    d_prj = float(np.abs(np.asarray(prj_c[rows]) - fresh_prj).max())
    d_shift = float(np.abs(np.asarray(emb_c[rows]) - np.stack(shifted)).max())

    # Scattered composition, for contrast: the float-noise floor of a naive re-embed.
    scattered = embed(model, [csv_smiles[r] for r in rows], 64)[0]
    d_scatter = float(np.abs(np.asarray(emb_c[rows]) - scattered).max())

    ok = d_emb == 0.0 and d_prj == 0.0 and d_shift > 1e-3
    return report.add(
        "cached row i == fresh(CSV row i)", ok,
        f"in-chunk on {cache_device}: max|delta| embeddings {d_emb:.3e}, projected "
        f"{d_prj:.3e} at rows {rows}. An off-by-one gives {d_shift:.3e} and a scattered "
        f"re-embed {d_scatter:.1e}, so the check separates a shift from float noise by "
        f"~{d_shift / max(d_scatter, 1e-12):.0f}x",
        "two-sided: bit-exact against the right row, and shown to reject the neighbouring "
        "one -- a tolerant check that never rejects anything tests nothing")


def check_loader_guards(report, root, meta):
    """The provenance guards must actually fire, not merely exist.

    Written the way `verify_metrics.py` writes its freeze negative test: a guard that has
    never been seen to reject anything is indistinguishable from a guard that is broken.
    """
    import tempfile
    outcomes = []

    with tempfile.TemporaryDirectory() as td:
        # A cache whose source CSV has changed must be refused: subset.py takes a --seed, so
        # a regenerated CSV is a different 331k set and every row would be mispaired.
        tampered = Path(td) / "tampered"
        tampered.mkdir()
        m = dict(meta)
        m["input_sha256"] = "0" * 64
        (tampered / "meta.json").write_text(json.dumps(m))
        for name in ("embeddings", "projected"):
            (tampered / meta["arrays"][name]["file"]).symlink_to(
                (root / meta["arrays"][name]["file"]).resolve())
        try:
            load_embeddings(tampered, "cls", allow_partial=True)
            outcomes.append(("changed CSV sha256", False, "ACCEPTED"))
        except ValueError:
            outcomes.append(("changed CSV sha256", True, "rejected"))

        # A --limit cache indexed with full-set fold indices would run off the end.
        partial = Path(td) / "partial"
        partial.mkdir()
        m = dict(meta)
        m["limit"] = 5000
        (partial / "meta.json").write_text(json.dumps(m))
        for name in ("embeddings", "projected"):
            (partial / meta["arrays"][name]["file"]).symlink_to(
                (root / meta["arrays"][name]["file"]).resolve())
        try:
            load_embeddings(partial, "cls")
            outcomes.append(("--limit cache", False, "ACCEPTED"))
        except ValueError:
            outcomes.append(("--limit cache", True, "rejected"))

    for bad in ("emb:nope", "pred:graph", "nonsense"):
        try:
            load_embeddings(root, bad, check_input=False, allow_partial=True)
            outcomes.append((f"readout {bad!r}", False, "ACCEPTED"))
        except ValueError:
            outcomes.append((f"readout {bad!r}", True, "rejected"))

    ok = all(o[1] for o in outcomes)
    return report.add("loader guards have teeth", ok,
                      "; ".join(f"{n}: {r}" for n, _, r in outcomes),
                      "each guard is shown rejecting something, not merely present")


def check_sanity(report, root, meta):
    """Finite everywhere, and no token collapsed to a constant.

    A collapsed token would be a 512-d column of zero variance -- it would train, score
    somewhere near the mean, and never look like an error. "Constant" is tested as
    `max == min` per (token, dimension) rather than as a variance near zero: it is exact,
    needs one pass, and has no tolerance to argue about.
    """
    nonfinite, worst, where = 0, None, None
    for name, spec in sorted(meta["arrays"].items()):
        arr = np.load(root / spec["file"], mmap_mode="r")
        lo_v = np.full(arr.shape[1:], np.inf, dtype=np.float32)
        hi_v = np.full(arr.shape[1:], -np.inf, dtype=np.float32)
        for lo in range(0, arr.shape[0], 20000):        # chunked: 8.8 GB per array
            blk = np.asarray(arr[lo:lo + 20000])
            nonfinite += int((~np.isfinite(blk)).sum())
            np.minimum(lo_v, blk.min(axis=0), out=lo_v)
            np.maximum(hi_v, blk.max(axis=0), out=hi_v)
        spread = (hi_v - lo_v).min(axis=1)              # per token, its flattest dimension
        i = int(np.argmin(spread))
        if worst is None or spread[i] < worst:
            worst, where = float(spread[i]), f"{name}:{meta['token_names'][i]}"
    ok = nonfinite == 0 and worst > 0.0
    return report.add("finite, and no collapsed token", ok,
                      f"{nonfinite} non-finite values; smallest per-dimension (max-min) "
                      f"{worst:.3e} (at {where})",
                      "a zero-variance token would train and score without ever erroring")


def check_split_coverage(report, root, meta, splits_dir):
    """The 5 validation folds tile the cache exactly once, and no index runs off the end."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from splits import load_fold

    n = meta["n_molecules"]
    seen = np.zeros(n, dtype=np.int32)
    hi = -1
    for fold in range(5):
        _, val_idx = load_fold(splits_dir, fold=fold)
        seen[val_idx] += 1
        hi = max(hi, int(val_idx.max()))
    ok = bool((seen == 1).all()) and hi < n
    return report.add("5 val folds tile the cache exactly once", ok,
                      f"{int((seen == 1).sum()):,}/{n:,} rows covered once, "
                      f"{int((seen == 0).sum()):,} uncovered, {int((seen > 1).sum()):,} "
                      f"duplicated; max index {hi:,}",
                      "this is what lets pooled out-of-fold metrics use all 100 potent "
                      "molecules rather than 20")


# --------------------------------------------------------------------------- report

def write_report(path, report, context):
    lines = [
        "# Mol-JEPA embedding cache verification",
        "",
        f"Generated by `src/verify_jepa_embed.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "The cache is computed once and every number on this branch is a function of it, so",
        "each claim below is a measurement rather than an assertion -- per the repo",
        "convention (CLAUDE.md \"Conventions\").",
        "",
        "## Environment",
        "",
    ]
    lines += [f"- `{k}`: {v}" for k, v in context.items()]
    lines += ["", "## Results", "",
              "| Check | Result | Measurement |", "|---|---|---|"]
    for r in report.rows:
        detail = r["detail"].replace("|", "\\|")
        lines.append(f"| {r['name']} | {'PASS' if r['passed'] else '**FAIL**'} | {detail} |")
    lines += ["", "## Notes", ""]
    for r in report.rows:
        if r["note"]:
            lines.append(f"- **{r['name']}** — {r['note']}")
    lines += ["", f"**Overall: {'PASS' if report.ok else 'FAIL'}**", ""]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    import torch

    ref_npz_path, ref_meta_path = args.ref.with_suffix(".npz"), args.ref.with_suffix(".json")
    have_ref = ref_npz_path.exists() and ref_meta_path.exists()

    root = args.embeddings
    meta = load_meta(root, check_input=True, allow_partial=args.partial)
    csv_smiles = pd.read_csv(meta["input_abspath"], usecols=["SMILES"])["SMILES"].tolist()

    report = Report()
    print(f"cache:  {root}  ({meta['n_molecules']:,} rows, {meta['embed_dim']}-d)")
    print(f"model:  {meta['hf_repo']}@{meta.get('hf_revision')}")
    print(f"device: {args.device} for the live-model checks\n")

    started = time.time()
    model = load_model(args.device)
    print(f"loaded the model in {time.time() - started:.1f}s\n")

    if have_ref:
        check_reference(report, model, np.load(ref_npz_path),
                        json.loads(ref_meta_path.read_text()))
    else:
        report.skip("model matches the frozen reference",
                    f"no fixture at {ref_npz_path}; run src/dump_jepa_reference.py")

    check_eval_mode(report, model)
    check_determinism(report, model, csv_smiles[:64], args.device)
    check_layout_matches_meta(report, model, meta, csv_smiles, args.device)
    check_alignment(report, model, root, meta, csv_smiles, ALIGNMENT_ROWS, args.device)
    model.to(args.device)
    check_loader_guards(report, root, meta)
    check_sanity(report, root, meta)
    if args.partial:
        report.skip("5 val folds tile the cache exactly once",
                    "--partial: a --limit cache cannot cover the folds")
    else:
        check_split_coverage(report, root, meta, args.splits)

    import transformers
    write_report(args.out, report, {
        "cache": str(root),
        "molecules": f"{meta['n_molecules']:,}",
        "arrays": ", ".join(f"{k} {tuple(v['shape'])}"
                            for k, v in sorted(meta["arrays"].items())),
        "hf revision": meta.get("hf_revision"),
        "cache built on": f"{meta['device']} "
                          f"(repeat max|delta| {meta.get('repeat_maxdiff')})",
        "verified on": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    })
    print(f"\nwrote {args.out}")
    print("OVERALL:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
