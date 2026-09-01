"""Run Mol-JEPA once over the whole subset and cache what it returns, in CSV row order.

This branch does **not** fine-tune Mol-JEPA. The model is frozen, so its output for a given
SMILES string is a pure function of the string and the checkpoint -- exactly the situation
`featurize.py` exists for on the MiniMol side, only more so: there, the cache held the
featurizer's output and the trunk still ran every epoch; here the cache holds the *entire*
forward pass, and training reads floats off disk.

WHAT IS STORED, AND WHY ALL OF IT

`model(smiles_list)` returns three tensors, and the relationship between them is not what
their names suggest. Measured against the checkpoint (`MolJEPA.predict`, and verified
bit-exactly at load time by `check_layout` below):

    embeddings   [B, 13, 512]   the RAW transformer output: CLS at column 0, then the 12
                                modality tokens
    cls          [B,     512]   == modality_pred[0](embeddings[:, 0])
    predictions  [B, 12, 512]   == modality_pred[i+1](embeddings[:, i+1])

So `cls` is **not** a slice of `embeddings` -- it is a *linear projection* of column 0, and
`predictions` are the projections of columns 1..12. `cls` and `predictions` together are
exactly the projected counterpart of `embeddings`, one row per token. Caching only
`embeddings` and `predictions` would silently lose `cls`, which is the model card's headline
output; caching `embeddings` and `cls` would lose the other twelve projections.

Both full tensors are therefore stored, aligned column-for-column under one name list:

    embeddings.npy   [N, 13, 512]   raw transformer output
    projected.npy    [N, 13, 512]   cat([cls, predictions]) -- the per-token linear heads

Only two of Mol-JEPA's seven modalities take `input: "smiles"` (`graph` and `ecfp`); the
other five are `input: "precomputed"` and unavailable to us, so from a bare SMILES they are
masked and the JEPA predictor guesses them. That makes columns 3..13 a genuinely different
kind of feature from anything MiniMol produces -- a structure-only model's estimate of
binding, ADMET and phenotype embeddings.

Which of these becomes *the* encoder is an open design question. The cost of keeping all of
it is ~17.6 GB of untracked, regenerable disk; the benefit is that changing the readout
later is a slice of a `.npy` rather than a re-run of this script.

ROW ORDER AND ALIGNMENT

Written in exact CSV row order, so `splits.py` indices -- which are positions into that
order -- index the cache directly with no mapping layer. If any molecule fails, this script
raises rather than skipping it: a hole would shift every subsequent row onto the wrong
target, which is silent. `meta.json` records the CSV's sha256 and `jepa_features.py`
re-hashes on load and refuses a mismatch, the same guard `features.py` and `splits.py` apply.

THE TOKEN LAYOUT IS MEASURED, NOT ASSUMED

Nothing in the model card documents any of the above. Guessing would produce a plausible
wrong answer rather than an error -- the first draft of this script assumed the card's
wording that `cls` is a column of `embeddings`, and only a bit-exactness check caught that
it is not. So `check_layout` re-derives the relationship from the loaded model on every run,
bit-exactly, and refuses to write a cache if it does not hold. The 12 modality names come
off `model.model.modalities_spec` -- the list the model actually encodes in, after
`apply_label_strategy` has appended the five label modalities to the seven real ones.
`token_names` goes into `meta.json`, and every readout downstream resolves through it.

DETERMINISM

Measured on this box, and the reason this script does not simply run on the GPU as-is:

    device   repeat @ same batch    batch 8/16/256 vs 64    reversed order
    cuda           2.1e-06                2.4e-06              1.7e-06
    cpu            0.000e+00              0.000e+00            0.000e+00

The GPU differs from *itself* on a plain repeat, so this is kernel non-determinism -- the
scatter-adds in message passing, and cuBLAS split-k -- not a batching artifact. It is ~5e-07
relative and would not change a trained model, but a cache is computed once and every number
downstream inherits it, so "rebuilding the cache changes the results slightly" is a bad
property to accept for free.

With `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)` the GPU
comes back bit-exact, at 218 mol/s against 281 (25 min for the full set, against 40 on CPU).
Both are set here by default, and the run *measures* a repeat rather than trusting them --
`warn_only=True` means a PyG op with no deterministic kernel warns instead of raising, so
determinism is checked, not assumed. `--allow-nondeterministic` turns the check into a
warning; the measured figure is recorded in `meta.json` either way.

Usage:
    python src/jepa_embed.py
    python src/jepa_embed.py --limit 5000 -o /tmp/jepa_smoke
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# Set before torch is imported anywhere: cuBLAS reads this when it initialises its
# workspace, and a value set afterwards is ignored. Without it `use_deterministic_algorithms`
# cannot make the matmuls reproducible. See DETERMINISM in the module docstring.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import _sha256_file  # noqa: E402  (same repo, reused rather than duplicated)

HF_REPO = "Flogrammer/Mol-JEPA"

# `trust_remote_code=True` executes Python fetched from the Hub, so an unpinned load means
# the definition of our encoder can change under us between runs. The resolved commit sha
# is always recorded in meta.json; pin it here once a cache is built that matters.
HF_REVISION = "4c912b450175f31b5ba913a5dc921c03b27b985a"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("-o", "--out", type=Path, default=Path("data/embeddings/moljepa_v1"),
                   help="output directory; embeddings.npy, predictions.npy and meta.json "
                        "are written into it")
    p.add_argument("--batch", type=int, default=256,
                   help="molecules per forward pass. Must not change the result -- "
                        "verify_jepa_embed.py checks batch-size invariance explicitly.")
    p.add_argument("--limit", type=int, default=None,
                   help="embed only the first N rows (smoke test). Recorded in meta.json, "
                        "which makes the partial cache self-identifying.")
    p.add_argument("--repo", default=HF_REPO)
    p.add_argument("--revision", default=HF_REVISION,
                   help="Hub revision. A branch name resolves to a sha, which is recorded.")
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--allow-nondeterministic", action="store_true",
                   help="downgrade the repeat-determinism check to a warning. The measured "
                        "max|delta| is recorded in meta.json either way.")
    return p.parse_args(argv)


def load_model(repo, revision, device):
    """Load Mol-JEPA frozen, and prove it is frozen rather than assuming it."""
    import torch
    from transformers import AutoModel

    # warn_only: a PyG scatter with no deterministic kernel warns rather than raising, so
    # this is a request, not a guarantee -- which is why main() measures a repeat.
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = AutoModel.from_pretrained(repo, revision=revision, trust_remote_code=True)
    model.eval().to(device)

    # config.json puts dropout 0.1 on all three expert encoders, so eval mode is the
    # difference between a deterministic cache and a random one. Assert rather than trust
    # `.eval()`: a submodule constructed after it, or re-trained by a stray call, would be
    # invisible here and poison every row downstream.
    live = [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Dropout) and m.training]
    if live:
        raise SystemExit(f"{len(live)} Dropout modules still in training mode after "
                         f".eval(): {live[:5]} -- the cache would not be reproducible")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def modality_names(model):
    """The 12 non-CLS token names, in the order the model actually encodes them.

    Taken from `model.model.modalities_spec`, not from `model.config`: the config keeps
    seven `modalities_spec` and five `labels_spec`, and `apply_label_strategy()` appends the
    latter to the former on the *model's* copy. The model's list is the one whose order the
    columns are in. The config path is kept as a fallback for a checkpoint that does not
    mutate, and it produces the same 12 names in the same order for this one.
    """
    core = getattr(model, "model", None)
    spec = getattr(core, "modalities_spec", None)
    if spec:
        return [m["name"] for m in spec]
    cfg = model.config
    return ([m["name"] for m in getattr(cfg, "modalities_spec", [])]
            + [l["name"] for l in getattr(cfg, "labels_spec", [])])


def check_layout(model, out):
    """Re-derive the token layout from the model, bit-exactly, and name the 13 columns.

    Asserts the three things every readout downstream depends on:

      * `embeddings` has exactly one more column than `predictions` has rows;
      * `cls` is `modality_pred[0]` applied to `embeddings[:, 0]`  -- i.e. CLS is column 0
        and the projections are per-token, so `projected` can be assembled as
        `cat([cls, predictions])` and stay aligned with `embeddings`;
      * the 12 modality names line up with the 12 prediction rows.

    Bit-exact rather than `allclose`: this is recomputing the model's own arithmetic in the
    same order and dtype, so anything but 0.0 means the structure is different, not that the
    tolerance is too tight.
    """
    import torch

    emb, cls, preds = out.embeddings, out.cls, out.predictions
    n_emb, n_pred = emb.shape[1], preds.shape[1]
    if n_emb != n_pred + 1:
        raise SystemExit(f"embeddings has {n_emb} columns, expected predictions "
                         f"({n_pred}) + 1 for cls")

    mp = model.model.transformer_head.modality_pred
    if len(mp) != n_emb:
        raise SystemExit(f"modality_pred has {len(mp)} heads for {n_emb} tokens")
    with torch.no_grad():
        d_cls = (cls - mp[0](emb[:, 0, :])).abs().max().item()
        d_prd = max((preds[:, i, :] - mp[i + 1](emb[:, i + 1, :])).abs().max().item()
                    for i in range(n_pred))
    if d_cls != 0.0 or d_prd != 0.0:
        raise SystemExit(
            f"token layout does not hold: max|cls - modality_pred[0](emb[:,0])| = {d_cls}, "
            f"max|pred[i] - modality_pred[i+1](emb[:,i+1])| = {d_prd}. This script assumes "
            "CLS is column 0 and the projections are per-token; refusing to write a cache.")

    mods = modality_names(model)
    if len(mods) != n_pred:
        raise SystemExit(
            f"model names {len(mods)} modalities {mods} but predictions has {n_pred} rows "
            "-- refusing to write a cache whose columns cannot be named")
    return ["cls"] + list(mods)


def main(argv=None):
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    started_all = time.time()
    print(f"hashing {args.input} ...", flush=True)
    input_sha = _sha256_file(args.input)

    df = pd.read_csv(args.input, usecols=["SMILES"])
    smiles = df["SMILES"].tolist()
    n_rows_csv = len(smiles)
    if args.limit is not None:
        smiles = smiles[:args.limit]
    n = len(smiles)
    print(f"{n:,} molecules to embed (CSV has {n_rows_csv:,} rows)")

    started = time.time()
    model = load_model(args.repo, args.revision, device)
    print(f"loaded {args.repo}@{args.revision} on {device} in {time.time() - started:.1f}s "
          f"({sum(p.numel() for p in model.parameters()):,} params, frozen)")

    # Resolve the branch name to the commit sha that was actually executed. Recorded so a
    # cache states which version of the remote code produced it.
    try:
        from huggingface_hub import HfApi
        resolved = HfApi().model_info(args.repo, revision=args.revision).sha
    except Exception as exc:                                    # offline, or no hub access
        resolved = None
        print(f"  ! could not resolve {args.revision} to a commit sha ({exc})")
    print(f"  revision {args.revision} -> {resolved}")

    # One probe batch before allocating 17.6 GB: it fixes the shapes, names the columns, and
    # fails on a layout surprise while the only cost is a few seconds.
    with torch.no_grad():
        probe = model(smiles[:min(8, n)])
    token_names = check_layout(model, probe)
    n_tok, dim = probe.embeddings.shape[1], probe.embeddings.shape[2]
    print("  layout verified bit-exactly: cls is column 0, projections are per-token")
    print(f"  [*, {n_tok}, {dim}] x2  tokens: {token_names}")

    # Determinism is measured, not assumed -- see DETERMINISM above. A repeat of the same
    # batch on the same device is the weakest possible version of the property, so anything
    # but 0.0 here means the cache is not reproducible at all.
    with torch.no_grad():
        again = model(smiles[:min(8, n)])
    repeat_maxdiff = max(
        (probe.embeddings - again.embeddings).abs().max().item(),
        (probe.cls - again.cls).abs().max().item(),
        (probe.predictions - again.predictions).abs().max().item(),
    )
    print(f"  repeat determinism on {device}: max|delta| = {repeat_maxdiff:.3e}")
    if repeat_maxdiff != 0.0:
        msg = (f"the model is not reproducible on {device}: embedding the same batch twice "
               f"differs by {repeat_maxdiff:.3e}. Every number downstream inherits this "
               "cache, so rebuilding it would move the results. Use --device cpu, or pass "
               "--allow-nondeterministic to write anyway.")
        if not args.allow_nondeterministic:
            raise SystemExit(msg)
        print(f"  ! {msg}")

    args.out.mkdir(parents=True, exist_ok=True)
    emb_path, proj_path = args.out / "embeddings.npy", args.out / "projected.npy"

    # Memory-mapped so peak RSS is a batch, not 17.6 GB, and so a reader can mmap one column
    # of the result without paging in the rest.
    emb_mm = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32,
                                       shape=(n, n_tok, dim))
    proj_mm = np.lib.format.open_memmap(proj_path, mode="w+", dtype=np.float32,
                                        shape=(n, n_tok, dim))

    started = time.time()
    done = 0
    for lo in range(0, n, args.batch):
        chunk = smiles[lo:lo + args.batch]
        with torch.no_grad():
            out = model(chunk)
        e = out.embeddings.detach().float().cpu().numpy()
        # cls is column 0 of the projected tensor, exactly as it is column 0 of the raw one,
        # so the two arrays stay aligned under a single token_names list.
        pr = torch.cat([out.cls.unsqueeze(1), out.predictions], dim=1)
        pr = pr.detach().float().cpu().numpy()

        # The alignment invariant the whole cache rests on. The model featurizes internally,
        # so a molecule rdkit cannot parse is exactly the case that would come back short --
        # and a short batch here silently shifts every later row onto the wrong target.
        if e.shape[0] != len(chunk) or pr.shape[0] != len(chunk):
            raise SystemExit(
                f"batch at row {lo} sent {len(chunk)} SMILES and got {e.shape[0]} "
                f"embeddings / {pr.shape[0]} projections back -- row alignment with the CSV "
                f"is broken, refusing to write. First SMILES of the batch: {chunk[0]!r}")
        if not (np.isfinite(e).all() and np.isfinite(pr).all()):
            raise SystemExit(f"non-finite values in the batch at row {lo}")

        emb_mm[lo:lo + len(chunk)] = e
        proj_mm[lo:lo + len(chunk)] = pr
        done += len(chunk)

        if lo % (args.batch * 20) == 0 or done == n:
            rate = done / (time.time() - started)
            eta = (n - done) / rate if rate else float("nan")
            print(f"  {done:>7,} / {n:,}  ({rate:,.0f} mol/s, eta {eta/60:4.1f} min)",
                  flush=True)

    emb_mm.flush(); proj_mm.flush()
    del emb_mm, proj_mm
    elapsed = time.time() - started
    print(f"embedded {n:,} molecules in {elapsed/60:.1f} min ({n/elapsed:,.0f} mol/s)")

    import rdkit
    import transformers
    try:
        import molfeat
        molfeat_version = molfeat.__version__
    except Exception:
        molfeat_version = None

    meta = {
        "script": "src/jepa_embed.py",
        "argv": sys.argv,
        "encoder": "moljepa-v1",
        "input": str(args.input),
        "input_abspath": str(args.input.resolve()),
        "input_sha256": input_sha,
        "input_rows": n_rows_csv,
        "limit": args.limit,
        "n_molecules": n,
        "row_order": "exact CSV row order; splits.py indices apply directly",
        "hf_repo": args.repo,
        "hf_revision_requested": args.revision,
        "hf_revision": resolved,
        "token_names": token_names,
        "cls_index": 0,
        "embed_dim": dim,
        "batch": args.batch,
        "device": device,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "repeat_maxdiff": repeat_maxdiff,
        "arrays": {
            "embeddings": {"file": emb_path.name, "shape": [n, n_tok, dim],
                           "dtype": "float32", "sha256": _sha256_file(emb_path),
                           "bytes": emb_path.stat().st_size,
                           "what": "raw transformer output, one row per token"},
            "projected": {"file": proj_path.name, "shape": [n, n_tok, dim],
                          "dtype": "float32", "sha256": _sha256_file(proj_path),
                          "bytes": proj_path.stat().st_size,
                          "what": "modality_pred[i](embeddings[:, i]); row 0 is the "
                                  "model's `cls` output, rows 1.. are its `predictions`"},
        },
        "embed_seconds": round(elapsed, 1),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "rdkit": rdkit.__version__,
            "molfeat": molfeat_version,
        },
        "host": platform.node(),
    }
    meta_path = args.out / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {emb_path} ({emb_path.stat().st_size/1e9:.2f} GB)")
    print(f"wrote {proj_path} ({proj_path.stat().st_size/1e9:.2f} GB)")
    print(f"wrote {meta_path}")
    if resolved and args.revision == "main":
        print(f"\n  NOTE: loaded from branch 'main'. Pin HF_REVISION = {resolved!r} in "
              "this script before building a cache whose numbers you intend to report.")
    print(f"total {(time.time() - started_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
