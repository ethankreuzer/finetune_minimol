"""Dump frozen MiniMol embeddings as a regression fixture for the trainable trunk.

`src/trunk.py` reimplements MiniMol's forward pass without `torch.inference_mode()` so
gradients can flow (NOTES.md §§4, 6.3). The only thing that makes that reimplementation
trustworthy is a numerical check against the original, and the original cannot be
constructed alongside it conveniently: `Minimol.load_config` calls `hydra.initialize`,
which is a per-process singleton (NOTES §9). So the reference is produced *here*, in its
own process, and saved.

Three artifacts, all consumed by `src/verify_trunk.py`:

  <out>.npy          [n, 512] float64 embeddings from the stock, frozen `Minimol`
  <out>.json         SMILES, argv, hashes, versions, and the observed dtypes
  <out>.features.pt  the featurized PyG `Data` list for the same molecules

The third is the one that is easy to skip and expensive to add later. If the trunk's
embeddings disagree with the reference, the featurized inputs localise the fault: identical
features with different embeddings means we read the wrong tensor out of the model;
differing features means our featurization path diverged before the model was even reached.
Those two have completely different fixes.

Usage:
    python src/dump_reference_embeddings.py
    python src/dump_reference_embeddings.py --n 128 --seed 1 -o data/reference/ref128
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import _sha256_file  # noqa: E402  (same repo, reused rather than duplicated)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path,
                   default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("-o", "--out", type=Path,
                   default=Path("data/reference/minimol_v1_ref64"),
                   help="output stem; .npy, .json and .features.pt are appended")
    p.add_argument("--n", type=int, default=64, help="molecules to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=100,
                   help="passed to Minimol(); 100 is the library default")
    return p.parse_args(argv)


def sample_smiles(input_csv, n, seed):
    """A seeded random sample of SMILES, never the first n rows.

    `ampc_subset_331k.csv` is sorted by `(-pprop, smiles)` (NOTES §9), so `head(n)` would
    return the 64 most potent molecules in the set — a fixture drawn entirely from the
    thin tail, and unrepresentative of the featurization paths a real batch exercises.
    """
    df = pd.read_csv(input_csv, usecols=["SMILES", "pprop"])
    if n > len(df):
        raise SystemExit(f"--n {n} exceeds {len(df)} rows in {input_csv}")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=n, replace=False))
    return idx, df["SMILES"].to_numpy()[idx].tolist(), df["pprop"].to_numpy()[idx]


def describe_dtypes(model, embeddings):
    """Record what precision the reference path actually ran at.

    This sets the tolerance `verify_trunk.py` compares against. `Minimol.__call__` upcasts
    half-precision *features* via `to_fp32`, but that says nothing about the weights, and
    guessing fp32 when the path is fp16 would turn a correct reimplementation into an
    apparent failure at a 1e-5 threshold.
    """
    info = {"embedding_dtype": str(embeddings[0].dtype)}

    network = getattr(model.predictor, "network", None)
    if network is not None:
        for name, param in network.named_parameters():
            if name.startswith("gnn."):
                info["gnn_weight_name"] = name
                info["gnn_weight_dtype"] = str(param.dtype)
                break
        info["param_dtypes"] = sorted({str(p.dtype) for p in network.parameters()})
    return info


def main(argv=None):
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    idx, smiles, pprop = sample_smiles(args.input, args.n, args.seed)
    print(f"sampled {len(smiles)} SMILES from {args.input} (seed {args.seed})")

    import torch
    from torch_geometric.data import Batch
    from minimol import Minimol

    started = time.time()
    model = Minimol(batch_size=args.batch_size)
    print(f"constructed Minimol in {time.time() - started:.1f}s")

    def embed(tag):
        started = time.time()
        out = model(smiles)
        if len(out) != len(smiles):
            raise SystemExit(
                f"Minimol returned {len(out)} embeddings for {len(smiles)} SMILES"
            )
        arr = torch.stack([e.detach().float().cpu() for e in out]).double().numpy()
        print(f"  {tag}: {arr.shape} in {time.time() - started:.1f}s")
        return arr, out

    # Is the stock model even deterministic? The pretrained config carries live dropout
    # (pre_nn 0.02, pre_nn_edges 0.02, gnn 0.02, la_pos/rw_pos 0.1) and minimol 1.3.4 never
    # calls .eval(), so `Minimol.__call__` fingerprints in *training* mode. If that makes
    # repeated calls disagree, then no fixed reference exists and any tolerance compared
    # against it is meaningless -- an eval-mode trunk would look "wrong" purely because the
    # reference was random. Measure it rather than reason about it.
    print("stock Minimol (as shipped, training mode):")
    stock_a, _ = embed("run 1")
    stock_b, _ = embed("run 2")
    stock_repeat_diff = float(np.abs(stock_a - stock_b).max())
    deterministic = stock_repeat_diff == 0.0
    print(f"  max|Δ| between identical calls: {stock_repeat_diff:.3e} "
          f"-> {'deterministic' if deterministic else 'NON-DETERMINISTIC'}")

    # Put the network in eval mode and re-measure. This is a deliberate deviation from
    # stock `Minimol.__call__`, recorded as such: fingerprinting with dropout active is
    # not a property anyone wants in a reference, and the fine-tuning trunk is compared
    # against it in eval mode too, so both sides are consistent.
    print("\neval mode (dropout off):")
    model.predictor.network.eval()
    eval_a, eval_out = embed("run 1")
    eval_b, _ = embed("run 2")
    eval_repeat_diff = float(np.abs(eval_a - eval_b).max())
    print(f"  max|Δ| between identical calls: {eval_repeat_diff:.3e}")
    if eval_repeat_diff != 0.0:
        raise SystemExit(
            f"eval-mode embeddings are still non-deterministic (max|Δ| = "
            f"{eval_repeat_diff:.3e}). Something other than dropout is stochastic -- "
            "likely Laplacian eigenvector sign flipping in the pe_encoders. A fixed "
            "reference cannot be produced until that is pinned down."
        )

    embeddings = eval_out
    dtype_info = describe_dtypes(model, embeddings)
    dtype_info["stock_train_mode_repeat_maxdiff"] = stock_repeat_diff
    dtype_info["eval_mode_repeat_maxdiff"] = eval_repeat_diff
    dtype_info["stock_vs_eval_maxdiff"] = float(np.abs(stock_a - eval_a).max())
    dtype_info["mode"] = "eval"

    # float64 so the fixture never limits the comparison's precision; the tolerance should
    # be set by the model's dtype, not by how we chose to store the reference.
    array = torch.stack([e.detach().float().cpu() for e in embeddings]).double().numpy()
    if array.shape != (len(smiles), 512):
        raise SystemExit(f"expected [{len(smiles)}, 512] embeddings, got {array.shape}")

    # Featurize separately from the embedding call. `Minimol.__call__` does this internally
    # and throws the result away, so this repeats it rather than intercepting it -- which
    # also means the saved features are exactly what `to_fp32` produces, the same object
    # the model consumed.
    features, _ = model.datamodule._featurize_molecules(smiles)
    features = model.to_fp32(features)
    n_bad = sum(1 for f in features if isinstance(f, str))
    if n_bad:
        raise SystemExit(f"{n_bad} molecules failed to featurize")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    npy_path = out.with_suffix(".npy")
    pt_path = out.with_suffix(".features.pt")
    json_path = out.with_suffix(".json")

    np.save(npy_path, array)
    torch.save(features, pt_path)

    import graphium
    import minimol
    meta = {
        "argv": sys.argv,
        "input": str(args.input),
        "input_abspath": str(args.input.resolve()),
        "input_sha256": _sha256_file(args.input),
        "n": len(smiles),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "row_idx": idx.tolist(),
        "smiles": smiles,
        "pprop": [float(p) for p in pprop],
        "embedding_shape": list(array.shape),
        "embedding_stored_dtype": str(array.dtype),
        "embedding_sha256": __import__("hashlib").sha256(array.tobytes()).hexdigest(),
        "runtime_dtypes": dtype_info,
        "mode": "eval",
        "stock_minimol_deterministic": deterministic,
        "versions": {
            "minimol": getattr(minimol, "__version__", "1.3.4"),
            "graphium": getattr(graphium, "__version__", "unknown"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nembeddings   {array.shape} {array.dtype}  -> {npy_path}")
    print(f"features     {len(features)} graphs          -> {pt_path}")
    print(f"metadata                                     -> {json_path}")
    print(f"\nruntime info: {json.dumps(dtype_info, indent=2)}")
    print(f"embedding norm: mean {np.linalg.norm(array, axis=1).mean():.4f}")
    if not deterministic:
        print("\nNOTE: stock Minimol is non-deterministic (it fingerprints with dropout "
              "live). The saved reference is the eval-mode embedding; verify_trunk.py "
              "compares against it in eval mode too.")


if __name__ == "__main__":
    main()
