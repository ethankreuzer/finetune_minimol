"""The shape an active-learning loop should have, and the one mistake that makes it slow.

    python examples/active_learning.py

THE POINT OF THIS FILE
----------------------
`enc.encode(smiles)` does two things: it converts SMILES to molecular graphs (CPU-bound,
single-threaded from your side, already parallel internally), and it runs those graphs
through the network (GPU-bound). Only the second depends on the model.

SMILES -> graph is a pure function of the string. In a loop that scores the same candidate
pool round after round, calling `encode(smiles)` every round re-derives an answer that cannot
have changed. Measured in the source project: featurization runs at ~775-1,745 mol/s, against
a full epoch over 265k graphs in ~1.7 min -- so on a fixed pool the conversion, not the
network, becomes the loop's cost.

So: **featurize the pool once, cache it, and encode from the cache every round.** Reloading a
cache is ~1 s where re-featurizing is minutes.

    round 0:  build_cache(pool) -> encode_graphs(...)      pay the CPU cost once
    round n:  load_cache(...)   -> encode_graphs(...)      ~1 s to reload

The embeddings themselves do not change either, as long as the encoder stays frozen -- which
is the intended use. If you are only ever using THIS model frozen, encode the pool once and
cache the vectors too; the loop below re-encodes each round so that it still illustrates the
right thing if you ever swap the model between rounds.
"""

import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from minimol_ampc import MiniMolAmpcEncoder, featurize      # noqa: E402


def make_pool(n=64):
    """Stand-in for your candidate pool. Replace with your own SMILES."""
    info = __import__("json").loads((PKG / "fixtures" / "fixture.json").read_text())
    pool = list(info["smiles"])
    return (pool * (n // len(pool) + 1))[:n]


def main():
    pool = make_pool()
    cache_dir = Path("al_cache/pool")

    enc = MiniMolAmpcEncoder.load(PKG / "model" / "final.pt")
    print(f"encoder on {enc.device}\n")

    # ---- once per pool: SMILES -> graphs -> disk -------------------------------------
    #
    # `trunk=enc.model.trunk` reuses the graphium object the encoder already built, saving
    # ~30 s of construction. `on_invalid="skip"` is the right default for an AL loop over
    # GENERATED molecules: graphium returns an error string rather than raising for a SMILES
    # it cannot featurize, and a generator will produce some. The cache records which ones.
    if not (cache_dir / "meta.json").exists():
        t0 = time.time()
        meta = featurize.build_cache(pool, cache_dir, on_invalid="skip",
                                     trunk=enc.model.trunk, quiet=True)
        print(f"featurized {meta['n_graphs']:,} of {meta['n_input']:,} molecules in "
              f"{time.time() - t0:.1f}s -> {cache_dir}"
              + (f" ({len(meta['invalid'])} unfeaturizable, recorded in meta.json)"
                 if meta["invalid"] else ""))

    # ---- every round: reload and encode ----------------------------------------------
    t0 = time.time()
    graphs, meta = featurize.load_cache(cache_dir)
    load_s = time.time() - t0

    # `graphs.smiles` is the cache's OWN row order -- the molecules that survived
    # featurization, in order. Use it, rather than your original `pool` list, to tie a row
    # back to a molecule. This is where a skipped SMILES silently misaligns everything.
    labelled = np.zeros(len(graphs), dtype=bool)
    labelled[:8] = True                                    # pretend round 0 is done

    for round_i in range(3):
        t0 = time.time()
        out = enc.encode_graphs(graphs, batch_size=256)
        x = out.pooled512                                   # (n_pool, 512) -- feed your GP
        encode_s = time.time() - t0

        # Your DKL/GP goes here: fit on x[labelled], score x[~labelled], pick a batch.
        # Stand-in acquisition so this file runs end to end: take the highest predicted pProp
        # among the unlabelled. A real loop would use the GP's posterior, not the point
        # prediction -- the uncertainty is the whole reason the encoder's geometry matters.
        scores = np.where(labelled, -np.inf, out.pprop)
        picked = np.argsort(scores)[-4:]
        labelled[picked] = True

        print(f"round {round_i}: cache load {load_s:.2f}s | encode {len(graphs):,} mols in "
              f"{encode_s:.2f}s ({len(graphs)/max(encode_s,1e-9):,.0f} mol/s) | "
              f"x {x.shape} | {labelled.sum()} labelled")
        load_s = 0.0

    print("\nBefore an RBF/Matern kernel, standardize: pooled512 is a max-pool output, so "
          "its\ndimensions have different scales and are not centred. model/"
          "pooled512_stats.npz\nholds the per-dimension mean and std to use:")
    print("    s = np.load('model/pooled512_stats.npz'); xs = (x - s['mean']) / s['std']")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
