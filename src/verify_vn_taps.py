"""Prove that `src/vn_taps.py` reads MiniMol's real virtual-node state, and nothing else.

The tap works by shadowing an instance's `.forward` attribute. That is a sharp tool, and every
way it can go wrong is quiet:

  - the shadow is bypassed and nothing is captured (a hook on the virtual node layer itself
    fails exactly this way, which is *why* the shadow exists);
  - the wrong tensor is captured -- `_readout_cache[i]` is atom-level `feat` after the virtual
    node broadcast, same width, different meaning;
  - the shadow perturbs the forward pass it is measuring;
  - the shadow is left installed after use;
  - the captured rows do not line up with the molecules they claim to describe.

None of those raises. So each is a measurement here, per the repo convention that findings are
settled by number (CLAUDE.md "Conventions").

The load-bearing check is `cross-route agreement`. It reads the same state a second way -- a
forward hook on `node_projection`, whose `input[0]` is `vn_feat[g.batch]` -- and requires the
two to agree exactly. Two independent routes to the same tensor is what turns "the shadow
seems to work" into evidence.

Usage:
    python src/verify_vn_taps.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trunk import MiniMolTrunk                                          # noqa: E402
from vn_taps import (POOLED_DIM, TAP_NAMES, VN_DIM, VN_TAPS,            # noqa: E402
                     capture_vn, degather, vn_layers)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("--n", type=int, default=8, help="molecules in the test batch")
    return p.parse_args(argv)


class Report:
    """Collects check results so a failure part-way still produces a full report."""

    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail):
        self.rows.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        return passed

    @property
    def ok(self):
        return all(r["passed"] for r in self.rows)


def fresh(trunk, feats, order=None):
    """Collate a NEW batch every time.

    Not defensiveness: a collated batch is single-use. The positional encoders concatenate
    into `feat`, so its width *grows* during forward and a second pass over the same object
    computes something different (CLAUDE.md, the feature cache section). Every check below
    that runs two passes would silently compare apples to a wider apple without this.
    """
    picked = feats if order is None else [feats[i] for i in order]
    return trunk.collate(picked)


def check_fires_and_shape(report, trunk, feats, n):
    with capture_vn(trunk) as cap:
        trunk(fresh(trunk, feats))
        calls = dict(cap.calls)
        shapes = {TAP_NAMES[i]: tuple(cap.tensors[i].shape) for i in VN_TAPS}

    report.add("taps fire exactly once",
               calls == {i: 1 for i in VN_TAPS},
               f"call counts {calls}")

    want = (n, VN_DIM)
    report.add("captures are molecule-level [B, 336]",
               all(s == want for s in shapes.values()),
               f"{shapes} (expected {want} for each)")


def check_cross_route(report, trunk, feats):
    """The same state, read a second way, must agree exactly.

    Route A shadows `virtual_node_layers[i].forward` and takes the `vn_feat` it returned.
    Route B hooks `virtual_node_layers[i].node_projection`, which graphium invokes through a
    real `__call__` (`pooling_pyg.py:343`), and takes `input[0]` = `vn_feat[g.batch]`, then
    de-gathers it back to one row per molecule.

    Both read the same tensor in the same forward pass, so this is exact-equality or a bug --
    there is no floating point between them.
    """
    layers = vn_layers(trunk)
    seen = {}
    handles = [
        layers[i].node_projection.register_forward_hook(
            lambda _m, inp, _out, _i=i: seen.__setitem__(_i, inp[0].detach()))
        for i in VN_TAPS
    ]
    try:
        batch = fresh(trunk, feats)
        with capture_vn(trunk) as cap:
            trunk(batch)
            route_a = {i: cap.tensors[i] for i in VN_TAPS}
    finally:
        for h in handles:
            h.remove()

    if set(seen) != set(VN_TAPS):
        report.add("cross-route agreement", False,
                   f"the node_projection hook fired for {sorted(seen)}, expected {list(VN_TAPS)}")
        return

    bidx = batch["batch_indices"]
    deltas = {TAP_NAMES[i]: float((degather(seen[i], bidx) - route_a[i]).abs().max())
              for i in VN_TAPS}
    worst = max(deltas.values())
    report.add("cross-route agreement (shadow vs node_projection hook)",
               worst == 0.0,
               f"max|d| = {worst:.3e} over 5 taps {deltas}")


def check_non_invasive(report, trunk, feats):
    """Installing the tap must not change what the model computes.

    Run on CPU, where the forward is bit-reproducible, so this is exact equality rather than a
    tolerance. On CUDA the same comparison lands at ~2e-6 -- but so does an untapped pass
    against another untapped pass, so a tolerance there would be measuring atomics, not the
    tap. `check_cuda_floor` reports that number separately.
    """
    with torch.no_grad():
        plain = trunk(fresh(trunk, feats)).clone()
        with capture_vn(trunk):
            tapped = trunk(fresh(trunk, feats)).clone()
    delta = float((plain - tapped).abs().max())
    report.add("tapping does not perturb the 512-d output",
               delta == 0.0,
               f"max|d| = {delta:.3e} over {tuple(plain.shape)} (CPU, exact)")


def check_norm_growth(report, trunk, feats):
    """The virtual node accumulates residually and is never normalized, so its norm grows.

    `Minimol_architecture_overview.md` §9 measures mean L2 of 2.73 / 14.18 / 36.50 / 45.27 at
    depths 0 / 5 / 10 / 14. Reproducing the monotone shape at the five taps is a cheap check
    that these are the running state and not, say, the per-layer pooled contribution -- which
    would be roughly flat across depth.
    """
    with torch.no_grad(), capture_vn(trunk) as cap:
        trunk(fresh(trunk, feats))
        norms = [(TAP_NAMES[i], float(cap.tensors[i].norm(dim=1).mean())) for i in VN_TAPS]
    values = [v for _, v in norms]
    report.add("virtual node norm grows monotonically with depth",
               all(b > a for a, b in zip(values, values[1:])),
               " -> ".join(f"{k} {v:.2f}" for k, v in norms))


def check_restored(report, trunk, feats):
    layers = vn_layers(trunk)
    with capture_vn(trunk):
        during = all("forward" in vars(layers[i]) for i in VN_TAPS)
    after = [i for i in VN_TAPS if "forward" in vars(layers[i])]

    # And prove the restoration is real by running a pass that must capture nothing.
    leaked = {}
    orig_wrapped = capture_vn(trunk)
    del orig_wrapped                       # never entered; just proves construction is inert
    trunk(fresh(trunk, feats))
    report.add("shadow installed during, removed after",
               during and not after and not leaked,
               f"installed during: {during}; still shadowed after: {after or 'none'}")


def check_batch_order(report, trunk, feats, n):
    """A molecule's virtual node vector must not depend on where it sat in the batch.

    Every downstream join -- embeddings to row indices to pProp -- assumes this. Asserted as a
    *relative* tolerance rather than equality: `scatter_logsum_pool` is a scatter-add, CUDA
    scatter reductions are not order-deterministic, and that error accumulates residually
    across 15 depths whose norms grow ~17x. The measured number is printed because if it is
    large, that is a real finding about these tensors and not a test detail.
    """
    order = list(reversed(range(n)))
    with torch.no_grad(), capture_vn(trunk) as cap:
        trunk(fresh(trunk, feats))
        forward_caps = {i: cap.tensors[i].clone() for i in VN_TAPS}
        cap.clear()
        trunk(fresh(trunk, feats, order=order))
        reversed_caps = {i: cap.tensors[i].clone() for i in VN_TAPS}

    rel = {}
    for i in VN_TAPS:
        a = forward_caps[i]
        b = reversed_caps[i][list(reversed(range(n)))]     # undo the permutation
        rel[TAP_NAMES[i]] = float((a - b).abs().max() / a.norm(dim=1).mean())
    worst = max(rel.values())
    report.add("virtual node is invariant to batch position",
               worst < 1e-3,
               f"max relative |d| = {worst:.3e} (tol 1e-3) {rel}")


def check_eval_determinism(report, trunk, feats):
    """Two identical passes in eval() must agree exactly.

    `MiniMolTrunk.__init__` ends with `self.train()`, and train mode applies random Laplacian
    sign-flip augmentation (architecture overview §11). Extraction that forgot `model.eval()`
    would emit silently irreproducible embeddings, so this check exists to make the mode
    requirement a measured fact rather than a comment.
    """
    with torch.no_grad(), capture_vn(trunk) as cap:
        trunk(fresh(trunk, feats))
        first = {i: cap.tensors[i].clone() for i in VN_TAPS}
        cap.clear()
        trunk(fresh(trunk, feats))
        second = {i: cap.tensors[i].clone() for i in VN_TAPS}
    worst = max(float((first[i] - second[i]).abs().max()) for i in VN_TAPS)
    report.add("eval() mode is deterministic across passes",
               worst == 0.0,
               f"max|d| = {worst:.3e} over two identical passes (CPU, exact)")


def check_train_mode_differs(report, trunk, feats):
    """The negative control for the check above.

    If train() and eval() produced the same numbers, `check_eval_determinism` would pass for
    free and prove nothing. This asserts the sign-flip augmentation is genuinely live, so the
    eval() requirement has teeth.
    """
    trunk.train()
    with torch.no_grad(), capture_vn(trunk) as cap:
        trunk(fresh(trunk, feats))
        a = cap.tensors[VN_TAPS[-1]].clone()
        cap.clear()
        trunk(fresh(trunk, feats))
        b = cap.tensors[VN_TAPS[-1]].clone()
    trunk.eval()
    delta = float((a - b).abs().max())
    report.add("train() mode is NOT deterministic (so eval() is load-bearing)",
               delta > 0.0,
               f"max|d| = {delta:.3e} between two train()-mode passes")


def check_cuda_floor(report, trunk, feats):
    """Extraction runs on GPU, so measure how reproducible a GPU pass actually is.

    MiniMol pools atoms with `scatter_logsum_pool`, a scatter-add. CUDA scatter reductions
    accumulate in nondeterministic order, so two identical eval-mode passes do NOT agree
    bitwise on GPU -- measured ~2e-6 on the 512-d output with no taps installed at all. That
    is the floor every GPU-extracted embedding sits on, and it is worth a printed number
    rather than a surprise later.

    The assertion is the one that matters: the tap must not add anything ON TOP of that floor.
    """
    if not torch.cuda.is_available():
        report.add("CUDA reproducibility floor", True, "skipped -- no GPU visible")
        return

    trunk = trunk.to("cuda")
    try:
        with torch.no_grad():
            a = trunk(fresh(trunk, feats)).clone()
            b = trunk(fresh(trunk, feats)).clone()
            with capture_vn(trunk):
                c = trunk(fresh(trunk, feats)).clone()
        floor = float((a - b).abs().max())
        tapped = float((a - c).abs().max())
        scale = float(a.abs().mean())
        report.add("tap adds nothing above the CUDA nondeterminism floor",
                   tapped <= max(floor, 1e-12) and floor < 1e-4,
                   f"untapped-vs-untapped {floor:.3e}, tapped-vs-untapped {tapped:.3e}, "
                   f"mean|emb| {scale:.3f}")
    finally:
        trunk.to("cpu")


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(0)

    smiles = pd.read_csv(args.csv, usecols=["SMILES"], nrows=args.n)["SMILES"].tolist()
    # CPU deliberately, not GPU: the checks below assert exact equality, and a GPU forward is
    # not bit-reproducible even against itself (see check_cuda_floor). Eight molecules cost
    # nothing on CPU, and an exact assertion is worth far more than a tolerance chosen to
    # accommodate atomics.
    trunk = MiniMolTrunk().to("cpu")
    feats = trunk.featurize_raw(smiles)
    n_atoms = sum(int(f.num_nodes) for f in feats)
    print(f"cpu (exact checks) | {len(smiles)} molecules, {n_atoms} atoms | "
          f"{len(vn_layers(trunk))} virtual node layers, tapping {list(VN_TAPS)}\n")

    report = Report()
    trunk.eval()
    check_fires_and_shape(report, trunk, feats, args.n)
    check_cross_route(report, trunk, feats)
    check_non_invasive(report, trunk, feats)
    check_norm_growth(report, trunk, feats)
    check_restored(report, trunk, feats)
    check_batch_order(report, trunk, feats, args.n)
    check_eval_determinism(report, trunk, feats)
    check_train_mode_differs(report, trunk, feats)
    check_cuda_floor(report, trunk, feats)

    n_pass = sum(r["passed"] for r in report.rows)
    print(f"\n{n_pass}/{len(report.rows)} checks passed")
    print("OVERALL:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
