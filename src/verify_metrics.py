"""Prove the ported loss, metrics and objective do what they claim. Writes a markdown verdict.

    .venv/bin/python src/verify_metrics.py

Companion to `verify_trunk.py`, and built the same way: every check is a function returning
`(ok, lines)`, the verdict lands in a markdown file, and the numbers are *measured* rather
than asserted from reasoning.

Two checks earn their place by being the ones a plausible-looking bug would survive:

- `check_metric_parity` compares against numbers produced by pProp_MLP's own code under
  pProp_MLP's own interpreter (`src/dump_metric_reference.py`). A reimplementation that is
  subtly wrong still produces plausible correlations and MAEs; only the source's outputs can
  distinguish "ported" from "rewritten from the docstring".

- `check_weight_flavour_ordering` fixes the *direction* of the weightings rather than a
  magnitude: `MAE_uniform < MAE_balanced`, because `balanced` up-weights the hard tail. An
  inverted weight vector still yields individually plausible numbers, so the check also
  pins the vectors down directly -- the unweighted base rate must equal the true positive
  rate, and the balanced one must be exactly 0.5.

`check_freeze_negative` is the negative test for `train.py`'s schedule assertion: it neuters
the freeze and requires the run to die. A passing assertion proves nothing unless it can be
made to fail.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import losses as L                                                          # noqa: E402
from dump_metric_reference import EDGE, make_inputs                         # noqa: E402
from metrics import flavoured_metrics                                       # noqa: E402
from objective import (OBJECTIVE_SPEC, OBJECTIVE_VERSION, REQUIRED_FLAVOURS,  # noqa: E402
                       compute_objective)

# Float64 accumulation order differs between sklearn's weighted mean and ours, so exact
# equality is the wrong bar. 1e-8 is ~7 orders below the quantities being compared and far
# tighter than any real porting error would land.
TOL = 1e-8

# The ESS table that fixed PPROP_EDGE at 3.5 (see losses.py). Re-derived here so a change to
# the weighting silently altering it would fail rather than pass unnoticed.
EXPECTED_ESS = {3.0: 38711, 3.5: 12492, 4.0: 3972, 5.0: 400}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path,
                   default=Path("data/reference/pprop_mlp_metrics_v1"))
    p.add_argument("--csv", type=Path, default=Path("data/ampc_subset_331k.csv"))
    p.add_argument("-o", "--out", type=Path, default=Path("verification_metrics.md"))
    p.add_argument("--skip-freeze", action="store_true",
                   help="skip check_freeze_negative, which needs the GPU feature cache")
    return p.parse_args(argv)


# -- checks -------------------------------------------------------------------------

def check_metric_parity(ctx):
    """Ported metrics reproduce pProp_MLP's own outputs on the same input."""
    ref_dir = ctx["args"].reference
    ref_file = ref_dir / "reference.json"
    if not ref_file.exists():
        return False, [f"missing {ref_file}; run it under the source interpreter first:",
                       "    /home/ethan2/pProp_MLP/.venv/bin/python "
                       "src/dump_metric_reference.py"]
    ref = json.loads(ref_file.read_text())
    meta = json.loads((ref_dir / "meta.json").read_text())

    y, pred = make_inputs(meta["n"], meta["seed"])
    w = {f: L.build_sample_weights(f, y) for f in ("uniform", "balanced")}
    m = flavoured_metrics(y, pred, pred, w, edge=EDGE)

    # pProp_MLP's `weighted_*` used grouped inverse frequency, which on two groups is our
    # `balanced`. Its `mae_skill` pair maps onto ours directly now that both score the same
    # constant-median baseline under their own weights.
    pairs = [("mae", "mae_uniform"), ("weighted_mae", "mae_balanced"),
             ("mae_skill", "mae_skill_uniform"),
             ("weighted_mae_skill", "mae_skill_balanced"),
             ("pearson", "pearson_uniform"), ("pearson_weighted", "pearson_balanced")]

    lines = [f"source: {meta['source_repo']} @ {meta.get('source_git_sha')}",
             f"input: n={meta['n']} seed={meta['seed']} edge={meta['edge']}", "",
             "| source key | ours | source | ours | abs delta |",
             "|---|---|---|---|---|"]
    ok = ref["n_positive"] == m["n_positive"]
    for src_key, our_key in pairs:
        delta = abs(ref[src_key] - m[our_key])
        ok &= delta < TOL
        lines.append(f"| `{src_key}` | `{our_key}` | {ref[src_key]:.12f} | "
                     f"{m[our_key]:.12f} | {delta:.2e} |")
    lines += ["", f"positives {m['n_positive']} vs source {ref['n_positive']}",
              f"tolerance {TOL:.0e} (float64 accumulation order, not a porting margin)"]
    return bool(ok), lines


def check_bce_matches_ce(ctx):
    """`weighted_bce_loss` is `CrossEntropyLoss(weight=...)` on two classes, not its mean form."""
    torch.manual_seed(0)
    n = 4000
    y = (torch.rand(n) < 0.0095).long()          # this dataset's base rate
    z = torch.randn(n)
    cw = L.grouped_frequency_weights(y.numpy(), 2, L.WEIGHT_GROUPS)
    sw = L.sample_weights_from_classes(y, cw)

    ours = float(L.weighted_bce_loss(z, y.float(), sw))
    ce = float(nn.CrossEntropyLoss(weight=cw)(torch.stack([torch.zeros(n), z], 1), y))
    naive = float(nn.BCEWithLogitsLoss(reduction="mean")(z, y.float()))

    lines = [f"weighted_bce_loss        {ours:.12f}",
             f"CrossEntropyLoss(weight) {ce:.12f}   delta {abs(ours - ce):.2e}",
             f"BCEWithLogits(mean)      {naive:.12f}   <- the form NOT used",
             "",
             f"class weights {cw.tolist()} (ratio {cw[1] / cw[0]:.1f}x)",
             "The mean form differs by a large constant at this ratio, which would move "
             "every inherited `w_cls` out of the units it was swept in."]
    return abs(ours - ce) < 1e-6, lines


def check_loss_reduces_to_mse(ctx):
    """With the other terms off and a huge delta, `combined_loss` is exactly half weighted MSE."""
    torch.manual_seed(1)
    n = 2000
    pred, tgt = torch.randn(n), torch.randn(n)
    w = torch.rand(n) + 0.1
    yb = (torch.rand(n) < 0.05).float()

    loss, _ = L.combined_loss(torch.randn(n), pred, yb, tgt, w,
                              w_cls=0.0, w_pair=0.0, w_std=0.0, huber_delta=1e6)
    mse = L.weighted_mse_loss(pred, tgt, w)
    ratio = float(loss) / float(mse)
    lines = [f"combined_loss {float(loss):.12f}", f"weighted_mse  {float(mse):.12f}",
             f"ratio {ratio:.12f} (expected exactly 0.5)",
             "",
             "The factor of 2 is `F.huber_loss`'s definition -- 0.5*r^2 below the "
             "transition, not r^2 -- and is inherited unchanged from pProp_MLP, which "
             "called the same function. So the loss scale `w_cls`/`w_pair`/`w_std` were "
             "swept against is preserved."]
    return abs(ratio - 0.5) < 1e-6, lines


def check_weight_flavour_ordering(ctx):
    """`balanced` emphasises the hard tail; MAE and the base rate must both show it."""
    y = ctx["y"]
    rng = np.random.default_rng(0)
    pred = y * 0.75 + rng.normal(0, 0.45, len(y))
    w = {f: L.build_sample_weights(f, y) for f in REQUIRED_FLAVOURS}
    m = flavoured_metrics(y, pred, pred, w)

    order = m["mae_uniform"] < m["mae_balanced"]
    # The direct statement of what `balanced` claims to do: equal total weight mass per
    # group, so the weighted positive rate is 1/2 whatever the true rate is. An inverted or
    # mis-indexed weight vector still produces a plausible MAE, but it cannot produce this
    # number. The tolerance is 1e-6 rather than exact because `grouped_frequency_weights`
    # builds the weights as torch float32 (eps 1.2e-7); measured deviation is 2.5e-9, and
    # the same weights in float64 give exactly 0.5. A float32 margin, not a porting margin.
    balanced_rate = abs(m["base_rate_balanced"] - 0.5) < 1e-6
    true_rate = m["n_positive"] / m["n"]
    uniform_rate = abs(m["base_rate_uniform"] - true_rate) < 1e-12

    lines = [f"MAE   uniform {m['mae_uniform']:.6f}  <  balanced {m['mae_balanced']:.6f}"
             f"   -> {order}",
             "",
             f"base rate  uniform {m['base_rate_uniform']:.8f} (true "
             f"{m['n_positive']}/{m['n']} = {true_rate:.8f}, match {uniform_rate})",
             f"base rate  balanced {m['base_rate_balanced']:.12f} (0.5 to 1e-6, match "
             f"{balanced_rate}; the 2.5e-9 gap is float32 in "
             "`grouped_frequency_weights`, exact in float64)",
             "",
             "Direction, not magnitude: `balanced` up-weights the hard tail 104x, so it must "
             "score *worse* than unweighted on a model that fits the easy bulk better. A "
             "sign error inverts the vector while leaving each number individually "
             "plausible, so the base rates pin the vector down as well -- unweighted must "
             "reproduce the true positive rate, and balanced must be exactly one half.",
             "",
             "This check previously fixed three flavours against each other "
             "(`MAE_ipw < MAE_uniform < MAE_balanced`). `ipw` was removed from the modelling "
             "code on 2026-08-12, so the ordering is now two-sided and the base-rate "
             "assertions carry the weight the third flavour used to."]
    return bool(order and balanced_rate and uniform_rate), lines


def check_objective(ctx):
    """`goal_metric` decomposes exactly, and a missing metric raises instead of going nan."""
    y = ctx["y"]
    rng = np.random.default_rng(2)
    pred = y * 0.75 + rng.normal(0, 0.45, len(y))
    w = {f: L.build_sample_weights(f, y) for f in REQUIRED_FLAVOURS}
    m = flavoured_metrics(y, pred, pred, w)
    o = compute_objective(m)

    sums = abs((o["goal_term_cls"] + o["goal_term_reg"]) - o["goal_metric"]) < 1e-12

    # The deleted key must be one OBJECTIVE_SPEC actually names, or this passes while
    # testing nothing. Taken from the spec rather than written out, so it cannot drift out
    # of the spec the way `ap_ipw` did when that flavour was removed.
    victim = OBJECTIVE_SPEC["ap_star"][0]
    raised = False
    try:
        compute_objective({k: v for k, v in m.items() if k != victim})
    except KeyError:
        raised = True

    lines = [f"objective version `{OBJECTIVE_VERSION}` (derived from the spec, so an edit "
             "re-stamps it automatically)", ""]
    lines += [f"{k:16s} {v:.6f}" for k, v in o.items()]
    lines += ["", f"cls + reg == goal_metric: {sums}",
              f"missing-metric guard raises when `{victim}` is dropped: {raised}",
              "",
              "pProp_MLP accumulated three incompatible `goal_metric` revisions under one "
              "name; the derived version string is what stops that recurring here."]
    return bool(sums and raised), lines


def check_ess_table(ctx):
    """The effective-sample-size table that fixed the group edge at 3.5."""
    y = ctx["y"]
    lines = ["| edge | groups | ratio | ESS | % of N |", "|---|---|---|---|---|"]
    ok = True
    for edge, expected in EXPECTED_ESS.items():
        w = L.balanced_sample_weights(y, edge)
        ess = L.effective_sample_size(w)
        pos = int((y >= edge).sum())
        ok &= abs(ess - expected) < 1.0
        lines.append(f"| {edge} | {len(y) - pos:,} / {pos:,} | "
                     f"{w.max() / w.min():.0f}x | {ess:,.0f} | {100 * ess / len(y):.2f}% |")
    lines += ["", "3.5 is the finest edge that leaves a usable effective sample; 5.0 would "
              "train on an effective 400 molecules."]
    return bool(ok), lines


def check_weighted_std(ctx):
    """The weighted std term targets a visibly different spread from the unweighted one."""
    y = ctx["y"]
    w = torch.tensor(L.balanced_sample_weights(y))
    t = torch.tensor(y)
    zero = torch.zeros_like(t)
    unweighted = float(torch.sqrt(L.std_match_loss(zero, t)))
    weighted = float(torch.sqrt(L.std_match_loss(zero, t, w)))
    ratio = weighted / unweighted
    lines = [f"target std unweighted {unweighted:.4f}  |  balanced-weighted "
             f"{weighted:.4f}  ({ratio:.2f}x)",
             "",
             "An unweighted std term paired with a weighted huber would pull prediction "
             "spread toward the smaller number while the huber pulled toward the larger. "
             "`train.py` passes weights to both."]
    return ratio > 1.5, lines


def check_freeze_negative(ctx):
    """Neutering the freeze must kill the run. A passing assertion proves nothing otherwise."""
    if ctx["args"].skip_freeze:
        return True, ["skipped (--skip-freeze)"]
    import train

    original = train.set_trunk_trainable
    train.set_trunk_trainable = lambda *a, **k: None      # the bug being simulated
    out = Path(ctx["tmp"]) / "freeze_negative"
    argv = ["--epochs", "2", "--freeze-epochs", "1", "--subset", "1500",
            "--batch-size", "256", "--num-workers", "2", "--no-wandb", "--out", str(out)]
    try:
        train.main(argv)
        died, why = False, "run completed -- the freeze assertion did NOT fire"
    except SystemExit as exc:
        died = "freeze violated" in str(exc)
        why = str(exc).split("\n")[0][:120]
    finally:
        train.set_trunk_trainable = original

    return died, [f"with `set_trunk_trainable` neutered: {'died' if died else 'SURVIVED'}",
                  f"  {why}",
                  "",
                  "The trunk trains through the frozen phase when the freeze is a no-op, so "
                  "`trunk_max_delta` is non-zero and the paired check fires. Confirms the "
                  "assertion in `train.py` has teeth."]


CHECKS = [
    ("metric parity vs pProp_MLP", check_metric_parity),
    ("BCE equals CrossEntropy(weight)", check_bce_matches_ce),
    ("combined loss reduces to weighted MSE", check_loss_reduces_to_mse),
    ("weighting flavours order correctly", check_weight_flavour_ordering),
    ("objective decomposes and guards", check_objective),
    ("effective sample size table", check_ess_table),
    ("weighted std differs from unweighted", check_weighted_std),
    ("freeze assertion has teeth", check_freeze_negative),
]


def main(argv=None):
    args = parse_args(argv)
    ctx = {"args": args,
           "y": pd.read_csv(args.csv, usecols=["pprop"])["pprop"].to_numpy(dtype=np.float64),
           "tmp": Path("outputs/_verify")}
    Path(ctx["tmp"]).mkdir(parents=True, exist_ok=True)

    started = time.time()
    results, body = [], []
    for name, fn in CHECKS:
        t0 = time.time()
        try:
            ok, lines = fn(ctx)
        except Exception as exc:                      # a crashing check is a failing check
            ok, lines = False, [f"raised {type(exc).__name__}: {exc}"]
        results.append((name, ok))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}  ({time.time() - t0:.1f}s)")
        body += [f"## {status} — {name}", ""] + lines + [""]

    overall = "PASS" if all(ok for _, ok in results) else "FAIL"
    header = [f"# Metric and loss verification: **OVERALL: {overall}**", "",
              f"{sum(ok for _, ok in results)}/{len(results)} checks pass. "
              f"Objective `{OBJECTIVE_VERSION}`.", "",
              "| check | result |", "|---|---|"]
    header += [f"| {n} | {'PASS' if ok else 'FAIL'} |" for n, ok in results]
    header += ["",
               f"Ran in {time.time() - started:.1f}s under python "
               f"{platform.python_version()}, numpy {np.__version__}, "
               f"torch {torch.__version__}.", "", "---", ""]

    args.out.write_text("\n".join(header + body) + "\n")
    print(f"\nOVERALL: {overall} -> {args.out}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
