"""Prove the transformer head actually uses the token axis, and that its loss is well-formed.

`TokenTransformerHead` introduces exactly one new failure mode over the MLP arm, and it is
completely silent: **`nn.TransformerEncoder` is permutation-equivariant.** If `token_embed`
were dropped, mis-shaped, or never received gradient, the 13 tokens would be an unordered bag
-- `graph` indistinguishable from `boltz`, a real encoding indistinguishable from a
reconstruction. The model would train, the loss would fall, `goal_metric` would land somewhere
plausible, and nothing would ever error.

So the position property is checked **two-sidedly**, in the shape of `verify_trunk.py`'s
`check_grad_flow`: permuting the tokens must change the output, AND the same permutation must
become a no-op once `token_embed` is zeroed under mean pooling. A check that has only ever
seen the passing case cannot distinguish a working position parameter from a dead one -- it
would pass just as happily on a head that ignores position entirely.

Usage:
    python src/verify_jepa_head.py
    python src/verify_jepa_head.py -o verification_jepa_head.md
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from head import TokenTransformerHead                                        # noqa: E402
from losses import PPROP_EDGE, binary_labels, combined_loss                  # noqa: E402
from verify_jepa_embed import Report                                         # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", type=Path, default=Path("verification_jepa_head.md"))
    p.add_argument("--tokens", type=int, default=13)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    return p.parse_args(argv)


def build(pooling="mean", **kw):
    torch.manual_seed(0)
    return TokenTransformerHead(pooling=pooling, **kw).eval()


def check_position_is_used(report, head, x):
    """Permuting the 13 tokens must change the output.

    The positive half. On its own it is weak -- see the negative half below.
    """
    perm = torch.randperm(x.shape[1])
    with torch.no_grad():
        a = head.forward_with_embedding(x)[1]
        b = head.forward_with_embedding(x[:, perm, :])[1]
    d = float((a - b).abs().max())
    return report.add("permuting tokens changes the output", d > 1e-4,
                      f"max|delta pred| = {d:.3e} under a random token permutation",
                      "if this were 0 the head would be treating the 13 modalities as an "
                      "unordered bag, and would still train and score plausibly")


def check_position_check_has_teeth(report, head, x):
    """Zeroing `token_embed` under mean pooling must make that permutation a no-op.

    The negative half, and the reason the positive half means anything. A mean-pooled
    permutation-equivariant encoder with no position parameter is exactly permutation
    INVARIANT, so this is the arithmetic the check above is meant to detect the absence of.
    Measured here rather than argued.
    """
    saved = head.token_embed.detach().clone()
    perm = torch.randperm(x.shape[1])
    try:
        with torch.no_grad():
            head.token_embed.zero_()
            a = head.forward_with_embedding(x)[1]
            b = head.forward_with_embedding(x[:, perm, :])[1]
        d = float((a - b).abs().max())
    finally:
        with torch.no_grad():
            head.token_embed.copy_(saved)
    # Not 0.0 exactly: attention softmax and the feedforward reassociate under a permutation,
    # so float noise survives. Three orders below the live signal is the discriminating fact.
    return report.add("the permutation check has teeth", d < 1e-5,
                      f"with token_embed zeroed, the same permutation moves the output by "
                      f"{d:.3e} -- so a live reading is the position parameter, not noise",
                      "pairs with the check above exactly as check_grad_flow is paired: a "
                      "tolerant test never seen to reject anything tests nothing")


def check_gradient_reaches_position(report, head, x, y):
    """`token_embed` must receive gradient, or it is frozen at its init and inert."""
    head.train()
    logits, pred, z = head.forward_with_embedding(x)
    loss, _ = combined_loss(logits, pred, y["bin"], y["norm"], y["w"],
                            w_cls=0.4418, w_pair=7.486, w_std=0.7911, huber_delta=1.0513,
                            embedding=z, w_vic=0.0)
    head.zero_grad(set_to_none=True)
    loss.backward()
    head.eval()

    g = head.token_embed.grad
    dead = [n for n, p in head.named_parameters() if p.grad is None]
    ok = g is not None and float(g.abs().max()) > 0 and not dead
    return report.add("gradient reaches every parameter", ok,
                      f"token_embed grad max|g| = "
                      f"{(float(g.abs().max()) if g is not None else float('nan')):.3e}; "
                      f"{len(dead)} parameters with grad None"
                      + (f" ({dead[:3]})" if dead else ""),
                      "a token_embed with no gradient would sit at its init forever, which "
                      "is the dead-position case in slow motion")


def check_pooling_modes(report, kw, x):
    """Both swept poolings run and disagree -- otherwise the axis is not an axis."""
    outs = {}
    for mode in TokenTransformerHead.POOLINGS:
        h = build(pooling=mode, **kw)
        with torch.no_grad():
            outs[mode] = h.forward_with_embedding(x)[1]
    d = float((outs["mean"] - outs["max"]).abs().max())
    return report.add("mean and max pooling both run, and differ", d > 1e-4,
                      f"max|delta pred| between poolings = {d:.3e} at identical weights",
                      "sweeping a knob that does nothing would waste half the trials")


def check_shape_contract(report, head, x):
    """A flat readout must be rejected, not silently broadcast."""
    outcomes = []
    for label, bad in (("flat [B, D]", x[:, 0, :]),
                       ("wrong token count", x[:, :5, :])):
        try:
            head.forward_with_embedding(bad)
            outcomes.append((label, False))
        except ValueError:
            outcomes.append((label, True))
    ok = all(o[1] for o in outcomes)
    return report.add("token-shape contract is enforced", ok,
                      "; ".join(f"{l}: {'rejected' if r else 'ACCEPTED'}"
                                for l, r in outcomes),
                      "the MLP arm's flattened readout and this head's sequence are both "
                      "'the cache'; only one is what this head means")


def check_loss_terms(report, head, x, y):
    """The five unscaled loss terms must be finite and non-degenerate."""
    with torch.no_grad():
        logits, pred, z = head.forward_with_embedding(x)
        _, terms = combined_loss(logits, pred, y["bin"], y["norm"], y["w"],
                                 w_cls=0.4418, w_pair=7.486, w_std=0.7911,
                                 huber_delta=1.0513, embedding=z, w_vic=1.0)
    bad = [k for k, v in terms.items() if not np.isfinite(v) or v == 0.0]
    return report.add("the five loss terms are finite and non-zero", not bad,
                      ", ".join(f"{k} {v:.4f}" for k, v in sorted(terms.items()))
                      + (f"; degenerate: {bad}" if bad else ""),
                      "a float64 slice or a collapsed token shows up here and nowhere else")


def main(argv=None):
    args = parse_args(argv)
    kw = dict(in_dim=args.dim, n_tokens=args.tokens, n_blocks=2, n_heads=4,
              dim_feedforward=256, embed_dim=128, hidden_dim=256)

    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(args.batch, args.tokens, args.dim)),
                     dtype=torch.float32)
    raw = np.clip(rng.gamma(1.2, 1.4, args.batch), 0, 7.0)
    y = {"norm": torch.tensor((raw - raw.mean()) / raw.std(), dtype=torch.float32),
         "bin": torch.tensor(binary_labels(raw, PPROP_EDGE), dtype=torch.float32),
         "w": torch.ones(args.batch)}

    head = build(**kw)
    report = Report()
    print(f"TokenTransformerHead: {sum(p.numel() for p in head.parameters()):,} params, "
          f"input {tuple(x.shape)}\n")

    check_position_is_used(report, head, x)
    check_position_check_has_teeth(report, head, x)
    check_gradient_reaches_position(report, head, x, y)
    check_pooling_modes(report, kw, x)
    check_shape_contract(report, head, x)
    check_loss_terms(report, head, x, y)

    lines = [
        "# Transformer-head verification",
        "",
        f"Generated by `src/verify_jepa_head.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "`nn.TransformerEncoder` is permutation-equivariant, so a head over Mol-JEPA's 13",
        "tokens can silently ignore which token is which. These are measurements that it",
        "does not.",
        "",
        "| Check | Result | Measurement |", "|---|---|---|",
    ]
    for r in report.rows:
        lines.append(f"| {r['name']} | {'PASS' if r['passed'] else '**FAIL**'} | "
                     f"{r['detail'].replace('|', chr(92) + '|')} |")
    lines += ["", "## Notes", ""]
    lines += [f"- **{r['name']}** — {r['note']}" for r in report.rows if r["note"]]
    lines += ["", f"**Overall: {'PASS' if report.ok else 'FAIL'}**", ""]
    Path(args.out).write_text("\n".join(lines) + "\n")

    print(f"\nwrote {args.out}")
    print("OVERALL:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
