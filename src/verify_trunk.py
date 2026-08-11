"""Prove that `src/trunk.py` is the same model as MiniMol, and that it trains.

Two independent things can go wrong with the reimplementation in `src/trunk.py`, and they
have opposite fixes:

  1. It computes something *other* than MiniMol's embedding -- wrong readout tensor, a
     diverging featurization path, a skipped dtype cast. The model would train happily and
     produce a fine-looking loss curve while no longer being a pretrained model at all.
  2. It computes the right embedding but gradients do not reach the trunk, so "fine-tuning
     the whole trunk" is silently just training the head -- the exact frozen-embedding
     workflow this repo exists to avoid (NOTES.md §1).

Neither announces itself. So every claim below is a measurement written to
`verification.md`, per the repo convention that findings are settled by number rather than
by argument (CLAUDE.md "Conventions", NOTES §11).

Run `src/dump_reference_embeddings.py` first -- it produces the frozen reference in its own
process.

Usage:
    python src/verify_trunk.py
    python src/verify_trunk.py --ref data/reference/minimol_v1_ref64 -o verification.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphium.nn.encoders.mlp_encoder import MLPEncoder    # noqa: E402
from head import MLPHead                                   # noqa: E402
from model import MiniMolRegressor                         # noqa: E402
from trunk import FINGERPRINT_LAYER, FINGERPRINT_MODULE, MiniMolTrunk  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", type=Path, default=Path("data/reference/minimol_v1_ref64"),
                   help="stem written by dump_reference_embeddings.py")
    p.add_argument("-o", "--out", type=Path, default=Path("verification.md"))
    p.add_argument("--tol", type=float, default=None,
                   help="override the dtype-derived embedding tolerance")
    p.add_argument("--batch", type=int, default=16,
                   help="molecules per batch for the gradient checks")
    return p.parse_args(argv)


class Report:
    """Collects check results so a failure part-way still produces a full report."""

    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail, note=None):
        self.rows.append({"name": name, "passed": bool(passed),
                          "detail": detail, "note": note})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if note:
            print(f"       {note}")
        return passed

    @property
    def ok(self):
        return all(r["passed"] for r in self.rows)


def check_config_equivalence(report):
    """`OmegaConf.load` must give what `hydra.compose` gives.

    `trunk.py` deliberately bypasses hydra to escape its once-per-process singleton
    (NOTES §9). That is only safe if the resulting config is identical -- the config has no
    `defaults:` list, so it should be, but 'should be' is what this repo does not accept.
    """
    from omegaconf import OmegaConf
    from trunk import load_minimol_config, minimol_paths

    ours = load_minimol_config(accelerator="cpu")

    try:
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        paths = minimol_paths()
        cfg_dir = str(Path(paths["config"]).parent)
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(cfg_dir, version_base=None)
        theirs = OmegaConf.to_container(hydra.compose(config_name="config.yaml"),
                                        resolve=True)
        GlobalHydra.instance().clear()
    except Exception as exc:
        return report.add("config: OmegaConf == hydra", False,
                          f"hydra route failed: {type(exc).__name__}: {exc}")

    # Normalise the two fields trunk.py sets by hand after loading.
    theirs["accelerator"]["type"] = "cpu"
    theirs["architecture"]["mup_base_path"] = ours["architecture"]["mup_base_path"]

    same = ours == theirs
    detail = "identical" if same else "DIFFER"
    if not same:
        diff = [k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k)]
        detail += f" in top-level sections: {diff}"
    return report.add("config: OmegaConf == hydra", same, detail,
                      "justifies trunk.py skipping hydra.initialize")


def check_load_report(report, trunk):
    """Every trunk parameter must have come from the checkpoint (NOTES §9)."""
    r = trunk.load_report
    missing_trunk = r.get("missing_trunk_keys", [])
    passed = not missing_trunk
    detail = (f"{len(r['missing_keys'])} missing "
              f"({len(missing_trunk)} of them trunk), "
              f"{len(r['unexpected_keys'])} unexpected, "
              f"{r['checkpoint_keys']} keys in checkpoint")
    note = None
    if r["unexpected_keys"]:
        note = "unexpected (harmless, not built here): " + ", ".join(
            r["unexpected_keys"][:6]) + (" ..." if len(r["unexpected_keys"]) > 6 else "")
    return report.add("state_dict covers the trunk", passed, detail, note)


def check_embedding_matches(report, trunk, ref_emb, ref_meta, features, tol):
    """The headline check: same numbers as the stock frozen model.

    A tight tolerance is the point. "Close but not equal" would mean a different tensor
    was read out -- a wrong-but-plausible embedding -- not floating-point noise, so the
    threshold must not be relaxed to make this pass.
    """
    trunk.eval()   # match the reference: no dropout during extraction
    with torch.no_grad():
        batch = trunk.collate(features)
        emb = trunk(batch).double().cpu().numpy()
    trunk.train()

    if emb.shape != ref_emb.shape:
        return report.add("embedding == frozen MiniMol", False,
                          f"shape {emb.shape} != reference {ref_emb.shape}")

    diff = np.abs(emb - ref_emb)
    max_diff = float(diff.max())
    passed = max_diff < tol

    scale = float(np.abs(ref_emb).max())
    detail = (f"max|Δ| = {max_diff:.3e} over {ref_emb.shape[0]}x{ref_emb.shape[1]} "
              f"(tol {tol:.1e}, embedding scale {scale:.3f})")
    note = f"reference dtypes: {json.dumps(ref_meta.get('runtime_dtypes', {}))}"
    if not passed:
        # Distinguish the two failure shapes rather than reporting one number: a wrong
        # tensor is structurally different, float noise is uniformly tiny.
        note += (f" | mean|Δ| = {float(diff.mean()):.3e}, "
                 f"frac of entries over tol = {float((diff > tol).mean()):.3%} "
                 "-- a large structured error means the wrong readout tensor; a small "
                 "diffuse one means a dtype difference")
    return report.add("embedding == frozen MiniMol", passed, detail, note)


def check_readout_identity(report, trunk, features):
    """Is `_readout_cache[15]` literally the same tensor as `g["feat"]`?

    `trunk.forward` reads `g["feat"]` on the strength of `gnn.depth: 16` making layer 15
    the last one. If that is right, the cache entry and the graph feature are the same
    object, and `is` / `data_ptr()` proves it outright.

    Deliberately done inside a *single* forward pass. Comparing two pooled outputs from two
    separate passes would be both weaker and — with dropout live in this config — sensitive
    to the very nondeterminism that would make it meaningless.
    """
    trunk._network._enable_readout_cache([FINGERPRINT_MODULE])
    try:
        trunk.eval()
        with torch.no_grad():
            batch = trunk.collate(features[:8])
            _, node_feats = trunk(batch, return_node_features=True)
            cache = trunk._network._module_map[FINGERPRINT_MODULE]._readout_cache
            layers = sorted(cache.keys())
            cached = cache[FINGERPRINT_LAYER]
        trunk.train()

        same_object = cached is node_feats
        same_storage = cached.data_ptr() == node_feats.data_ptr()
        passed = same_object and same_storage and max(layers) == FINGERPRINT_LAYER

        detail = (f"cache holds layers {min(layers)}..{max(layers)}; "
                  f"layer {FINGERPRINT_LAYER} shape {tuple(cached.shape)}; "
                  f"`cache[{FINGERPRINT_LAYER}] is g['feat']` = {same_object}, "
                  f"same storage = {same_storage}")
        return report.add("readout cache is g['feat']", passed, detail,
                          f"confirms layer {FINGERPRINT_LAYER} is the last of "
                          f"{max(layers) + 1} GNN layers, so stopping before task_heads "
                          "cannot change the embedding")
    finally:
        trunk._network._disable_readout_cache()


def check_features_match(report, trunk, ref_features, smiles):
    """Our featurization must be the same as the reference run's.

    Only diagnostic value if the embedding check passed, but it is what separates "we read
    the wrong tensor" from "we fed the model different inputs" when it did not.
    """
    ours_list = trunk.featurize_raw(smiles)

    if len(ours_list) != len(ref_features):
        return report.add("featurization == reference", False,
                          f"{len(ours_list)} graphs vs {len(ref_features)}")

    worst, worst_key, mismatched = 0.0, None, []
    for a, b in zip(ours_list, ref_features):
        keys_a = {k for k in a.keys() if isinstance(a[k], torch.Tensor)}
        keys_b = {k for k in b.keys() if isinstance(b[k], torch.Tensor)}
        if keys_a != keys_b:
            mismatched.append(f"keys {sorted(keys_a ^ keys_b)}")
            continue
        for k in sorted(keys_a):
            ta, tb = a[k], b[k]
            if ta.shape != tb.shape:
                mismatched.append(f"{k}: shape {tuple(ta.shape)} vs {tuple(tb.shape)}")
                continue
            if ta.is_floating_point():
                d = float((ta.double() - tb.double()).abs().max()) if ta.numel() else 0.0
            else:
                d = 0.0 if torch.equal(ta, tb) else float("inf")
            if d > worst:
                worst, worst_key = d, k

    passed = not mismatched and worst == 0.0
    detail = (f"{len(ours_list)} graphs, max|Δ| = {worst:.3e}"
              + (f" on '{worst_key}'" if worst_key else ""))
    if mismatched:
        detail += f"; {len(mismatched)} structural mismatches, e.g. {mismatched[:3]}"
    return report.add("featurization == reference", passed, detail)


def orphaned_norm_params(trunk):
    """Trunk parameters that *no* forward pass can reach, derived from the module tree.

    graphium's `MLPEncoder.__init__` builds `self.first_normalization` (via `BaseEncoder`,
    base_encoder.py:49) and then hands that module to `MLP(...)`. But `MLP.__init__` does
    not store what it was given -- it calls `get_norm()` again (global_architectures.py:174),
    producing a *second* LayerNorm. `MLPEncoder.forward` then only ever calls
    `self.pe_encoder`, so the encoder's own copy is never executed. The normalization the
    author intended still happens; it is performed by the inner duplicate.

    Measured on minimol 1.3.4 / graphium 2.4.7 (see CLAUDE.md "The rw_pos dead norm"):
      - `outer is inner` is False -- two distinct LayerNorm((16,)) objects
      - the outer sits at LayerNorm init (weight all 1, bias all 0) while the inner holds
        loaded values, and `missing_keys` is only the two `task_heads` entries -- so the
        checkpoint *contains* the outer, saved still at init. It was dead during MiniMol's
        own pretraining too.
      - `.grad` comes back `None` rather than zero, so no optimizer can move it, weight
        decay included.

    Derived structurally rather than hardcoded by name, so that if a future graphium stops
    duplicating the norm this set empties on its own and the check below tightens
    automatically, instead of quietly excusing a tensor that has come back to life.
    """
    orphans = set()
    for name, module in trunk.named_modules():
        norm = getattr(module, "first_normalization", None)
        if isinstance(module, MLPEncoder) and norm is not None:
            orphans.update(f"{name}.first_normalization.{p}"
                           for p, _ in norm.named_parameters())
    return orphans


def partition_grad_flow(dead, expected_dead):
    """The two-sided verdict, as pure set logic over parameter *names*.

    Split out from `check_grad_flow` so it can be exercised directly with synthetic sets --
    see `check_grad_flow_has_teeth`. Driving it through a real backward pass cannot isolate
    the two branches, because perturbing the exemption to test one branch tends to trip the
    other at the same time.

    Returns `(unexpected_dead, unexpectedly_live, passed)`:
      unexpected_dead    -- a tensor that should train and did not. The regression this
                            check exists to catch.
      unexpectedly_live  -- a tensor written off as unreachable that now has a gradient.
                            Not a training bug, but the exemption is stale and must be
                            re-derived rather than trusted.
    """
    unexpected_dead = sorted(dead - expected_dead)
    unexpectedly_live = sorted(expected_dead - dead)
    return unexpected_dead, unexpectedly_live, not unexpected_dead and not unexpectedly_live


def check_grad_flow_has_teeth(report):
    """Prove the gradient verdict can fail, in each direction independently.

    A check that has only ever returned PASS is indistinguishable from a check that cannot
    fail. `check_grad_flow` guards the central claim of this repo -- that gradients reach
    the whole trunk -- and it carries an exemption, which is exactly the shape of a check
    that quietly stops working. So assert the verdict's behaviour on known inputs.

    Each case isolates one branch: the exemption is held fixed at `{"orphan"}` and only the
    observed-dead set varies.
    """
    orphan, live = {"orphan"}, "gnn.layers.0.linear.weight"
    cases = [
        # (label, dead set, expect_pass, branch that must fire)
        ("orphan dead, nothing else", orphan, True, None),
        ("a real tensor also dead", orphan | {live}, False, "unexpected_dead"),
        ("orphan now has gradient", set(), False, "unexpectedly_live"),
    ]
    failures = []
    for label, dead, expect_pass, branch in cases:
        u_dead, u_live, passed = partition_grad_flow(dead, orphan)
        fired = {"unexpected_dead": bool(u_dead), "unexpectedly_live": bool(u_live)}
        if passed != expect_pass:
            failures.append(f"{label}: passed={passed}, expected {expect_pass}")
        # The branch under test must fire, and the *other* must stay silent -- otherwise a
        # case can pass for the wrong reason.
        for name, did in fired.items():
            if did != (name == branch):
                failures.append(f"{label}: {name} fired={did}, expected {name == branch}")

    detail = (f"{len(cases)} cases: verdict PASSes when only the exemption is dead, and "
              "FAILs via exactly one branch for a genuinely dead tensor and for a stale "
              "exemption")
    return report.add("grad-flow verdict can fail", not failures, detail,
                      "; ".join(failures) if failures else None)


def check_grad_flow(report, trunk, features, batch_size):
    """NOTES §8, over the trunk only.

    The denominator matters. Run over the full `PredictorModule` this would count the five
    pretraining heads and `graph_output_nn`, report something like 60/180, and read as a
    broken chain when it is correct by construction. `trunk.parameters()` excludes them, so
    every *reachable* trunk tensor having a gradient is the honest criterion.

    The assertion is deliberately two-sided. Simply skipping the known-orphaned tensors
    would be a plain loosening -- it would stay silent if some other tensor went dead later,
    which is the exact failure this check exists to catch. So we require the dead set to be
    *equal* to the predicted one: nothing else dead, and nothing predicted-dead alive.
    """
    trunk.train()
    trunk.zero_grad(set_to_none=True)

    batch = trunk.collate(features[:batch_size])
    emb = trunk(batch, check_grad=True)

    if emb.grad_fn is None:
        return report.add("embedding carries grad_fn", False, "grad_fn is None")
    report.add("embedding carries grad_fn", True, f"grad_fn = {type(emb.grad_fn).__name__}")

    emb.sum().backward()

    named = [(n, p) for n, p in trunk.named_parameters() if p.requires_grad]
    dead = {n for n, p in named if p.grad is None or float(p.grad.abs().sum()) == 0.0}
    expected_dead = orphaned_norm_params(trunk) & {n for n, _ in named}
    unexpected_dead, unexpectedly_live, passed = partition_grad_flow(dead, expected_dead)

    reachable = [(n, p) for n, p in named if n not in expected_dead]
    got = len(reachable) - len(unexpected_dead)
    detail = (f"{got} / {len(reachable)} reachable trunk tensors received nonzero gradient "
              f"({sum(p.numel() for _, p in reachable):,} params)")
    if expected_dead:
        n_orphan = sum(p.numel() for n, p in named if n in expected_dead)
        detail += (f"; {len(expected_dead)} orphaned graphium MLPEncoder norm tensors "
                   f"({n_orphan:,} params) excluded, exactly as predicted")

    note = None
    if unexpected_dead:
        by_module = {}
        for n in unexpected_dead:
            by_module[n.split(".")[0]] = by_module.get(n.split(".")[0], 0) + 1
        note = f"no gradient: {by_module}; first few: {unexpected_dead[:8]}"
    if unexpectedly_live:
        note = ((note + " | ") if note else "") + (
            "predicted-unreachable tensors now have gradient, so orphaned_norm_params() is "
            f"stale: {unexpectedly_live[:8]}")
    return report.add("gradients reach the whole trunk", passed, detail, note)


def check_excluded_unreachable(report, trunk):
    """Paired check: dead modules have no gradient *while* the trunk does.

    `.grad is None` on its own proves nothing -- it is equally what a parameter looks like
    when no backward pass has run at all. Asserting it in the same pass as a live `gnn`
    gradient is what makes it evidence that `task_heads` is genuinely unreachable.
    """
    excluded = trunk.excluded_modules()
    ex_params = [(f"{mod}.{n}", p) for mod, m in excluded.items()
                 for n, p in m.named_parameters()]
    with_grad = [n for n, p in ex_params
                 if p.grad is not None and float(p.grad.abs().sum()) != 0.0]

    gnn_live = [n for n, p in trunk.gnn.named_parameters()
                if p.grad is not None and float(p.grad.abs().sum()) != 0.0]

    in_parameters = {id(p) for p in trunk.parameters()}
    leaked = [n for n, p in ex_params if id(p) in in_parameters]

    passed = not with_grad and not leaked and bool(gnn_live)
    detail = (f"{len(ex_params)} excluded tensors "
              f"({sum(p.numel() for _, p in ex_params):,} params): "
              f"{len(with_grad)} with gradient, {len(leaked)} leaked into "
              f"trunk.parameters(); meanwhile {len(gnn_live)} gnn tensors have gradient")
    note = None
    if with_grad:
        note = f"unexpectedly got gradient: {with_grad[:5]}"
    elif not gnn_live:
        note = "no gnn gradient in the same pass, so this check proves nothing"
    return report.add("excluded modules unreachable", passed, detail, note)


def check_head(report):
    """Shape and parameter count of the requested 512 -> 1200 -> 32 -> 1 head."""
    head = MLPHead()
    x = torch.randn(8, 512)
    out = head(x)

    expected = (512 * 1200 + 1200) + (1200 * 32 + 32) + (32 * 1 + 1)
    actual = sum(p.numel() for p in head.parameters())

    linear = MLPHead(hidden_dims=())
    linear_ok = sum(p.numel() for p in linear.parameters()) == 512 * 1 + 1

    passed = out.shape == (8, 1) and actual == expected and linear_ok
    detail = (f"512->1200->32->1 gives {out.shape} and {actual:,} params "
              f"(expected {expected:,}); hidden_dims=() gives a {512 * 1 + 1}-param "
              f"linear probe: {linear_ok}")
    return report.add("head shape and size", passed, detail)


def check_optimizer_step(report, trunk, features, batch_size):
    """An end-to-end step must actually move pretrained trunk weights.

    Everything above could pass while the optimizer is handed the wrong parameter groups.
    This is the check that the composed `MiniMolRegressor` trains the trunk, which is the
    whole point of the repo.
    """
    model = MiniMolRegressor(trunk, MLPHead())
    counts = model.n_parameters()

    groups = model.param_groups(trunk_lr=1e-4, head_lr=1e-3)
    names = [g["name"] for g in groups]
    lrs = {g["name"]: g["lr"] for g in groups}
    opt = torch.optim.AdamW(groups)

    probe_name, probe = next((n, p) for n, p in model.trunk.gnn.named_parameters()
                             if p.dim() > 1)
    before = probe.detach().clone()

    model.train()
    opt.zero_grad(set_to_none=True)
    batch = model.trunk.collate(features[:batch_size])
    target = torch.randn(batch_size, device=model.trunk.device)
    pred = model(batch)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    opt.step()

    delta = float((probe.detach() - before).abs().max())
    passed = (pred.shape == (batch_size,) and delta > 0
              and names == ["trunk", "head"] and lrs["trunk"] != lrs["head"])
    detail = (f"pred {tuple(pred.shape)}, loss {float(loss):.4f}, "
              f"max|Δw| on {probe_name} = {delta:.3e}; "
              f"groups {names} at lrs {lrs}")
    note = (f"trunk {counts['trunk']:,} + head {counts['head']:,} "
            f"= {counts['total']:,} trainable params")
    return report.add("optimizer step updates the trunk", passed, detail, note)


def write_report(path, report, context):
    lines = [
        "# Trunk verification",
        "",
        f"Generated by `src/verify_trunk.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "Checks that `src/trunk.py` computes the same embedding as the stock frozen",
        "`Minimol` and that gradients reach every trunk parameter. Numbers, not claims --",
        "see NOTES.md §§4, 6.3, 8, 9 for why each of these is worth measuring.",
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

    ref_npy = args.ref.with_suffix(".npy")
    ref_json = args.ref.with_suffix(".json")
    ref_pt = args.ref.with_suffix(".features.pt")
    for p in (ref_npy, ref_json, ref_pt):
        if not p.exists():
            raise SystemExit(
                f"missing {p}. Run: python src/dump_reference_embeddings.py"
            )

    ref_emb = np.load(ref_npy)
    ref_meta = json.loads(ref_json.read_text())
    ref_features = torch.load(ref_pt, weights_only=False)
    smiles = ref_meta["smiles"]

    # Tolerance from the dtype the reference actually ran at, not from habit. fp32 end to
    # end justifies 1e-5; anything narrower would make a correct implementation look wrong.
    dtypes = set(ref_meta.get("runtime_dtypes", {}).get("param_dtypes", []))
    dtypes.add(ref_meta.get("runtime_dtypes", {}).get("embedding_dtype", ""))
    reduced = any(d in ("torch.float16", "torch.bfloat16") for d in dtypes)
    tol = args.tol if args.tol is not None else (1e-2 if reduced else 1e-5)

    report = Report()
    print(f"reference: {ref_emb.shape} from {ref_npy}")
    print(f"tolerance: {tol:.1e} (reduced precision detected: {reduced})\n")

    check_config_equivalence(report)

    started = time.time()
    trunk = MiniMolTrunk()
    build_seconds = time.time() - started
    print(f"built MiniMolTrunk in {build_seconds:.1f}s "
          f"({trunk.n_parameters():,} trainable params)\n")

    check_load_report(report, trunk)
    check_embedding_matches(report, trunk, ref_emb, ref_meta, ref_features, tol)
    check_features_match(report, trunk, ref_features, smiles)
    check_readout_identity(report, trunk, ref_features)

    batch_size = min(args.batch, len(ref_features))
    check_grad_flow_has_teeth(report)
    check_grad_flow(report, trunk, ref_features, batch_size)
    check_excluded_unreachable(report, trunk)
    check_head(report)
    check_optimizer_step(report, trunk, ref_features, batch_size)

    import graphium
    context = {
        "trunk trainable params": f"{trunk.n_parameters():,}",
        "trunk build time": f"{build_seconds:.1f}s",
        "reference": str(ref_npy),
        "reference molecules": ref_emb.shape[0],
        "embedding tolerance": f"{tol:.1e}",
        "torch": torch.__version__,
        "graphium": getattr(graphium, "__version__", "unknown"),
        "device": "cpu",
    }
    write_report(args.out, report, context)
    print(f"\nwrote {args.out}")
    print("OVERALL:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
