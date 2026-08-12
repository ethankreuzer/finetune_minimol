"""The selection objective, and the version stamp that keeps runs comparable.

    AP*        = mean over OBJECTIVE_SPEC["ap_star"]
    Pearson*   = mean over OBJECTIVE_SPEC["pearson_star"]
    MAE_skill* = mean over OBJECTIVE_SPEC["mae_skill_star"]

    goal_metric = AP*  +  1/2 * (Pearson* + MAE_skill*)

Classification and regression each contribute one unit, and within each the flavours are
averaged so a model has to do well under both rather than trading one off. Inherited from
pProp_MLP, whose objective this reproduces in structure.

WHY THE VERSION STRING IS DERIVED, NOT WRITTEN
----------------------------------------------
pProp_MLP's `runs/` accumulated three incompatible revisions of `goal_metric` under one
metric name, and its CLAUDE.md carries the resulting warning: *"`goal_metric` values are NOT
comparable across the objective's revisions -- compare only runs whose config logs the same
objective."* Enforcing that by hand means remembering to bump a string every time the spec
changes, which is exactly the kind of discipline that fails silently.

So `OBJECTIVE_VERSION` is a hash of the spec itself. Edit any term below and the version
changes automatically; runs stamped with different versions are, by construction, not
comparable, and `src/collect_runs.py`-style consumers can filter on equality without
trusting anyone's bookkeeping.

WHICH FLAVOURS GO INTO EACH TERM
--------------------------------
- `AP*` is `ap_uniform` alone. `ap_balanced` is deliberately absent: forcing a 50/50 base
  rate makes AP a statement about the weighting rather than about the model. That leaves one
  interpretable reading, so the term is a single metric rather than an average — kept in the
  spec's tuple form so a second reading can be added without changing the machinery.
- `Pearson*` and `MAE_skill*` average `uniform` and `balanced`, mirroring pProp_MLP's
  unweighted-plus-class-balanced pair. `balanced` is also what the loss optimises, so this
  keeps the objective aligned with training while the unweighted half stops it from
  ignoring the bulk.

The `ipw` flavour was removed on 2026-08-12 (it previously sat beside `ap_uniform` in
`AP*`). `ipw` is the per-bin subsampling rate `subset.py` applied when it built the 331k
file — a record of how the data was made, not a modelling choice — so it stays a CSV column
and stays in the split diagnostics, but no longer weights any loss or any reported metric.
Removing it re-stamped `OBJECTIVE_VERSION`, which cost nothing because no run had yet been
scored under the old spec.
"""

import hashlib
import json

# Human-readable tag. The machine-readable half is appended automatically.
OBJECTIVE_TAG = "v1-binary3.5"

# Each starred term is the unweighted mean of these metric keys, as produced by
# `metrics.flavoured_metrics`. Editing this dict re-stamps OBJECTIVE_VERSION.
OBJECTIVE_SPEC = {
    "ap_star": ("ap_uniform",),
    "pearson_star": ("pearson_uniform", "pearson_balanced"),
    "mae_skill_star": ("mae_skill_uniform", "mae_skill_balanced"),
}

GOAL_FORMULA = "ap_star + 0.5 * (pearson_star + mae_skill_star)"

# Every flavour any term needs -- so train.py can build exactly these weight vectors and
# no others, rather than hardcoding a list that drifts from the spec.
REQUIRED_FLAVOURS = tuple(sorted({
    key.rsplit("_", 1)[1] for keys in OBJECTIVE_SPEC.values() for key in keys
}))


def _spec_hash():
    payload = json.dumps({"spec": {k: list(v) for k, v in sorted(OBJECTIVE_SPEC.items())},
                          "formula": GOAL_FORMULA},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


OBJECTIVE_VERSION = f"{OBJECTIVE_TAG}-{_spec_hash()}"


def compute_objective(metrics):
    """`flavoured_metrics(...) -> {ap_star, pearson_star, mae_skill_star, goal_*}`.

    Missing keys raise rather than defaulting to nan: a typo in `OBJECTIVE_SPEC` would
    otherwise produce a plausible-looking objective computed from fewer terms, which is the
    same class of silent-wrong-number failure the version stamp exists to prevent.

    A nan *value* does propagate to `goal_metric`, deliberately. Pearson is nan when
    predictions are constant -- a real early-epoch possibility here -- and a nan objective
    is the honest report of "this model has no rank signal". Sweep drivers must decide what
    to do with that (prune, or treat as worst); they must not silently coerce it to 0, which
    would rank a degenerate model above a genuinely negative one.
    """
    missing = [k for keys in OBJECTIVE_SPEC.values() for k in keys if k not in metrics]
    if missing:
        raise KeyError(f"objective needs metrics {missing}, which flavoured_metrics did not "
                       f"produce; check that --weights built every flavour in "
                       f"REQUIRED_FLAVOURS={REQUIRED_FLAVOURS}")

    stars = {name: sum(metrics[k] for k in keys) / len(keys)
             for name, keys in OBJECTIVE_SPEC.items()}

    # The two-task decomposition, logged so the objective's balance is visible directly
    # rather than inferred: these sum to goal_metric by construction.
    goal_cls = stars["ap_star"]
    goal_reg = 0.5 * (stars["pearson_star"] + stars["mae_skill_star"])

    return {**stars,
            "goal_term_cls": goal_cls,
            "goal_term_reg": goal_reg,
            "goal_metric": goal_cls + goal_reg}
