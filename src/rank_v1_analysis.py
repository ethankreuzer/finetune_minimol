"""Apply the pre-registered decision rules of the S1 screen and write the results report.

    .venv/bin/python src/emb_readout.py --runs outputs/rank_v1 -o reports/rank_v1_runs.csv
    .venv/bin/python src/rank_v1_analysis.py -o reports/rank_v1_results.md

Reads the readout CSV, runs the randomized-complete-block analysis specified in
`reports/embedding_collapse_experiment.md`, and emits the report. Nothing here is chosen after
seeing the data -- the rules, the response variable, the margins and the escalation triggers
were all written into the plan before S1 launched, which is the whole point of pre-registering
them. This file is the mechanical statement of those rules, so what the plan promised and what
was computed cannot drift.

THE DESIGN. 8 cells x 3 seeds, fold 0, every cell on the same seeds. The error term for a
paired contrast is the *cell x seed interaction*, which a single-config seed scan cannot
estimate -- that measures sigma_seed instead, several times larger and the wrong quantity. The
additive two-way model leaves (8-1)(3-1) = 14 residual df, against 2 df from any isolated
pairwise comparison.

THE RESPONSE IS log(effective rank). Seed effects here are multiplicative (1.32 against 2.85
at one configuration), and the target is "several times better", not "+6 units".

WHY THE DAMAGE GUARD IS NOT `goal_metric`. The stated success bar was "no predictive cost", but
`goal_metric` cannot test it: across the six-run w_vic scan its entire spread (0.9598-1.0065)
rides on `ap_uniform` over ~620 positives in one fold, while `pearson_uniform` moves 0.002 in
total. A cost stated in `goal_metric` is unfalsifiable at that noise level, so the primary
guard is non-inferiority on `pearson_uniform` against an explicit margin and `goal_metric` is
reported as a two-sided bound, never as a claim.

The contrast standard errors use the RCBD pooled error (df = 14) rather than a three-point
paired t (df = 2), for both the response and the guards. That is the same argument the design
rests on -- pooling the interaction across cells is what buys the df -- and it is applied to
the guards for consistency rather than switching estimators mid-report. The per-seed
differences are printed beside every contrast so a reader can check the pooling did not carry
a conclusion.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

CONTROL = "A_base"
# The plan pre-registers 2.145 -- two-sided 95% at the 14 residual df an 8x3 block gives.
# The critical value is taken from the df the analysis ACTUALLY has, because a dropped cell
# (E_w10 is a deliberate ceiling probe, and a probe that fails is a result) lowers the df and
# keeping 2.145 there would be quietly anti-conservative. Any departure is flagged in the
# report rather than absorbed.
T_CRIT_PLAN = 2.145
PLAN_DF = 14
PEARSON_MARGIN = 0.005         # 2.5x the full observed spread of pearson_uniform
MSE_MARGIN = 0.010             # ~5% of the observed val MSE (0.185)


def md_table(df, index_label="", floats=3):
    """A markdown table without pulling in `tabulate`.

    `pandas.to_markdown` needs an optional dependency this environment does not have, and
    adding one for table borders would mean a lock change on a 0.5 MB/s link.
    """
    df = df.copy()
    cols = [index_label or (df.index.name or "")] + [str(c) for c in df.columns]
    body = []
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else f"{v:.{floats}f}")
            else:
                cells.append(str(v))
        body.append(cells)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("reports/rank_v1_runs.csv"))
    p.add_argument("-o", "--out", type=Path, default=Path("reports/rank_v1_results.md"))
    return p.parse_args(argv)


def rcbd(df, value, cell="cell", block="cfg_seed"):
    """Additive two-way (cell + block) fit; returns cell means and the pooled residual sigma.

    Balanced by construction, so the fit is arithmetic: fitted = grand + (cell mean - grand)
    + (block mean - grand). Unbalanced input is refused rather than silently mis-estimated --
    an incomplete block would make the "block" adjustment absorb a cell effect.
    """
    tab = df.pivot_table(index=cell, columns=block, values=value, aggfunc="mean")
    if tab.isna().any().any():
        missing = [(c, b) for c in tab.index for b in tab.columns if pd.isna(tab.loc[c, b])]
        raise ValueError(f"design is incomplete for {value}: missing {missing}")
    grand = tab.values.mean()
    cell_eff = tab.mean(axis=1) - grand
    block_eff = tab.mean(axis=0) - grand
    fitted = grand + cell_eff.values[:, None] + block_eff.values[None, :]
    resid = tab.values - fitted
    df_resid = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    sigma = float(np.sqrt((resid ** 2).sum() / df_resid))
    return tab, cell_eff + grand, sigma, df_resid


def contrasts(tab, means, sigma, n, t_crit):
    """Every cell against the control, with the pooled-error confidence half-width."""
    half = t_crit * sigma * np.sqrt(2.0 / n)
    rows = []
    for cell in tab.index:
        delta = float(means[cell] - means[CONTROL])
        per_seed = (tab.loc[cell] - tab.loc[CONTROL])
        rows.append({"cell": cell, "delta": delta, "half_width": half,
                     "lo": delta - half, "hi": delta + half,
                     "significant": abs(delta) > half,
                     "all_same_direction": bool((per_seed > 0).all() or (per_seed < 0).all()),
                     "per_seed": [round(float(v), 4) for v in per_seed.values]})
    return pd.DataFrame(rows).set_index("cell")


def fmt(df, cols, floats=3):
    return md_table(df[cols], index_label="cell", floats=floats)


def main(argv=None):
    args = parse_args(argv)
    d = pd.read_csv(args.csv)

    # Keep only cells that hold the full block. The RCBD fit refuses an incomplete design
    # anyway (an absent block would let the block adjustment absorb a cell effect), so
    # dropping them here makes the omission explicit and named instead of an exception.
    seeds_per_cell = d.groupby("cell")["cfg_seed"].nunique()
    n = int(seeds_per_cell.max())
    # A run that diverged writes NaN metrics rather than failing, and one NaN would poison
    # the pooled sigma for every cell. Those cells are excluded and reported by name --
    # E_w10 is a deliberate ceiling probe, so its failing IS a measurement.
    guarded = ["emb_effective_rank", "val/pearson_uniform", "val/mse", "val/goal_metric",
               "tanimoto_partial"]
    bad = d[d[guarded].isna().any(axis=1)]["cell"].unique().tolist()
    dropped = sorted(set(seeds_per_cell[seeds_per_cell < n].index) | set(bad))
    d = d[~d["cell"].isin(dropped)]
    if d.empty:
        raise SystemExit(f"no complete cells to analyse; dropped {dropped}")
    d["log_rank"] = np.log(d["emb_effective_rank"])

    L = ["# S1 — the screen: results", "",
         f"Generated by `src/rank_v1_analysis.py` from `{args.csv}`. "
         f"{len(d)} runs, {d['cell'].nunique()} cells x {n} seeds, fold 0.", "",
         "Every rule applied below was written into "
         "`reports/embedding_collapse_experiment.md` before these runs started.", ""]
    if dropped:
        L += [f"**Excluded from the analysis: {', '.join(dropped)}** — an incomplete seed "
              "block or a non-finite metric. An excluded cell is a result in its own right "
              "(`E_w10` is a deliberate ceiling probe); the raw rows are in the readout CSV.",
              ""]

    # ---- provenance, before any number is quoted -----------------------------------
    triples = d[["objective_version", "split_sha256", "input_sha256"]].drop_duplicates()
    L += ["## Provenance", "",
          f"`objective_version` = `{triples.iloc[0]['objective_version']}`, "
          f"`split_sha256` = `{str(triples.iloc[0]['split_sha256'])[:12]}…`, "
          f"{len(triples)} distinct triple(s)."
          + ("" if len(triples) == 1 else "  **RUNS ARE NOT COMPARABLE — see the table below.**"), ""]
    if len(triples) > 1:
        L += [md_table(triples.set_index("objective_version")), ""]

    # ---- the response --------------------------------------------------------------
    tab, means, sigma, dfree = rcbd(d, "log_rank")
    t_crit = float(student_t.ppf(0.975, dfree))
    con = contrasts(tab, means, sigma, n, t_crit)
    if dfree != PLAN_DF:
        L += [f"**Residual df is {dfree}, not the pre-registered {PLAN_DF}** (a cell was "
              f"excluded), so the two-sided 95% critical value is {t_crit:.3f} rather than "
              f"{T_CRIT_PLAN}. Everything else is applied as written.", ""]
    raw = d.pivot_table(index="cell", columns="cfg_seed", values="emb_effective_rank")
    L += ["## Effective rank", "",
          f"Response `log(emb_effective_rank)`; RCBD pooled sigma = **{sigma:.4f}** on "
          f"{dfree} df, so a contrast against `{CONTROL}` clears at "
          f"|Delta log| > {t_crit:.3f} * sigma * sqrt(2/{n}) = "
          f"**{t_crit * sigma * np.sqrt(2 / n):.4f}** "
          f"(a factor of {np.exp(t_crit * sigma * np.sqrt(2 / n)):.3f}x).", "",
          "Effective rank by cell and seed:", "", md_table(raw, index_label="cell"), "",
          "Contrasts against the control, on the log scale:", "",
          fmt(con.assign(ratio=np.exp(con["delta"])),
              ["delta", "lo", "hi", "ratio", "significant", "all_same_direction", "per_seed"],
              4), ""]

    # ---- damage guards -------------------------------------------------------------
    guards, guard_notes = {}, []
    for metric, margin, direction in (("val/pearson_uniform", PEARSON_MARGIN, "higher_better"),
                                      ("val/mse", MSE_MARGIN, "lower_better"),
                                      ("val/goal_metric", None, "higher_better")):
        t2, m2, s2, _ = rcbd(d, metric)
        c2 = contrasts(t2, m2, s2, n, t_crit)
        guards[metric] = c2
        L += [f"## Guard — `{metric}`", "",
              f"Pooled sigma = {s2:.5f}; contrast half-width = "
              f"{t_crit * s2 * np.sqrt(2 / n):.5f}."
              + ("" if margin is None else
                 f"  Non-inferiority margin **{margin}**, "
                 f"{'lower' if direction == 'higher_better' else 'upper'} bound must stay "
                 f"{'above -' if direction == 'higher_better' else 'below +'}{margin}."),
              "", fmt(c2, ["delta", "lo", "hi", "per_seed"], 5), ""]

    def guard_pass(cell):
        p = guards["val/pearson_uniform"].loc[cell]
        m = guards["val/mse"].loc[cell]
        return bool(p["lo"] > -PEARSON_MARGIN), bool(m["hi"] < MSE_MARGIN)

    # ---- the four clearing conditions ----------------------------------------------
    partial = d.pivot_table(index="cell", values="tanimoto_partial", aggfunc="mean")
    tan = d.pivot_table(index="cell", values="tanimoto_spearman", aggfunc="mean")
    knn = d.pivot_table(index="cell", values="knn20_jaccard", aggfunc="mean")
    scal = d.pivot_table(index="cell", values="scalarness", aggfunc="mean")
    base_partial = float(partial.loc[CONTROL, "tanimoto_partial"])

    rows = []
    for cell in tab.index:
        c = con.loc[cell]
        p_ok, m_ok = guard_pass(cell)
        rows.append({
            "cell": cell,
            "1_significant_up": bool(c["significant"] and c["delta"] > 0),
            "2_same_direction": bool(c["all_same_direction"] and c["delta"] > 0),
            "3_guards": p_ok and m_ok,
            "4_partial_up": bool(partial.loc[cell, "tanimoto_partial"] > base_partial),
            "tanimoto_partial": float(partial.loc[cell, "tanimoto_partial"]),
            "tanimoto_rho": float(tan.loc[cell, "tanimoto_spearman"]),
            "knn20": float(knn.loc[cell, "knn20_jaccard"]),
            "scalarness": float(scal.loc[cell, "scalarness"]),
        })
    verdict = pd.DataFrame(rows).set_index("cell")
    verdict["CLEARS"] = verdict[["1_significant_up", "2_same_direction",
                                 "3_guards", "4_partial_up"]].all(axis=1)
    L += ["## Pre-registered verdict", "",
          "A cell clears iff all four hold: (1) paired mean Delta log(rank) exceeds the "
          "pooled-error half-width, (2) all seeds move the same way, (3) both damage guards "
          "pass, (4) `tanimoto_partial` rises above the control — rank without structural "
          "information is a noise embedding and does not count.", "",
          md_table(verdict, index_label="cell", floats=4), ""]

    clearing = verdict[verdict["CLEARS"]]
    if len(clearing):
        best = clearing["tanimoto_partial"].idxmax()
        L += [f"**Best configuration: `{best}`** — highest `tanimoto_partial` "
              f"({clearing.loc[best, 'tanimoto_partial']:.4f}) among the "
              f"{len(clearing)} clearing cell(s), which is the pre-registered criterion. "
              f"Rank is the proxy; the readout is the thing.", ""]
    else:
        best = None
        L += ["**No cell clears all four conditions.** See the escalation check below.", ""]

    # ---- diagnosis: where the per-dimension stds sit -------------------------------
    diag = d.groupby("cell")[["emb_std_p5", "emb_std_p50", "emb_std_p95", "emb_std_max",
                              "n_dims_below_0.1", "n_dims_below_0.5", "n_dims_below_gamma",
                              "emb_trace", "cfg_vic_gamma"]].mean()
    L += ["## Diagnosis — per-dimension spread", "",
          "| observation | conclusion | action |", "|---|---|---|",
          "| stds pile up **below** gamma | force-limited | raise `w_vic` |",
          "| stds sit **at** gamma, rank still low | covariance binding | raise `--w-cov` (or gamma) |",
          "| a stable subset stuck near 0.06 at every dose | GELU-dead dims | **architecture**, not loss |",
          "", md_table(diag, index_label="cell"), ""]

    # ---- escalation triggers, evaluated mechanically -------------------------------
    strongest = means.idxmax()
    fires = []
    sat = diag.loc[strongest, "n_dims_below_gamma"] <= 4
    dead = diag.loc[strongest, "n_dims_below_0.1"] >= 4
    if sat and dead and float(np.exp(means[strongest])) < 12:
        fires.append("at the strongest dose the hinge is saturated, a subset of dims is still "
                     "dead, and rank plateaus below 12 — the LayerNorm scale budget plus "
                     "GELU-dead dimensions, which no loss coefficient reaches")
    up = verdict[verdict["1_significant_up"]]
    if len(up) and not up["4_partial_up"].any():
        fires.append("rank rises while `tanimoto_partial` stays flat — the term is "
                     "manufacturing orthogonal noise")
    if len(up) and not up["3_guards"].any():
        fires.append("every cell that raises rank fails a damage guard — the trade is real "
                     "and 15-25 is unreachable through the loss")
    L += ['## "The loss cannot fix this; the architecture must change"', "",
          ("**Not fired.**" if not fires else "**FIRED:**"), ""]
    L += [f"- {f}" for f in fires] or ["No trigger condition is met on this evidence."]
    L += ["", "Escalation order if it fires: (a) export pre-GELU or change the activation; "
          "(b) renegotiate `embed_dim` with the DKL project; (c) add an information-bearing "
          "auxiliary target (a 32 -> 2048 ECFP-bit decoder, with the readout moved to a "
          "fingerprint family the decoder never saw).", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"script": "src/rank_v1_analysis.py", "argv": sys.argv[1:], "csv": str(args.csv),
         "n_seeds": n, "sigma_log_rank": sigma, "resid_df": dfree,
         "best_cell": best, "escalation_fired": fires,
         "cell_mean_rank": {k: float(np.exp(v)) for k, v in means.items()},
         "verdict": verdict.reset_index().to_dict("records")}, indent=2))
    print("\n".join(L))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
