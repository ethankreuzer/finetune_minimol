"""Apply the pre-registered decision rules of the embedding-collapse experiment and render the
single consolidated results document.

    .venv/bin/python src/emb_readout.py --runs outputs/rank_v1 outputs/rank_v2 \
        -o reports/collapse_runs.csv
    .venv/bin/python src/collapse_analysis.py --report -o reports/embedding_collapse_results.md

Reads the readout CSV, runs the randomized-complete-block analysis specified in
`reports/embedding_collapse_experiment.md`, and emits the report. Nothing here is chosen after
seeing the data -- the rules, the response variable, the margins and the escalation triggers
were all written into the plan before the runs they judge, which is the whole point of
pre-registering them. This file is the mechanical statement of those rules, so what the plan
promised and what was computed cannot drift.

WHY ONE SCRIPT FOR THREE SECTIONS. S1, S2a and S2b are the same analysis on different subsets
of one CSV. A second implementation per stage could drift from the first, and the S1 numbers
are the baseline every later claim is stated against -- so the S1 section is regenerated from
the same code path as the new ones, and the regression test in `--selftest` pins it to the
values the pre-registered S1 report published (sigma = 0.3795, threshold = 0.6646, and an
identical CLEARS column).

THE DESIGN. Every section is a complete cells x seeds block, fold 0, every cell on the same
seeds. The error term for a paired contrast is the *cell x seed interaction*, which a single
config seed scan cannot estimate -- that measures sigma_seed instead, several times larger and
the wrong quantity. The additive two-way model leaves (cells-1)(seeds-1) residual df: 14 for
S1's 8x3, and 8 for both of S2's blocks.

THE RESPONSE IS log(effective rank). Seed effects here are multiplicative (1.32 against 2.85
at one configuration), and the target is "several times better", not "+6 units".

WHY THE DAMAGE GUARD IS NOT `goal_metric`. The stated success bar was "no predictive cost", but
`goal_metric` cannot test it: across the six-run w_vic scan its entire spread (0.9598-1.0065)
rides on `ap_uniform` over ~620 positives in one fold, while `pearson_uniform` moves 0.002 in
total. A cost stated in `goal_metric` is unfalsifiable at that noise level, so the primary
guard is non-inferiority on `pearson_uniform` against an explicit margin and `goal_metric` is
reported as a two-sided bound, never as a claim.

TWO RULE SETS, BOTH PRE-REGISTERED, AND THE SECOND SUPERSEDES THE FIRST.

  `s1`  clears on four conditions and ranks the survivors on `tanimoto_partial`.
  `s2`  adds a fifth condition -- the worst seed's effective rank must reach
        WORST_CORNER_MIN -- demotes `tanimoto_partial` to a gate, ranks on `knn20_jaccard`,
        and only accepts that ranking when the top two are separated by more than the
        pooled-error half-width. Otherwise it takes the terminal branch: pin the more robust
        survivor and escalate the global-vs-local question to the DKL project.

The `s1` set is kept executable rather than deleted, because the S1 report published under it
is the pre-registration artifact and must stay reproducible. Its defect -- ranking on a mean
whose 0.018 gap needs ~36 seeds to resolve -- is recorded in the plan, not patched here.

The contrast standard errors use the RCBD pooled error rather than a paired t on the seeds
alone. That is the same argument the design rests on -- pooling the interaction across cells is
what buys the df -- and it is applied to the guards for consistency rather than switching
estimators mid-report. The per-seed differences are printed beside every contrast so a reader
can check the pooling did not carry a conclusion.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

CONTROL = "A_base"
# S1 pre-registers 2.145 -- two-sided 95% at the 14 residual df an 8x3 block gives. The
# critical value is taken from the df the analysis ACTUALLY has, because a dropped cell (or a
# smaller section) lowers the df and keeping 2.145 there would be quietly anti-conservative.
# Any departure is flagged in the report rather than absorbed.
T_CRIT_PLAN = 2.145
PLAN_DF = 14
PEARSON_MARGIN = 0.005         # 2.5x the full observed spread of pearson_uniform
MSE_MARGIN = 0.010             # ~5% of the observed val MSE (0.185)
# S2-R2. Not invented for S2: 12 is the number S4 already uses as the point below which "the
# loss cannot fix this; the architecture must change" fires. A configuration whose worst seed
# lands in territory S4 calls unfixable cannot be the configuration S4 pins.
WORST_CORNER_MIN = 12.0

# The three sections of the consolidated report. Each is a complete block on its own.
SECTIONS = {
    "s1": dict(
        cells=["A_base", "B_w0.3", "C_w1", "D_w3", "E_w10", "F_w3_g1", "G_w3_drop", "H_drop"],
        seeds=[0, 1, 2], rules="s1",
        heading="S1 — the screen",
        intro="Eight cells x three seeds, fold 0. Every rule applied below was written into "
              "`reports/embedding_collapse_experiment.md` before these runs started."),
    "s2a": dict(
        cells=["A_base", "C_w1", "D_w3"], seeds=[0, 1, 2, 3, 4], rules="s2",
        heading="S2a — replication to n=5",
        intro="The control and the two cells S1 could not choose between, at five seeds. "
              "Seeds 0-2 are S1's runs, reused; seeds 3-4 are new. Judged under the **S2 rule "
              "set**, which adds the worst-corner condition S1's rule lacked."),
    "s2b": dict(
        cells=["A_base", "D_w3", "I_w3_cov4", "J_w3_cov16", "K_wd0.1"], seeds=[0, 1, 2],
        rules="s2",
        heading="S2b — covariance dose and the weight-decay null",
        intro="`I`/`J` raise `--w-cov` at fixed `w_vic=3` and gamma=0.5, isolating covariance "
              "pressure from the scale inflation that confounded `F_w3_g1`; `emb_trace` is the "
              "pre-registered column that separates them. `K_wd0.1` is the null probe that "
              "makes P6.3 empirical. `A_base` and `D_w3` are S1's runs, reused."),
    "s27": dict(
        cells=["A_base", "D_w3", "L_nonorm", "M_nonorm_w3"], seeds=[0, 1, 2], rules="s2",
        heading="S2.7 — removing the export-point LayerNorm",
        intro="P6.3 named LayerNorm as the likely collapse mechanism; `--bottleneck-norm none` "
              "removes it. `L_nonorm` is the test (no norm, no `vic`), `M_nonorm_w3` the "
              "diagnostic (no norm, `w_vic=3`). `A_base` and `D_w3` are S1's runs, reused. "
              "**Read `emb_trace` beside every rank** — without the norm the bottleneck's scale "
              "is unconstrained, and S2 established that rank tracks scale."),
}


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


# ---------------------------------------------------------------- one section of the report

def analyse(d, heading, intro, rules="s1", level="##"):
    """Run one complete block through the pre-registered rules; return (lines, summary)."""
    sub = level + "#"
    seeds_per_cell = d.groupby("cell")["cfg_seed"].nunique()
    n = int(seeds_per_cell.max())
    # A run that diverged writes NaN metrics rather than failing, and one NaN would poison the
    # pooled sigma for every cell. Those cells are excluded and reported by name -- E_w10 is a
    # deliberate ceiling probe, so its failing IS a measurement.
    guarded = ["emb_effective_rank", "val/pearson_uniform", "val/mse", "val/goal_metric",
               "tanimoto_partial", "knn20_jaccard"]
    bad = d[d[guarded].isna().any(axis=1)]["cell"].unique().tolist()
    dropped = sorted(set(seeds_per_cell[seeds_per_cell < n].index) | set(bad))
    d = d[~d["cell"].isin(dropped)]
    if d.empty:
        raise SystemExit(f"no complete cells to analyse; dropped {dropped}")
    d = d.copy()
    d["log_rank"] = np.log(d["emb_effective_rank"])

    L = [f"{level} {heading}", "",
         f"{intro}  {len(d)} runs, {d['cell'].nunique()} cells x {n} seeds.", ""]
    if dropped:
        L += [f"**Excluded: {', '.join(dropped)}** — an incomplete seed block or a non-finite "
              "metric. An excluded cell is a result in its own right; the raw rows are in the "
              "readout CSV.", ""]

    triples = d[["objective_version", "split_sha256", "input_sha256"]].drop_duplicates()
    L += [f"Provenance: `objective_version` = `{triples.iloc[0]['objective_version']}`, "
          f"`split_sha256` = `{str(triples.iloc[0]['split_sha256'])[:12]}…`, "
          f"{len(triples)} distinct triple(s)."
          + ("" if len(triples) == 1 else
             "  **RUNS ARE NOT COMPARABLE — see the table below.**"), ""]
    if len(triples) > 1:
        L += [md_table(triples.set_index("objective_version")), ""]

    # ---- the response --------------------------------------------------------------
    tab, means, sigma, dfree = rcbd(d, "log_rank")
    t_crit = float(student_t.ppf(0.975, dfree))
    con = contrasts(tab, means, sigma, n, t_crit)
    raw = d.pivot_table(index="cell", columns="cfg_seed", values="emb_effective_rank")
    worst = raw.min(axis=1)
    half_rank = t_crit * sigma * np.sqrt(2 / n)
    if dfree != PLAN_DF:
        L += [f"Residual df is {dfree} for this block (S1's pre-registered 14 assumes its 8x3 "
              f"shape), so the two-sided 95% critical value is {t_crit:.3f} rather than "
              f"{T_CRIT_PLAN}. Everything else is applied as written.", ""]
    L += [f"{sub} Effective rank", "",
          f"Response `log(emb_effective_rank)`; RCBD pooled sigma = **{sigma:.4f}** on "
          f"{dfree} df, so a contrast against `{CONTROL}` clears at "
          f"|Delta log| > {t_crit:.3f} * sigma * sqrt(2/{n}) = **{half_rank:.4f}** "
          f"(a factor of {np.exp(half_rank):.3f}x).", "",
          "Effective rank by cell and seed (`worst` is the S2-R2 column):", "",
          md_table(raw.assign(worst=worst), index_label="cell"), "",
          "Contrasts against the control, on the log scale:", "",
          fmt(con.assign(ratio=np.exp(con["delta"])),
              ["delta", "lo", "hi", "ratio", "significant", "all_same_direction", "per_seed"],
              4), ""]

    # ---- damage guards -------------------------------------------------------------
    # Non-inferiority is a ONE-sided question ("is it not worse by more than the margin"), so
    # the guards use a one-sided 95% bound. The response above stays two-sided, because "did
    # rank move" is genuinely two-sided.
    t_one = float(student_t.ppf(0.95, dfree))
    guards, guard_lines = {}, []
    for metric, margin, direction in (("val/pearson_uniform", PEARSON_MARGIN, "higher_better"),
                                      ("val/mse", MSE_MARGIN, "lower_better"),
                                      ("val/goal_metric", None, "higher_better")):
        t2, m2, s2, _ = rcbd(d, metric)
        c2 = contrasts(t2, m2, s2, n, t_one)
        guards[metric] = c2
        half = t_one * s2 * np.sqrt(2 / n)
        # A margin narrower than the half-width does not test "within the margin" -- it tests
        # "indistinguishable from zero", which is a stricter question than the plan asked.
        # Said out loud rather than left for a reader to derive from sigma.
        resolution = "" if margin is None else (
            f"  The one-sided half-width ({half:.5f}) is "
            + (f"**wider than the margin ({margin})**, so this guard is effectively testing "
               "'no detectable change at all', not 'within the margin' — a real cost smaller "
               "than the margin would still fail it."
               if half > margin else
               f"narrower than the margin ({margin}), so the guard tests the margin as "
               "intended."))
        guard_lines += [f"{sub} Guard — `{metric}`", "",
                        f"Pooled sigma = {s2:.5f}; one-sided (95%) contrast half-width = "
                        f"{half:.5f}."
                        + ("" if margin is None else
                           f"  Non-inferiority margin **{margin}**: the "
                           f"{'lower' if direction == 'higher_better' else 'upper'} bound must "
                           f"stay {'above -' if direction == 'higher_better' else 'below +'}"
                           f"{margin}.")
                        + resolution,
                        "", fmt(c2, ["delta", "lo", "hi", "per_seed"], 5), ""]
    L += guard_lines

    def guard_pass(cell):
        p = guards["val/pearson_uniform"].loc[cell]
        m = guards["val/mse"].loc[cell]
        return bool(p["lo"] > -PEARSON_MARGIN), bool(m["hi"] < MSE_MARGIN)

    # ---- the clearing conditions ---------------------------------------------------
    agg = d.groupby("cell")[["tanimoto_partial", "tanimoto_spearman", "knn20_jaccard",
                             "scalarness"]].mean()
    base_partial = float(agg.loc[CONTROL, "tanimoto_partial"])

    rows = []
    for cell in tab.index:
        c = con.loc[cell]
        p_ok, m_ok = guard_pass(cell)
        row = {"cell": cell,
               "1_significant_up": bool(c["significant"] and c["delta"] > 0),
               "2_same_direction": bool(c["all_same_direction"] and c["delta"] > 0),
               "3_guards": p_ok and m_ok,
               "4_partial_up": bool(agg.loc[cell, "tanimoto_partial"] > base_partial)}
        if rules == "s2":
            row["5_worst_corner"] = bool(worst[cell] >= WORST_CORNER_MIN)
        row.update({"worst_rank": float(worst[cell]),
                    "tanimoto_partial": float(agg.loc[cell, "tanimoto_partial"]),
                    "knn20": float(agg.loc[cell, "knn20_jaccard"]),
                    "scalarness": float(agg.loc[cell, "scalarness"])})
        rows.append(row)
    verdict = pd.DataFrame(rows).set_index("cell")
    conds = [c for c in verdict.columns if c[0].isdigit()]
    verdict["CLEARS"] = verdict[conds].all(axis=1)

    if rules == "s1":
        L += [f"{sub} Pre-registered verdict — S1 rule set", "",
              "A cell clears iff all four hold: (1) paired mean Delta log(rank) exceeds the "
              "pooled-error half-width, (2) all seeds move the same way, (3) both damage "
              "guards pass, (4) `tanimoto_partial` rises above the control — rank without "
              "structural information is a noise embedding and does not count.", ""]
    else:
        L += [f"{sub} Pre-registered verdict — S2 rule set", "",
              "S1's four conditions, plus **S2-R2**: the *worst* seed's effective rank must "
              f"reach **{WORST_CORNER_MIN:.0f}** — the threshold S4 already uses as the point "
              "below which the loss is declared unable to fix the collapse. Ranking moves to "
              "`knn20_jaccard` under **S2-R3**, and `tanimoto_partial` is demoted to a gate: "
              "its `C_w1`-`D_w3` gap of 0.018 against a per-seed sd of 0.050 would need ~36 "
              "seeds to resolve, so ranking on it is not a measurement.", ""]
    L += [md_table(verdict, index_label="cell", floats=4), ""]

    # ---- selection -----------------------------------------------------------------
    clearing = verdict[verdict["CLEARS"]]
    best, escalate = None, None
    if not len(clearing):
        L += ["**No cell clears every condition.** See the escalation check below.", ""]
    elif rules == "s1":
        best = clearing["tanimoto_partial"].idxmax()
        L += [f"**Best configuration: `{best}`** — highest `tanimoto_partial` "
              f"({clearing.loc[best, 'tanimoto_partial']:.4f}) among the {len(clearing)} "
              "clearing cell(s), which is the pre-registered criterion. Rank is the proxy; "
              "the readout is the thing.", ""]
    else:
        _, _, s_knn, _ = rcbd(d, "knn20_jaccard")
        half_knn = t_crit * s_knn * np.sqrt(2 / n)
        order = clearing["knn20"].sort_values(ascending=False)
        top = order.index[0]
        # A sole survivor needs no tie-break: R3 exists to decide BETWEEN clearing cells, so
        # with one there is nothing to resolve and quoting an infinite gap would dress a
        # walkover up as a measurement.
        if len(order) == 1:
            best = top
            L += [f"**`{best}` is the only cell that clears.** S2-R3 has nothing to "
                  f"tie-break: the selection was made by the conditions, not by "
                  f"`knn20_jaccard` (which reads {order.iloc[0]:.4f} here).", ""]
        elif (gap := float(order.iloc[0] - order.iloc[1])) > half_knn:
            best = top
            L += [f"**S2-R3 resolves: `{best}`** — highest `knn20_jaccard` "
                  f"({order.iloc[0]:.4f}) among {len(clearing)} clearing cells, and the gap "
                  f"to the runner-up ({gap:.4f}) exceeds the pooled half-width on that metric "
                  f"({half_knn:.4f}).", ""]
        else:
            best = clearing["worst_rank"].idxmax()
            escalate = (f"`knn20_jaccard` cannot separate the survivors: the top-two gap "
                        f"({gap:.4f}) is inside the pooled half-width ({half_knn:.4f}).")
            L += [f"**S2-R4, the terminal branch: pin `{best}`.** {escalate} The rule then "
                  "takes the more robust survivor — the highest worst-corner rank "
                  f"({clearing.loc[best, 'worst_rank']:.2f}) — and the global-vs-local "
                  "question (`tanimoto_partial` falling while `knn20_jaccard` rises) is "
                  "**escalated to the DKL project**, not to more fold-0 seeds.", ""]

    # ---- diagnosis: where the per-dimension stds sit -------------------------------
    diag = d.groupby("cell")[["emb_std_p5", "emb_std_p50", "emb_std_p95", "emb_std_max",
                              "n_dims_below_0.1", "n_dims_below_0.5", "n_dims_below_gamma",
                              "emb_trace", "cfg_vic_gamma"]].mean()
    L += [f"{sub} Diagnosis — per-dimension spread", "",
          "| observation | conclusion | action |", "|---|---|---|",
          "| stds pile up **below** gamma | force-limited | raise `w_vic` |",
          "| stds sit **at** gamma, rank still low | covariance binding | raise `--w-cov` (or gamma) |",
          "| a stable subset stuck near 0.06 at every dose | GELU-dead dims | **architecture**, not loss |",
          "", md_table(diag, index_label="cell"), "",
          "`n_dims_below_0.5` is the column to read when cells disagree: the `emb_std_p*` "
          "entries are means of per-run percentiles, which is not the percentile of the pooled "
          "distribution, and `n_dims_below_gamma` uses each cell's own gamma so it is not "
          "comparable across a gamma=1.0 cell.", ""]

    # ---- escalation triggers, evaluated mechanically -------------------------------
    strongest = means.idxmax()
    fires = []
    # `n_dims_below_0.5`, not `n_dims_below_gamma`: a gamma=1.0 cell would measure saturation
    # against a different target than every other cell and make the trigger depend on which
    # cell happened to be strongest.
    sat = diag.loc[strongest, "n_dims_below_0.5"] <= 4
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
    L += [f'{sub} "The loss cannot fix this; the architecture must change"', "",
          ("**Not fired.**" if not fires else "**FIRED:**"), ""]
    L += [f"- {f}" for f in fires] or ["No trigger condition is met on this evidence."]
    L += [""]

    summary = {"heading": heading, "rules": rules, "n_seeds": n, "n_runs": int(len(d)),
               "sigma_log_rank": sigma, "resid_df": dfree, "dropped": dropped,
               "best_cell": best, "escalation_fired": fires, "unresolved": escalate,
               "cell_mean_rank": {k: float(np.exp(v)) for k, v in means.items()},
               "worst_rank": {k: float(v) for k, v in worst.items()},
               "verdict": verdict.reset_index().to_dict("records")}
    return L, summary


def utility_section(csv, pin="D_w3", level="##", heading="S2.5 — the featurizer utility probe"):
    """S2.5: the featurizer utility probe, judged by the pre-registered U1/U2/U3.

    Separate from `analyse` because the design is different in kind -- no RCBD, no cells x
    seeds block. The error scale here is the sd across seeds of the same cell, which is what
    the S2.5 rules name.
    """
    sub = level + "#"
    d = pd.read_csv(csv)
    knn = d[d.probe == "knn"]
    shared = knn[knn.cell == "(shared)"].set_index("feat")
    per = knn[knn.cell != "(shared)"]
    cols = [c for c in ["pred1", "pca1", "pca2", "pca4", "pca8", "pca16", "pca32", "emb32"]
            if c in set(per["feat"])]
    piv = per.pivot_table(index="cell", columns="feat", values="spearman")[cols]
    sd = per.pivot_table(index="cell", columns="feat", values="spearman", aggfunc="std")

    L = [f"{level} {heading}", "",
         "Held-out-**cluster** skill on fold 0: fit a probe on molecules in one half of the "
         "validation clusters, score it on the other half, then swap and average. Generated by "
         f"`src/feature_utility.py` from `{csv}`. The response is **Spearman under the kNN "
         "probe** — a GP kernel consumes distances, so that is the probe the deliverable turns "
         "on; ridge is reported beside it as the linear-extractability check.", ""]

    L += [f"{sub} Baselines that do not depend on the run", "",
          "| featurization | dim | kNN Spearman | kNN AP@3.5 |", "|---|---|---|---|",
          f"| raw frozen MiniMol | 512 | {shared.loc['minimol512','spearman']:.4f} | "
          f"{shared.loc['minimol512','ap']:.4f} |",
          f"| ECFP4 | 2048 | {shared.loc['ecfp4','spearman']:.4f} | "
          f"{shared.loc['ecfp4','ap']:.4f} |", "",
          f"{sub} The k-curve — kNN Spearman by principal components retained", "",
          md_table(piv, index_label="cell", floats=4), "",
          "Seed sd of the same quantities (the pre-registered error scale):", "",
          md_table(sd[["pred1", "pca2", "pca32", "emb32"]], index_label="cell", floats=4), ""]

    m = per[per.cell == pin].groupby("feat")["spearman"].agg(["mean", "std"])
    e32, s_e = float(m.loc["emb32", "mean"]), float(m.loc["emb32", "std"])
    rules = [("U1", e32 - float(m.loc["pred1", "mean"]),
              "`emb32` beats the model's own predicted pProp as a 1-d featurization — the floor"),
             ("U2", e32 - float(m.loc["pca2", "mean"]),
              "`emb32` beats its own top-2 PCs — the recovered dimensions earn their place"),
             ("U3", e32 - float(shared.loc["minimol512", "spearman"]),
              "`emb32` is not below the raw frozen 512-d trunk — fine-tuning destroys nothing")]
    L += [f"{sub} Pre-registered verdict — U1/U2/U3 at `{pin}`", "", "| rule | delta | "
          f"seed sd | verdict | claim |", "|---|---|---|---|---|"]
    for name, delta, claim in rules:
        v = "**PASS**" if delta > s_e else ("**FAIL**" if delta < -s_e else "inconclusive")
        L += [f"| {name} | {delta:+.4f} | {s_e:.4f} | {v} | {claim} |"]
    L += ["", f"All three are evaluated at the S2 pin `{pin}`, against the seed sd of that "
          "cell's own `emb32` score, exactly as S2.5 specifies.", ""]
    return L


def uncertainty_section(csv, nfit_glob="reports/uncertainty_nfit_*.csv", level="##"):
    """S2.6: what the geometry buys in *uncertainty*, under the pre-registered V1/V2/V3."""
    import glob as _glob
    sub = level + "#"
    d = pd.read_csv(csv)
    order = [c for c in ["A_base", "C_w1", "D_w3", "E_w10"] if c in set(d["cell"])]
    g = d.groupby("cell").agg(
        calib_rho=("calib_rho", "mean"), novelty_rho=("novelty_rho", "mean"),
        predext_rho=("predext_rho", "mean"), embnn_rho=("embnn_rho", "mean"),
        coverage95=("coverage95", "mean"), gp_spearman=("gp_spearman", "mean")).loc[order]
    L = [f"{level} S2.6 — the GP uncertainty probe", "",
         "An exact GP (Matern 5/2 + white noise) fitted on labelled molecules from one half of "
         "fold 0's validation clusters, predicting the other half. Generated by "
         f"`src/uncertainty_probe.py` from `{csv}`. `calib_rho` is V1, `novelty_rho` is V2 "
         "(with `predext_rho` beside it as the failure mode in its own currency), and the "
         "acquisition table is V3.", "",
         md_table(g, index_label="cell", floats=3), ""]
    hits = [c for c in ("hits_ucb", "hits_maxvar", "hits_greedy", "hits_random",
                        "n_pool_positives") if c in d.columns]
    if hits:
        L += [f"{sub} V3 — simulated acquisition (cumulative pProp >= 3.5 found)", "",
              md_table(d.groupby("cell")[hits].mean().loc[order], index_label="cell", floats=1),
              ""]
    D = d[d.cell == "D_w3"]
    A = d[d.cell == "A_base"]
    sd_pool = float(np.sqrt((D.novelty_rho.std() ** 2 + A.novelty_rho.std() ** 2) / 2))
    v1, v1sd = float(D.calib_rho.mean()), float(D.calib_rho.std())
    dv2 = float(D.novelty_rho.mean() - A.novelty_rho.mean())
    L += [f"{sub} Pre-registered verdict — V1/V2/V3", "",
          "| rule | statistic | error scale | verdict |", "|---|---|---|---|",
          f"| V1 calibration at `D_w3` | {v1:+.4f} | {v1sd:.4f} | "
          f"{'**PASS**' if v1 > v1sd else '**FAIL**'} |",
          f"| V2 novelty, `D_w3` - `A_base` | {dv2:+.4f} | {sd_pool:.4f} | "
          f"{'**PASS**' if dv2 > sd_pool else ('**FAIL**' if dv2 < -sd_pool else 'inconclusive')} |",
          f"| V3 acquisition, `D_w3` vs `A_base` UCB | "
          f"{float(D.hits_ucb.mean()):.1f} vs {float(A.hits_ucb.mean()):.1f} | — | "
          f"{'**PASS**' if D.hits_ucb.mean() > A.hits_ucb.mean() else '**FAIL**'} |", ""]

    files = sorted(_glob.glob(nfit_glob), key=lambda s: int(s.split("_")[-1].split(".")[0]))
    if files:
        parts = []
        for f in files:
            n = int(f.split("_")[-1].split(".")[0])
            x = pd.read_csv(f)
            x["n_fit"] = n
            parts.append(x)
        s = pd.concat(parts).groupby(["cell", "n_fit"])[
            ["calib_rho", "novelty_rho", "embnn_rho", "mean_std"]].mean()
        s.index = [f"{c} @ n_fit={n}" for c, n in s.index]
        L += [f"{sub} The sparsity check", "",
              "**Prediction made before running:** if `D_w3`'s novelty-blindness were an "
              "artefact of 1,000 points spread over ~28 dimensions, its `embnn_rho` and "
              "`novelty_rho` should **rise** with `n_fit` while `A_base`'s stayed flat.", "",
              md_table(s, index_label="cell @ n_fit", floats=3), ""]
    return L


def trunk_pca_section(csv, level="##"):
    """S2.8: the learned bottleneck against a PCA of the fine-tuned trunk."""
    sub = level + "#"
    d = pd.read_csv(csv)
    cols = ["pred1", "trunkpca2", "trunkpca8", "trunkpca32", "trunkpca32_white",
            "trunk512_ft", "emb32"]
    cols = [c for c in cols if c in set(d["feat"])]
    L = [f"{level} S2.8 — the learned bottleneck against a PCA of the fine-tuned trunk", "",
         "Does `head.shared`'s learned 32-d bottleneck earn its place over an unsupervised PCA "
         "of the trunk's 512-d output? PCA cannot produce a dead or duplicate dimension, so it "
         "sidesteps the collapse entirely rather than penalising it. Basis fitted on 20,000 "
         "**training-fold** molecules; each config compared against PCA of **its own** trunk, "
         f"since `--w-vic` reaches the trunk too. Generated by `src/trunk_pca_probe.py` from "
         f"`{csv}`, on the same held-out-cluster split as S2.5.", ""]
    for probe, label in (("knn", "kNN Spearman"), ("knn_ap", "kNN AP@3.5"),
                         ("ridge", "ridge Spearman")):
        sel = d[d.probe == ("knn" if probe.startswith("knn") else "ridge")]
        val = "ap" if probe == "knn_ap" else "spearman"
        L += [f"{sub} {label}", "",
              md_table(sel.pivot_table(index="cell", columns="feat", values=val)[cols],
                       index_label="cell", floats=4), ""]
    geo = d[d.feat.isin(["emb32", "trunkpca32", "trunkpca32_white"])].pivot_table(
        index="cell", columns="feat", values="eff_rank")
    L += [f"{sub} Effective rank of the three 32-d arms", "",
          md_table(geo, index_label="cell", floats=1), "",
          "`trunkpca32` lands near 20, not 32 — PCA guarantees orthogonality, not an even "
          "spectrum, and the trunk's own output is variance-concentrated. `trunkpca32_white` "
          "divides each component by its std and so reaches ~31.5 by construction.", ""]
    g = d[d.probe == "knn"].pivot_table(index=["cell", "seed"], columns="feat",
                                        values="spearman")
    rows = []
    for cell in sorted({c for c, _ in g.index}):
        s = g.loc[cell]
        for arm in [c for c in ("trunkpca32", "trunkpca32_white", "trunk512_ft") if c in s]:
            dl = s["emb32"] - s[arm]
            rows.append({"comparison": f"{cell}: emb32 - {arm}", "mean_delta": float(dl.mean()),
                         "per_seed": [round(float(v), 4) for v in dl.values],
                         "emb32_wins_every_seed": bool((dl > 0).all())})
    L += [f"{sub} Paired within config", "",
          md_table(pd.DataFrame(rows).set_index("comparison"), index_label="comparison",
                   floats=4), ""]
    return L


# ------------------------------------------------------------------------------- the report

def subset(d, spec):
    """One section's rows, plus every (cell, seed) the section asks for and does not have.

    Missing cells AND missing seeds are both reported. Checking only cells would let a
    half-finished section render as a smaller complete design -- S2a at seeds 0-2 instead of
    0-4 is a valid 3x3 block, so nothing downstream would notice, and its df, threshold and
    worst-corner column would all quietly answer a different question than the one asked.
    """
    sel = d[d["cell"].isin(spec["cells"]) & d["cfg_seed"].isin(spec["seeds"])]
    have = set(zip(sel["cell"], sel["cfg_seed"]))
    missing = [f"{c}/seed{s}" for c in spec["cells"] for s in spec["seeds"]
               if (c, s) not in have]
    return sel, missing


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("reports/collapse_runs.csv"))
    p.add_argument("-o", "--out", type=Path,
                   default=Path("reports/embedding_collapse_results.md"))
    p.add_argument("--report", action="store_true",
                   help="render the consolidated report -- the default, named so the command "
                        "in the docstring says what it does")
    p.add_argument("--section", choices=sorted(SECTIONS), default=None,
                   help="render one section alone instead of the consolidated report")
    p.add_argument("--narrative", type=Path, default=Path("reports/collapse_narrative.md"),
                   help="hand-written frame; {{S1}}/{{S2A}}/{{S2B}} are replaced by the "
                        "generated sections")
    p.add_argument("--utility", type=Path, default=Path("reports/feature_utility.csv"),
                   help="the S2.5 probe's CSV; the section renders as 'not yet run' without it")
    p.add_argument("--selftest", action="store_true",
                   help="assert the S1 section still reproduces the published S1 numbers")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    d = pd.read_csv(args.csv)

    if args.selftest:
        sel, missing = subset(d, SECTIONS["s1"])
        if missing:
            raise SystemExit(f"selftest needs the full S1 block; missing {missing}")
        _, s = analyse(sel, **{k: v for k, v in SECTIONS["s1"].items()
                               if k in ("heading", "intro", "rules")})
        checks = [("sigma", s["sigma_log_rank"], 0.3795, 5e-4),
                  ("resid_df", s["resid_df"], 14, 0),
                  ("best_cell", s["best_cell"], "C_w1", None)]
        ok = True
        for name, got, want, tol in checks:
            good = (got == want) if tol is None else abs(got - want) <= tol
            print(f"{'PASS' if good else 'FAIL'}  {name}: {got!r} (expected {want!r})")
            ok &= good
        clears = {r["cell"]: r["CLEARS"] for r in s["verdict"]}
        want_clears = {"A_base": False, "B_w0.3": False, "C_w1": True, "D_w3": True,
                       "E_w10": True, "F_w3_g1": True, "G_w3_drop": False, "H_drop": False}
        good = clears == want_clears
        print(f"{'PASS' if good else 'FAIL'}  CLEARS column: {clears}")
        ok &= good
        print("OVERALL:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.section:
        spec = dict(SECTIONS[args.section])
        sel, missing = subset(d, spec)
        if missing:
            raise SystemExit(f"section {args.section} is missing cells {missing}")
        lines, summary = analyse(sel, spec["heading"], spec["intro"], spec["rules"], level="#")
        summaries = {args.section: summary}
        text = "\n".join(lines) + "\n"
    else:
        rendered, summaries, absent = {}, {}, {}
        for key, spec in SECTIONS.items():
            sel, missing = subset(d, spec)
            if missing:
                absent[key] = missing
                rendered[key.upper()] = (
                    f"## {spec['heading']}\n\n**NOT YET RUN** — missing cells "
                    f"{', '.join(missing)}. This section renders as soon as they land.\n")
                continue
            lines, summaries[key] = analyse(sel, spec["heading"], spec["intro"], spec["rules"])
            rendered[key.upper()] = "\n".join(lines)
        if Path("reports/trunk_pca.csv").exists():
            rendered["S28"] = "\n".join(trunk_pca_section("reports/trunk_pca.csv"))
        else:
            rendered["S28"] = ("## S2.8 — the learned bottleneck against a PCA of the "
                               "fine-tuned trunk\n\n**NOT YET RUN**\n")
        if Path("reports/uncertainty_probe.csv").exists():
            rendered["S26"] = "\n".join(uncertainty_section("reports/uncertainty_probe.csv"))
        else:
            rendered["S26"] = "## S2.6 — the GP uncertainty probe\n\n**NOT YET RUN**\n"
        if Path("reports/feature_utility_v3.csv").exists():
            rendered["S27U"] = "\n".join(utility_section(
                "reports/feature_utility_v3.csv", pin="L_nonorm",
                heading="S2.7 — utility of the no-LayerNorm variants", level="###"))
        else:
            rendered["S27U"] = ""
        if args.utility.exists():
            rendered["S25"] = "\n".join(utility_section(args.utility))
        else:
            rendered["S25"] = ("## S2.5 — the featurizer utility probe\n\n**NOT YET RUN** — "
                               f"`{args.utility}` does not exist.\n")
        text = args.narrative.read_text()
        for token, body in rendered.items():
            if "{{" + token + "}}" not in text:
                raise SystemExit(f"narrative {args.narrative} has no {{{{{token}}}}} slot")
            text = text.replace("{{" + token + "}}", body)
        if absent:
            print(f"sections not yet renderable: {absent}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"script": "src/collapse_analysis.py", "argv": sys.argv[1:], "csv": str(args.csv),
         "sections": summaries}, indent=2, default=str))
    print(f"wrote {args.out} ({len(text.splitlines())} lines) and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
