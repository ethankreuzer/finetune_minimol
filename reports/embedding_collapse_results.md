# Embedding collapse — all results

**This file is generated.** `src/collapse_analysis.py --report` renders it from
`reports/collapse_runs.csv` plus the hand-written frame in `reports/collapse_narrative.md`.
**Edit the narrative, never this file.** Every table between the narrative sections is computed,
not typed.

```bash
VENV=/home/ethan2/finetune_minimol/.venv/bin/python
$VENV src/emb_readout.py --runs outputs/rank_v1 outputs/rank_v2 -o reports/collapse_runs.csv
$VENV src/collapse_analysis.py --report -o reports/embedding_collapse_results.md
$VENV src/collapse_analysis.py --csv reports/collapse_runs.csv --selftest   # S1 regression
```

The plan, the pre-registered rules and the open checkboxes live in
`reports/embedding_collapse_experiment.md`. **This file holds the results; that one holds the
design.** The rules quoted here were written into it before the runs they judge.

---

## The question

The deliverable is a frozen 32-d embedding — the output of `head.shared` — consumed by a
separate deep-kernel-learning GP. It was collapsing: `emb_effective_rank` measured **1.32–2.85
of 32** against a 15–25 target.

That is not a cosmetic problem. A GP uses exactly one property of this space, distance between
molecules. If the embedding is a scalar in 32 slots, "far apart" degenerates into "different
predicted pProp", two unrelated compounds with the same predicted score become
indistinguishable, and posterior variance stops tracking genuine ignorance — which is the
entire mechanism active learning runs on. Measured at the seed-1 baseline,
ρ(embedding distance, |Δ predicted pProp|) = **0.9986**. That is the failure stated as a number.

**Scope:** loss and regularization only — `w_vic`, `vic_gamma`, `w_cov`, `weight_decay`,
`dropout`. Head architecture is out of scope; `embed_dim=32` and linear task heads are
deliverable requirements. Fold 0 throughout unless stated, ~16 min/run on an RTX A6000.

---

## P1–P6 — before the designed experiment

Six preliminary steps, all complete. They are why the experiment is shaped the way it is, and
three of them **falsified claims that were in the repo at the time**.

| # | step | result |
|---|---|---|
| P1 | verification suites | `verify_trunk` 11/11, `verify_metrics` 8/8, both `OVERALL: PASS`. Embedding still reproduces frozen MiniMol at max\|Δ\| = 0.000e+00; head = 1,611,938 params |
| P2 | smoke run | artifact and metric confirmed. **Export point verified by tracing**: `predict()` returns the same `z` that `embedding_metrics` scores and `val_embeddings.npy` saves, post-LayerNorm/post-GELU — they cannot diverge |
| P3 | the measurement | fold 0 seed 0, 20 epochs: `emb_effective_rank` = **2.78 / 32**, `top1_share` 0.633. The predicted collapse |
| P4 | `w_vic` scan, 6 values | tops out at **9.46 / 32** at `w_vic=0.3`; no measurable `goal_metric` cost |
| P5 | seed-1 discriminator | the 0→0.03 differences are **noise**; the baseline is as bad as **1.32** with 93% of variance in one direction |
| P6 | zero-GPU forensics | four findings from artifacts already on disk |

**P4's per-value numbers** (fold 0, seed 0, `--vic-gamma 0.5`):

| `w_vic` | eff_rank | top1_share | goal_metric | pearson_uniform |
|---|---|---|---|---|
| 0 | 2.85 | 0.624 | 0.9598 | 0.8706 |
| 0.003 | 2.53 | 0.664 | 0.9860 | 0.8710 |
| 0.01 | 3.32 | 0.549 | 0.9609 | 0.8705 |
| 0.03 | 2.25 | 0.685 | 1.0065 | 0.8703 |
| 0.1 | 4.35 | 0.421 | 0.9851 | 0.8710 |
| 0.3 | **9.46** | 0.216 | 0.9882 | 0.8707 |

P5 is what stopped this from being read as a dose–response curve: at seed 1, `w_vic=0` gives
1.32 and `w_vic=0.03` gives 1.33 — indistinguishable, and both below every seed-0 number. Seed
spread exceeded the whole apparent effect below `w_vic=0.1`. Two runs that differ only in an
inert flag read 2.78 and 2.85, so **run-to-run nondeterminism is ≈0.07** — replication effort
belongs on seeds, not repeats.

### P6 — the four forensic findings

**P6.1 — `vic` at real collapse is ~0.41, not ~28, so the ceiling that forbade the fix was
wrong.** `reports/embedding_geometry.md` §7 derived "`w_vic ∈ [0.003, 0.3]`; anything ≥ 1
optimises the wrong thing" from *synthetic unit-scale rank-1* data. Measured `val/vic` is
**0.409** (seed 0, `w_vic=0`), **0.537** (seed 1), 0.191 (`w_vic=0.3`). At the real operating
scale `w_vic=1.0` contributes ~9% of the loss, not 600%. **Falsified by ~60×** — and that
falsification is what made S1's `w_vic ∈ {1, 3, 10}` cells legal to run.

**P6.2 — the whole `vic` term is underpowered, not one half of it.** Decomposed at γ=0.5, the
variance and covariance halves measure comparably (ratio 1.3–2.4) and the hinge is **not**
saturated — 28 of 32 dims sit below γ even at the strongest tested setting. The fix is more
total force, not a re-weighting.

**P6.3 — weight decay cannot be the collapse mechanism.** AdamW's decoupled decay shrinks by
`lr·wd` per step: head `1e-3 · 0.01 = 1e-5` × 4,420 steps = **4.3%** total; trunk **0.33%**;
cosine roughly halves both. A 2–4% shrink cannot erase 30 dimensions. The likelier shrinker is
**LayerNorm**, which normalises across the 32 dims *within each row*, so one dominant
pre-activation divides every other dim down — exactly the baseline std profile (0.934 → 0.017).
S2b's `K_wd0.1` probe exists to make this empirical rather than only arithmetic.

**P6.4 — the Tanimoto readout works and de-circularises `emb_effective_rank`.**

| run | rank | ρ(emb_dist, 1−T) | ρ(\|Δpred\|, 1−T) *control* | ρ(emb_dist, \|Δpred\|) *scalarness* |
|---|---|---|---|---|
| seed1 `w_vic=0` | 1.32 | 0.062 | 0.067 | **0.9986** |
| seed0 `w_vic=0` | 2.85 | 0.079 | 0.070 | 0.920 |
| seed0 `w_vic=0.1` | 4.35 | 0.127 | 0.073 | 0.730 |
| seed0 `w_vic=0.3` | 9.46 | **0.221** | 0.070 | 0.525 |

Structural correlation rises 3.6× with rank while the prediction-only control stays flat at
~0.07 — added rank is genuinely new chemical information, not restated pProp.

### Why the readout exists at all

`emb_effective_rank` is **circular as a success criterion**: it is computed from the same
covariance the `vic` term minimises, so a term that inflates it has by construction optimised
the metric that scores it. Rank could rise because the embedding gained chemistry, or because
the penalty manufactured 30 directions of orthogonal noise, and the eigenspectrum cannot tell
those apart.

`src/emb_readout.py` scores the embedding against **ECFP4 Tanimoto** — a fingerprint family the
model never saw — over 5,000 validation molecules. With A = embedding distance,
B = |Δ predicted pProp|, C = 1 − Tanimoto:

| column | definition | what it is for |
|---|---|---|
| `tanimoto_spearman` | ρ(A, C) | does embedding distance track structure at all |
| `pred_tanimoto_spearman` | ρ(B, C) | **the control** — the correlation a good predictor earns for free. Flat at ~0.07 in every run ever measured |
| `tanimoto_partial` | (ρ_AC − ρ_AB·ρ_BC) / √(1 − ρ_AB²) | structural information **beyond** the prediction. A semi-partial: the correlation between structure and the part of embedding distance that \|Δpred\| does not already explain |
| `scalarness` | ρ(A, B) | the failure mode itself. Lower is better |
| `knn20_jaccard` | overlap of each molecule's 20 nearest by ECFP vs by embedding | **local** rather than global — closer to what a GP kernel actually consumes. Sample-size dependent, so comparable only at fixed n (5,000 everywhere here) |

Spearman throughout: Euclidean-vs-Tanimoto is monotone at best, and Tanimoto is heavily skewed.
No pair-count standard errors are quoted — millions of pairs are not independent, effective n is
the molecule count, and run-to-run variability comes from the seed replicates like every other
number here.

---

## S1 — the screen

Eight cells x three seeds, fold 0. Every rule applied below was written into `reports/embedding_collapse_experiment.md` before these runs started.  24 runs, 8 cells x 3 seeds.

Provenance: `objective_version` = `v1-binary3.5-c917327f`, `split_sha256` = `3ef97e78a85d…`, 1 distinct triple(s).

### Effective rank

Response `log(emb_effective_rank)`; RCBD pooled sigma = **0.3795** on 14 df, so a contrast against `A_base` clears at |Delta log| > 2.145 * sigma * sqrt(2/3) = **0.6646** (a factor of 1.944x).

Effective rank by cell and seed (`worst` is the S2-R2 column):

| cell | 0 | 1 | 2 | worst |
|---|---|---|---|---|
| A_base | 2.833 | 1.526 | 2.386 | 1.526 |
| B_w0.3 | 10.090 | 1.465 | 5.752 | 1.465 |
| C_w1 | 21.295 | 5.849 | 21.213 | 5.849 |
| D_w3 | 30.139 | 23.330 | 29.707 | 23.330 |
| E_w10 | 30.697 | 30.363 | 30.729 | 30.363 |
| F_w3_g1 | 31.025 | 31.046 | 30.957 | 30.957 |
| G_w3_drop | 10.049 | 6.834 | 11.888 | 6.834 |
| H_drop | 1.905 | 2.021 | 2.076 | 1.905 |

Contrasts against the control, on the log scale:

| cell | delta | lo | hi | ratio | significant | all_same_direction | per_seed |
|---|---|---|---|---|---|---|---|
| A_base | 0.0000 | -0.6646 | 0.6646 | 1.0000 | False | False | [0.0, 0.0, 0.0] |
| B_w0.3 | 0.7031 | 0.0385 | 1.3677 | 2.0200 | True | False | [1.2703, -0.0411, 0.8801] |
| C_w1 | 1.8487 | 1.1841 | 2.5133 | 6.3514 | True | True | [2.0173, 1.3436, 2.1852] |
| D_w3 | 2.5379 | 1.8732 | 3.2025 | 12.6525 | True | True | [2.3646, 2.727, 2.5219] |
| E_w10 | 2.6431 | 1.9785 | 3.3077 | 14.0567 | True | True | [2.383, 2.9905, 2.5558] |
| F_w3_g1 | 2.6565 | 1.9919 | 3.3211 | 14.2465 | True | True | [2.3936, 3.0128, 2.5632] |
| G_w3_drop | 1.4572 | 0.7926 | 2.1218 | 4.2938 | True | True | [1.2662, 1.4992, 1.6061] |
| H_drop | -0.0848 | -0.7494 | 0.5798 | 0.9187 | False | False | [-0.3966, 0.281, -0.1388] |

### Guard — `val/pearson_uniform`

Pooled sigma = 0.00053; one-sided (95%) contrast half-width = 0.00076.  Non-inferiority margin **0.005**: the lower bound must stay above -0.005.  The one-sided half-width (0.00076) is narrower than the margin (0.005), so the guard tests the margin as intended.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.00076 | 0.00076 | [0.0, 0.0, 0.0] |
| B_w0.3 | 0.00002 | -0.00074 | 0.00077 | [-0.0001, -0.0007, 0.0009] |
| C_w1 | 0.00021 | -0.00055 | 0.00096 | [-0.0002, -0.0002, 0.0011] |
| D_w3 | -0.00027 | -0.00103 | 0.00048 | [-0.0006, -0.0001, -0.0001] |
| E_w10 | -0.00040 | -0.00116 | 0.00035 | [-0.0017, 0.0004, 0.0] |
| F_w3_g1 | -0.00039 | -0.00115 | 0.00036 | [-0.001, 0.0003, -0.0004] |
| G_w3_drop | 0.00125 | 0.00050 | 0.00201 | [0.001, 0.0011, 0.0016] |
| H_drop | 0.00155 | 0.00080 | 0.00231 | [0.0013, 0.0017, 0.0017] |

### Guard — `val/mse`

Pooled sigma = 0.00177; one-sided (95%) contrast half-width = 0.00255.  Non-inferiority margin **0.01**: the upper bound must stay below +0.01.  The one-sided half-width (0.00255) is narrower than the margin (0.01), so the guard tests the margin as intended.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.00255 | 0.00255 | [0.0, 0.0, 0.0] |
| B_w0.3 | 0.00051 | -0.00203 | 0.00306 | [-0.0003, 0.0037, -0.0018] |
| C_w1 | -0.00010 | -0.00265 | 0.00244 | [0.001, 0.0006, -0.0019] |
| D_w3 | -0.00006 | -0.00260 | 0.00249 | [0.0004, -0.001, 0.0004] |
| E_w10 | 0.00118 | -0.00137 | 0.00372 | [0.0034, 0.0005, -0.0004] |
| F_w3_g1 | 0.00297 | 0.00042 | 0.00552 | [0.0045, 0.0025, 0.0019] |
| G_w3_drop | 0.01823 | 0.01568 | 0.02078 | [0.0158, 0.0201, 0.0188] |
| H_drop | 0.01181 | 0.00926 | 0.01435 | [0.0132, 0.0136, 0.0086] |

### Guard — `val/goal_metric`

Pooled sigma = 0.01518; one-sided (95%) contrast half-width = 0.02182.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.02182 | 0.02182 | [0.0, 0.0, 0.0] |
| B_w0.3 | -0.00139 | -0.02321 | 0.02043 | [0.024, 0.0132, -0.0414] |
| C_w1 | 0.00806 | -0.01377 | 0.02988 | [0.0287, 0.0137, -0.0182] |
| D_w3 | 0.00891 | -0.01291 | 0.03073 | [0.0365, 0.005, -0.0148] |
| E_w10 | 0.01015 | -0.01167 | 0.03197 | [0.0338, 0.004, -0.0074] |
| F_w3_g1 | 0.00938 | -0.01244 | 0.03121 | [0.0479, -0.0146, -0.0052] |
| G_w3_drop | 0.02166 | -0.00016 | 0.04348 | [0.0591, 0.0041, 0.0018] |
| H_drop | 0.03365 | 0.01183 | 0.05548 | [0.0732, 0.0188, 0.0089] |

### Pre-registered verdict — S1 rule set

A cell clears iff all four hold: (1) paired mean Delta log(rank) exceeds the pooled-error half-width, (2) all seeds move the same way, (3) both damage guards pass, (4) `tanimoto_partial` rises above the control — rank without structural information is a noise embedding and does not count.

| cell | 1_significant_up | 2_same_direction | 3_guards | 4_partial_up | worst_rank | tanimoto_partial | knn20 | scalarness | CLEARS |
|---|---|---|---|---|---|---|---|---|---|
| A_base | False | False | True | False | 1.5261 | 0.0137 | 0.0130 | 0.9603 | False |
| B_w0.3 | True | False | True | True | 1.4646 | 0.1336 | 0.0256 | 0.7371 | False |
| C_w1 | True | True | True | True | 5.8490 | 0.1650 | 0.0534 | 0.4554 | True |
| D_w3 | True | True | True | True | 23.3296 | 0.1474 | 0.0843 | 0.2833 | True |
| E_w10 | True | True | True | True | 30.3632 | 0.1187 | 0.0947 | 0.2364 | True |
| F_w3_g1 | True | True | True | True | 30.9568 | 0.0948 | 0.0940 | 0.1351 | True |
| G_w3_drop | True | True | False | True | 6.8336 | 0.1878 | 0.0369 | 0.6544 | False |
| H_drop | False | False | False | False | 1.9052 | -0.0037 | 0.0108 | 0.9590 | False |

**Best configuration: `C_w1`** — highest `tanimoto_partial` (0.1650) among the 4 clearing cell(s), which is the pre-registered criterion. Rank is the proxy; the readout is the thing.

### Diagnosis — per-dimension spread

| observation | conclusion | action |
|---|---|---|
| stds pile up **below** gamma | force-limited | raise `w_vic` |
| stds sit **at** gamma, rank still low | covariance binding | raise `--w-cov` (or gamma) |
| a stable subset stuck near 0.06 at every dose | GELU-dead dims | **architecture**, not loss |

| cell | emb_std_p5 | emb_std_p50 | emb_std_p95 | emb_std_max | n_dims_below_0.1 | n_dims_below_0.5 | n_dims_below_gamma | emb_trace | cfg_vic_gamma |
|---|---|---|---|---|---|---|---|---|---|
| A_base | 0.034 | 0.179 | 0.762 | 1.005 | 8.667 | 28.333 | 28.333 | 4.024 | 0.500 |
| B_w0.3 | 0.137 | 0.322 | 0.439 | 0.451 | 2.667 | 30.667 | 30.667 | 3.640 | 0.500 |
| C_w1 | 0.390 | 0.448 | 0.512 | 0.530 | 0.333 | 17.000 | 17.000 | 6.755 | 0.500 |
| D_w3 | 0.505 | 0.532 | 0.600 | 0.644 | 0.000 | 3.333 | 3.333 | 9.349 | 0.500 |
| E_w10 | 0.514 | 0.545 | 0.609 | 0.641 | 0.000 | 0.667 | 0.667 | 9.750 | 0.500 |
| F_w3_g1 | 0.937 | 1.044 | 1.124 | 1.195 | 0.000 | 0.000 | 5.000 | 34.927 | 1.000 |
| G_w3_drop | 0.078 | 0.306 | 0.485 | 0.496 | 9.333 | 31.333 | 31.333 | 3.389 | 0.500 |
| H_drop | 0.079 | 0.192 | 0.673 | 0.840 | 3.000 | 28.000 | 28.000 | 3.466 | 0.500 |

`n_dims_below_0.5` is the column to read when cells disagree: the `emb_std_p*` entries are means of per-run percentiles, which is not the percentile of the pooled distribution, and `n_dims_below_gamma` uses each cell's own gamma so it is not comparable across a gamma=1.0 cell.

### "The loss cannot fix this; the architecture must change"

**Not fired.**

No trigger condition is met on this evidence.


### Reading S1

**The collapse is solved by the loss.** Effective rank goes from 2.25 (control mean) to ~30 of
32, at no measurable predictive cost, in the `w_vic ∈ {3, 10}` range P6.1 made legal.

**Both damage guards resolve the margin they were asked to test** — `pearson_uniform` half-width
0.00076 against a 0.005 margin, `mse` 0.00255 against 0.010. "No detectable cost" is therefore a
real non-inferiority result, not an underpowered one. No escalation trigger fired, and
`n_dims_below_0.1` reaches 0.0 at `D_w3`, so there are no GELU-dead dimensions and escalation
(a) — changing the activation or the export point — is off the table.

**`[!]` The S1 rule named `C_w1`, and the S1 rule was defective.** The output above is correct:
it is the mechanical result of the pre-registered rule, and it is deliberately not rewritten,
because editing it would destroy exactly what pre-registration bought. The defect is recorded
instead. The rule ranks clearing cells on the *mean* of `tanimoto_partial` with **no dispersion
and no worst-corner condition**, and so selected on a gap of 0.018:

| cell | `tanimoto_partial` per seed | mean | sd |
|---|---|---|---|
| `C_w1` | 0.2122 / 0.1188 / 0.1639 | 0.1650 | **0.0467** |
| `D_w3` | 0.1438 / 0.1500 / 0.1484 | 0.1474 | **0.0032** |

The paired differences are +0.0684 / −0.0312 / +0.0155 — mean +0.0176, sd 0.0498, t = 0.61.
Resolving that mean at 95% needs **~36 seeds per cell**, more than this experiment's entire
budget. The repair is not more replicates; it is to stop ranking on an unresolvable mean. S2's
rule set does that, and adds the worst-corner condition: `C_w1`'s worst seed reads **5.85**,
a collapsed embedding by every standard set here, against `D_w3`'s **23.33**.

**Dropout is a rank antagonist, as predicted before the runs.** `_mlp_blocks` appends `Dropout`
as the last layer of every hidden block including the bottleneck, and `forward_with_embedding`
returns `self.shared(x)` — so the `z` fed to `variance_covariance_loss` is **post-dropout during
training**, with inverted scaling inflating a dead dimension's apparent std out of its own mean.
At fixed `w_vic=3`, dropout 0.2 costs two-thirds of the rank (`D_w3` ~30 → `G_w3_drop` ~10), and
`G`/`H` are the only cells to fail a damage guard. The val-time metric stays honest because
`eval()` makes dropout identity, so this corrupts the *loss*, not the measurement — and it means
`bayes_v2` cannot document `dropout` and `w_vic` as two independent axes.

**γ=1.0 buys scale, not information.** `F_w3_g1` has the highest rank (31.0) and `emb_trace`
34.9 against `D_w3`'s 9.3 — the embedding is simply larger — while its `tanimoto_partial` is the
*lowest* of the four clearing cells. That is the cleanest confirmation in the screen that rank
alone would have been the wrong criterion. But γ changes two things at once, scale **and**
covariance pressure (the covariance half scales as ~s⁴, so 0.5 → 1.0 multiplies it ~16×), so
`F` cannot say which caused the drop. S2b's `I`/`J` separate them.

**The disagreement the plan pre-committed to treating as a finding.** Across `C_w1` → `D_w3` →
`E_w10` → `F_w3_g1`, `tanimoto_partial` *falls* monotonically (0.165 → 0.147 → 0.119 → 0.095)
while `knn20_jaccard` *rises* (0.053 → 0.084 → 0.095 → 0.094). Global rank correlation and local
neighbourhood agreement point opposite ways. Which one the DKL GP actually consumes decides the
pin, and that is a question for the downstream project rather than for more fold-0 seeds.

---

## S2a — replication to n=5

The control and the two cells S1 could not choose between, at five seeds. Seeds 0-2 are S1's runs, reused; seeds 3-4 are new. Judged under the **S2 rule set**, which adds the worst-corner condition S1's rule lacked.  15 runs, 3 cells x 5 seeds.

Provenance: `objective_version` = `v1-binary3.5-c917327f`, `split_sha256` = `3ef97e78a85d…`, 1 distinct triple(s).

Residual df is 8 for this block (S1's pre-registered 14 assumes its 8x3 shape), so the two-sided 95% critical value is 2.306 rather than 2.145. Everything else is applied as written.

### Effective rank

Response `log(emb_effective_rank)`; RCBD pooled sigma = **0.2495** on 8 df, so a contrast against `A_base` clears at |Delta log| > 2.306 * sigma * sqrt(2/5) = **0.3638** (a factor of 1.439x).

Effective rank by cell and seed (`worst` is the S2-R2 column):

| cell | 0 | 1 | 2 | 3 | 4 | worst |
|---|---|---|---|---|---|---|
| A_base | 2.833 | 1.526 | 2.386 | 2.706 | 2.605 | 1.526 |
| C_w1 | 21.295 | 5.849 | 21.213 | 20.316 | 23.868 | 5.849 |
| D_w3 | 30.139 | 23.330 | 29.707 | 30.067 | 29.950 | 23.330 |

Contrasts against the control, on the log scale:

| cell | delta | lo | hi | ratio | significant | all_same_direction | per_seed |
|---|---|---|---|---|---|---|---|
| A_base | 0.0000 | -0.3638 | 0.3638 | 1.0000 | False | False | [0.0, 0.0, 0.0, 0.0, 0.0] |
| C_w1 | 1.9554 | 1.5916 | 2.3193 | 7.0669 | True | True | [2.0173, 1.3436, 2.1852, 2.016, 2.2151] |
| D_w3 | 2.4927 | 2.1289 | 2.8566 | 12.0944 | True | True | [2.3646, 2.727, 2.5219, 2.408, 2.4421] |

### Guard — `val/pearson_uniform`

Pooled sigma = 0.00045; one-sided (95%) contrast half-width = 0.00052.  Non-inferiority margin **0.005**: the lower bound must stay above -0.005.  The one-sided half-width (0.00052) is narrower than the margin (0.005), so the guard tests the margin as intended.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.00052 | 0.00052 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| C_w1 | 0.00002 | -0.00051 | 0.00054 | [-0.0002, -0.0002, 0.0011, 0.0003, -0.0008] |
| D_w3 | -0.00060 | -0.00112 | -0.00007 | [-0.0006, -0.0001, -0.0001, -0.0011, -0.0011] |

### Guard — `val/mse`

Pooled sigma = 0.00103; one-sided (95%) contrast half-width = 0.00122.  Non-inferiority margin **0.01**: the upper bound must stay below +0.01.  The one-sided half-width (0.00122) is narrower than the margin (0.01), so the guard tests the margin as intended.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.00122 | 0.00122 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| C_w1 | 0.00044 | -0.00077 | 0.00166 | [0.001, 0.0006, -0.0019, 0.0009, 0.0016] |
| D_w3 | 0.00088 | -0.00033 | 0.00210 | [0.0004, -0.001, 0.0004, 0.0026, 0.002] |

### Guard — `val/goal_metric`

Pooled sigma = 0.01327; one-sided (95%) contrast half-width = 0.01560.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.01560 | 0.01560 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| C_w1 | 0.00478 | -0.01083 | 0.02038 | [0.0287, 0.0137, -0.0182, 0.0138, -0.0141] |
| D_w3 | 0.00822 | -0.00738 | 0.02383 | [0.0365, 0.005, -0.0148, -0.0006, 0.015] |

### Pre-registered verdict — S2 rule set

S1's four conditions, plus **S2-R2**: the *worst* seed's effective rank must reach **12** — the threshold S4 already uses as the point below which the loss is declared unable to fix the collapse. Ranking moves to `knn20_jaccard` under **S2-R3**, and `tanimoto_partial` is demoted to a gate: its `C_w1`-`D_w3` gap of 0.018 against a per-seed sd of 0.050 would need ~36 seeds to resolve, so ranking on it is not a measurement.

| cell | 1_significant_up | 2_same_direction | 3_guards | 4_partial_up | 5_worst_corner | worst_rank | tanimoto_partial | knn20 | scalarness | CLEARS |
|---|---|---|---|---|---|---|---|---|---|---|
| A_base | False | False | True | False | False | 1.5261 | 0.0222 | 0.0142 | 0.9504 | False |
| C_w1 | True | True | True | True | False | 5.8490 | 0.1811 | 0.0581 | 0.4267 | False |
| D_w3 | True | True | True | True | True | 23.3296 | 0.1393 | 0.0873 | 0.2714 | True |

**`D_w3` is the only cell that clears.** S2-R3 has nothing to tie-break: the selection was made by the conditions, not by `knn20_jaccard` (which reads 0.0873 here).

### Diagnosis — per-dimension spread

| observation | conclusion | action |
|---|---|---|
| stds pile up **below** gamma | force-limited | raise `w_vic` |
| stds sit **at** gamma, rank still low | covariance binding | raise `--w-cov` (or gamma) |
| a stable subset stuck near 0.06 at every dose | GELU-dead dims | **architecture**, not loss |

| cell | emb_std_p5 | emb_std_p50 | emb_std_p95 | emb_std_max | n_dims_below_0.1 | n_dims_below_0.5 | n_dims_below_gamma | emb_trace | cfg_vic_gamma |
|---|---|---|---|---|---|---|---|---|---|
| A_base | 0.041 | 0.187 | 0.803 | 1.035 | 8.400 | 28.000 | 28.000 | 4.348 | 0.500 |
| C_w1 | 0.422 | 0.472 | 0.522 | 0.557 | 0.400 | 13.200 | 13.200 | 7.309 | 0.500 |
| D_w3 | 0.507 | 0.539 | 0.607 | 0.649 | 0.000 | 2.200 | 2.200 | 9.556 | 0.500 |

`n_dims_below_0.5` is the column to read when cells disagree: the `emb_std_p*` entries are means of per-run percentiles, which is not the percentile of the pooled distribution, and `n_dims_below_gamma` uses each cell's own gamma so it is not comparable across a gamma=1.0 cell.

### "The loss cannot fix this; the architecture must change"

**Not fired.**

No trigger condition is met on this evidence.


## S2b — covariance dose and the weight-decay null

`I`/`J` raise `--w-cov` at fixed `w_vic=3` and gamma=0.5, isolating covariance pressure from the scale inflation that confounded `F_w3_g1`; `emb_trace` is the pre-registered column that separates them. `K_wd0.1` is the null probe that makes P6.3 empirical. `A_base` and `D_w3` are S1's runs, reused.  15 runs, 5 cells x 3 seeds.

Provenance: `objective_version` = `v1-binary3.5-c917327f`, `split_sha256` = `3ef97e78a85d…`, 1 distinct triple(s).

Residual df is 8 for this block (S1's pre-registered 14 assumes its 8x3 shape), so the two-sided 95% critical value is 2.306 rather than 2.145. Everything else is applied as written.

### Effective rank

Response `log(emb_effective_rank)`; RCBD pooled sigma = **0.4426** on 8 df, so a contrast against `A_base` clears at |Delta log| > 2.306 * sigma * sqrt(2/3) = **0.8333** (a factor of 2.301x).

Effective rank by cell and seed (`worst` is the S2-R2 column):

| cell | 0 | 1 | 2 | worst |
|---|---|---|---|---|
| A_base | 2.833 | 1.526 | 2.386 | 1.526 |
| D_w3 | 30.139 | 23.330 | 29.707 | 23.330 |
| I_w3_cov4 | 26.346 | 7.711 | 30.673 | 7.711 |
| J_w3_cov16 | 14.951 | 3.157 | 30.352 | 3.157 |
| K_wd0.1 | 2.848 | 1.531 | 2.194 | 1.531 |

Contrasts against the control, on the log scale:

| cell | delta | lo | hi | ratio | significant | all_same_direction | per_seed |
|---|---|---|---|---|---|---|---|
| A_base | 0.0000 | -0.8333 | 0.8333 | 1.0000 | False | False | [0.0, 0.0, 0.0] |
| D_w3 | 2.5379 | 1.7046 | 3.3711 | 12.6525 | True | True | [2.3646, 2.727, 2.5219] |
| I_w3_cov4 | 2.1347 | 1.3014 | 2.9679 | 8.4544 | True | True | [2.2301, 1.62, 2.554] |
| J_w3_cov16 | 1.6446 | 0.8114 | 2.4779 | 5.1790 | True | True | [1.6636, 0.7268, 2.5434] |
| K_wd0.1 | -0.0252 | -0.8585 | 0.8080 | 0.9751 | False | False | [0.0052, 0.003, -0.0839] |

### Guard — `val/pearson_uniform`

Pooled sigma = 0.00053; one-sided (95%) contrast half-width = 0.00080.  Non-inferiority margin **0.005**: the lower bound must stay above -0.005.  The one-sided half-width (0.00080) is narrower than the margin (0.005), so the guard tests the margin as intended.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.00080 | 0.00080 | [0.0, 0.0, 0.0] |
| D_w3 | -0.00027 | -0.00107 | 0.00053 | [-0.0006, -0.0001, -0.0001] |
| I_w3_cov4 | -0.00041 | -0.00121 | 0.00039 | [-0.0012, -0.0005, 0.0004] |
| J_w3_cov16 | 0.00005 | -0.00075 | 0.00085 | [-0.0011, 0.0001, 0.0011] |
| K_wd0.1 | 0.00001 | -0.00079 | 0.00081 | [0.0001, -0.0004, 0.0003] |

### Guard — `val/mse`

Pooled sigma = 0.00127; one-sided (95%) contrast half-width = 0.00193.  Non-inferiority margin **0.01**: the upper bound must stay below +0.01.  The one-sided half-width (0.00193) is narrower than the margin (0.01), so the guard tests the margin as intended.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.00193 | 0.00193 | [0.0, 0.0, 0.0] |
| D_w3 | -0.00006 | -0.00198 | 0.00187 | [0.0004, -0.001, 0.0004] |
| I_w3_cov4 | 0.00103 | -0.00090 | 0.00296 | [0.0008, 0.003, -0.0007] |
| J_w3_cov16 | 0.00051 | -0.00142 | 0.00243 | [0.0014, 0.002, -0.0018] |
| K_wd0.1 | 0.00062 | -0.00131 | 0.00255 | [-0.0002, 0.0016, 0.0004] |

### Guard — `val/goal_metric`

Pooled sigma = 0.02089; one-sided (95%) contrast half-width = 0.03171.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.03171 | 0.03171 | [0.0, 0.0, 0.0] |
| D_w3 | 0.00891 | -0.02280 | 0.04062 | [0.0365, 0.005, -0.0148] |
| I_w3_cov4 | 0.02028 | -0.01143 | 0.05200 | [0.0522, 0.0094, -0.0007] |
| J_w3_cov16 | 0.02816 | -0.00355 | 0.05988 | [0.0715, 0.0053, 0.0078] |
| K_wd0.1 | -0.00510 | -0.03681 | 0.02661 | [-0.0181, 0.002, 0.0008] |

### Pre-registered verdict — S2 rule set

S1's four conditions, plus **S2-R2**: the *worst* seed's effective rank must reach **12** — the threshold S4 already uses as the point below which the loss is declared unable to fix the collapse. Ranking moves to `knn20_jaccard` under **S2-R3**, and `tanimoto_partial` is demoted to a gate: its `C_w1`-`D_w3` gap of 0.018 against a per-seed sd of 0.050 would need ~36 seeds to resolve, so ranking on it is not a measurement.

| cell | 1_significant_up | 2_same_direction | 3_guards | 4_partial_up | 5_worst_corner | worst_rank | tanimoto_partial | knn20 | scalarness | CLEARS |
|---|---|---|---|---|---|---|---|---|---|---|
| A_base | False | False | True | False | False | 1.5261 | 0.0137 | 0.0130 | 0.9603 | False |
| D_w3 | True | True | True | True | True | 23.3296 | 0.1474 | 0.0843 | 0.2833 | True |
| I_w3_cov4 | True | True | True | True | False | 7.7112 | 0.1646 | 0.0678 | 0.3096 | False |
| J_w3_cov16 | True | True | True | True | False | 3.1567 | 0.1238 | 0.0653 | 0.4122 | False |
| K_wd0.1 | False | False | True | True | False | 1.5306 | 0.0488 | 0.0126 | 0.9662 | False |

**`D_w3` is the only cell that clears.** S2-R3 has nothing to tie-break: the selection was made by the conditions, not by `knn20_jaccard` (which reads 0.0843 here).

### Diagnosis — per-dimension spread

| observation | conclusion | action |
|---|---|---|
| stds pile up **below** gamma | force-limited | raise `w_vic` |
| stds sit **at** gamma, rank still low | covariance binding | raise `--w-cov` (or gamma) |
| a stable subset stuck near 0.06 at every dose | GELU-dead dims | **architecture**, not loss |

| cell | emb_std_p5 | emb_std_p50 | emb_std_p95 | emb_std_max | n_dims_below_0.1 | n_dims_below_0.5 | n_dims_below_gamma | emb_trace | cfg_vic_gamma |
|---|---|---|---|---|---|---|---|---|---|
| A_base | 0.034 | 0.179 | 0.762 | 1.005 | 8.667 | 28.333 | 28.333 | 4.024 | 0.500 |
| D_w3 | 0.505 | 0.532 | 0.600 | 0.644 | 0.000 | 3.333 | 3.333 | 9.349 | 0.500 |
| I_w3_cov4 | 0.281 | 0.420 | 0.531 | 0.567 | 0.667 | 15.000 | 15.000 | 6.293 | 0.500 |
| J_w3_cov16 | 0.160 | 0.263 | 0.432 | 0.455 | 3.000 | 23.000 | 23.000 | 3.954 | 0.500 |
| K_wd0.1 | 0.047 | 0.172 | 0.679 | 0.899 | 8.000 | 28.000 | 28.000 | 3.359 | 0.500 |

`n_dims_below_0.5` is the column to read when cells disagree: the `emb_std_p*` entries are means of per-run percentiles, which is not the percentile of the pooled distribution, and `n_dims_below_gamma` uses each cell's own gamma so it is not comparable across a gamma=1.0 cell.

### "The loss cannot fix this; the architecture must change"

**Not fired.**

No trigger condition is met on this evidence.


### Reading S2

15 new runs, all `rc=0`, 2026-08-19 12:29 → 13:52. **`D_w3` (`--w-vic 3 --vic-gamma 0.5`) is
the only cell that clears in either block**, and it clears on conditions rather than on a
tie-break.

**`C_w1` is disqualified, and the pre-registered rule is what disqualifies it.** At five seeds
it reads 21.3 / **5.85** / 21.2 / 20.3 / 23.9. Seeds 3 and 4 came back healthy, so seed 1 is a
genuine outlier and not a trend — but that is precisely the case S2-R2 was written for. `C_w1`
collapses on **one seed in five**; `D_w3`'s worst of five is 23.33. Pinning a 20%-collapse
configuration into a 250-trial sweep is the failure the rule exists to prevent, and it fires
without any argument made after the fact. The S1 report's `C_w1` and this file's `D_w3` are now
reconciled by measurement rather than by override.

**`w_cov` fails, and fails informatively — it is a scale shrinker, not a rank raiser.** The
block's pre-registered prediction table asked `emb_trace` to separate scale from covariance
pressure, and it answers cleanly:

| cell | `w_cov` | rank mean | rank **sd** | rank worst | `emb_trace` | `n_dims_below_0.5` | `tanimoto_partial` |
|---|---|---|---|---|---|---|---|
| `A_base` | — | 2.41 | 0.52 | 1.53 | 4.35 | 28.0 | 0.022 |
| `D_w3` | 1 | 28.64 | **2.97** | 23.33 | 9.56 | 2.2 | 0.139 |
| `I_w3_cov4` | 4 | 21.58 | **12.20** | 7.71 | 6.29 | 15.0 | 0.165 |
| `J_w3_cov16` | 16 | 16.15 | **13.64** | 3.16 | 3.95 | 23.0 | 0.124 |
| `F_w3_g1` | 1, γ=1.0 | 31.01 | 0.05 | 30.96 | **34.93** | 0.0 | 0.095 |

`A_base` and `D_w3` are five-seed means in that table (every seed each cell has run); `I`,
`J` and `F_w3_g1` are three-seed. **The `n` differs by row because the evidence does** — the
S2a and S2b sections above keep their designs balanced, and this table is a summary across
them, not a fourth analysis.

`emb_trace` falls monotonically with `w_cov` (9.56 → 6.29 → 3.95) while `n_dims_below_0.5`
rises (2.2 → 15 → 23). **The covariance penalty is satisfied most cheaply by shrinking the
embedding**, and the variance hinge cannot hold the line against it: the hinge is linear below
γ (`relu(γ−σ)` has slope −1) while the covariance half is quartic in scale, so at high `w_cov`
shrinking wins, dimensions fall back below γ, and rank follows the scale down.

That closes a loop with S1's γ finding, and the two together are the strongest structural
result of the whole experiment. `F_w3_g1` inflated the scale and bought rank without buying
chemistry (trace 34.9, the *lowest* `tanimoto_partial` of S1's clearing cells).
`J_w3_cov16` deflates the scale and loses rank. **In this architecture effective rank is
substantially a function of operating scale** — which is the deepest reason rank alone was
never an adequate criterion, and an argument for reporting `emb_trace` beside every rank number.

One nuance against reading trace as the whole story: `J_w3_cov16`'s trace (3.95) is
essentially `A_base`'s (4.35), yet its rank is 16.2 against 2.41. Same total variance,
distributed very differently. Scale sets the budget; the `vic` term still decides how it is
spread.

**The instability is the disqualifying finding, not the mean.** `w_cov`'s effect on *mean* rank
is modest, but its effect on *dispersion* is severe — seed sd goes 2.97 → 12.20 → 13.64, and
both `I` and `J` fail S2-R2 on a single bad seed exactly as `C_w1` does. A configuration this
seed-sensitive cannot be pinned regardless of where its mean sits.

**`I_w3_cov4` also reproduces S1's rule defect in miniature, and R3 catches it.** `I` has the
highest mean `tanimoto_partial` of any cell in the experiment (0.165 against `D_w3`'s 0.139) —
and a seed sd of 0.043 against `D_w3`'s 0.015. That is the same pattern that let S1's rule
select `C_w1`: the noisiest cell posts the highest mean. Under the S2 rule set
`tanimoto_partial` is a gate rather than a ranking, so it cannot carry a selection, and `I` is
disqualified on the worst corner instead. The repair worked on data it was not designed against.

**The weight-decay null is confirmed, cleanly and in the paired form.** `K_wd0.1` against
`A_base` at the same three seeds: 2.85/2.83, 1.53/1.53, 2.19/2.39. Ten times the weight decay
moves effective rank by less than the 0.07 run-to-run nondeterminism measured in P5. The
contrast is not significant and the direction is not consistent. **P6.3 is now empirical as
well as arithmetic**, and S4 item 4 can be written as planned: `weight_decay` stays pinned at
0.01, and the `lr·wd·steps` arithmetic replaces the "least safe change in this file" caveat in
`sweeps/bayes_v2.yaml`, which describes a mechanism that does not operate at these learning
rates. (`K`'s `tanimoto_partial` reads 0.049 against `A_base`'s 0.022, which trips the rule-4
gate — but both are inside `A_base`'s own seed sd of 0.031 and neither embedding has any rank
to speak of. It means nothing, and it changes nothing, because `K` fails on rank first.)

**A real but negligible predictive cost is now visible, which is what n=5 bought.** In S2a the
`pearson_uniform` guard has a half-width of 0.00052, sharp enough that `D_w3`'s contrast
against the control is **−0.00060 with a 95% interval of [−0.00112, −0.00007] — excluding
zero.** So the cost is not zero; it is **8× inside the 0.005 margin** pre-registered as
negligible, and the `mse` guard's interval still includes zero. This is the guard working as
designed: at n=3 it reported "no detectable cost", at n=5 it reports "a detectable cost of
0.0006, far inside the margin". The second statement is the more honest one and points the
same way.

**Two bookkeeping notes, so nobody quotes the wrong figure.** First, `D_w3` appears in three
sections at two seed counts: **the n=5 figures are the ones to cite** — rank 28.64,
`tanimoto_partial` 0.1393, `knn20_jaccard` 0.0873 — while S1's and S2b's tables show the n=3
values (27.73 / 0.1474 / 0.0843) because those blocks are balanced at three seeds. The
differences are seed-count arithmetic, not disagreement. Second, the rule-4 gate compares each
cell against the control's `tanimoto_partial`, and that control value itself moves with seed
count (0.0222 at n=5, 0.0137 at n=3) by more than either number is worth, since `A_base`'s own
seed sd is 0.031. **Checked: no cell's rule-4 verdict changes under either control value** — every
clearing cell sits 4–13× above both — so the gate carries no section-membership artifact here.
It is still a weak gate, and `K_wd0.1` tripping it at 0.049 is the visible symptom.

**A limitation to carry into S3: seed 1 is the binding corner almost everywhere.** It is the
worst seed in 8 of the 11 cells run (the exceptions are cells with essentially no seed spread).
So `min` over seeds is, in practice, close to "how does this configuration behave on seed 1",
and S2-R2 is a one-seed test wearing a five-seed coat. It is still the right rule — a
configuration that collapses on a reproducible hard seed is a configuration that collapses —
but the *number* 23.33 should not be quoted as if five independent corners had been probed.
S3's fold-1 and fold-2 pairs are the check that matters here, because they vary the data rather
than the initialisation.

---

## S2.5 — the featurizer utility probe

Held-out-**cluster** skill on fold 0: fit a probe on molecules in one half of the validation clusters, score it on the other half, then swap and average. Generated by `src/feature_utility.py` from `reports/feature_utility.csv`. The response is **Spearman under the kNN probe** — a GP kernel consumes distances, so that is the probe the deliverable turns on; ridge is reported beside it as the linear-extractability check.

### Baselines that do not depend on the run

| featurization | dim | kNN Spearman | kNN AP@3.5 |
|---|---|---|---|
| raw frozen MiniMol | 512 | 0.7730 | 0.0467 |
| ECFP4 | 2048 | 0.6733 | 0.0488 |

### The k-curve — kNN Spearman by principal components retained

| cell | pred1 | pca1 | pca2 | pca4 | pca8 | pca16 | pca32 | emb32 |
|---|---|---|---|---|---|---|---|---|
| A_base | 0.8342 | 0.8241 | 0.8652 | 0.8698 | 0.8707 | 0.8711 | 0.8711 | 0.8711 |
| C_w1 | 0.8340 | 0.8085 | 0.8528 | 0.8648 | 0.8686 | 0.8683 | 0.8690 | 0.8690 |
| D_w3 | 0.8334 | 0.8153 | 0.8545 | 0.8629 | 0.8618 | 0.8614 | 0.8619 | 0.8619 |
| E_w10 | 0.8340 | 0.7170 | 0.8162 | 0.8482 | 0.8524 | 0.8521 | 0.8565 | 0.8565 |

Seed sd of the same quantities (the pre-registered error scale):

| cell | pred1 | pca2 | pca32 | emb32 |
|---|---|---|---|---|
| A_base | 0.0008 | 0.0028 | 0.0015 | 0.0015 |
| C_w1 | 0.0013 | 0.0043 | 0.0022 | 0.0022 |
| D_w3 | 0.0017 | 0.0023 | 0.0009 | 0.0009 |
| E_w10 | 0.0017 | 0.0035 | 0.0016 | 0.0016 |

### Pre-registered verdict — U1/U2/U3 at `D_w3`

| rule | delta | seed sd | verdict | claim |
|---|---|---|---|---|
| U1 | +0.0286 | 0.0009 | **PASS** | `emb32` beats the model's own predicted pProp as a 1-d featurization — the floor |
| U2 | +0.0075 | 0.0009 | **PASS** | `emb32` beats its own top-2 PCs — the recovered dimensions earn their place |
| U3 | +0.0890 | 0.0009 | **PASS** | `emb32` is not below the raw frozen 512-d trunk — fine-tuning destroys nothing |

All three are evaluated at the S2 pin `D_w3`, against the seed sd of that cell's own `emb32` score, exactly as S2.5 specifies.


### Reading S2.5 — the uncomfortable part

**All three pre-registered rules pass at `D_w3`**, and one of them passes enormously. But the
cross-cell comparison — which S2.5 pre-registered as *reporting only, never a selection rule* —
shows something the rules did not anticipate, and it has to be stated plainly.

**Fine-tuning is emphatically worth it (U3).** The 32-d embedding reaches kNN Spearman 0.862 on
held-out clusters against **0.773 for the raw frozen 512-d MiniMol** and **0.673 for ECFP4**.
That is the frozen-embedding workflow this repo exists to beat, beaten by a wide margin on the
splits it was measured on. Nothing in this project was previously evidence for that.

**But utility falls monotonically with `w_vic`, and the collapsed control is the best
featurizer on this probe:**

| cell | rank | kNN Spearman | kNN AP@3.5 |
|---|---|---|---|
| `A_base` (`w_vic=0`) | 2.4 | **0.8711** | **0.2052** |
| `C_w1` | 18.5 | 0.8690 | 0.2068 |
| `D_w3` | 28.6 | 0.8619 | 0.1997 |
| `E_w10` | 29.9 | 0.8565 | 0.1721 |

So the anti-collapse term does **not** add pProp-relevant information. It **redistributes** it,
at a small cost in total. The k-curve says exactly that: `A_base` reaches 84% of its skill by
k=2 and 96% by k=4, while `E_w10` is *below* the 1-d baseline at k=2 and needs all 32 components
to catch up. Spreading the signal across the space is precisely what the deliverable asked for
geometrically — and it costs about **0.009 Spearman and 0.006 AP**.

**A correction to U1 as it was written.** U1 compares `emb32` against the *kNN-smoothed* 1-d
prediction (0.833), and passes by +0.029. But 20-neighbour smoothing in one dimension is a
handicap, and the fairer floor is the model's own held-out Spearman, which the ridge probe
recovers as **0.878**. Against that floor, no featurization beats simply reading the model's
scalar output: ridge on `emb32` ties it at 0.878, kNN on `emb32` falls short at 0.862. U1 passes
as written, but as written it was a weaker test than intended, and the stronger version does not
pass. That is a defect in a rule this file wrote, found by executing it — the same way S1's
ranking defect was found — and it is recorded rather than patched.

**Why this does not overturn the S2 pin, and what it does change.** The probe measures
*mean-prediction* utility only; S2.5 said so before it ran. The entire argument for fixing the
collapse was about **uncertainty**, not accuracy: at `A_base`, `scalarness` is 0.96, so a GP's
notion of "these molecules are far apart" *is* "these molecules have different predicted pProp",
and posterior variance cannot distinguish a genuinely novel compound from one with an unusual
score. That is the mechanism active learning runs on, and this probe is blind to it by
construction.

What the probe does is **put a price on the fix**, which was not previously known:

| | `A_base` | `D_w3` |
|---|---|---|
| effective rank | 2.4 | 28.6 |
| `scalarness` (lower better) | 0.960 | 0.283 |
| `knn20_jaccard` | 0.014 | 0.087 |
| held-out kNN Spearman | 0.871 | 0.862 |
| held-out AP@3.5 | 0.205 | 0.200 |

**`w_vic=3` buys a non-degenerate geometry for about 0.009 Spearman.** Whether that trade is
correct is now a well-posed question for the DKL project rather than an article of faith here —
and it is the first time the cost side has had a number at all. It also sharpens the case
against `E_w10`, which pays roughly three times as much (0.857, AP 0.172) for no additional
geometric benefit over `D_w3`, independently reinforcing S2's choice.

---

## S2.6 — the GP uncertainty probe

An exact GP (Matern 5/2 + white noise) fitted on labelled molecules from one half of fold 0's validation clusters, predicting the other half. Generated by `src/uncertainty_probe.py` from `reports/uncertainty_probe.csv`. `calib_rho` is V1, `novelty_rho` is V2 (with `predext_rho` beside it as the failure mode in its own currency), and the acquisition table is V3.

| cell | calib_rho | novelty_rho | predext_rho | embnn_rho | coverage95 | gp_spearman |
|---|---|---|---|---|---|---|
| A_base | 0.007 | 0.118 | 0.310 | 0.660 | 0.944 | 0.878 |
| C_w1 | 0.071 | 0.086 | -0.032 | 0.471 | 0.944 | 0.872 |
| D_w3 | 0.121 | -0.025 | -0.061 | 0.330 | 0.945 | 0.870 |
| E_w10 | 0.164 | -0.016 | -0.093 | 0.343 | 0.947 | 0.869 |

### V3 — simulated acquisition (cumulative pProp >= 3.5 found)

| cell | hits_ucb | hits_maxvar | hits_greedy | hits_random | n_pool_positives |
|---|---|---|---|---|---|
| A_base | 35.6 | 23.6 | 35.7 | 5.0 | 51.0 |
| C_w1 | 34.4 | 16.1 | 33.6 | 5.0 | 51.0 |
| D_w3 | 33.5 | 16.6 | 32.9 | 5.0 | 51.0 |
| E_w10 | 34.2 | 18.7 | 33.3 | 5.0 | 51.0 |

### Pre-registered verdict — V1/V2/V3

| rule | statistic | error scale | verdict |
|---|---|---|---|
| V1 calibration at `D_w3` | +0.1211 | 0.0160 | **PASS** |
| V2 novelty, `D_w3` - `A_base` | -0.1429 | 0.0578 | **FAIL** |
| V3 acquisition, `D_w3` vs `A_base` UCB | 33.5 vs 35.6 | — | **FAIL** |

### The sparsity check

**Prediction made before running:** if `D_w3`'s novelty-blindness were an artefact of 1,000 points spread over ~28 dimensions, its `embnn_rho` and `novelty_rho` should **rise** with `n_fit` while `A_base`'s stayed flat.

| cell @ n_fit | calib_rho | novelty_rho | embnn_rho | mean_std |
|---|---|---|---|---|
| A_base @ n_fit=250 | 0.021 | 0.088 | 0.637 | 0.447 |
| A_base @ n_fit=500 | -0.009 | 0.109 | 0.624 | 0.425 |
| A_base @ n_fit=1000 | 0.007 | 0.118 | 0.660 | 0.427 |
| A_base @ n_fit=2000 | 0.002 | 0.105 | 0.676 | 0.417 |
| D_w3 @ n_fit=250 | 0.153 | 0.045 | 0.551 | 0.467 |
| D_w3 @ n_fit=500 | 0.114 | -0.007 | 0.360 | 0.432 |
| D_w3 @ n_fit=1000 | 0.121 | -0.025 | 0.330 | 0.435 |
| D_w3 @ n_fit=2000 | 0.117 | -0.026 | 0.317 | 0.420 |


### Reading S2.6 — the fix makes uncertainty *meaningful* but not *novelty-aware*

Two pre-registered predictions were made and **both were wrong**, which is the strongest
argument that the pre-registration was worth writing.

**V1 passes, monotonically in the dose, and it is the important one.** `A_base`'s posterior
variance is *uncorrelated with its own error* (ρ = +0.007). It is decoration. With `w_vic` the
correlation rises 0.007 → 0.071 → 0.121 → 0.164 across `A_base` → `C_w1` → `D_w3` → `E_w10`.
**A collapsed embedding really does produce meaningless confidence, and fixing the collapse
really does fix that.** Until this run, that claim — the one the whole project rests on — had
never been measured.

**V2 fails, in the direction opposite to the prediction.** `A_base` tracks structural novelty
better (+0.118) than `D_w3` (−0.025). The mechanism is visible in the columns beside it:
`A_base`'s variance correlates 0.310 with *prediction extremity* and 0.660 with distance in its
own space. It is a one-dimensional space, so "far from training data" means "unusual predicted
score" — and unusual scores travel with unusual chemistry, because potent binders are a
distinct chemotype. It looks like novelty detection and is the predicted score in disguise.

**The sparsity explanation was also predicted, and also falsified.** The obvious defence of
`D_w3` was that 1,000 points over ~28 dimensions is too sparse to resolve distance. If so, more
data would help. It does not: across `n_fit` 250 → 2000, `D_w3`'s `embnn_rho` *falls*
0.551 → 0.317 and `novelty_rho` *falls* 0.045 → −0.026, while `A_base`'s hold flat. **V2's
failure is real, not an artefact.**

**But the honest reading is narrower than "the control is better at novelty".** Both are poor
at it — +0.105 is a weak correlation — and mean posterior std sits near 0.42–0.47 for every
cell, so the GP's variance is dominated by its fitted noise term rather than by distance. The
robust, replicated finding is the calibration one. **V3 is the weakest leg** and was labelled so
before it ran: UCB and greedy-on-mean score alike everywhere (≈34 of 51 against random's 5), so
the variance contributes almost nothing to acquisition in this setup, and V3 mostly measures
that the surrogate is already good enough for exploitation to win.

**A defect in this file's own rules.** The V2 failure branch said "the anti-collapse work has no
demonstrated benefit". **That is falsified by V1 in the same run.** The rule set was not
self-consistent, and half of it is not being applied as though it were the whole. Recorded, not
patched.

---

## S2.7 — removing the export-point LayerNorm

P6.3 named LayerNorm as the likely collapse mechanism; `--bottleneck-norm none` removes it. `L_nonorm` is the test (no norm, no `vic`), `M_nonorm_w3` the diagnostic (no norm, `w_vic=3`). `A_base` and `D_w3` are S1's runs, reused. **Read `emb_trace` beside every rank** — without the norm the bottleneck's scale is unconstrained, and S2 established that rank tracks scale.  12 runs, 4 cells x 3 seeds.

Provenance: `objective_version` = `v1-binary3.5-c917327f`, `split_sha256` = `3ef97e78a85d…`, 1 distinct triple(s).

Residual df is 6 for this block (S1's pre-registered 14 assumes its 8x3 shape), so the two-sided 95% critical value is 2.447 rather than 2.145. Everything else is applied as written.

### Effective rank

Response `log(emb_effective_rank)`; RCBD pooled sigma = **0.1342** on 6 df, so a contrast against `A_base` clears at |Delta log| > 2.447 * sigma * sqrt(2/3) = **0.2682** (a factor of 1.308x).

Effective rank by cell and seed (`worst` is the S2-R2 column):

| cell | 0 | 1 | 2 | worst |
|---|---|---|---|---|
| A_base | 2.833 | 1.526 | 2.386 | 1.526 |
| D_w3 | 30.139 | 23.330 | 29.707 | 23.330 |
| L_nonorm | 2.670 | 1.227 | 2.680 | 1.227 |
| M_nonorm_w3 | 30.015 | 19.386 | 29.674 | 19.386 |

Contrasts against the control, on the log scale:

| cell | delta | lo | hi | ratio | significant | all_same_direction | per_seed |
|---|---|---|---|---|---|---|---|
| A_base | 0.0000 | -0.2682 | 0.2682 | 1.0000 | False | False | [0.0, 0.0, 0.0] |
| D_w3 | 2.5379 | 2.2697 | 2.8060 | 12.6525 | True | True | [2.3646, 2.727, 2.5219] |
| L_nonorm | -0.0536 | -0.3218 | 0.2145 | 0.9478 | False | False | [-0.0592, -0.2181, 0.1163] |
| M_nonorm_w3 | 2.4744 | 2.2062 | 2.7426 | 11.8746 | True | True | [2.3605, 2.5419, 2.5209] |

### Guard — `val/pearson_uniform`

Pooled sigma = 0.00034; one-sided (95%) contrast half-width = 0.00055.  Non-inferiority margin **0.005**: the lower bound must stay above -0.005.  The one-sided half-width (0.00055) is narrower than the margin (0.005), so the guard tests the margin as intended.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.00055 | 0.00055 | [0.0, 0.0, 0.0] |
| D_w3 | -0.00027 | -0.00082 | 0.00028 | [-0.0006, -0.0001, -0.0001] |
| L_nonorm | 0.00046 | -0.00009 | 0.00101 | [-0.0004, 0.0006, 0.0012] |
| M_nonorm_w3 | 0.00062 | 0.00007 | 0.00117 | [0.0002, 0.0009, 0.0007] |

### Guard — `val/mse`

Pooled sigma = 0.00423; one-sided (95%) contrast half-width = 0.00671.  Non-inferiority margin **0.01**: the upper bound must stay below +0.01.  The one-sided half-width (0.00671) is narrower than the margin (0.01), so the guard tests the margin as intended.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.00671 | 0.00671 | [0.0, 0.0, 0.0] |
| D_w3 | -0.00006 | -0.00676 | 0.00665 | [0.0004, -0.001, 0.0004] |
| L_nonorm | 0.00322 | -0.00349 | 0.00993 | [0.0014, 0.011, -0.0028] |
| M_nonorm_w3 | 0.00359 | -0.00312 | 0.01030 | [0.0022, 0.0109, -0.0024] |

### Guard — `val/goal_metric`

Pooled sigma = 0.01977; one-sided (95%) contrast half-width = 0.03136.

| cell | delta | lo | hi | per_seed |
|---|---|---|---|---|
| A_base | 0.00000 | -0.03136 | 0.03136 | [0.0, 0.0, 0.0] |
| D_w3 | 0.00891 | -0.02245 | 0.04027 | [0.0365, 0.005, -0.0148] |
| L_nonorm | 0.02311 | -0.00825 | 0.05447 | [0.0525, 0.0234, -0.0065] |
| M_nonorm_w3 | 0.01393 | -0.01743 | 0.04530 | [0.0659, -0.0108, -0.0133] |

### Pre-registered verdict — S2 rule set

S1's four conditions, plus **S2-R2**: the *worst* seed's effective rank must reach **12** — the threshold S4 already uses as the point below which the loss is declared unable to fix the collapse. Ranking moves to `knn20_jaccard` under **S2-R3**, and `tanimoto_partial` is demoted to a gate: its `C_w1`-`D_w3` gap of 0.018 against a per-seed sd of 0.050 would need ~36 seeds to resolve, so ranking on it is not a measurement.

| cell | 1_significant_up | 2_same_direction | 3_guards | 4_partial_up | 5_worst_corner | worst_rank | tanimoto_partial | knn20 | scalarness | CLEARS |
|---|---|---|---|---|---|---|---|---|---|---|
| A_base | False | False | True | False | False | 1.5261 | 0.0137 | 0.0130 | 0.9603 | False |
| D_w3 | True | True | True | True | True | 23.3296 | 0.1474 | 0.0843 | 0.2833 | True |
| L_nonorm | False | False | True | False | False | 1.2271 | -0.0265 | 0.0116 | 0.8309 | False |
| M_nonorm_w3 | True | True | False | True | True | 19.3860 | 0.2258 | 0.0974 | 0.2979 | False |

**`D_w3` is the only cell that clears.** S2-R3 has nothing to tie-break: the selection was made by the conditions, not by `knn20_jaccard` (which reads 0.0843 here).

### Diagnosis — per-dimension spread

| observation | conclusion | action |
|---|---|---|
| stds pile up **below** gamma | force-limited | raise `w_vic` |
| stds sit **at** gamma, rank still low | covariance binding | raise `--w-cov` (or gamma) |
| a stable subset stuck near 0.06 at every dose | GELU-dead dims | **architecture**, not loss |

| cell | emb_std_p5 | emb_std_p50 | emb_std_p95 | emb_std_max | n_dims_below_0.1 | n_dims_below_0.5 | n_dims_below_gamma | emb_trace | cfg_vic_gamma |
|---|---|---|---|---|---|---|---|---|---|
| A_base | 0.034 | 0.179 | 0.762 | 1.005 | 8.667 | 28.333 | 28.333 | 4.024 | 0.500 |
| D_w3 | 0.505 | 0.532 | 0.600 | 0.644 | 0.000 | 3.333 | 3.333 | 9.349 | 0.500 |
| L_nonorm | 0.000 | 0.191 | 2.182 | 2.513 | 16.000 | 18.000 | 18.000 | 34.445 | 0.500 |
| M_nonorm_w3 | 0.499 | 0.527 | 0.668 | 0.705 | 0.000 | 4.000 | 4.000 | 9.973 | 0.500 |

`n_dims_below_0.5` is the column to read when cells disagree: the `emb_std_p*` entries are means of per-run percentiles, which is not the percentile of the pooled distribution, and `n_dims_below_gamma` uses each cell's own gamma so it is not comparable across a gamma=1.0 cell.

### "The loss cannot fix this; the architecture must change"

**Not fired.**

No trigger condition is met on this evidence.


### S2.7 — utility of the no-LayerNorm variants

Held-out-**cluster** skill on fold 0: fit a probe on molecules in one half of the validation clusters, score it on the other half, then swap and average. Generated by `src/feature_utility.py` from `reports/feature_utility_v3.csv`. The response is **Spearman under the kNN probe** — a GP kernel consumes distances, so that is the probe the deliverable turns on; ridge is reported beside it as the linear-extractability check.

#### Baselines that do not depend on the run

| featurization | dim | kNN Spearman | kNN AP@3.5 |
|---|---|---|---|
| raw frozen MiniMol | 512 | 0.7730 | 0.0467 |
| ECFP4 | 2048 | 0.6733 | 0.0488 |

#### The k-curve — kNN Spearman by principal components retained

| cell | pred1 | pca1 | pca2 | pca4 | pca8 | pca16 | pca32 | emb32 |
|---|---|---|---|---|---|---|---|---|
| A_base | 0.8337 | 0.8283 | 0.8651 | 0.8689 | 0.8699 | 0.8703 | 0.8703 | 0.8703 |
| D_w3 | 0.8341 | 0.8182 | 0.8554 | 0.8630 | 0.8620 | 0.8620 | 0.8624 | 0.8624 |
| L_nonorm | 0.8354 | 0.5441 | 0.8642 | 0.8705 | 0.8712 | 0.8713 | 0.8713 | 0.8713 |
| M_nonorm_w3 | 0.8344 | 0.7382 | 0.8143 | 0.8412 | 0.8552 | 0.8580 | 0.8592 | 0.8592 |

Seed sd of the same quantities (the pre-registered error scale):

| cell | pred1 | pca2 | pca32 | emb32 |
|---|---|---|---|---|
| A_base | 0.0007 | 0.0017 | 0.0015 | 0.0015 |
| D_w3 | 0.0005 | 0.0027 | 0.0007 | 0.0007 |
| L_nonorm | 0.0030 | 0.0009 | 0.0008 | 0.0009 |
| M_nonorm_w3 | 0.0016 | 0.0766 | 0.0022 | 0.0022 |

#### Pre-registered verdict — U1/U2/U3 at `L_nonorm`

| rule | delta | seed sd | verdict | claim |
|---|---|---|---|---|
| U1 | +0.0360 | 0.0009 | **PASS** | `emb32` beats the model's own predicted pProp as a 1-d featurization — the floor |
| U2 | +0.0072 | 0.0009 | **PASS** | `emb32` beats its own top-2 PCs — the recovered dimensions earn their place |
| U3 | +0.0984 | 0.0009 | **PASS** | `emb32` is not below the raw frozen 512-d trunk — fine-tuning destroys nothing |

All three are evaluated at the S2 pin `L_nonorm`, against the seed sd of that cell's own `emb32` score, exactly as S2.5 specifies.


### Reading S2.7 — LayerNorm is exonerated, and that is more useful than a win

P6.3 named LayerNorm as the likely collapse mechanism, and the experiment then spent 39 runs
building a penalty to fight it. **The hypothesis is false.**

| cell | LayerNorm | `w_vic` | rank mean | worst | `emb_trace` | held-out kNN Spearman |
|---|---|---|---|---|---|---|
| `A_base` | on | 0 | 2.25 | 1.53 | 4.02 | 0.8703 |
| `L_nonorm` | **off** | 0 | **2.19** | 1.23 | **34.44** | 0.8713 |
| `D_w3` | on | 3 | 27.73 | 23.33 | 9.35 | 0.8624 |
| `M_nonorm_w3` | **off** | 3 | **26.36** | 19.39 | **9.97** | 0.8592 |

Read in pairs: removing the norm changes nothing about the collapse, alone (2.19 vs 2.25) or in
combination (26.36 vs 27.73). On utility it is neutral for the pair without `vic` (+0.0010
against a pooled seed sd of 0.0012) and mildly **harmful** for the pair with it (−0.0032 against
0.0016). **N1 fails, so by N3 `D_w3` stands and `--bottleneck-norm` reverts to its default.**

**The scale story closes here, as arithmetic.** Without `w_vic`, removing the norm inflates the
scale 8.6× (trace 4.02 → 34.44) while leaving the *concentration* untouched. Turn `w_vic` on and
the trace lands at ~9–10 **whether the norm is present or not** — the `vic` term overrides the
normalisation entirely as a scale regulator, and it lands where the hinge says it should:

| γ | predicted trace = 32γ² | measured |
|---|---|---|
| 0.5 | 8.0 | 9.35 (`D_w3`), 9.97 (`M_nonorm_w3`) |
| 1.0 | 32.0 | 34.93 (`F_w3_g1`) |

So **`vic_gamma` sets the operating scale, and scale is the primary lever on effective rank.**
That was inferred across S1 and S2 from `emb_trace` correlations; it is now predicted from the
hinge's definition and confirmed twice.

**What the negative buys, by elimination.** Weight decay: inert (P6.3's arithmetic, confirmed by
`K_wd0.1`). LayerNorm: exonerated (here). What survives is the *original* hypothesis in
`CLAUDE.md`, which P6.3 displaced: the supervised signal reaching the bottleneck is near rank-1,
and **nothing ever asks the other 30 dimensions to do anything.** It is collapse by **neglect**,
not by compression.

That explains cleanly why an unsupervised expansion pressure works and why removing a
normalisation does not — and it names the next idea. If the problem is that nothing asks the
dimensions to carry information, the direct remedy is **to ask them**: supervised auxiliary
targets (RDKit descriptors, or the 32 → 2048 ECFP-bit decoder) rather than an unsupervised
variance penalty. That is the best-motivated untried option on the table, and S2.5's "rank
without relevance" finding argues the same way from the other end.

---

## S2.8 — the learned bottleneck against a PCA of the fine-tuned trunk

Does `head.shared`'s learned 32-d bottleneck earn its place over an unsupervised PCA of the trunk's 512-d output? PCA cannot produce a dead or duplicate dimension, so it sidesteps the collapse entirely rather than penalising it. Basis fitted on 20,000 **training-fold** molecules; each config compared against PCA of **its own** trunk, since `--w-vic` reaches the trunk too. Generated by `src/trunk_pca_probe.py` from `reports/trunk_pca.csv`, on the same held-out-cluster split as S2.5.

### kNN Spearman

| cell | pred1 | trunkpca2 | trunkpca8 | trunkpca32 | trunkpca32_white | trunk512_ft | emb32 |
|---|---|---|---|---|---|---|---|
| A_base | 0.8343 | 0.7614 | 0.8165 | 0.8470 | 0.8270 | 0.8475 | 0.8714 |
| D_w3 | 0.8337 | 0.6841 | 0.7962 | 0.8402 | 0.8249 | 0.8424 | 0.8615 |

### kNN AP@3.5

| cell | pred1 | trunkpca2 | trunkpca8 | trunkpca32 | trunkpca32_white | trunk512_ft | emb32 |
|---|---|---|---|---|---|---|---|
| A_base | 0.1176 | 0.0212 | 0.0390 | 0.0991 | 0.0689 | 0.1041 | 0.2356 |
| D_w3 | 0.1352 | 0.0166 | 0.0331 | 0.0758 | 0.0621 | 0.0850 | 0.1991 |

### ridge Spearman

| cell | pred1 | trunkpca2 | trunkpca8 | trunkpca32 | trunkpca32_white | trunk512_ft | emb32 |
|---|---|---|---|---|---|---|---|
| A_base | 0.8782 | 0.7711 | 0.7921 | 0.8537 | 0.8537 | 0.8625 | 0.8783 |
| D_w3 | 0.8782 | 0.7041 | 0.7482 | 0.8207 | 0.8207 | 0.8518 | 0.8775 |

### Effective rank of the three 32-d arms

| cell | emb32 | trunkpca32 | trunkpca32_white |
|---|---|---|---|
| A_base | 2.1 | 19.8 | 31.5 |
| D_w3 | 28.7 | 21.3 | 31.5 |

`trunkpca32` lands near 20, not 32 — PCA guarantees orthogonality, not an even spectrum, and the trunk's own output is variance-concentrated. `trunkpca32_white` divides each component by its std and so reaches ~31.5 by construction.

### Paired within config

| comparison | mean_delta | per_seed | emb32_wins_every_seed |
|---|---|---|---|
| A_base: emb32 - trunkpca32 | 0.0244 | [0.0288, 0.0227, 0.0218] | True |
| A_base: emb32 - trunkpca32_white | 0.0444 | [0.0492, 0.0407, 0.0433] | True |
| A_base: emb32 - trunk512_ft | 0.0239 | [0.0261, 0.0232, 0.0224] | True |
| D_w3: emb32 - trunkpca32 | 0.0213 | [0.0228, 0.0142, 0.027] | True |
| D_w3: emb32 - trunkpca32_white | 0.0366 | [0.0375, 0.0288, 0.0436] | True |
| D_w3: emb32 - trunk512_ft | 0.0191 | [0.0192, 0.0152, 0.0229] | True |


### Reading S2.8 — the bottleneck earns its place, decisively

**The learned bottleneck beats every PCA arm, in every config, on every seed.**

| arm | `A_base` | `D_w3` |
|---|---|---|
| `emb32` (learned bottleneck) | **0.8714** | **0.8615** |
| `trunk512_ft` (fine-tuned trunk, all 512) | 0.8475 | 0.8424 |
| `trunkpca32` | 0.8470 | 0.8402 |
| `trunkpca32_white` | 0.8270 | 0.8249 |
| `pred1` (the model's own scalar) | 0.8343 | 0.8337 |

*(held-out-cluster kNN Spearman)*

**The 512 → 32 squeeze is not what costs anything.** `trunkpca32` ≈ `trunk512_ft` to within
0.0005 — PCA does its job, and 32 components carry essentially everything the full 512 do for
this task. The gap is entirely that **the trunk's raw representation is worse for pProp than
the head's supervised bottleneck**, by +0.024 (`A_base`) and +0.021 (`D_w3`), positive on all
six seeds.

**On the tail the gap is not close.** `AP@3.5` reads 0.236 for `emb32` against 0.099 for
`trunkpca32` at `A_base` — a factor of **2.4**. Note that `trunkpca32` scores *below* `pred1`
(0.118) there: an unsupervised projection of the trunk is worse at retrieving potent molecules
than simply reading the model's predicted score. Whatever the head learns, the trunk does not
expose it to a distance-based probe.

**Whitening hurts, as predicted.** `trunkpca32_white` loses another 0.020 against plain
`trunkpca32` on kNN and is identical under ridge — which is exactly right, since whitening is an
invertible linear rescaling that a linear probe undoes and a distance-based probe cannot. It
buys effective rank **31.5, essentially the theoretical maximum**, and pays for it in skill. The
low-variance directions it amplifies are noise.

**The caveat raised before running was correct.** `trunkpca32`'s effective rank is **19.8–21.3,
not 32.** PCA guarantees orthogonality, not an even spectrum, and the trunk's own 512-d output
is variance-concentrated enough that the top-32 spectrum stays skewed.

**And this is the third independent decoupling of rank from usefulness.** `A_base`'s
`emb32` has effective rank **2.1** and beats `trunkpca32` at rank 19.8 and
`trunkpca32_white` at rank 31.5 — on both bulk correlation and tail retrieval. Together with
S2.5's dose curve (utility falls as `w_vic` raises rank) and S1's γ result (`F_w3_g1` bought
rank by inflating scale), **effective rank has now failed as a proxy for featurizer quality in
three unrelated ways.** It remains a valid detector of the *pathology* — S2.6's V1 shows a
rank-2 embedding produces uncalibrated GP variance — but it must never again be read as "higher
is better".

**Verdict: keep the learned bottleneck.** The architecture alternative is rejected on the same
evidence standard as the LayerNorm one. What survives from the earlier architecture list is the
option neither of these tested: **supervised auxiliary targets**, which changes *what the
bottleneck is asked to encode* rather than where the vector is taken from or how it is
normalised.

---

## Where this leaves the experiment

**The loss question is settled.** `--w-vic 3 --vic-gamma 0.5 --w-cov 1` (the default `w_cov`)
takes effective rank from 2.4 to 28.6 mean / 23.3 worst of 32, raises structural information
beyond the prediction ~6× over the control, cuts `scalarness` from 0.96 to 0.28, and costs
0.0006 of `pearson_uniform`. Nothing in the tested space does better, and the three obvious
alternatives each fail for a stated reason: `w_vic=1` on seed stability, `w_cov>1` on scale
shrinkage and dispersion, γ=1.0 on buying rank without chemistry.

**What is *not* settled, and should not be claimed.**

1. **Usefulness — measured in S2.5, and the answer is mixed.** The dimensions are real and
   fine-tuning beats the frozen 512-d workflow by a wide margin, but `w_vic` does not *add*
   pProp-relevant information; it redistributes it for about 0.009 Spearman. The remaining
   unmeasured half is **uncertainty quality** — whether a GP on `D_w3`'s geometry gives better
   calibrated posterior variance than one on `A_base`'s near-scalar. That is the claim the
   whole anti-collapse effort rests on, it is the half this probe is blind to by construction,
   and it needs a GP, not a kNN. It belongs to the DKL project.
2. **Scale for the structural readout.** `tanimoto_partial` = 0.139 still has no upper
   reference. S2.5 anchored the *utility* axis (MiniMol-512 at 0.773, ECFP4 at 0.673); the
   equivalent anchor for the structural axis — the same readout on MiniMol-512 and on a random
   32-d projection of it — is still cheap and still unmeasured.
3. **Transferability.** Every number in this file is fold 0 at 20 epochs with dropout 0. S3's
   paired corners — 40 unfrozen epochs, dropout 0.3, folds 1 and 2 — are what decide whether
   `w_vic=3` can be pinned or must move. S1 already showed dropout costs two-thirds of the rank
   at fixed `w_vic`, so the dropout corner is the one most likely to bite.

**S3 and S4 remain blocked for review** — see the plan file's progress table.
