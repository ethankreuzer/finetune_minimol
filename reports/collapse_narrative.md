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

{{S1}}

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

{{S2A}}

{{S2B}}

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

{{S25}}

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

{{S26}}

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

{{S27}}

{{S27U}}

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
