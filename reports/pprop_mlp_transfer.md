# What pProp_MLP's 2,247 comparable runs say about this repo's sweep ranges

**Which hyperparameter ranges to keep, which to cut, and which to stop sweeping — argued
from the frozen-embedding sweeps rather than from their single winner.**

Analysed 2026-08-17 from `ethan_personal/pprop-mlp-minimol-multitask` (11,784 runs, July
2026). Every number below traces to `reports/pprop_mlp_runs.csv` and is reprinted by
`python reports/pprop_mlp_transfer.py`; the two tables needing per-epoch history come from
`--history`, which re-reads wandb.

**Provenance of the transfer claims**, since they carry the most weight and are the easiest
thing here to overstate:

| claim | status |
|---|---|
| the run counts, bands, regrets and correlations below | **measured** — 2,247 runs, reproducible offline from the CSV |
| `train.py`'s four loss defaults came from run `zfs9n2ln` | **verified** — asserted by the script, all four match to 4 s.f. |
| pProp_MLP's `init_lr` is this repo's phase-1 `head_lr` | **argued** — same regime, not the same code; §2 states the one way it could fail |
| the loss weights are in z-units and so transfer | **inherited** — NOTES §12.3, `pProp_MLP/src/sweep_train.py:460-465` |
| any of this predicts fine-tuning behaviour | **no** — every pProp_MLP run had a frozen trunk. §7 |

Nothing in this report was measured on a trainable trunk. It narrows a search space; it does
not predict a result.

---

## Summary

| | |
|---|---|
| Runs pulled / usable | 11,784 / **2,247** — one objective revision discarded 9,466 |
| Swept dimensions, `bayes_v1` → `bayes_v2` | **11 → 8** |
| Ranges narrowed | `head_lr` (−0.52 decades), `w_cls` (−0.70), `w_pair` (−0.48), `dropout` (moved) |
| Axes pinned as noise | `huber_delta`, `w_std`, `weight_decay` — max regret ≤ 0.0046 each |
| Biggest single finding | `head_lr` above 1.2e-3: **0 of 493 runs** came near best |
| Decisions confirmed, not changed | `batch_size` 1200, `pprop_norm=zscore`, no `early_terminate`, final-epoch selection |
| Biggest risk taken | pinning `weight_decay` — pProp_MLP could not have measured what it does here |

The output is `sweeps/bayes_v2.yaml`. `bayes_v1.yaml` is kept: sweep `c63i6zoh` ran under it.

---

## 1. What is comparable, and what had to be thrown away

Three sweeps live in that project and **only two are mutually comparable**:

| sweep | runs | finished | split | scheme | `goal_metric` range | usable |
|---|---|---|---|---|---|---|
| `9okf022y` | 1,441 | 1,431 | `split_1` | 4-class `[0,1,2,0]` | 1.3045 … 1.4413 | **yes** |
| `j6z4dh1u` | 877 | 816 | `split_1` | 4-class `[0,1,2,0]` | 1.3162 … **1.4483** | **yes** |
| `nihehqst` | 9,466 | 9,446 | `split_6` | 6-class, no `objective_classes` | **−4.3847 … 0.3978** | no |

`nihehqst` is 80% of the project and had to go. Its `goal_metric` does not overlap the
others' at all — a different objective revision reported under the same metric name, on a
different split. This is exactly the failure `src/objective.py` describes inheriting:

> pProp_MLP's `runs/` accumulated three incompatible revisions of `goal_metric` under one
> metric name […] Enforcing that by hand means remembering to bump a string every time the
> spec changes, which is exactly the kind of discipline that fails silently.

It is worth noting that here the discipline *did* fail silently and the evidence survived
only because the config recorded `objective_classes` and `split_dir`. `OBJECTIVE_VERSION`'s
spec-hash is the right fix, and this is the concrete cost of not having had it.

`j6z4dh1u` is the later and wider of the two survivors, so per-axis tables below use it
(816 runs, 8 quantile bins ≈ 102 runs each) and `9okf022y` serves as independent
corroboration. Aggregate tables pool both (2,247).

**The inherited defaults trace to one run.** `train.py`'s loss defaults are `zfs9n2ln`'s
config, `j6z4dh1u`'s best at `goal_metric` 1.4483 — `w_cls` 0.441849, `w_pair` 7.48645,
`w_std` 0.79108, `huber_delta` 1.05132, against defaults 0.4418 / 7.486 / 0.7911 / 1.0513.
The script asserts this rather than trusting the prose: if it ever stops matching, the
premise of this whole report needs re-examining.

**That is also the problem this report exists to fix.** `bayes_v1`'s ranges were built by
centring on that one run and widening a decade each way. One draw from a 2,247-run
population is a weak prior, and §§2–4 show it was misplaced on four axes.

---

## 2. The learning rate — the one direct regime match

**Phase 1 of this repo is not analogous to pProp_MLP's setting, it *is* that setting**: a
randomly initialised head reading a frozen 512-d MiniMol embedding, one cosine annealed to
`eta_min` 1e-8, a z-scored pProp target, and the same four-term loss. Its `init_lr` and this
repo's `head_lr` are the same knob on the same problem.

P(within 0.005 of the global best), 2,247 runs:

| `init_lr` band | n | P(near-best) | p90 | max |
|---|---|---|---|---|
| 5e-5 – 1.5e-4 | 387 | 1.6% | 1.4368 | 1.4459 |
| 1.5e-4 – 3e-4 | 385 | **2.6%** | 1.4387 | 1.4468 |
| 3e-4 – 6e-4 | 484 | 2.3% | **1.4395** | **1.4483** |
| 6e-4 – 1.2e-3 | 498 | **0.2%** | 1.4351 | 1.4449 |
| 1.2e-3 – 3.1e-3 | 493 | **0.0%** | 1.4276 | 1.4386 |

The per-bin regret table on `j6z4dh1u` alone says the same thing — 0.0014–0.0034 for every
band below 8.6e-4, then 0.0066 and 0.0096. `9okf022y` agrees independently: regret 0.0082
above 1.2e-3 and 0.0146 above 1.9e-3.

The p90 penalty above 1.2e-3 is **0.012**, about six times what final-epoch selection costs
(§6). `bayes_v1` sampled `1e-4 – 1e-2` log-uniformly, so roughly **60% of its draws landed
in territory where 991 runs found essentially nothing.**

The 0.005 tolerance is not arbitrary: it is ~2.5× the measured final-epoch selection cost,
so "near-best" means genuinely good rather than lucky in the last epoch.

**Recommendation: `head_lr` → `1e-4 – 3e-3`**, not the `1e-4 – 1.5e-3` the table implies.
Two hedges, both deliberate:

1. **The regimes differ in schedule length.** Phase 1 here is 1–10 epochs against
   pProp_MLP's 20–100. A short cosine can legitimately want a higher peak, and this is the
   one mechanism by which an otherwise-direct transfer could fail.
2. **Sweep `c63i6zoh`'s best trial used `head_lr` 2.4e-3.** Six finished trials is not
   evidence, but excluding your own current best on a borrowed prior is a bad trade.

3e-3 is also exactly where pProp_MLP's own search stopped — both sweeps declared
`init_lr` max `0.003` (highest value actually drawn: 2.996e-3). So this ceiling is the edge
of the evidence rather than a step past it, and the recommendation extrapolates nowhere.

`head_lr_unfrozen` and `trunk_lr` are **unchanged**. Neither has any pProp_MLP analog —
there was no second phase and no trunk. The only check available is a consistency one: the
head is warm rather than random at the phase-2 restart, so its peak should sit at or below
`head_lr`, and `1e-5 – 1e-3` already does.

---

## 3. Three axes that are noise

Max regret is the most `goal_metric` you give up by restricting the axis to any one band —
a flat profile means there is nothing to find, whatever the rank correlation says.

| axis | range explored | bins | **max regret** |
|---|---|---|---|
| `weight_decay` | 1.0e-6 – 9.5e-2 (**six decades**) | 8 | **0.0045** |
| `huber_delta` | 0.100 – 4.98 | 8 | **0.0046** |
| `w_std` | 0.0051 – 0.995 | 8 | **0.0046** |

For scale, final-epoch selection alone costs 0.0019 (§6). All three sit at the noise floor.
`weight_decay` is the starkest: best-in-bin runs 1.4437–1.4483 across six decades, and the
bin containing the global best (1.2e-5 – 4.0e-5) is not distinguishable from the bin at
1.1e-2 – 9.5e-2.

**Recommendation: pin all three** at their existing `train.py` defaults — `huber_delta`
1.0513, `w_std` 0.7911, `weight_decay` 0.01. Pinning changes no behaviour relative to a
trial that drew those values; it only stops bayes spending trials on empty axes. Three of
eleven dimensions removed.

Note in passing that `bayes_v1`'s `weight_decay` floor of 1e-4 already excluded pProp_MLP's
best value (1.34e-5) by an order of magnitude. On a flat axis that cost nothing — but it is
a sign the ranges were being set by intuition rather than by the data that existed.

### The `weight_decay` pin is the least safe change here

pProp_MLP's null result was measured with a **frozen trunk**, where weight decay reached
only the head. Here it is the main thing restraining a 7,919,912-parameter pretrained trunk
from drifting off its initialisation — a mechanism those runs could not have observed at all.
The null result is evidence about a different question.

Pinned anyway, because an axis with no local evidence and no transferred evidence is not
worth a sweep dimension on the first pass. **Reopen it if** the trunk drifts far from its
pretrained weights, if the frozen baseline (`--unfrozen-epochs 0`) matches or beats the
fine-tuned arm, or if the val/train gap widens with `unfrozen_epochs`.
`MiniMolRegressor.param_groups` already accepts a separate `trunk_weight_decay` and is the
sharper knob if it does reopen; it has no CLI flag yet.

---

## 4. Three axes with real structure

### `dropout` — both ends of `bayes_v1`'s range were wrong

| band | n | max | p95 | median | regret |
|---|---|---|---|---|---|
| 0.001 – 0.079 | 102 | 1.4312 | **1.4263** | 1.4091 | **0.0171** |
| 0.079 – 0.125 | 102 | 1.4430 | 1.4372 | 1.4266 | 0.0053 |
| 0.125 – 0.173 | 102 | 1.4432 | 1.4413 | 1.4299 | 0.0051 |
| 0.173 – 0.225 | 102 | **1.4483** | 1.4432 | **1.4348** | 0.0000 |
| 0.225 – 0.272 | 102 | 1.4456 | 1.4433 | 1.4352 | 0.0026 |
| 0.272 – 0.324 | 102 | 1.4468 | **1.4437** | 1.4292 | 0.0015 |
| 0.324 – 0.382 | 102 | 1.4468 | **1.4439** | 1.4295 | 0.0014 |
| 0.382 – 0.500 | 102 | 1.4437 | 1.4413 | 1.4153 | 0.0046 |

The near-zero bin is **the worst region on any axis in this report** — regret 0.0171, more
than 3× the next-worst. Near-zero dropout is not neutral, it is actively harmful, and
`bayes_v1` sampled it. At the other end, the best band runs 0.17–0.38, so the ceiling of
0.30 clipped good territory.

**Recommendation: `dropout` → `uniform 0.05 – 0.35`.**

This also makes `train.py`'s `--dropout` default of **0.0 the worst value pProp_MLP ever
measured**. Not changed here — moving a `train.py` default re-hashes every `config_id` and
orphans the existing output buckets — but it matters for anything driven by hand rather than
by the sweep, the frozen baseline especially. Flagged as a follow-up in §8.

**Caveat.** pProp_MLP's head carried the whole model (13.9M params against a frozen trunk),
so it plausibly needed more regularisation than a 2.1M-param head sitting in front of a
trainable 7.9M-param trunk. The floor at 0.05 is the safe half of this change; if the sweep
pushes hard against it, that is a real signal rather than a range artefact.

### `w_cls` — the top of the range is actively harmful

Bad only below ~0.014 (regret 0.0057), then broadly flat to 1.0. The decisive evidence is
the *other* sweep: `9okf022y` searched 0.1–4.0 and found Spearman **−0.178** (p = 1.1e-11)
against `goal_metric`, with its top-5% at a median of 0.134. Higher `w_cls` is worse over
precisely the stretch `bayes_v1` added.

**Recommendation: `w_cls` → `0.05 – 1.0`** (was 0.05–5.0).

**This is the weakest of the three loss-weight transfers** and the first to distrust. This
repo's `cls` term is binary BCE where pProp_MLP's was 4-class CE.
`losses.weighted_bce_loss` normalises by `sample_weights.sum()` specifically to preserve
`w_cls`'s units across that change — verified equal to `CrossEntropyLoss(weight=…)` on two
classes — but the class ratio is 104× here against a scheme retaining 18.5% there. The units
are preserved; the regime is not.

### `w_pair` — a plateau, not a trend

Spearman is +0.238 and the global best sits in the top bin, which reads as "more is better".
The regret and p95 columns say otherwise:

| band | max | p95 | regret |
|---|---|---|---|
| 0.005 – 0.061 | 1.4390 | 1.4334 | **0.0093** |
| 0.061 – 0.215 | 1.4415 | 1.4370 | **0.0068** |
| 0.215 – 0.570 | 1.4441 | 1.4418 | 0.0041 |
| 0.570 – 1.101 | 1.4463 | 1.4430 | 0.0020 |
| 1.101 – 2.125 | 1.4456 | 1.4432 | 0.0026 |
| 2.125 – 3.620 | 1.4468 | **1.4439** | 0.0015 |
| 3.620 – 6.305 | 1.4457 | 1.4436 | 0.0025 |
| 6.305 – 9.998 | **1.4483** | 1.4424 | 0.0000 |

p95 is flat at 1.4424–1.4439 from 0.57 all the way to 10. Only the bottom two bins are
genuinely bad. The top bin holds the global best by one lucky run, not by being a better
place to sample.

**Recommendation: `w_pair` → `1.0 – 10.0`** (was 1.0–30.0). 10 is where pProp_MLP's search
actually stopped, so `bayes_v1`'s 10–30 was extrapolation past the evidence rather than a
wider prior. The floor stays at 1.0, comfortably inside the plateau.

---

## 5. The epoch budget — the ceiling is flat, the hit rate is not

`bayes_v1.yaml` states that "`goal_metric` is close to monotone in budget", and concludes
that a winner on the boundary means the range was too narrow and should be widened.
**Measured, the premise does not hold:**

| epochs | n | max | p90 | P(near-best) |
|---|---|---|---|---|
| 20 – 35 | 151 | 1.4463 | 1.4329 | 0.7% |
| 35 – 50 | 347 | **1.4483** | 1.4364 | 0.9% |
| 50 – 70 | 1,018 | 1.4468 | 1.4350 | 0.5% |
| 70 – 101 | 731 | 1.4468 | **1.4385** | **2.6%** |

Best-achievable is flat within 0.002 across a 5× range of budget. What rises with budget is
the **hit rate** — the chance that a given draw of the other hyperparameters lands near the
ceiling. Spearman is +0.221 and positive, but it is measuring robustness, not attainment.

**A longer schedule buys robustness, not a higher peak.** So a winner pinned at
`unfrozen_epochs = 40` is expected behaviour, not evidence the range was binding, and
widening to 80 would double per-trial cost for a ceiling that is already reached. Fine-tuning
epochs cost ~10× what pProp_MLP's did, which makes this the difference between a sweep that
fits the compute budget and one that does not.

**Recommendation: keep `unfrozen_epochs` 3–40 unchanged, and rewrite the yaml guidance.**
The range stands; the instruction to widen on boundary-pinning does not. Widen only on a
*rising ceiling* across the range — a different observation, and one this repo has not yet
made for itself.

`freeze_epochs` is a different matter, and §5.1 moves it.

### 5.1 `freeze_epochs` — phase 1 *is* pProp_MLP's regime, and 1–10 was the wrong window

The two-phase structure has no analog there, but **phase 1 alone has an exact one**: a random
head on a frozen MiniMol embedding, annealing its own cosine to `eta_min`. `nihehqst` is the
only sweep that ran that regime at short budgets — the other two start at 20 epochs — so it
is the only evidence on how long a head takes. Its scale is not comparable to §§2–4 (different
objective revision, `split_6`, `pprop_norm=none`), but the shape within it is valid, and the
structural match is exact: each run anneals a complete cosine over its own length, so a
2-epoch run is a *complete* 2-epoch schedule, not the first two epochs of a long one.

Note `effective = L − 1`: the last epoch of any phase trains at `eta_min` ≈ 0.

| `L` | effective | max | % of best | p95 |
|---|---|---|---|---|
| 1 | 0 | −0.1554 | −39% | −0.214 |
| **2** | **1** | **0.0005** | **0.1%** | **−0.206** |
| 3 | 2 | 0.1786 | 44.9% | −0.045 |
| 4 | 3 | 0.2601 | 65.4% | 0.091 |
| 5 | 4 | 0.3323 | 83.5% | 0.161 |
| 10 | 9 | 0.3593 | 90.3% | **0.245** |
| 15 | 14 | 0.3683 | 92.6% | 0.235 |
| 20 | 19 | 0.3124 | 78.5% | 0.241 |
| 30 | 29 | 0.2681 | 67.4% | 0.208 |

Read `p95`, not `max` — these are bayes draws, so run counts and companion-hyperparameter
quality differ per `L`, and a single-run maximum is noisy. p95 climbs to ~0.245 by `L=10` and
is flat-to-declining after.

**`bayes_v1` sampled the wrong window at both ends**, and the log spacing hid it. On `[1,10]`
the modal draw is `L=2` at 22.2%, `P(1)=17.6%` — and since a length-2 phase equals a length-1
phase, **~40% of v1 trials handed phase 2 a head with no rank signal at all.** Meanwhile only
2.2% of draws reached the saturated region:

| range | P(`L≤2`, dead) | P(`L≤3`, ≤45%) | P(`L≥10`, saturated) |
|---|---|---|---|
| **1–10** (`bayes_v1`) | **39.8%** | **54.4%** | **2.2%** |
| 2–30 | 8.2% | 20.7% | 42.5% |
| **3–20** (`bayes_v2`) | **0.0%** | 8.1% | 39.2% |
| 4–20 | 0.0% | 0.0% | 46.3% |

**Recommendation: `freeze_epochs` → `3 – 20`.** Floor at 3 because `L≤2` is measurably dead;
ceiling at 20 because p95 stops improving past ~15 and `bayes_v1`'s 10 stopped short of it.
The residual 8.1% at `L=3` is kept deliberately — see the counter-argument below.

**Counter-argument, unsettled.** Phase 1's job may be narrower than convergence. The head is
not frozen afterwards — it keeps training through phase 2 on a restarted cosine — so phase 1
may only need to make the head non-random enough that its gradient doesn't distort the
pretrained features (LP-FT, Kumar et al. 2022). "Non-random" could need far fewer epochs than
"converged", and the fold-0 prototype's first unfrozen epoch went val MSE 0.2430 → 0.1896,
the opposite of a distortion collapse. `nihehqst` cannot settle this, because it never had a
phase 2. The 8% probe at `L=3` is what lets the sweep answer it.

**A trap worth recording.** Because every phase anneals to `eta_min` regardless of length,
**phase 1's own curve always plateaus and always looks converged.** "Phase 1 was too short"
is invisible in a single run and only appears when comparing across `freeze_epochs`. That is
plausibly why the default of 5 has stayed untested despite CLAUDE.md flagging it as the first
measurement to take. Do not read a flat phase-1 curve as evidence the phase was long enough.

Widening here is also the cheap direction: a frozen epoch costs ~2× less than an unfrozen one
(32 s vs 65 s) because autograd builds no graph through the trunk. `bayes_v1` was stingy on
the cheap axis and generous on the expensive one.

---

## 6. Decisions this confirms rather than changes

**`batch_size` 1200 — and two measurements that de-risk pinning it.** `batch_size` is fixed
here while pProp_MLP swept it 1000–10000, so any prior taken from its runs is valid only if
the optima are batch-independent. Best-achievable, by band:

| | bs 1000–1600 | bs 1600–2600 | bs 2600–10000 |
|---|---|---|---|
| `init_lr` 5e-5 – 2e-4 | 1.4468 (130) | 1.4459 (68) | 1.4432 (55) |
| `init_lr` 2e-4 – 5e-4 | 1.4463 (112) | **1.4483** (66) | 1.4468 (52) |
| `init_lr` 5e-4 – 1e-3 | 1.4421 (60) | 1.4449 (51) | 1.4450 (50) |
| `init_lr` 1e-3 – 3.1e-3 | 1.4416 (76) | 1.4390 (43) | 1.4396 (53) |

The degradation above 5e-4 appears in **every** column, and the same holds for `w_pair`
(rising in all three bands). Neither optimum moves with batch size, so pinning the batch does
not invalidate either transfer. pProp_MLP also independently prefers small batches — flat
from 1000 to 3800, regret 0.0070 above — corroborating 1200 from a third direction.

**`pprop_norm = zscore`.** Swept there against `none`: median 1.4283 vs 1.4211, p95 1.4430
vs 1.4406, max 1.4483 vs 1.4468. Corroborates the settled choice without reopening it.

**Final-epoch selection, and no `early_terminate`.** Over `j6z4dh1u`'s top 20 runs, the
best-minus-final gap is a median of **0.0019** (mean 0.0025, max 0.0065) — *tighter* than the
0.003–0.005 CLAUDE.md quotes. And 20/20 peaked before their last epoch, at a median of
**77%** through the schedule, only 30% in the final fifth.

Both halves of the current design survive this. The cost of scoring the final epoch is real
and small, worth paying to keep the fixed-budget guarantee. And a peak at 77% is exactly why
hyperband would mislead: the intermediate value read early is not a ranking signal.

---

## 7. What does not transfer — and one warning it carries anyway

**Head shape does not transfer.** NOTES §12.3 already says so; pProp_MLP quantifies *why*.
Its head strongly preferred to be large — regret **0.0132** at `n_layers` 1–2 and **0.0079**
at `hidden_dim` < 783 — and its winner was 13,951,442 parameters, **1.76× the trunk**. That
was correct there: with a frozen embedding, the MLP did all the learning. Here the trunk is
trainable and the division of labour reverses.

**But there is a corollary this repo has not recorded.** `--unfrozen-epochs 0` reproduces
pProp_MLP's regime exactly — that is the point of it. So the frozen baseline **inherits
pProp_MLP's head-size sensitivity**, while using a `DualHead` of **2,105,346** parameters at
`train.py`'s defaults — **6.6× smaller** than the head pProp_MLP tuned, and 0.27× the trunk
where that one was 1.76×. CLAUDE.md calls that baseline

> the only honest baseline for the fine-tuned arm

and on this evidence it is currently not one: an undersized head handicaps the frozen arm on
an axis measured to be worth ~0.013 of `goal_metric` there, which would **overstate the
fine-tuning win**. The fix is not to sweep head shape in the main sweep — that stays blocked
on the supervisor's shared-vs-per-head bottleneck question — but to give the baseline its own
head-size treatment before quoting the comparison. Recorded as a follow-up, not actioned
here.

**Also not transferring, for completeness:** the freeze schedule and `trunk_lr`
(no analog at all), and everything NOTES §12.5 already lists.

---

## 8. The recommended sweep

Implemented in `sweeps/bayes_v2.yaml`. **8 swept dimensions, down from 11.**

| parameter | `bayes_v1` | `bayes_v2` | basis |
|---|---|---|---|
| `freeze_epochs` | q_log **1 – 10** | q_log **3 – 20** | phase 1 *is* pProp's regime; L≤2 is dead (§5.1) |
| `unfrozen_epochs` | q_log 3 – 40 | *unchanged*, guidance rewritten | flat ceiling (§5) |
| `head_lr` | log 1e-4 – **1e-2** | log 1e-4 – **3e-3** | 0/493 near-best above 1.2e-3 (§2) |
| `head_lr_unfrozen` | log 1e-5 – 1e-3 | *unchanged* | no analog (§2) |
| `trunk_lr` | log 1e-6 – 3e-4 | *unchanged* | no analog (§2) |
| `dropout` | uniform **0 – 0.3** | uniform **0.05 – 0.35** | <0.08 is the worst region measured (§4) |
| `w_cls` | log 0.05 – **5.0** | log 0.05 – **1.0** | Spearman −0.178 over 0.1–4 (§4) |
| `w_pair` | log 1.0 – **30.0** | log 1.0 – **10.0** | plateau above 0.5; >10 unexplored (§4) |
| `huber_delta` | uniform 0.5 – 2.0 | **pinned 1.0513** | max regret 0.0046 (§3) |
| `w_std` | log 0.1 – 3.0 | **pinned 0.7911** | max regret 0.0046 (§3) |
| `weight_decay` | log 1e-4 – 1e-1 | **pinned 0.01** | max regret 0.0045 — *but see §3* |

All three pins take their existing `train.py` default, so pinning changes no behaviour
relative to a trial that drew that value.

### Follow-ups this report implies but does not action

1. **`train.py`'s `--dropout` default of 0.0** is the worst value pProp_MLP measured (§4).
   Changing it re-hashes every `config_id`, so it is a deliberate decision, not a drive-by.
2. **The frozen baseline needs its own head-size treatment** before it can be quoted as the
   comparison justifying the repo (§7).
3. **NOTES §12.3** frames the loss-hyperparameter transfer around one winner. It now has a
   2,247-run population behind it, and §§3–4 show the winner was unrepresentative on three
   of the four terms.
4. **NOTES §12.5's "replace wandb with Optuna"** bullet contradicts the shipped
   `sweeps/*.yaml` and CLAUDE.md — stale, unrelated to this analysis, noticed in passing.

---

## Reproducing this

```bash
# offline: every table above, from the committed CSV (numpy + scipy only)
python reports/pprop_mlp_transfer.py

# re-pull from wandb (~4 min) and rewrite the evidence file
python reports/pprop_mlp_transfer.py --fetch

# the two tables needing per-epoch history (§6)
python reports/pprop_mlp_transfer.py --history
```

Evidence: `reports/pprop_mlp_runs.csv` (11,784 rows, one per run) and its sibling
`reports/pprop_mlp_runs.meta.json`.
