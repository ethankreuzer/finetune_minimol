# Sweep `lyc0lh2d` — what the hyperparameter tune actually found

Read on 2026-08-27 from `models-mila5723/finetune_minimol/lyc0lh2d`, snapshotted while the
sweep still read `RUNNING`: **143 runs, 131 finished and scored, 12 unfinished**. One trial is
one configuration over the full 5x2 grid, so 131 configurations were compared. Median trial
wall-clock **59.9 min**, which confirms the ~55 min estimate and refutes the ~40 min in
`sweeps/bayes_v1.yaml`'s header.

Objective is `final/goal_metric_mean`, range over the 131: **0.6941 .. 1.0311**.

Reproduce with `python src/sweep_top.py --sweep models-mila5723/finetune_minimol/lyc0lh2d`.

## The finding: this sweep discovered one axis, and that axis hit its ceiling

Correlation of each swept axis against the objective, over all 131 scored configurations:

| axis | corr with `goal_metric_mean` |
|---|---:|
| **`log trunk_lr`** | **+0.852** |
| `log head_lr_unfrozen` | +0.136 |
| `log freeze_epochs` | −0.122 |
| `log unfrozen_epochs` | −0.100 |
| `log head_lr` | −0.087 |

`trunk_lr` explains essentially all of the variation. Every other axis is inside the noise --
and `head_lr` in particular spans **1.1e-4 to 8.5e-3 within the top 10 alone**, a 75x range at
indistinguishable quality. That is consistent with what `CLAUDE.md` already argues from the
schedule: with `freeze_epochs` settling at 1-3, phase 1 barely happens, so `head_lr` matters
only through the head it hands over.

**`trunk_lr` is pinned against the yaml's 3.0e-4 cap:**

| tier | median `trunk_lr` | fraction >= 2e-4 (top third of the log range) |
|---|---:|---:|
| top 10 | 2.85e-4 | **100%** |
| top 30 | 2.82e-4 | **100%** |
| all 131 | 2.73e-4 | 84% |

Best configuration with `trunk_lr < 1e-4`: **1.0071**. Best with `trunk_lr >= 2e-4`: **1.0311**.
The gap, 0.024, is roughly 1 sigma of the across-model spread.

By this repo's own rule (`CLAUDE.md`, "Reading the epoch result") a winner on a boundary means
**the range was the answer, not the sweep**. The honest reading is that the optimal `trunk_lr`
lies at or above 3e-4 and was never explored. **Re-sweeping with a widened cap is the first
thing to do if the retrained models underperform.**

Decided with Ethan 2026-08-27: **proceed with these configurations anyway.** `CLAUDE.md` states
this tune's only job is to make the downstream MiniMol-feature analysis trustworthy and that it
"does not need to be rigorous". A model at `trunk_lr` 2.9e-4 is a good model even if 8e-4 would
be better; it is not a badly configured one, which is the bar that matters here.

## What came out clean

- **`unfrozen_epochs` converged interior**: the top 10 sit at 6-11 against a swept range of
  3-40. `CLAUDE.md` warned that "goal_metric is close to monotone in budget, so bayes will push
  `unfrozen_epochs` toward the top of its range" and that a winner on that boundary would mean
  the range was the answer. **That did not happen** -- the epoch budget found a genuine interior
  optimum, and the range does not need widening.
- **`freeze_epochs` settled at its floor**, 1-3 in the top 10 (median 2). Read `2` as `1`: the
  final epoch of a phase trains at ~zero learning rate, so a 2-epoch phase is a 1-epoch phase
  with a no-op appended. The supervisor ranked the freeze schedule as knob #1; the sweep's answer
  is "freeze barely at all".

## The top 3 are a statistical tie

| rank | `goal_metric_mean` | std | run | `freeze`/`unfrozen` | `head_lr` | `head_lr_unfrozen` | `trunk_lr` |
|---|---:|---:|---|---|---:|---:|---:|
| 0 | 1.0311 | 0.0212 | `cfg-84d980dd` | 1 / 9 | 1.02e-3 | 4.68e-4 | 2.94e-4 |
| 1 | 1.0310 | 0.0255 | `cfg-5b94e92b` | 2 / 7 | 1.38e-4 | 9.47e-4 | 2.65e-4 |
| 2 | 1.0289 | 0.0236 | `cfg-78469d72` | 2 / 8 | 1.14e-4 | 8.37e-4 | 2.14e-4 |

rank 0 − rank 2 = **0.0022** against a mean across-model std of **0.0235** — a **0.09 sigma**
gap. These are three samples from a tie, not a ranking. They are kept as the three configurations
to retrain (decided with Ethan) precisely because they are quality-equivalent but
hyperparameter-distinct: rank 0's `head_lr` is 9x ranks 1-2's. The spread across them is
therefore a free robustness check on whatever the feature analysis concludes — if a result holds
across all three, it is not an artifact of one configuration.

**Do not report the rank ordering as a result.** Nothing in this table separates rank 0 from
rank 1 at 0.0001.

---

## What happened when the top 3 were retrained (2026-08-27)

Thirty models: three configurations x ten bootstrap resamples of fold 0's training side, each
drawn with replacement to the full 265,184 rows (**167,263 .. 167,733 unique**, i.e. 63.1-63.3%,
against the 63.2% a bootstrap predicts). Fold 0's 66,296-row validation side is held out and
untouched. 85.6 min wall on 3x RTX A6000, 246 GPU-minutes, 29/29 plus the gate model.

| config | sweep mean | sweep std | retrain mean | retrain std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| cfg0 (sweep rank 0) | 1.0311 | 0.0212 | 1.0059 | 0.0110 | 0.9886 | 1.0211 |
| cfg1 (sweep rank 1) | 1.0310 | 0.0255 | **1.0186** | 0.0128 | 0.9998 | 1.0339 |
| cfg2 (sweep rank 2) | 1.0289 | 0.0236 | 1.0105 | 0.0134 | 0.9906 | 1.0331 |

All 30: mean **1.0117**, std **0.0132**, range 0.9886 .. 1.0339.

**The sweep's rank order did not survive retraining** -- cfg1 comes out ahead of cfg0 by 0.013,
having trailed it by 0.0001. That is exactly what a 0.09 sigma gap predicts, and it is the
cleanest possible confirmation that the top 3 were a tie rather than a ranking. Continue to treat
them as three equivalent encoders.

Two differences from the sweep numbers, both expected and neither a problem:

- **Retrain means sit ~0.02 below sweep means.** A bootstrap training set holds only ~63.2%
  distinct molecules, so each model sees less unique data than the sweep's models did; and these
  are fold 0 alone rather than a mean over five folds.
- **Retrain std is about half the sweep std** (0.011-0.013 against 0.021-0.026). The sweep's
  spread was across 5 folds x 2 seeds, so it contains fold-to-fold variation -- a different data
  partition each time. These ten share one partition and differ only in the resample and the
  seed, which is the narrower quantity by construction.
