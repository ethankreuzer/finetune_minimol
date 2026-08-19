# Embedding collapse: experiment plan and progress log

**Status file. Hand this back to Claude to resume — it is self-contained.**
Created 2026-08-18. Update the checkboxes and the Results log as steps complete.

---

## How to resume

Give Claude this file and say which step you want next. Everything needed to continue is
here: what was measured, what it means, what is still unrun, and the exact commands.

Conventions used below: `[x]` done, `[ ]` not started, `[~]` partially done.
**Every completed step records its number in the Results log** — that is the repo convention
(`CLAUDE.md` → Conventions).

```bash
cd /home/ethan2/finetune_minimol
VENV=/home/ethan2/finetune_minimol/.venv/bin/python
```

---

## Progress at a glance

| # | Step | Status | Key result |
|---|---|---|---|
| P1 | Verification suites | `[x]` | 11/11 and 8/8 PASS |
| P2 | Smoke run | `[x]` | artifact + metric confirmed, export point verified |
| P3 | **The measurement** (fold 0 seed 0) | `[x]` | **`emb_effective_rank` = 2.78/32 — collapse confirmed** |
| P4 | `w_vic` scan, 6 values | `[x]` | tops out at **9.46/32**, no measurable `goal_metric` cost |
| P5 | Seed-1 discriminator | `[x]` | low-`w_vic` scatter is **noise**; baseline as bad as **1.32** |
| P6 | Zero-GPU forensics (4 findings) | `[x]` | **§7 ceiling falsified; weight decay is not the mechanism** |
| S0 | Stage 0 — doc corrections + code | `[x]` | flags landed (`d80ca10`), gate 11/11 + 8/8, readout reproduces P6.4 exactly |
| S1 | Stage 1 — the screen (24 runs) | `[~]` | launched 2026-08-18 23:17 under `nohup scripts/run_rank_v1.sh` |
| S2 | Stage 2 — dose / refine (16 runs) | `[ ]` | — |
| S3 | Stage 3 — transferability (12 runs) | `[ ]` | — |
| S4 | Deliverable — update `bayes_v2.yaml` | `[ ]` | — |

**Open decision, SETTLED 2026-08-18 — option 2, "S1 only, then stop".** S0 + S1 run, the
readout and analysis are written to `reports/rank_v1_results.md`, and the experiment halts for
review before S2. All three options run S1 identically, so the choice only binds at the S1->S2
boundary; option 1 (coding the pre-registered rules) stays available later. Unattended runs
write to `outputs/rank_v1/**` and `reports/rank_v1_*` only.

---

## Context

The deliverable is a frozen 32-d embedding (`head.shared` output) consumed by a downstream
deep-kernel-learning GP. It is collapsing: `emb_effective_rank` measures **1.32–2.85 of 32**
against a 15–25 target. A near-scalar embedding makes GP distance degenerate into "difference
in predicted pProp", destroying the posterior-variance signal active learning runs on.

The `w_vic` scan topped out at 9.46 and showed no measurable `goal_metric` cost, but could not
settle the question: one seed, and seed spread (1.32 vs 2.85) exceeded every `w_vic` effect
below 0.1.

This experiment determines what loss configuration prevents collapse, with enough replication
to be believable and enough interaction evidence to decide whether it can be **pinned** in
`sweeps/bayes_v2.yaml` or must be **swept**.

**Scope** (settled with Ethan): loss + regularization — `w_vic`, `vic_gamma`, `weight_decay`,
`dropout`. Head architecture is out of scope (`embed_dim=32` and linear task heads are
deliverable requirements). **Budget** ~60 runs / ~6 h on 3 A6000s (16 min/run), fold 0 unless
stated. **Success bar:** maximise rank subject to no predictive cost; 15–25 is aspirational,
and an honest "unreachable" is an acceptable outcome.

---

## Results log — completed steps

### `[x]` P1 — Verification suites

```bash
$VENV src/verify_trunk.py   -o verification.md          # 11/11 (the runbook's 10 + grad-flow-has-teeth)
$VENV src/verify_metrics.py -o verification_metrics.md  # 8/8
```

Both `OVERALL: PASS`. Embedding still reproduces frozen MiniMol at max|Δ| = **0.000e+00**;
gradients reach 284/284 reachable trunk tensors. The two at-risk checks (loss parity, freeze
negative test) both held against the real fixtures. Head = **1,611,938** params as predicted.

### `[x]` P2 — Smoke run

`val_embeddings.npy` at `[5000, 32]`, `val/emb_effective_rank` present in `meta.json` history.

**Export point verified by tracing, not assumption:** `predict()` returns `z` from
`forward_with_embedding`; that single array is passed to `embedding_metrics` *and* saved to
`val_embeddings.npy` — they cannot diverge. `_mlp_blocks` appends LayerNorm→GELU after every
width including the final 32, and `shared` has no output projection, so `z` is
post-LayerNorm/post-GELU exactly as the deliverable spec requires.

**Hardware:** RTX A6000 (49 GB), not the H100 the "~4 min" estimate assumed. Peak **6.42 GB**
for the full-width unfrozen backward at batch 1200. ~48 s/epoch unfrozen → **16 min/run**.

### `[x]` P3 — THE MEASUREMENT (fold 0, seed 0, 20 epochs)

`outputs/_no_sweep/fold0_seed0/` — final **`val/emb_effective_rank` = 2.78 / 32**,
`emb_top1_share` = 0.633, `emb_min_std` = 0.0128, `goal_metric` = 0.9707.

Per-epoch: 1.74 → 2.31 → 2.07 → 2.98 → 2.97 *(freeze boundary)* → 1.40 → 1.66 → 1.80 → 1.95 →
2.00 → 2.16 → 2.31 → 2.51 → 2.63 → 2.76 → 2.77 → 2.82 → 2.80 → 2.78 → 2.78.

**This is the predicted collapse, not the healthy range.**

### `[x]` P4 — `w_vic` scan (fold 0, seed 0, `--vic-gamma 0.5`, `outputs/wvic_scan/`)

| `w_vic` | eff_rank | top1_share | min_std | goal_metric | pearson_uniform |
|---|---|---|---|---|---|
| 0 | 2.85 | 0.624 | 0.0167 | 0.9598 | 0.8706 |
| 0.003 | 2.53 | 0.664 | 0.0030 | 0.9860 | 0.8710 |
| 0.01 | 3.32 | 0.549 | 0.0367 | 0.9609 | 0.8705 |
| 0.03 | 2.25 | 0.685 | 0.0076 | 1.0065 | 0.8703 |
| 0.1 | 4.35 | 0.421 | 0.0489 | 0.9851 | 0.8710 |
| 0.3 | **9.46** | 0.216 | 0.0598 | 0.9882 | 0.8707 |

**Does not reach 15–25 anywhere in the tested range.** No measurable `goal_metric` cost — but
`pearson_uniform` moves only 0.002 across all six while `ap_uniform` spans 0.237–0.306, so the
entire `goal_metric` spread is tail-metric noise over ~620 positives on one fold.

> **Runbook bug found and fixed here:** all six values default to
> `outputs/_no_sweep/fold0_seed0` and clobber each other. `--out` per value is mandatory.
> CLAUDE.md's step-4 command has been corrected.

### `[x]` P5 — Seed-1 discriminator (`outputs/wvic_scan/seed1_check/`)

| config | rank | top1_share | goal_metric |
|---|---|---|---|
| seed0 `w_vic=0` | 2.85 | 0.624 | 0.9598 |
| **seed1 `w_vic=0`** | **1.32** | **0.932** | 1.0203 |
| seed0 `w_vic=0.03` | 2.25 | 0.685 | 1.0065 |
| **seed1 `w_vic=0.03`** | **1.33** | **0.930** | 1.0300 |

Two conclusions. **(a)** The 0→0.03 scatter is *noise*: seed 1 gives 1.32 vs 1.33, and
seed-to-seed spread exceeds the apparent `w_vic` effect across that whole range. **(b)** The
baseline collapse is **worse than P3 suggested** — 93% of variance in one direction, below even
the 2–4 "predicted failure" band.

Also free: `outputs/_no_sweep/fold0_seed0` (2.78) and `wvic_scan/w0` (2.85) are *behaviourally
identical* runs (`w_vic=0` makes `gamma` irrelevant), so **run-to-run nondeterminism ≈ 0.07** —
far below seed spread. Replication effort belongs on seeds, not repeats.

### `[x]` P6 — Zero-GPU forensics

Four findings from artifacts already on disk. All independently verified. **Three contradict
text currently in the repo.**

**P6.1 — `vic` at real collapse is ~0.41, not ~28.** `reports/embedding_geometry.md` §7 derived
its "defensible range `w_vic ∈ [0.003, 0.3]`; anything ≥1 optimises the wrong thing" from
*synthetic unit-scale rank-1* data. Measured `val/vic`: **0.409** (seed 0 `w_vic=0`), **0.537**
(seed 1), 0.191 (`w_vic=0.3`). At real scale `w_vic=1.0` contributes ~9% of the loss, not 600%.
**The §7 ceiling is falsified — 1.0/3.0/10 are legitimately testable.**

**P6.2 — the whole `vic` term is underpowered, not one half of it.** Decomposed on the full val
split at γ=0.5:

| run | rank | variance half | covariance half | ratio | dims below γ |
|---|---|---|---|---|---|
| baseline `w_vic=0` | 2.85 | 0.269 | 0.141 | 1.9 | 28/32 |
| seed1 `w_vic=0` | 1.32 | 0.333 | 0.204 | 1.6 | 29/32 |
| `w_vic=0.1` | 4.35 | 0.195 | 0.081 | 2.4 | 32/32 |
| `w_vic=0.3` | 9.46 | 0.109 | 0.083 | 1.3 | 28/32 |

The halves are **comparable** and the hinge is **not saturated** — 28 of 32 dims sit below γ
even at the strongest setting. At `w_vic=0.3` the term is ~1.2% of a ~4.6 loss. **The fix is
more total force, not a re-balance.** Per-dim stds do compress toward γ (0.017–0.934 at
baseline → mostly 0.26–0.52 at `w_vic=0.3`) while rank stays at 9.46.

**P6.3 — weight decay cannot be the collapse mechanism.** AdamW decoupled decay shrinks by
`lr·wd` per step: head `1e-3 · 0.01 = 1e-5` × 4,420 steps = **4.3% total shrink**; trunk
`1e-4 · 0.01` → **0.33%**. Cosine roughly halves both. A 2–4% shrink cannot erase 30
dimensions. The likelier shrinker is **LayerNorm** — it normalizes across the 32 dims *within
each row*, so one dominant pre-activation divides every other dim down, which is exactly the
baseline std profile (0.934 → 0.017).

**P6.4 — the Tanimoto readout works and de-circularizes `emb_effective_rank`.** 2,500 val
molecules, 3.1M pairs, ECFP4 from the existing `fingerprints.npy`:

| run | rank | ρ(emb_dist, 1−T) | ρ(\|Δpred\|, 1−T) *control* | ρ(emb_dist, \|Δpred\|) *scalarness* |
|---|---|---|---|---|
| seed1 `w_vic=0` | 1.32 | 0.062 | 0.067 | **0.9986** |
| seed0 `w_vic=0` | 2.85 | 0.079 | 0.070 | 0.920 |
| seed0 `w_vic=0.1` | 4.35 | 0.127 | 0.073 | 0.730 |
| seed0 `w_vic=0.3` | 9.46 | **0.221** | 0.070 | 0.525 |

Structural correlation rises **3.6×** with rank while the prediction-only control stays flat at
~0.07 — added rank is genuinely new chemical information, not restated pProp. And ρ = **0.9986**
at seed-1 baseline is the "disguised scalar" failure stated numerically.

---

## `[ ]` S0 — Stage 0: corrections and code (no GPU, ~1 h)

- `[x]` **S0.1 Record P6 and correct the stale claims.** All four done 2026-08-18.
  - `[x]` `CLAUDE.md` — all three claims corrected: the falsified §7 `w_vic ≤ 0.3` ceiling
    (now states real `vic` ≈ 0.41 vs the synthetic 28, and that 1–10 is untried), the
    weight-decay mechanism (now gives the 4.3% / 0.33% arithmetic and names LayerNorm as the
    likelier shrinker, in both the step-4 section and "The risk this design manages"), and the
    "settled values, not truncated climbs" line (now states rank was still climbing at epoch 20
    and that `unfrozen_epochs` is itself a rank lever). Also added: `pearson_uniform` is the
    right damage instrument, not `goal_metric`; the state table points here.
  - `[x]` `src/losses.py` — the "weight decay actively shrinks them" claim is replaced by the
    4.3% / 0.33% arithmetic and LayerNorm; the measured 2.78 / 1.32 are recorded; `gamma` is
    documented as a scale target, not a strength knob.
  - `[x]` `src/head.py` — same correction in the `DualHead` docstring, plus the measured
    ρ(emb_dist, |Δpred|) = 0.9986 that states the failure numerically.
  - `[x]` `reports/embedding_geometry.md` — a correction banner in the header, §2's mechanism
    corrected in place, and §7's ceiling table superseded by the measured one (real
    contributions ~60× smaller; `w_vic ∈ {1,3,10}` testable). The original §7 table is retained
    struck-through so the correction is checkable.

  Deferred deliberately: the three source files are touched by S0.2 anyway, so correcting the
  docstrings in the same commit as the `--w-cov` change keeps one reviewable diff per file.

- `[x]` **S0.2 Two CLI flags, in ONE commit.** Landed as `d80ca10`. `--w-cov` verified
  bit-identical at its default (8.939805031 either way) and exactly linear in `w_cov`. Every new `train.py` flag re-hashes every
  `config_id` (`run_config.mirror_train_arguments` copies the whole parser), so all CLI
  additions must land before the sweep is registered.
  - `--trunk-weight-decay` (default `None`): argparse at `src/train.py:194`, pass through at
    `src/train.py:645-646`. `model.param_groups` (`src/model.py:75`) already accepts it.
  - `--w-cov` (default `1.0`, keyword-only, behaviour bit-identical): `src/losses.py:333`
    `variance_covariance_loss(embedding, gamma=1.0, w_cov=1.0)` returning
    `variance + w_cov * covariance`; thread through `combined_loss` (`src/losses.py:384`);
    `src/train.py:416` **and `src/train.py:482`** — `loss_terms_from_arrays` calls the same
    function, and missing it makes `val/vic` report a different function than the one trained.

  P6.2 says `w_cov` is *not* the lever today, so **no S1 cell uses it**. It is added now as
  cheap insurance: once `w_vic` is large enough to drive every dim to γ, the hinge saturates and
  covariance becomes binding. Adding it later would re-hash every `config_id`.

  > **Constraint:** `verify_metrics.check_loss_reduces_to_mse` passes neither `embedding` nor
  > `w_vic`, surviving on the defaults. Do not make `embedding` required, do not reorder
  > positionals before it, do not move `w_vic`'s default off `0.0`. Keyword-only appends are safe.

- `[x]` **S0.3 `src/emb_readout.py` written and validated.** It reproduces P6.4's table
  *exactly* from the 9 runs on disk — rank 1.323/2.851/4.350/9.457, ρ(emb,1−T)
  0.062/0.079/0.127/0.221, scalarness 0.999/0.920/0.730/0.525, control flat at 0.067–0.073.
  8.6 s for 9 runs at n=2500. Adds `tanimoto_partial` (−0.091 at the seed-1 baseline: that
  embedding carries *no* chemistry beyond pProp) and `knn20_jaccard` (0.013 → 0.048).

- `[x]` **S0.4 Gate passed:** `verify_trunk.py` → 10/10, `verify_metrics.py` → 8/8, plus a smoke run
  (`--subset 5000 --freeze-epochs 1 --unfrozen-epochs 1 --w-vic 1 --w-cov 1`) confirming
  `val/vic` moves and `meta.json["config"]` records both new flags.
  **Measured:** `verify_trunk` **11/11** (the gate text above says 10/10; P1 already recorded
  11 — the extra is `grad-flow-has-teeth`, not a regression), `verify_metrics` **8/8**, smoke
  run `val/vic` 0.4423 → 0.4457 with `w_cov: 1.0` and `trunk_weight_decay: null` in
  `meta.json["config"]`.

---

## `[~]` S1 — Stage 1: the screen (24 runs, ~2.2 h)

**Launched 2026-08-18 23:17 as `nohup bash scripts/run_rank_v1.sh`** — the loop below, made
durable as a script (the plan's inline version dies with the SSH session). Logs:
`/home/ethan2/logs/rank_v1_driver.log` (one line per run) and `rank_v1_gpu{0,1,2}.log`.

A **randomized complete block design**: every cell on the same seeds `{0,1,2}`, fold 0.

There is deliberately **no separate noise-floor stage**. The error term for a paired contrast is
the *seed × config interaction*, which a single-config seed scan cannot estimate — it measures
σ_seed instead, 4–5× larger and the wrong quantity. An 8×3 RCBD gives **14 residual df** free,
versus 2 df from any isolated pairwise comparison.

| cell | `w_vic` | `vic_gamma` | `dropout` | decides | done |
|---|---|---|---|---|---|
| `A_base` | 0 | 0.5 | 0 | control / block anchor | `[ ]` |
| `B_w0.3` | 0.3 | 0.5 | 0 | replicate the known 9.46 at n=3 | `[ ]` |
| `C_w1` | 1.0 | 0.5 | 0 | the dose §7 wrongly forbade | `[ ]` |
| `D_w3` | 3.0 | 0.5 | 0 | strong dose | `[ ]` |
| `E_w10` | 10.0 | 0.5 | 0 | ceiling probe — expected to trip a guard, so the ceiling is *measured* | `[ ]` |
| `F_w3_g1` | 3.0 | 1.0 | 0 | higher std target, where force is adequate | `[ ]` |
| `G_w3_drop` | 3.0 | 0.5 | 0.2 | dropout × vic interaction | `[ ]` |
| `H_drop` | 0 | 0.5 | 0.2 | dropout's main effect on rank | `[ ]` |

**Deliberately absent:** `weight_decay` (P6.3 predicts a null below ~0.1; no `n` in this budget
resolves a null — one `wd=0.1` probe sits in S2), and any `w_vic` < 0.3 (P5 showed it
unresolvable against seed noise; do not re-buy that measurement).

**Why γ=1.0 only at `w_vic=3`.** γ does *not* change the hinge's gradient (`relu(γ−σ)` has slope
−1 for every dim below γ), so it is a **scale target, not a strength knob**. Its real effect is
indirect: the covariance half scales as ~s⁴, so lifting the operating scale 0.5 → 1.0 multiplies
it ~16×. But P6.2 shows the model is already force-starved at γ=0.5 (28/32 dims below target,
trace 5.34 against the 8.0 that "all dims at 0.5" needs). Testing γ=1.0 at low `w_vic` would
read as "γ does nothing" for the wrong reason.

**Why dropout gets two cells.** `_mlp_blocks` appends `Dropout` as the last layer of *every*
hidden block including the bottleneck, and `forward_with_embedding` returns `self.shared(x)` —
so the `z` fed to `variance_covariance_loss` is **post-dropout during training**. With inverted
scaling, `Var(z') = Var(z)/(1−p) + E[z]²·p/(1−p)` while covariance is unchanged. Measured
`mean|dim mean| ≈ 0.4`, so a dead dim (std 0.017) shows an apparent std of ~0.20 at p=0.2 —
inflated ~12× out of its own mean. **Falsifiable prediction: dropout weakens the hinge, making
rank no better and plausibly worse at fixed `w_vic`.** The val-time metric is honest (`eval()`
makes dropout identity), so this corrupts the *loss*, not the measurement. It is also the
strongest argument against pinning `w_vic` while `bayes_v2` sweeps dropout 0.05–0.35.

### Execution

**Drive `train.py` directly, one process per GPU, 3 in flight.** `run_config.py` trains
in-process *sequentially*, so it would serialize each config's seeds — the opposite of what a
screen wants. Driving `train.py` and reading `meta.json` directly also defuses all five known
`run_config` traps at once (stale `already_done` reuse that ignores `val_embeddings.npy`; the
`f{fold}s{seed}` label collision; `final/*_std` conflating folds and seeds). Reserve
`run_config.py --aggregate-only` for the winner's final 5×2, where pooling and wandb are wanted.

**Buckets: human-readable**, `outputs/rank_v1/<cell>/fold{F}_seed{S}`. `config_id` buys nothing
here (it changes the moment `--w-cov` lands) and costs legibility. **The analysis reads
hyperparameters from `meta.json["config"]`, never from the directory name** — so a mislabelled
directory is a caught error, not a silent one.

**Ordering: seed-major** — all 8 cells at seed 0, then seed 1, then seed 2. Stopping early then
leaves a complete unreplicated block rather than three finished cells and five empty ones.

```bash
cd /home/ethan2/finetune_minimol
VENV=/home/ethan2/finetune_minimol/.venv/bin/python
ROOT=outputs/rank_v1
CELLS=(
 "A_base|--w-vic 0   --vic-gamma 0.5"
 "B_w0.3|--w-vic 0.3 --vic-gamma 0.5"
 "C_w1|--w-vic 1     --vic-gamma 0.5"
 "D_w3|--w-vic 3     --vic-gamma 0.5"
 "E_w10|--w-vic 10   --vic-gamma 0.5"
 "F_w3_g1|--w-vic 3  --vic-gamma 1.0"
 "G_w3_drop|--w-vic 3 --vic-gamma 0.5 --dropout 0.2"
 "H_drop|--w-vic 0   --vic-gamma 0.5 --dropout 0.2"
)
for S in 0 1 2; do
  for G in 0 1 2; do
   ( for i in "${!CELLS[@]}"; do
       [ $(( i % 3 )) -eq $G ] || continue
       NAME=${CELLS[$i]%%|*}; FLAGS=${CELLS[$i]#*|}
       OUT=$ROOT/$NAME/fold0_seed$S
       [ -f "$OUT/val_embeddings.npy" ] && continue   # resume on the file already_done forgets
       CUDA_VISIBLE_DEVICES=$G $VENV src/train.py --fold 0 --seed $S $FLAGS \
         --no-wandb --no-save-checkpoint --out $OUT >> /home/ethan2/logs/rank_v1_gpu$G.log 2>&1
     done ) &
  done
  wait                                                 # one complete seed block at a time
done
```

`train.py` has no `--device`; `CUDA_VISIBLE_DEVICES` is the pin. `--no-wandb` throughout —
`meta.json` holds everything.

---

## `[ ]` S2 — Stage 2: dose and refine (~16 runs, ~1.5 h)

Shape committed now, values after S1.

| block | runs | done |
|---|---|---|
| upgrade best cell + control `A` to n=5 (seeds 3, 4) | 4 | `[ ]` |
| dose refinement along whichever axis cleared — bracket the best two `w_vic`, or γ at the winning dose | 9 | `[ ]` |
| one `wd=0.1` null probe (≈35% shrink) at n=3, making P6.3 empirical rather than only arithmetic | 3 | `[ ]` |

If S1 shows per-dim stds piling up **at** γ at high `w_vic` (hinge finally saturated), spend the
refinement block on `--w-cov ∈ {4, 16}` instead — that is the case the flag was added for.

---

## `[ ]` S3 — Stage 3: transferability (~12 runs, ~2 h)

**This stage decides pin-vs-sweep**, so every corner runs as a *pair*: chosen config vs control.

| corner | why | runs | done |
|---|---|---|---|
| `--unfrozen-epochs 40` | rank was still climbing at 20; the sweep goes to 40 | 4 | `[ ]` |
| `--dropout 0.3` | top of the swept range, plus the S1 dropout confound | 4 | `[ ]` |
| folds 1 and 2, seed 0 | the only fold-variance measurement, taken at the config that matters | 4 | `[ ]` |

**Reserve (~8):** `trunk_lr ∈ {3e-5, 3e-4}` paired corner.

**Follow-on, not in budget:** the winner re-run at 5 folds × 2 seeds with `--keep-checkpoints`.
`run_config.py` forces `--no-save-checkpoint`, so **nothing on disk today is exportable** to the
DKL project.

---

## Analysis — `src/emb_readout.py` (new)

One CSV row per run: cell, fold, seed, every hyperparameter from `meta.json["config"]`, every
`val/*` from `history[-1]`, plus:

| column | definition |
|---|---|
| `tanimoto_spearman` | ρ(emb_dist, 1−T) — structural information |
| `pred_tanimoto_spearman` | ρ(\|Δpred\|, 1−T) — **the control**, flat at ~0.07 |
| `tanimoto_partial` | ρ of rank-residuals of emb_dist on \|Δpred\| vs rank(1−T) — information *beyond* the prediction |
| `scalarness` | ρ(emb_dist, \|Δpred\|) — the failure mode itself |
| `knn20_jaccard` | mean overlap of each molecule's 20 nearest by ECFP vs by embedding — what a GP kernel actually uses |
| geometry | `eff_rank`, `top1_share`, `trace`, per-dim std percentiles, `n_dims_below_0.1`, `n_dims_below_gamma` |

**Mechanics.** Sample 5,000 molecules per fold **by dataset row id, not position**: sort
`val_indices.npy`, draw with `default_rng(20260818 + fold)`, map back per run via
`np.searchsorted`, and `assert` the run's indices match the reference. Unpack fingerprints with
`np.unpackbits(fp[rows], axis=1)`. Tanimoto by Gram matrix (`inter = B@B.T`); distances by the
`x²+y²−2xy` trick — never an `[n,n,32]` broadcast. **Cache `T` and `rank(1−T)` per fold**, not
per run; `rankdata` over millions of pairs is the expensive step.

**Spearman throughout** — Euclidean-vs-Tanimoto is monotone at best and Tanimoto is heavily
skewed (median 0.127). Report global ρ *and* `knn20_jaccard` (stratification by construction,
no bins, directly interpretable for the DKL consumer); treat disagreement as a finding. **Do not
quote pair-count standard errors** — millions of pairs are not independent; effective n is the
molecule count, and run-to-run variability comes from the seed replicates like every other
metric here.

**Statistics.** Response is **`log(emb_effective_rank)`** at the final epoch — seed effects are
multiplicative (1.32 vs 2.85 at one config) and the target is "3× better", not "+6 units".
Two-way ANOVA cell × seed; σ = √(residual MS), df = 14; contrast vs `A` with `SE = σ√(2/n)`.

At n=3 with pooled σ this detects **+23% rank** (95% CI excluding 0) and has 80% power for
**+35%**. For scale, baseline → `w_vic=0.3` is ≈10σ. **n=3 is ample for every effect that
matters — buy doses, not replicates.**

---

## Pre-registered decision rules

Written before S1 runs, so the analysis is not post-hoc.

**A cell "clears" iff all four hold:**
1. paired mean Δlog(rank) vs `A` exceeds `2.145 · σ · √(2/n)`;
2. **all n seeds move the same direction** (blocks one chaotic trajectory carrying a cell);
3. both damage guards pass;
4. `tanimoto_partial` increases — *rank without structural information is a noise embedding and
   does not count.*

**Damage guards.** The stated success bar was "no `goal_metric` cost". `goal_metric` cannot
serve as that test: its entire observed spread (0.9598–1.0065) is carried by `ap_uniform` over
~620 positives on one fold, while `pearson_uniform` moves only 0.002 across the same eight runs.
These guards serve the same intent with metrics that can actually detect a cost:

- **Primary:** paired non-inferiority on `pearson_uniform`, margin **0.005** (2.5× its observed
  full spread) — lower 95% bound of paired Δ must exceed −0.005.
- **Secondary:** paired Δ`mse`, upper bound < **+0.010** (5% of 0.185).
- `goal_metric` reported as a **two-sided bound**, never as a claim.

**Best config** = highest `tanimoto_partial` among clearing cells — *not* highest rank. Rank is
the proxy; the readout is the thing.

**Diagnosis table** (per-dim stds and trace dumped for every run):

| observation | conclusion | action |
|---|---|---|
| stds pile up **below** γ | force-limited | raise `w_vic` |
| stds sit **at** γ, rank still low | covariance binding | raise `--w-cov` (or γ) |
| a stable subset stuck near 0.06 at every dose | GELU-dead dims | **architecture**, not loss |

---

## `[ ]` S4 — Deliverable: `sweeps/bayes_v2.yaml`, decided in advance

1. **Pin `vic_gamma` and `w_cov`** at the winner's values. Neither trades against `goal_metric`
   in a way bayes can read (§9.2 stands), so sweeping them wastes trials.
2. **Pin `w_vic`** iff S3 shows the paired Δlog(rank) at each corner within 2σ of the
   fold-0 / dropout-0 / 20-epoch value.
3. **If unstable, still pin — at the value that suffices in the *worst* corner.** Do **not** add
   `w_vic` as a sweep axis: a bayes run maximizing `goal_metric` drives it to the floor of
   whatever range it is given, so "sweep [floor, ceiling]" is an expensive way to write "pin at
   floor". Delete the yaml's "most important axis in the file" note.
4. **`weight_decay` stays pinned at 0.01**, with the `lr·wd·steps ≈ 2–4%` arithmetic replacing
   the current "least safe change in this file" caveat — that caveat describes a mechanism that
   does not operate at these learning rates.
5. **`dropout`:** if `G`/`H` show it degrades rank at fixed `w_vic`, the yaml must say so and
   either lower its ceiling or record that `w_vic` was pinned at a dropout-robust value. Two
   axes that interact through the loss cannot be documented independently.

**"The loss cannot fix this; the architecture must change"** fires if any of:
- at the strongest dose, stds sit at γ, a stable subset remains near 0.06, and rank plateaus
  **below ~12** — that is the LayerNorm scale budget plus GELU-dead dims, and no loss
  coefficient reaches it;
- rank rises while `tanimoto_partial` stays flat — the term is manufacturing orthogonal noise;
- every clearing cell fails a damage guard — the trade is real and 15–25 is unreachable via the
  loss.

Escalation order if it fires: (a) export pre-GELU or change the activation (kills the dead-dim
mechanism); (b) renegotiate `embed_dim` with the DKL project; (c) add an information-bearing
auxiliary target — a `32 → 2048` ECFP-bit decoder, noting the readout must then move to a
fingerprint family the decoder never saw.

---

## Verification

- **S0 gate:** `verify_trunk.py` 10/10 and `verify_metrics.py` 8/8 after the `losses.py` edit —
  the only automated protection on that file. Nothing currently tests
  `variance_covariance_loss` or `embedding_metrics` at all.
- **Bit-identity:** one run at `--w-cov 1` must reproduce an existing `w_vic=0.3` run's final
  rank to within run-to-run nondeterminism (~0.07), proving the refactor changed nothing.
- **Readout validation:** `emb_readout.py` must reproduce P6.4's table from runs already on disk
  before any new run is scored.
- **Per-cell sanity:** every run writes `val_embeddings.npy` at `[66296, 32]`; the resume guard
  keys on that file.
- **End-to-end:** the analysis reproduces `B_w0.3` ≈ 9.46 at seed 0, matching P4.

---

## Cost

| stage | runs | wall (3 GPUs) |
|---|---|---|
| S0 — corrections + code | 0 | ~1 h |
| S1 — screen (8 cells × 3 seeds) | 24 | ~2.2 h |
| S2 — dose / refine | 16 | ~1.5 h |
| S3 — transferability | 12 | ~2 h |
| reserve | 8 | — |
| **total** | **52 + 8** | **~5.5–6 h** |

Later stages are deliberately *not* fully specified — a staged design whose cells depend on
earlier results cannot be committed up front.

---

## Unattended execution — OPEN, unanswered

Ethan asked whether this can run to completion after disconnecting from rabelais.

**What survives:** the GPU work. `nohup`-ed processes outlive the SSH session (that is how P4
and P5 ran). A single driver script can run every specified cell, the readout, and the analysis
unattended.

**What does not:** the model. S2's values and S3's chosen config currently require judgment at
the stage boundaries.

The pre-registered rules above are **mechanical by design** — that is what pre-registration is
for — so they *can* be coded, letting the driver select the winner itself. Three options, not
yet chosen:

1. **Code the pre-registered rules** — driver runs S1, computes ANOVA/contrasts/guards, applies
   the rules, launches S2–S3 on its own choice. Fully autonomous, ~6 h, faithful to the design.
   Risk: a bug in the selection logic sends S2–S3 down a wrong branch — though S1's results
   survive regardless and the script logs its reasoning.
2. **S1 only, then stop** — 24 runs + analysis (~2.5 h), results written, halt. Zero wasted
   compute; S2–S3 wait for review.
3. **Fixed grid, no adaptivity** — pre-specify all ~52 cells and run regardless of intermediate
   results. No selection logic to get wrong, but spends runs on axes S1 might have ruled out,
   and S3 pairs against an a-priori config guess rather than the measured winner.

Also undecided: whether an unattended run may edit tracked files (`CLAUDE.md`,
`sweeps/bayes_v2.yaml`) or should write only to `outputs/rank_v1/**` and
`reports/rank_v1_results.md`. **Default assumption until told otherwise: results files only.**
