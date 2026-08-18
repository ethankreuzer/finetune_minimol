# finetune_minimol

Fine-tune the **entire MiniMol trunk** (~10M params) on AmpC docking-score regression —
not the usual frozen-embedding + MLP workflow. Gradients must reach all trunk parameters.

`NOTES.md` is the reference document: background, source-level findings, and the full plan.
This file is the operational summary. When they disagree, `NOTES.md` §§1–11 is authoritative
on *why*; this file is authoritative on *what currently exists*.

---

## NEXT: what to run on rabelais

**Nothing in the 2026-08-17/18 work has been run on real data.** It was written on a MacBook
with no `.venv` and no `data/`, so only the parts that need neither (pure `torch.nn` shape and
loss algebra, 17/17 checks) are verified. Everything below is unrun.

```bash
cd /home/ethan2/finetune_minimol && git pull
VENV=/home/ethan2/finetune_minimol/.venv/bin/python
```

**1 — the two verification suites.** Fast, no training, and they gate everything else.

```bash
$VENV src/verify_trunk.py   -o verification.md            # expect 10/10
$VENV src/verify_metrics.py -o verification_metrics.md    # expect 8/8
```

`verify_metrics` is the one at risk: the head was reshaped and a fifth loss term added, so its
loss-parity check (`combined_loss` == exactly half `weighted_mse_loss`) and its freeze-assertion
negative test are the checks that would catch a mistake. Parity was confirmed locally at
`0.500000000000` with `w_vic=0`, but not against the real fixtures.

**2 — smoke run.** Confirms the new artifact and metric appear at all.

```bash
$VENV src/train.py --freeze-epochs 1 --unfrozen-epochs 1 --subset 5000 --no-wandb
```

Check `val_embeddings.npy` exists at `[n_val, 32]` and `val/emb_effective_rank` is in
`meta.json`'s `history`.

**3 — THE MEASUREMENT. This is the point of the whole architecture change.**

```bash
$VENV src/train.py --fold 0 --seed 0
```

Read `emb_effective_rank` at the final epoch:

| value | meaning | what to do |
|---|---|---|
| **15–25** | the architecture is sufficient on its own | leave `--w-vic 0`, go to step 5 |
| **2–4** | the predicted collapse; the GP would receive a disguised scalar | do step 4 |

Record the number here in CLAUDE.md either way — that is the repo convention, and this is the
single most decision-relevant number outstanding.

**4 — CONDITIONAL, only if step 3 comes back low.** A 1-D scan, not a sweep — see
`reports/embedding_geometry.md` §9.2 for why `w_vic` cannot be swept against `goal_metric`.

```bash
for W in 0 0.003 0.01 0.03 0.1 0.3; do
  $VENV src/train.py --fold 0 --seed 0 --w-vic $W --vic-gamma 0.5
done
```

Note `--vic-gamma 0.5`, **not** the shipped default of 1.0 — see the open items below. Read off
the `emb_effective_rank` vs `goal_metric` trade curve and pick a point on it.

**5 — register the sweep.** Creates a real sweep on wandb.

```bash
wandb sweep --project finetune_minimol --entity ethan_personal sweeps/bayes_v2.yaml
python -m wandb agent ethan_personal/finetune_minimol/<sweep_id>
```

**Do not do step 5 before step 3.** If the embedding collapses and `w_vic` must be non-zero,
the loss changes and so does every `config_id` — a sweep launched first would be invalidated
and its trials unusable. Steps 3–4 settle the architecture; the sweep explores hyperparameters
within it.

### Open items, deliberately not fixed

Both are argued from measurement in `reports/embedding_geometry.md` §9. Neither is active
today, because `--w-vic` defaults to `0.0`.

1. **`--vic-gamma` defaults to 1.0 and should be 0.5.** Measured: a healthy bottleneck's
   per-dimension std is **0.585** post-LayerNorm+GELU, so the `gamma=1.0` hinge fires
   (`vic` = 0.421) even at effective rank 30.78/32. At 0.5 it reads 0.0057 — correctly inert —
   and loses ~1% sensitivity at true collapse. Pass `--vic-gamma 0.5` explicitly until the
   default is changed.
2. **`sweeps/bayes_v2.yaml` calls `w_vic` "the most important axis in the file".** That is
   wrong. `vic` trades prediction quality for embedding quality, so a sweep maximising
   `goal_metric` drives it to 0 regardless of collapse. It is a constrained 1-D choice
   (step 4), not a sweep axis.

Also note: `run_config.py` forces `--no-save-checkpoint`, so **sweep trials leave no weights on
disk.** The winning configuration must be re-run with `--keep-checkpoints` before anything can
be exported to the downstream DKL project.

---

## State as of 2026-08-18

| Piece | Status |
|---|---|
| Dataset subset (331,480 molecules) | **done** — `src/subset.py` |
| 5-fold cluster CV splits | **done, verified, frozen** — `src/split.py`, `src/splits.py` |
| Environment (uv, in-repo `.venv`) | **done, verified** — `pyproject.toml` + `uv.lock` |
| Trainable trunk | **done, 10/10 checks pass** — `src/trunk.py`, `src/verify_trunk.py` |
| Feature cache (Phase 2) | **done, verified** — `src/featurize.py`, `src/features.py` |
| Prototype training run | **done, verified** — `src/train.py`, 5 epochs, fold 0 seed 0 |
| Loss / metrics / objective ported from `pProp_MLP` | **done, 8/8 checks pass** — `src/losses.py`, `src/metrics.py`, `src/objective.py`, `src/verify_metrics.py` |
| Dual head (binary @ 3.5 + regression) | **done** — `head.DualHead` |
| Pooled out-of-fold tail metrics | **written, code path exercised; no real runs pooled yet** — `src/pool_oof.py` |
| Compute profile / benchmarks | **done** — `src/benchmark.py`, `reports/compute_profile.md` |
| 5×2 grid runner / SLURM | `scripts/run_grid.sbatch` exists; sweep driver not started |
| Config-level runner (1 config = 1 wandb run) | **done, verified 2026-08-13** — `src/run_config.py` |
| Hyperparameter sweep (wandb bayes) | `bayes_v1` ran as test sweep `c63i6zoh` (6 of 9 trials finished). **`sweeps/bayes_v2.yaml` written 2026-08-17, not launched** |
| pProp_MLP range transfer | **done, 2,247 runs analysed** — `reports/pprop_mlp_transfer.md`, `reports/pprop_mlp_transfer.py` |
| 32-d embedding head (the deliverable) | **built 2026-08-17, 17/17 local checks; NOT yet run on real data** — `head.DualHead` |
| Embedding geometry term (`--w-vic`) | **implemented, inert by default** — `losses.variance_covariance_loss`, `reports/embedding_geometry.md` |
| `emb_effective_rank` on real data | **NOT MEASURED — this is the next thing to do.** See the runbook at the top |
| Layer-wise freeze/unfreeze | **not started, deferred 2026-08-13** — see "Deferred: layer-wise freeze/unfreeze" |

The trunk reproduces frozen MiniMol embeddings **exactly** (max|Δ| = 0.000e+00 over 64×512),
gradients reach all 284 reachable trunk tensors (7,919,912 params), and an optimizer step
moves trunk weights. Two further tensors are unreachable inside graphium itself — see "The
rw_pos dead norm" below. `verification.md` reads `OVERALL: PASS`.

Commit `c00bc4e` tracks the env (`pyproject.toml` + `uv.lock`), the trunk, the feature cache
code and training, so `git clone` + `uv sync` reproduces on TamIA.

---

## Environments

Conda lives at `/home/ethan2/local/conda`. **Do not `conda activate` in tooling** — call
interpreters by absolute path.

```bash
# data / splits work — has rdkit 2024.03.5, numpy 2.2.6, pandas 2.3.0, matplotlib
/home/ethan2/local/conda/envs/my_conda_env/bin/python

# model work — the uv venv in this repo. Recreate with `uv sync` (see below).
/home/ethan2/finetune_minimol/.venv/bin/python
```

Do not install minimol into `my_conda_env` — graphium pins torch/PyG tightly enough to
break it. The old `minimol_ft` conda env is **dead** — a `pip install minimol` there
resolved torch 2.13.0/CUDA-13 over the pinned cu124 build; delete it, do not use it.

### The uv environment

`pyproject.toml` + `uv.lock` are the source of truth. On TamIA (login nodes have internet):
`git clone` then `uv sync` — no wheelhouse needed.

```bash
UV_HTTP_TIMEOUT=3600 uv sync --extra dev     # timeout matters: see below
```

Pinned stack: **Python 3.11**, `torch 2.6.0+cu124` (arch list includes **sm_90**, so it
covers TamIA's H100/H200), `torch-scatter/sparse/cluster` at `+pt26cu124` from an explicit
`data.pyg.org` flat index, graphium 2.4.7, minimol 1.3.4.

NOTES §6.1 says torch 2.5.1+cu124 — **superseded**. 2.6.0 is equivalent for sm_90 and is
what this box's uv cache already held, which on a 0.5 MB/s link is the whole ballgame.

Three pins exist only to hold back the modern world, and removing any of them breaks the
env silently rather than loudly:

- **`torch==` and `torchvision==` exact.** graphium declares *no* torch dependency (only
  the PyG extensions), so an unpinned resolve picks the newest torch and orphans the
  compiled extensions. This actually happened.
- **`setuptools<81`.** graphium pins `torchmetrics<0.11`, which imports `pkg_resources`;
  setuptools 81 deprecated it and 84 removed it. uv venvs ship no setuptools at all.
- **`scipy<1.14`.** graphium's featurizer builds the adjacency matrix as float16 and passes
  it to `scipy.sparse.coo_matrix`; newer scipy enforces a dtype whitelist excluding float16.
  Measured per version: 1.13.1 ✅, 1.14.1 ✅, 1.15.3 ❌, 1.17.1 ❌ — so the real boundary is
  **1.15**, and `<1.15` would also be correct. Pinned one lower because 1.13.1 is what the
  verification actually ran under. This pin drags numpy to 2.2.6, matching `my_conda_env`.

**Always read `uv.lock` before syncing** — grep for `cu13` and `cuda-toolkit`; both must be
absent and every `nvidia-*` must be on the cu12/12.4.x line. Resolution is metadata-only and
costs nothing, so the audit is free; the sync is ~1 hour.

**`UV_HTTP_TIMEOUT=3600` is mandatory here.** uv's default is 30s, shorter than a ~780 MB
torch wheel takes at 0.5 MB/s. It fails as `operation timed out (Connect)`, which reads like
an unreachable host and is not.

`pyarrow` is **not** installed anywhere here, which is why artifacts are CSV + `.npy`, never
parquet. Machine has 128 cores and 500 GB RAM; the split pipeline is CPU-only.

---

## Layout

```
data/
  ampc_unif_random_10M*.csv          source docking data (695 MB each, untracked)
  ampc_subset_331k.csv               the training set — 331,480 rows
  ampc_subset_331k.meta.json         provenance for the above
  splits/cluster_kfold_v1/           the frozen CV splits (see below)
  split_v1/                          empty leftover, superseded — ignore
  features/minimol_v1/               cached PyG graphs, 4.8 GB (regenerable, untracked)
src/
  subset.py                          10M -> 331k enriched subset
  split.py                           clusters + folds + diagnostics
  splits.py                          loader with provenance guard  <- use this
  featurize.py                       SMILES -> cached graphs, once
  features.py                        cache loader with provenance guard  <- use this
  fold_histograms.py                 per-fold pProp distribution PNGs
  cluster_histograms.py              ORPHANED — per-cluster version, superseded

  trunk.py / model.py / head.py      the trainable trunk, the two-group optimizer, DualHead
  train.py                           one fold, one seed  <- the entry point
  losses.py                          5-term loss (vic off by default) + the weighting flavours
  metrics.py                         AP / correlation / error / enrichment, all suffixed
  objective.py                       goal_metric + the derived OBJECTIVE_VERSION
  normalization.py                   --pprop-norm; ported verbatim
  run_paths.py                       outputs/<sweep_id>/<run_id>  <- never hardcode a path
  pool_oof.py                        pooled out-of-fold tail metrics across the 5 folds
  run_config.py                      one hyperparameter config -> ONE wandb run  <- sweep entry
  verify_trunk.py / verify_metrics.py    the two verification suites
  dump_metric_reference.py           run under pProp_MLP's venv; feeds verify_metrics
  benchmark.py / concurrency.py / collect_runs.py / report_charts.py   the compute profile
sweeps/
  bayes_v1.yaml                      the wandb bayes sweep over the two-phase schedule
scripts/
  run_grid.sbatch                    the 5×2 grid as a SLURM array (gpu:1, NOT mps)
  sample_gpu.sh                      nvidia-smi telemetry sampler
reports/                             compute_profile.md + its evidence
NOTES.md                             the reference document; §12 is the pProp_MLP translation
```

---

## The data

`data/ampc_subset_331k.csv` — columns `SMILES, score, pprop, bin, ipw`.

- `score` — AmpC docking score, kcal/mol, **more negative is better**
- `pprop` — `-log10(rank_max / 1e7)`, a quantile; monotone in `score`; max 7.0.
  **This is the regression target** (settled 2026-08-11). `score` is kept for reporting.
- `ipw` — **a record of how the subsample was drawn, not a modelling input** (settled
  2026-08-12). It is the per-bin subsampling rate `subset.py` applied: how many library
  molecules each retained row stands for. `subset.py` kept the whole potent tail and
  subsampled the bulk, so the file is deliberately *not* distributed like the library.
  Measured: **75.0 over `[0,1)`, 7.5 over `[1,2)`, 1.14 over `[2,2.5)`, and exactly 1.0
  everywhere above 2.5** — `--take-all-above 2.5` made that whole region a census, so every
  molecule in the 10M library with pProp ≥ 2.5 is present, and `sum(ipw)` is exactly
  10,000,000. The ≥ 3.5 positive class is therefore complete rather than sampled (`sum(ipw)`
  over its 3,153 rows is 3,154).

  **Nothing in the training or metric code reads this column.** It was briefly a loss
  weighting option and a metric flavour, and was removed from both on 2026-08-12: it
  describes the sampling design, not how the model should be trained or scored. It survives
  in the CSV and in `split.py`'s diagnostics as the provenance of the subsample. The cost of
  removing it is stated plainly below — subset metrics flatter the model by ~30× tail
  enrichment and there is now no reweighting that corrects for it.

**The loss is weighted, and the weighting is `balanced`** — two-group inverse frequency at
pProp 3.5, chosen 2026-08-12 over the alternatives. The training step accepts a
**per-sample weight vector**, never a hardcoded unweighted mean. See NOTES §1.

SMILES are RDKit-canonical (isomeric) and deduplicated. The tail is thin: **3,153** rows at
pProp ≥ 3.5, **100** at ≥ 5.0.

---

## The splits — `data/splits/cluster_kfold_v1/`

5-fold cluster CV. **Precomputed and frozen — load them, never re-derive per run.**

```python
import sys; sys.path.insert(0, "src")
from splits import load_fold, load_assignments, load_fingerprints

train_idx, val_idx = load_fold("data/splits/cluster_kfold_v1", fold=0)
```

Indices are positions into `ampc_subset_331k.csv` row order.

Method: ECFP4/2048 → sphere exclusion (`LeaderPicker`, Tanimoto distance 0.65) → nearest
centroid → 32,254 clusters dealt whole into 5 folds by stratified LPT. Clustering is on
ECFP, deliberately **not** MiniMol embeddings — the partition must not depend on the model
being evaluated.

Result: folds of exactly **66,296** each, with exactly **20** pProp ≥ 5.0 molecules per fold.

Files: `assignments.csv`, `fingerprints.npy` (packed ECFP4, uint8[331480, 256]),
`meta.json`, `diagnostics.md`, `fold_{0..4}_pprop_distribution.png`.

Regenerate (~2.5 min, deterministic, CPU only):

```bash
/home/ethan2/local/conda/envs/my_conda_env/bin/python src/split.py
```

A rerun must reproduce `split_sha256 = 3ef97e78a85d…` in `meta.json`. That hash covers the
`(row_idx, cluster_id, fold)` content — file byte-identity is the wrong criterion.

### How to use them correctly

- **Splits are fixed across model seeds.** The plan is 5 folds × 2 model seeds = 10 runs.
  The seed governs head init, dropout, and shuffling only; it never touches the partition.
  Variation down a column is data variance, across a row is optimisation variance.
- **Pool out-of-fold predictions for tail metrics.** 20 potent molecules per fold is noise.
  The 5 validation folds cover all 331,480 rows exactly once, so pooling gives one honest
  estimate over all 100; two seeds give two such estimates. Per-fold numbers remain right
  for bulk metrics.
- **This is a plain K-fold** — each fold is used for both early stopping and reporting, so
  the CV score is mildly optimistic. Fix the epoch budget instead of early-stopping to
  remove the bias.
- **Describe results as generalisation to new clusters within this library, not to new
  chemistry.** The split removes analog leakage (molecules with a training neighbour ≥ 0.7
  drop 5.15% → 2.11%) but the median nearest-neighbour similarity barely moves. See
  NOTES §11.2 for the pool-size-matched evidence.

---

## Training — `src/train.py`

One fold, one seed, staged fine-tuning in **two phases with independent lengths**: the head
trains alone on the frozen embedding for `--freeze-epochs`, then the trunk unfreezes and both
train together for `--unfrozen-epochs`.

```bash
python src/train.py --fold 0 --seed 0                                       # 5 + 15
python src/train.py --fold 0 --seed 0 --freeze-epochs 20 --unfrozen-epochs 0  # frozen BASELINE
python src/train.py --freeze-epochs 1 --unfrozen-epochs 1 --subset 5000 --no-wandb  # smoke
```

**There is no `--epochs`.** The total is derived as `--freeze-epochs + --unfrozen-epochs` and
stamped into `meta.json` like any other setting. Two lengths rather than a total and a cut
point, because a bayes sweep samples independently and `(epochs, freeze_epochs)` carries the
constraint `freeze_epochs ≤ epochs` — every violating draw would have died mid-sweep. Two
non-negative lengths have no cross-constraint. Changed 2026-08-13; `scripts/run_grid.sbatch`
now takes `FREEZE` / `UNFROZEN`.

`--unfrozen-epochs 0` never unfreezes, so it reproduces the frozen-embedding workflow **on
these splits**. That is the only honest baseline for the fine-tuned arm — `pProp_MLP`'s
numbers were measured on different data and a different partition, and cannot be quoted
against these.

### The deliverable is a 32-d embedding, not a predictive model

**Settled 2026-08-17.** This repo produces a function `SMILES → R³²`, deployed **frozen** into
a separate generative + active-learning project, where a **deep-kernel-learning GP** trains
its own network on top of the 32-d input. MiniMol is never trained again downstream. Training
happens here; there it is fixed.

That inverts what "good" means. `goal_metric` scores the *predictions*, and the predictions
are not the product — they are a training signal for the representation. Treat `goal_metric`
as a proxy and `emb_*` as the thing being optimised.

The head is `512 → 1024 → 1024 → **32** → {Linear(32→1) cls, Linear(32→1) reg}`, **1,611,938
params** (measured). Three choices follow from the deliverable rather than from accuracy:

- **The bottleneck is shared, not per-task.** The deliverable is one vector; a per-task
  bottleneck would be shaped by a single objective, discarding the auxiliary-classifier
  argument in `head.py` that is the classifier's whole justification.
- **The task heads are linear** (`--cls-n-layers 0 --reg-n-layers 0`). This makes "pProp is
  linear in the exported embedding" literally true, which is the geometry an RBF/Matérn kernel
  wants. Deeper branches would likely score better on `goal_metric` while letting the
  bottleneck encode the target in a shape a GP reads poorly.
- **The export point is the output of `head.shared`** — post-LayerNorm, post-GELU, the exact
  tensor the linear heads consume. Exporting the pre-activation would make the target
  linear-*after*-GELU, which is not the property above. Dropout is identity under `eval()`.

**The risk this design manages, and the number that decides it.** Regression and
"pProp ≥ 3.5" are near-collinear — the same axis, the second with its gradient concentrated at
the boundary — so the *supervised* signal reaching the bottleneck is close to rank-1, and
weight decay shrinks whatever else the other ~30 dimensions might hold. The predicted failure
is effective rank 2–4 of 32, which would hand the GP a disguised scalar: distance between
molecules would reduce to difference in predicted pProp, so two unrelated compounds with equal
predicted score become indistinguishable and the posterior variance stops tracking genuine
ignorance — which is the entire mechanism active learning runs on.

`val/emb_effective_rank` is therefore logged every epoch (`metrics.embedding_metrics`:
`exp(entropy)` of the covariance eigenspectrum, so it reads on the same scale as 32).
**Roughly 15–25 is healthy; 2–4 is the failure.** `val_embeddings.npy` is written per run so
it is checkable after the fact. **Not yet measured — take this number first.**

The countermeasure is `--w-vic`, a VICReg-style variance + covariance penalty on the `[B, 32]`
matrix (`losses.variance_covariance_loss`). It adds no target; it forbids dimensions from
being flat or duplicating each other, letting the trunk's own chemical variation occupy them.
**Default 0.0 — inert**, so enabling it is a sweep value rather than a code change. Verified on
synthetic geometry: 0.011 on isotropic 32-d, **14.4 on rank-1**.

`emb_*` is deliberately **not** in `OBJECTIVE_SPEC` — adding it would re-stamp
`OBJECTIVE_VERSION` and make every scored run un-poolable. It is a reported diagnostic, like
`enrichment_factor`.

**Not built yet:** `src/export.py`. Note the head is trivially portable but the trunk is not —
reconstructing it needs graphium 2.4.7, the minimol 1.3.4 wheel, and `trunk.py`'s exact
construction ordering, so the downstream project must vendor `trunk.py`/`head.py`/`model.py`
or put this `src/` on `sys.path`. Also note `run_config.py` forces `--no-save-checkpoint`, so
**sweep runs have no weights on disk** — the winning config must be re-run with
`--keep-checkpoints` before anything can be exported.

### The task is joint, and binary at pProp ≥ 3.5

`head.DualHead` puts a **classification logit** and a **regression scalar** on one shared MLP
over the 512-d embedding. 3.5 is the threshold at which binders become possible, and it is
also the finest split leaving a learnable positive class: **3,153 molecules, 619–643 per
validation fold**, against only **100 in the whole dataset** at pProp ≥ 5.0.

`pProp_MLP`'s 4-class scheme does not survive the move. pProp caps at **7.0** here (rank 1 of
10M), so its `7.5+` artifact class is *impossible* — and `WEIGHT_GROUPS = [0,1,2,0]`, which
existed solely to demote those 46 artifacts, has nothing left to do. It is ported as
`[0, 1]`, an identity grouping, purely so the shape of the argument survives a rebin.

### The loss — `src/losses.py`

```
loss = w_cls · cls  +  huber  +  w_pair · pair  +  w_std · std  +  w_vic · vic
```

`vic` is the bottleneck's anti-collapse term and is **off by default** (`--w-vic 0.0`) — see
"The deliverable" above. The four terms below are the ones that were swept.

Huber is **grounded at weight 1** — the only term anchoring the absolute pProp level, so the
other three are measured against it and only its `delta` is swept. Three deliberate
deviations from the port:

- **`cls` is BCE normalised by `w.sum()`, not by `N`.** `CrossEntropyLoss(weight=w)` divides
  by `Σw`; `BCEWithLogitsLoss(reduction="mean")` divides by `N`. At a 104× class ratio those
  differ by a large constant, which would move every inherited `w_cls` out of the units it
  was swept in. Verified equal to `CrossEntropyLoss(weight=…)` on 2 classes.
- **`std_match_loss` takes the sample weights.** The group-weighted target std is **1.455**
  against **0.864** unweighted, so an unweighted std term would pull prediction spread toward
  the smaller number while the weighted huber pulled toward the larger. They would fight.
- **`pair` stays unweighted**, exactly as ported. It is O(B²) — ~1.4M pairs at batch 1200.

`combined_loss` returns the four term values **unscaled**, and they are logged that way: a
term that has collapsed and a term whose weight is tiny look identical once multiplied.

### Weighting — `--weights {uniform,balanced}`, default `balanced`

Two-group inverse frequency at pProp 3.5: class weights `[0.0190, 1.9810]`, a **104×** ratio.
One threshold drives the classifier, the loss weights and the metric weighting.

**Weights are derived per fold, from that fold's own composition.** A single global vector
sliced per fold would encode the validation fold's tail fraction into the training loss.

Why the edge is 3.5 and not 5.0, measured as effective sample size `(Σw)² / Σw²`:

| edge | groups | ratio | ESS | % of N |
|---|---|---|---|---|
| 3.0 | 321,502 / 9,978 | 32× | 38,711 | 11.68% |
| **3.5** | **328,327 / 3,153** | **104×** | **12,492** | **3.77%** |
| 4.0 | 330,484 / 996 | 332× | 3,972 | 1.20% |
| 5.0 | 331,380 / 100 | 3,314× | **400** | **0.12%** |

At 5.0 the scheme would train on an effective 400 molecules. For scale, `pProp_MLP`'s scheme
retained **18.5%** — the same idea costs ~5× more here because the tail is 138× thinner.

That is also why `--batch-size` defaults to **1200**: ~11.4 positives per batch instead of
2.4 at 256, so the up-weighted half of the loss is not riding on two or three molecules.
`reports/compute_profile.md` independently wants ~1024 for throughput, so both arguments
agree. A `--weight-cap` is deliberately **not** implemented — add one only if the
fold-to-fold spread on tail metrics turns out large.

### Learning rates — one cosine per phase, three peaks

**Each phase anneals its own cosine, from its own peak down to `--eta-min` (1e-8).** Three
peaks, because the head has one in each phase and the trunk exists only in the second:

| knob | phase | default |
|---|---|---|
| `--head-lr` | 1 (head only) | 1e-3 |
| `--head-lr-unfrozen` | 2 (trunk + head) | falls back to `--head-lr`; expected lower |
| `--trunk-lr` | 2 only — it is 0 in phase 1 by construction | 1e-4 |

The phases are **one trajectory, not two runs**: weights carry straight over, and phase 1's
final weights *are* phase 2's initialization. What restarts at the boundary is the learning
rate, deliberately — a warm restart inside a continuous run. So `e1` and `head_lr` matter
almost entirely through the head they hand over, which is why phase 1's own val numbers are
not a result (except at `--unfrozen-epochs 0`, where phase 1 is the whole run).

Before 2026-08-13 a single cosine ran over the whole run, and the trunk unfroze into it
already partly annealed — at freeze 5 / total 20 it *began* phase 2 at ~85% of `--trunk-lr`
and fell from there, having "decayed" through five epochs in which its lr was pinned to 0.
That was the head's schedule with a hole in it.

**The denominator is `length - 1`, deliberately unlike torch.** `CosineAnnealingLR(T_max=N)`
puts the last epoch at `t = (N-1)/N`, which never reaches `eta_min`: at N = 20 that is ~0.6%
of base and passes for converged, but at N = 3 it is **25% of base** and does not. Phase
lengths are swept down to 1, so the torch form would leave short phases ending hot.
`t = (epoch-1)/(length-1)` lands the final epoch of every phase exactly on `eta_min`; a
length-1 phase cannot anneal and stays at `base`. Measured (freeze 3 / unfrozen 5): head
1.000e-3 → 5.000e-4 → **1e-8**, then restart 1.000e-3 → … → **1e-8**, trunk 1.000e-4 → …
→ **1e-8**.

The cost is that **the last epoch of a phase trains at ~zero lr**, which is what "settled"
means but is pure waste at short lengths — a 2-epoch phase is a 1-epoch phase with a no-op
appended (verified: identical val metrics at epochs 1 and 2 of a length-2 phase). Read
`freeze_epochs = 2` in a sweep result as `1`.

**The provenance triple does not cover this.** `objective_version`, `split_sha256` and
`input_sha256` are all unchanged by the schedule rewrite, so a run from before 2026-08-13 and
one from after pass every provenance filter and are still not comparable — they were trained
under different schedule shapes. The fold-0 prototype in this file is the only such run, and
it is quoted as a direction, not a number.

**Annealing is load-bearing given final-epoch selection.** With no early stopping, the final
epoch is what gets scored and saved; annealed to `eta_min` that is a settled model, whereas
at a live LR it is an arbitrary point on a still-moving trajectory. `pProp_MLP` measured
final-epoch selection as costing 0.003–0.005 of `goal_metric` (30/30 of its top-30 runs
peaked earlier) — without annealing that figure would be a floor here, not an estimate. It
applies to phase 1 too: at `--unfrozen-epochs 0` phase 1 is terminal, and the baseline has to
be as settled as the arm it is compared against.

The schedule is computed in closed form (`scheduled_lr`) rather than delegated to a torch
scheduler, because a scheduler writes `group["lr"]` every step and would silently fight the
freeze, which writes the same field. **`apply_lrs` is the single authority**, and it
delegates the trunk to `set_trunk_trainable` so the freeze keeps exactly one implementation —
which is also what keeps `verify_metrics.py`'s negative test meaningful, since that test
neuters `set_trunk_trainable` and expects the run to die. `phase_bounds` is the one place the
phase boundary is arithmetic rather than a comparison, so a schedule and a freeze cannot
disagree about which epoch belongs to which phase.

**AdamW state is not symmetric across the boundary.** Frozen params get no `.grad`, so AdamW
never creates a state entry for them: at unfreeze the trunk starts at `step = 0` with zero
moments, and bias correction makes its first step ≈ `trunk_lr` per coordinate regardless of
gradient magnitude, while the head carries warm moments across. That is the mechanism a
warmup at unfreeze would address. **Not implemented, on purpose** — the fold-0 prototype went
val MSE 0.2430 → 0.1896 in the first unfrozen epoch at `trunk_lr 1e-4`, which is the opposite
of a forgetting collapse. Mechanism identified, problem not observed; adding it now would buy
a sweep dimension no measurement asks for.

## One configuration = one wandb run — `src/run_config.py`

`train.py` trains **one model**. `run_config.py` trains a **configuration** — every
`(fold, seed)` pair — and logs all of them as a **single** wandb run.

```bash
python src/run_config.py --fold-list 0 1 2 3 4 --seed-list 0 1     # the full 5×2 grid
python src/run_config.py --fold-list 0 --seed-list 0 --head-lr 3e-4  # one cheap trial
python src/run_config.py --aggregate-only outputs/_no_sweep/<cid>    # log, do not train
```

Before this (2026-08-13) the 5×2 design produced **ten** wandb runs per configuration,
separated only by tags, and nothing on wandb answered *which configuration is best* — comparing
two meant eyeballing ten rows against ten others. Now the runs table has one row per
configuration.

What the run holds: per-model curves (`models/f{fold}s{seed}/val/*`), the across-model curve
(`agg/val/{metric}_mean`, `_std`), a `wandb.Table` of every model's final epoch, pooled
out-of-fold metrics (`pooled/*`), and the summary (`final/{metric}_{mean,std,min,max}`).
Everything logged is also written to `<bucket>/aggregate.json`, so `WANDB_MODE=offline` loses
nothing and the numbers can be checked without wandb in the loop.

- **Models go through disk, deliberately.** Each is trained by calling `train.main()`
  in-process with `--no-wandb`, then read back from the files it wrote — the same files
  `pool_oof.py` scores. An in-memory path would be a second implementation that could drift
  from the offline one. It also makes `--aggregate-only` free, which is what allows a SLURM
  array to train the ten models in parallel and aggregate afterwards — the shape
  `reports/compute_profile.md` says TamIA actually wants (whole-node, 4 GPUs, ≥1 h jobs).
- **Ten trunks in one process is safe** only because `trunk.py` uses `OmegaConf.load` rather
  than `hydra.initialize`. Hydra's once-per-process singleton would make this script
  impossible; see NOTES §9 and the `trunk.py` docstring.
- **`config_id`** is an 8-char hash of the hyperparameters — same construction as
  `OBJECTIVE_VERSION` — and names the bucket `outputs/<sweep_id>/<config_id>/`. Fold and seed
  lists, paths, `--subset` and `--num-workers` are excluded, so a cheap 1-model search trial
  and the full 5×2 confirmation at the same hyperparameters **share a bucket**. A complete
  model directory with a matching provenance triple is reused, not retrained (`--force`
  overrides). That is why widening the grid at the winner costs only the models it adds — and
  why `final/n_models` can exceed `len(fold_list) × len(seed_list)`: the bucket *is* the
  configuration, so a cheap trial aggregates every model that configuration has ever trained.
  `--subset` is part of the hash for the same reason — excluding it would let a full-data run
  silently reuse smoke models, which the provenance triple cannot catch.
- **Provenance is asserted.** All models must share the
  `(objective_version, split_sha256, input_sha256)` triple, or the aggregate raises rather
  than averaging incomparables — the same rule `pool_oof.group_key` enforces.
- **`pooled/*` appears only when a seed holds all 5 folds** and they tile the dataset exactly
  once. A partial set logs the mean and reports why it was not pooled; it never quietly pools
  four folds.
- Checkpoints are **off by default** here (`train.py --no-save-checkpoint`): `final.pt` is
  34 MB, and 10 per trial across a 250-trial sweep is ~85 GB nothing reads. `--keep-checkpoints`
  restores them. The predictions the pooled metrics need are ~2 MB and always written.

**`scripts/run_grid.sbatch` is the old path** and still opens one wandb run per array task —
that is the behaviour this replaces. To run the grid in parallel and still get one run, point
the array at a shared bucket with `--no-wandb --out <bucket>/fold{F}_seed{S}`, then finish with
`python src/run_config.py --aggregate-only <bucket>`. Not yet done; the sequential driver is
what has been exercised.

### The sweep — `sweeps/bayes_v1.yaml`

wandb bayes, on TamIA (**`wandb agent` does work there** — confirmed 2026-08-13, superseding
`reports/compute_profile.md`'s "a wandb sweep cannot run on TamIA", which is why this is wandb
and not Optuna).

```bash
wandb sweep --project finetune_minimol --entity ethan_personal sweeps/bayes_v1.yaml
python -m wandb agent ethan_personal/finetune_minimol/<sweep_id>
```

`program:` is **`src/run_config.py`**, so one trial is one configuration is one run.

**The objective is `final/goal_metric_mean`** — the mean over models of each model's
final-epoch `goal_metric`. Two things it is deliberately not:

- **Not the best epoch.** `agg/val/goal_metric_mean` carries `summary="max"`; optimising that
  would be early stopping on the same validation fold the run reports, exactly the bias the
  fixed epoch budget exists to avoid. Both are logged so the gap stays visible.
- **Not the pooled score**, though pooling is the more honest tail estimate (100 potent
  molecules against 20 per fold). Pooling is *undefined* unless a seed holds all five folds,
  so it would not exist for a cheap search trial. The mean is defined for any subset, which is
  what lets search and confirmation share one sortable column. Report `pooled/*` from the
  winner.

`fold_list` / `seed_list` are **the cost dial**, pinned as parameters: `"0"` × `"0"` is one
model per trial (~4 min on H100), `"0,1,2,3,4"` × `"0,1"` is the full grid (~40 min). Spell
them as **comma-separated strings, not YAML lists** — a yaml list reaches the agent as
`--fold_list=[0, 1]`, which the shell splits and argparse rejects.

Swept: `freeze_epochs`, `unfrozen_epochs` (both `q_log_uniform_values`, integer), the three LR
peaks, `weight_decay`, `dropout`, and the four loss terms. Deliberately not swept, each with
its reason in the yaml: head shape (**blocked on the architecture question**), `batch_size`
(1200 is argued twice over, and moving it would confound every LR axis), `weights`,
`pprop-norm`, `eta_min`, `lr_schedule`.

No `early_terminate`: hyperband prunes on the intermediate value, which systematically
penalises long-anneal schedules — a 40-epoch cosine is far from its best at epoch 5 while an
8-epoch one is nearly done. Schedule length is the axis being swept, so pruning on it would
decide the sweep before it started.

Three mechanical things the agent needs, all in `train.py` and reused by `run_config.py`:

- **`dashed()`** rewrites `--head_lr=…` to `--head-lr=…`. The agent spells flags exactly as
  the yaml spells its keys, i.e. underscored, and argparse does not accept an underscore
  variant of a dashed option. Without it every agent-launched run dies on
  `unrecognized arguments`.
- **`sweep_int`** parses `"3.0"`. `q_log_uniform_values` quantizes as `q * round(x/q)` and
  emits a float, so a plain `type=int` dies on every integer hyperparameter.
- **`build_parser()`** exposes the parser so `run_config.mirror_train_arguments` can copy
  every flag onto its own CLI. A hyperparameter added to `train.py` therefore reaches the
  sweep with no second edit.

**Reading the epoch result:** with final-epoch selection and full annealing, `goal_metric` is
close to monotone in budget, so bayes will push `unfrozen_epochs` toward the top of its range.
A winner sitting *on* the boundary means the range was the answer, not the sweep — widen and
re-run rather than reporting convergence.

### Deferred: layer-wise freeze/unfreeze

**Not started. Noted 2026-08-13 from supervisor feedback; circle back before the sweep is
specified.** The freeze is all-or-nothing today — `set_trunk_trainable` flips every tensor
under `model.trunk` at once.

The supervisor ranked the knobs as **(1) the freeze schedule, (2) *which* layers freeze and
unfreeze, (3) learning rates, (4) total epochs** — the last of which he declined to bound at
all ("ask the sweep, use `q_log_uniform`" — integer + log-spaced, which in Optuna is
`suggest_int(..., log=True)`). So partial unfreezing outranks the LR values this repo has
been tuning, and the one measurement on hand supports his #1: the fold-0 prototype unfroze
after **3** frozen epochs with val Pearson still climbing (0.8037 → 0.8228), so 3 was too
short. The default of **5 is untested** — that is the measurement to take first.

The shape to build is gradual unfreezing (ULMFiT). `gnn.depth` is 16, plus `encoder_manager`,
`pre_nn` and `pre_nn_edges`, so **"unfreeze the top *k* blocks" is one integer knob** rather
than ~19 booleans — and one integer is sweepable.

Two things make it real work rather than a flag flip:

- **The optimizer ordering trap generalises badly.** `param_groups` (model.py:74) filters on
  `requires_grad` at construction and *drops* empty groups, and `train.py:500` asserts the
  group set is exactly `{trunk, head}`. Every per-layer group must be constructed **before**
  any freezing, and that assertion must be widened rather than deleted — a silently-untrained
  block with a healthy loss curve is precisely what it exists to catch.
- **It must stay routed through `set_trunk_trainable`.** `verify_metrics.py` neuters that name
  to prove the freeze assertion has teeth; a second write path to `group["lr"]` would make
  that negative test pass while testing nothing.

Ambiguous in the supervisor's guidance, and worth resolving with him before writing code:

1. **Progressive or one-shot?** Unfreeze the top *k* at the handoff and hold that set for the
   rest of the run, or unfreeze one block at a time on a cadence? These are different
   mechanisms with different knobs (`k` versus a rate), and the sweep cost differs.
2. **Binary freeze, or per-depth learning rates?** ULMFiT's other half is discriminative
   fine-tuning — every block trains, with the LR decayed by a fixed factor per depth. That is
   the soft version of the same idea and composes with the existing two-group optimizer far
   more cheaply than N groups.
3. **Where do the non-GNN modules sit on the depth axis?** `encoder_manager` (the positional
   encoders), `pre_nn` and `pre_nn_edges` are not GNN layers. "Top *k*" presumes a linear
   order; these sit structurally *below* layer 0, and `pre_nn_edges` is arguably on a separate
   axis entirely — edge features rather than node depth. Always frozen, unfrozen last, or on
   the same axis?
4. **Does the frozen bottom ever unfreeze**, or stay frozen for the whole run?
5. **How does this interact with the deliverable?** **Settled 2026-08-17 — the product IS a
   pProp-specific 32-d embedding** (see "The deliverable" below). Freezing 14 of 16 GNN blocks
   leaves the representation mostly pretrained, which argues for unfreezing more; feature
   distortion (LP-FT, Kumar et al. 2022) and the fact that a generative loop will query novel
   chemistry argue for less. That trade-off is a research question, not an implementation
   detail — and it should be decided on `emb_*` over held-out clusters, not on `goal_metric`,
   because `goal_metric` scores the predictions and the predictions are not the product.

### Metrics — `src/metrics.py`, `src/objective.py`

**Every metric is computed under both weightings, and every key is suffixed.** There is no
unsuffixed `weighted_mae` to fall back on, because a sign error in a weight vector is
invisible in any single number. `verify_metrics.py` asserts `mae_uniform < mae_balanced`
*and* pins both vectors by their base rates — unweighted must reproduce the true positive
rate, balanced must be exactly 0.5 (measured 0.499999997; the 2.5e-9 gap is float32 in
`grouped_frequency_weights`, exact in float64).

- `*_uniform` estimates this 331k subset, which is ~30× tail-enriched by construction, **so
  it flatters the model rather than being neutral**. Read it as a subset number, never as a
  library number. Since `ipw` was removed there is no reweighting that corrects back to the
  10M library — that is the accepted cost of keeping the sampling design out of the metrics.
- `*_balanced` is tail-emphasising and matches what the loss optimises.
- `ap_balanced` is **not a performance number** — forcing a 50/50 base rate makes AP a
  statement about the weighting. `ap_uniform` is the one to read.

`goal_metric = AP* + ½(Pearson* + MAE_skill*)`, at the **final** epoch, no early stopping.
The starred terms, from `OBJECTIVE_SPEC` — **`AP*` = `ap_uniform` alone**; `Pearson*` and
`MAE_skill*` each average `uniform` and `balanced`. `objective.py` still flags the
composition of the two averaged terms as a defensible starting point rather than a settled
choice; changing either is a one-line edit that re-stamps the version automatically.
`OBJECTIVE_VERSION` is a **hash of the objective spec**, so editing any term re-stamps it
automatically — `pProp_MLP` accumulated three incompatible revisions under one metric name,
and its own CLAUDE.md warns they must never be compared. Every run stamps
`objective_version`, `split_sha256` and `input_sha256` into `meta.json`; filter on all three
before comparing two runs.

**The current stamp is `v1-binary3.5-c917327f`.** It replaced `adb3da05` on 2026-08-12 when
`ap_ipw` left `AP*`, so `AP*` went from a two-metric mean to `ap_uniform` alone. Treat
`goal_metric` as **not comparable across that boundary** — the two AP flavours measured close
(0.4861 vs 0.4968), so the magnitude barely moves, which makes an accidental comparison look
plausible rather than obviously wrong. In practice nothing is at risk: no run had been scored
under `adb3da05` (the only runs on disk were compute benchmarks with null provenance), which
is exactly why the change was free to make then and will not be later.

`--pprop-norm` defaults to `zscore` and **metrics denormalize first**, so every reported
number is on the raw pProp scale. Not cosmetic: `val/mse` is declared `summary="min"`, and a
normalized report would silently make that summary incomparable across settings. The
inherited loss hyperparameters are in **z-units** for the same reason — `pProp_MLP` trained
its huber/pair/std terms against the normalized target (`sweep_train.py:460-465`), which is
what lets `huber_delta 1.05` / `w_pair 7.49` / `w_std 0.79` transfer across two different
pProp distributions.

**First result** (fold 0, seed 0, target `pprop`, head 512→1024→32→1, head_lr 1e-3 /
trunk_lr 1e-4, [run](https://wandb.ai/ethan_personal/finetune_minimol/runs/uj7pnjiw)):

| epoch | phase | val MSE | val Pearson |
|---|---|---|---|
| 1–3 | head only | 0.3013 → 0.2430 | 0.8037 → **0.8228** |
| 4–5 | trunk + head | 0.1896 → **0.1828** | 0.8654 → **0.8714** |

Epoch 3 is effectively the frozen-trunk baseline. Unfreezing bought **+0.049 Pearson** and
cut val MSE by **25%** in two epochs — the comparison NOTES §7 Phase 3 calls the
justification for the repo, and it points the right way. Not yet a result: one fold, one
seed, 5 epochs, no tail metrics.

### Two ordering traps, both enforced in code

- **Build the optimizer BEFORE freezing.** `param_groups` (model.py:74) filters on
  `requires_grad` at construction and *drops* an empty group. Freeze first and you get a
  one-group optimizer; unfreezing then sets `requires_grad=True` on parameters the optimizer
  has never seen, so the trunk never trains — with no error and a healthy-looking loss curve.
  `train.py` asserts the group set is exactly `{trunk, head}`.
- **The schedule is asserted, not assumed.** Across the frozen phase the trunk must be
  bit-for-bit unchanged *while the head provably moves*, and must change once unfrozen.
  Measured on the real run: trunk Δ = **0.000e+00** / head Δ = 4.958e-01, then trunk Δ =
  1.185e-02. Paired for the same reason as `check_excluded_unreachable` — an unchanged trunk
  is also what a broken loop looks like. Negative-tested: neutering the freeze makes the run
  die with `freeze violated`.

### Other things worth knowing

- **Freezing is ~2× faster per epoch** (32 s vs 65 s), because with no trunk parameter
  requiring grad, autograd builds no graph through the trunk at all — skipping the backward
  *and* the stored activations.
- **The loss is weighted, defaulting to `balanced`** (`weighted_mse`, normalised by `w.sum()`
  so the loss scale does not move when a scheme is swapped in). Never replace it with a bare
  `.mean()` — see NOTES §1.
- `val_embeddings.npy` (`[n_val, 32]`, ~8 MB) is written per run beside the predictions — the
  exported artifact itself, final-epoch, in the same row order.
- `val_predictions.npy` + `val_indices.npy` are written per run, so pooled out-of-fold tail
  metrics need no re-running.
- Pearson is `nan` when predictions are constant; `val/pred_std` is logged beside it so that
  is diagnosable rather than mysterious.

---

## The feature cache — `data/features/minimol_v1/`

All 331,480 molecules featurized once, in **exact CSV row order**, so `splits.py` indices
index it directly with no mapping layer.

```python
from features import load_features
ds = load_features("data/features/minimol_v1")     # 1.4 s, incl. the CSV re-hash
loader = DataLoader(Subset(ds, train_idx), batch_size=256,
                    shuffle=True, collate_fn=trunk.collate)
```

Rebuild in ~3.5 min: `python src/featurize.py` (1,745 mol/s — graphium sets
`featurization_n_jobs = -1`, so it already uses all 128 cores; do not add a second pool).

- **Stored PyG-collated** as `(data, slices)`, not a list of `Data`. Benchmarked: 4.81 GB /
  **1.0 s** to load versus 5.44 GB / **81.7 s** for a list. 80× matters because the 5×2 grid
  is 10 processes each paying it once. `collate`→`separate` verified to round-trip every key
  of every graph exactly and give bit-identical embeddings.
- **Graphs only, no target.** `pprop` comes from the CSV at train time by row index, so the
  target choice does not invalidate the cache.
- **`load_features` re-hashes the source CSV and refuses a mismatch**, exactly as
  `splits.py` does, and for the same reason: `subset.py` takes a `--seed`, so a regenerated
  CSV is a different 331k set and would leave every graph bound to the wrong target. A
  `--limit` cache is likewise rejected unless `allow_partial=True`.
- Verified: cached row *i* is bit-identical to freshly featurizing CSV row *i* (checked at
  rows 0, 1, 3152, 165740, 331478, 331479), and the 5 validation folds cover all 331,480
  rows exactly once.

**A collated batch is single-use** — the positional encoders concatenate into `feat`, so its
width grows during `forward`. The *source* graphs are not mutated (`Batch.from_data_list`
copies, measured), so the cache can be served every epoch with no defensive copying. Never
reuse a batch returned by `trunk.collate`; re-collate instead.

---

## The rw_pos dead norm

`verify_trunk.py`'s `gradients reach the whole trunk` check reports 284/286. The two
tensors without gradient are
`encoder_manager.pe_encoders.rw_pos.first_normalization.{weight,bias}` (32 params), and
they are **vestigial inside graphium 2.4.7 itself** — not a defect in `trunk.py`. Measured:

- `MLPEncoder.__init__` builds `self.first_normalization` (via `BaseEncoder`), then passes
  that module into `MLP(...)`, whose `__init__` calls `get_norm(...)` *again* —
  `outer is inner` is **False**, two distinct LayerNorm(16) objects.
- `MLPEncoder.forward` only calls `self.pe_encoder(...)`. The outer norm is never executed.
- The outer norm sits at pristine init (weight all 1, bias all 0) while the inner one holds
  loaded values. Not a load failure: `missing_keys` is exactly the two unrelated
  `task_heads.graph_output_nn...layers.1.normalization.*` entries, so the checkpoint
  **contains** the outer norm, saved still at init — MiniMol never trained it either.
- `.grad` comes back **`None`**, not zero, so no optimizer can move it — weight decay
  included. The 32 params are inert, not merely small.
- `la_pos` is unaffected: `LaplacePosEncoder.forward` *does* call its own
  `first_normalization` (`laplace_pos_encoder.py:197`), and its is `None` here anyway.
- The normalization the author intended still happens — the inner duplicate performs it
  (`global_architectures.py:354`). Nothing is missing from the computation.

**Resolved** in `verify_trunk.py::orphaned_norm_params`, which derives the orphan set from
the module tree (any `MLPEncoder` with a non-None `first_normalization`) rather than
hardcoding names — so if graphium ever stops duplicating the norm, the set empties itself
and the check tightens automatically.

`check_grad_flow` then asserts the partition **two-sidedly**: every reachable tensor must
have gradient, *and* the dead set must equal the predicted one exactly. Simply skipping the
two would have been a plain loosening — silent if some other tensor went dead later, which
is the very failure the check exists to catch.

The verdict itself lives in `partition_grad_flow`, pure set logic over parameter names, and
`check_grad_flow_has_teeth` exercises it on synthetic inputs as an 11th check that runs
every time. Driving the two branches through a real backward pass *cannot* isolate them —
perturbing the exemption to test one branch trips the other simultaneously, so the case
passes for the wrong reason. The synthetic cases hold the exemption fixed, vary only the
observed-dead set, and require the branch under test to fire **while the other stays
silent**. Verified that stubbing `partition_grad_flow` to always-pass makes this check fail.

---

## Footguns

Full list in NOTES §9. The ones that bite hardest:

- **`ampc_subset_331k.csv` is sorted by `(-pprop, smiles)`** — i.e. sorted by the target. A
  DataLoader that forgets to shuffle trains on target-sorted batches. Never rely on file
  order.
- **`splits.py` re-hashes the source CSV on load and raises if it changed.** This is
  deliberate: `subset.py` takes `--seed`, so rerunning it silently produces a different 331k
  set and invalidates every split. Do not bypass the check — regenerate the splits.
- **`load_state_dict(..., strict=False)`** in `minimol/model.py` silently tolerates missing
  and unexpected keys, yielding a partly-random trunk with no error. Always capture and
  assert on the returned `_IncompatibleKeys`.
- **`Fingerprinter.get_fingerprints_for_batch` uses `torch.inference_mode()`** — verified in
  graphium 2.4.7 source. `Minimol.__call__` therefore **cannot** be used for fine-tuning.
  See NOTES §§4–5 for the two ways around it.
- **`hydra.initialize()` runs once per process** — constructing the model twice in one
  process raises. Will bite in sweeps and test suites.
- **Morgan fingerprint details shift across rdkit releases.** The splits were built under
  rdkit 2024.03.5; `minimol_ft` will carry a different one. This is why
  `fingerprints.npy` is persisted rather than recomputed, and why `meta.json` records the
  version.
- **`data/` is 1.5 GB and untracked.** `data/splits/cluster_kfold_v1/` alone is 112 MB.
  `.gitignore` excludes `data/*` wholesale and un-ignores exactly two things: `data/reference/`
  (the frozen MiniMol embeddings `verify_trunk.py` checks against, a few hundred KB, useless
  unless it travels with the code) and `data/*.meta.json`. Splits are regenerable in ~2.5 min
  from `src/split.py`, which is why they are not tracked.

---

## Conventions

- Scripts take argparse CLIs, are deterministic given their arguments, and write a sibling
  `meta.json` recording inputs, hashes, versions, and argv. `subset.py` and `split.py` both
  follow this — match it.
- Verify claims by measurement and record the number, rather than asserting from reasoning.
  The 0.65 clustering threshold, the rejection of Bemis–Murcko scaffolds, and
  `LeaderPicker`'s thread-determinism were all settled this way, and the evidence is written
  into NOTES §11.
- SLURM: `sbatch`, GPU as `--gres=mps:20`, job arrays, logs to `/home/ethan2/logs/`. Note
  that `/home/ethan2/job.sh` sanitizes inherited venvs from `PATH` before activating conda —
  keep that. Anything reading the splits must work from a foreign CWD; `splits.py` resolves
  paths via `meta["input_abspath"]` for this reason.
- wandb entity `ethan_personal`; sweeps via `python -m wandb agent`.
