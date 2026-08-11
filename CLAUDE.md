# finetune_minimol

Fine-tune the **entire MiniMol trunk** (~10M params) on AmpC docking-score regression —
not the usual frozen-embedding + MLP workflow. Gradients must reach all trunk parameters.

`NOTES.md` is the reference document: background, source-level findings, and the full plan.
This file is the operational summary. When they disagree, `NOTES.md` §§1–11 is authoritative
on *why*; this file is authoritative on *what currently exists*.

---

## State as of 2026-08-10

| Piece | Status |
|---|---|
| Dataset subset (331,480 molecules) | **done** — `src/subset.py` |
| 5-fold cluster CV splits | **done, verified, frozen** — `src/split.py`, `src/splits.py` |
| Environment (uv, in-repo `.venv`) | **done, verified** — `pyproject.toml` + `uv.lock` |
| Trainable trunk | **done, 10/10 checks pass** — `src/trunk.py`, `src/verify_trunk.py` |
| Feature cache (Phase 2) | **done, verified** — `src/featurize.py`, `src/features.py` |
| Prototype training run | **done, verified** — `src/train.py`, 5 epochs, fold 0 seed 0 |
| 5×2 grid runner / SLURM | not started (NOTES §7 Phase 4) |

The trunk reproduces frozen MiniMol embeddings **exactly** (max|Δ| = 0.000e+00 over 64×512),
gradients reach all 284 reachable trunk tensors (7,919,912 params), and an optimizer step
moves trunk weights. Two further tensors are unreachable inside graphium itself — see "The
rw_pos dead norm" below. `verification.md` reads `OVERALL: PASS`.

Nothing is committed beyond the initial commit; `pyproject.toml` and `uv.lock` are still
untracked, so the env is not yet reproducible on TamIA.

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
NOTES.md                             the reference document
```

---

## The data

`data/ampc_subset_331k.csv` — columns `SMILES, score, pprop, bin, ipw`.

- `score` — AmpC docking score, kcal/mol, **more negative is better**
- `pprop` — `-log10(rank_max / 1e7)`, a quantile; monotone in `score`; max 7.0.
  **This is the regression target** (settled 2026-08-11). `score` is kept for reporting.
- `ipw` — inverse-probability weight; how many library molecules each row stands for.
  `subset.py` kept the whole potent tail and subsampled the bulk, so the file is
  deliberately *not* distributed like the library. `ipw` corrects metrics back.
  Measured: bulk rows (pProp < 1) carry `ipw = 75`, the entire tail carries `ipw = 1`, and
  `sum(ipw)` is exactly 10,000,000. Potent molecules are ~30× over-represented relative to
  the library, which is why unweighted subset metrics flatter the model.

**Loss weighting is an open design point, not an absent one.** A weighting scheme is
planned but unchosen (`ipw` is one candidate, not a decision). The training step must
therefore accept a **per-sample weight vector defaulting to uniform** — never a hardcoded
unweighted mean. See NOTES §1.

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

One fold, one seed, staged fine-tuning. `--freeze-epochs 3 --epochs 5`: the head trains
alone on the frozen embedding, then the trunk unfreezes and both train together.

```bash
python src/train.py --fold 0 --seed 0                                    # 3.7 min
python src/train.py --epochs 2 --freeze-epochs 1 --subset 5000 --no-wandb  # smoke, <1 min
```

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

- **Build the optimizer BEFORE freezing.** `param_groups` (model.py:69) filters on
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
- **The loss is weighted, defaulting to uniform** (`weighted_mse`, normalised by `w.sum()`
  so the loss scale does not move when a scheme is swapped in). `--weights ipw` is wired.
  Never replace it with a bare `.mean()` — see NOTES §1.
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
- **Graphs only, no target.** `score`/`pprop`/`ipw` come from the CSV at train time by row
  index, so the still-open target choice (NOTES §1) does not invalidate the cache.
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
- **`data/` is 1.5 GB and untracked, with no `.gitignore`.** `data/splits/cluster_kfold_v1/`
  alone is 112 MB. Decide on `.gitignore` vs LFS before committing anything under `data/`.

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
