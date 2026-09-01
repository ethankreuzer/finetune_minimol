# finetune_minimol — the `frozen-moljepa` branch

**Swap the encoder, hold everything else fixed.** Every arc in this repo so far has
fine-tuned MiniMol's ~10M-parameter trunk. This branch asks whether a newer molecular
foundation model — **Mol-JEPA**, used **frozen** — carries more pProp signal in its
embeddings, by running it once offline and training a network on the cached vectors.

**Nothing is fine-tuned here.** There is no trainable trunk, no freeze schedule, no
`--trunk-lr`. Settled with Ethan 2026-09-01: forward-pass inference, then a model on top.

The comparison is only worth something if nothing else moves, so the data, the frozen 5-fold
cluster splits, the 5-term weighted loss, the metrics, `goal_metric` and the wandb bayes
machinery are **reused by import, not reimplemented** — see "What is shared, and how" below.
The benchmark this branch is measured against is **MiniMol at `--unfrozen-epochs 0`**: frozen
against frozen, same loss, same folds, same objective.

The MiniMol arm is **kept and still runnable** — `train.py`, `trunk.py` and the graphium/
minimol pins are untouched and stay installed, so the baseline can be re-run from this branch
rather than quoted from an old wandb page.

---

## Mol-JEPA, measured

[arXiv 2608.22642](https://arxiv.org/abs/2608.22642) (Boehringer Ingelheim), weights at
[`Flogrammer/Mol-JEPA`](https://huggingface.co/Flogrammer/Mol-JEPA) — 45,406,721 params,
182 MB safetensors, CC BY-NC 4.0. Loaded with `AutoModel.from_pretrained(...,
trust_remote_code=True)`; `model(smiles_list)` takes SMILES directly. **No 3D conformers and
no MoKa at inference** — those are pretraining-side only.

**Seven modalities, of which only two are computable from a SMILES string:**

| modality | input | encoder |
|---|---|---|
| `graph` | **smiles** | TransformerConv x3, node_dim 82 / edge_dim 17, max-pool -> 512 |
| `ecfp` | **smiles** | 2048 -> 3-layer MLP -> 512 |
| `uma`, `boltz`, `boltz_preds`, `moe`, `bioxmol` | precomputed | unavailable to us |
| `chembl/tdc/pcba/nabla/xtb_targets` | label modalities | unavailable to us |

From a bare SMILES the other eleven are **masked, and the JEPA predictor guesses them**. That
is the design, not a limitation — and it makes those eleven columns a genuinely different kind
of feature from anything MiniMol produces: a structure-only model's estimate of binding,
ADMET, phenotype and quantum-chemistry embeddings.

### What the three output tensors actually are

**The model card's wording is misleading and cost a rewrite; the layout below is measured**
(`MolJEPA.predict`, and re-derived bit-exactly on every run by `jepa_embed.check_layout`):

```
embeddings   [B, 13, 512]   the RAW transformer output: cls at column 0, then 12 modalities
cls          [B,     512]   == modality_pred[0](embeddings[:, 0])
predictions  [B, 12, 512]   == modality_pred[i+1](embeddings[:, i+1])
```

So **`cls` is NOT a slice of `embeddings`** — it is a *linear projection* of column 0, and
`predictions` are the projections of columns 1..12. Verified: max|Δ| = 0.0 for both
relationships. Caching `embeddings` + `predictions` would have silently dropped `cls`, which
is the card's headline output. The cache therefore stores two **column-aligned** `[N, 13, 512]`
arrays under one shared name list.

---

## NEXT: what to run

The cache and the plumbing exist. What is **not** settled is the encoder itself.

1. **Decide the readout** — `--readout` defaults to `cls` *as a placeholder so the pipeline
   can be smoke-tested*, not as a design choice. Pin it in `sweeps/bayes_jepa_v1.yaml`
   before reporting anything. Every option is a slice of the cache, so this costs no
   re-inference: `cls`, `emb:graph`, `emb:*`, `proj:*`, `emb:graph+emb:ecfp`, ...
2. **Decide what model sits on top.** `head.DualHead` is the starting point because it makes
   the loss identical to the MiniMol arm's, not because it has been argued.
3. **Run the sweep** (`sweeps/bayes_jepa_v1.yaml`), then report it **beside** frozen-MiniMol
   on the same folds. A single number alone is not a result.

---

## A first readout scan — orientation, NOT a result

Fold 0, seed 0, default head, 20 epochs. **Every caveat applies at once**: one fold, one seed,
an untuned head, and `goal_metric` scores the *predictions* rather than the encoder. Treat the
ordering as a hint about where to look, and nothing as measured until the sweep and the full
5x2 grid have run.

| readout | dim | `goal_metric` |
|---|---|---|
| **frozen MiniMol baseline** (`--unfrozen-epochs 0`) | 512 | **0.9070** (fold 0 only) |
| `emb:graph+emb:ecfp` | 1024 | 0.8166 |
| `emb:*` (all 13 raw tokens) | 6656 | 0.8084 |
| `emb:graph` | 512 | 0.8005 |
| `proj:*` (all 13 projections) | 6656 | 0.7782 |
| `emb:cls` | 512 | 0.7742 |
| `cls` (= `proj:cls`, the card's headline output) | 512 | 0.7736 |

Three things worth noticing, all of which want confirming rather than believing:

- **The two modalities actually computed from SMILES carry it.** `emb:graph+emb:ecfp` beats
  all 13 tokens together. The 11 *predicted* modalities are the interesting part of this model
  and, on this evidence, they add noise rather than signal for AmpC — which is a hypothesis
  to test properly, not a conclusion. They may well carry signal a linear head cannot reach.
- **Raw beats projected.** `emb:*` > `proj:*` and `emb:cls` ≈ `cls`. The `modality_pred` heads
  are trained for JEPA's latent prediction objective, not to be a representation.
- **Frozen MiniMol is ahead, and by more than the noise.** 0.9070 against 0.8166 on fold 0 is
  a gap of 0.09 — roughly 4x the 0.0242 fold-to-fold spread measured below, so the ordering is
  unlikely to be a lucky fold. That is the comparison this branch exists to make, and **it
  currently favours the incumbent.** Before reading too much into it: the MiniMol number is
  still fold 0 only, the head is untuned for either arm, and `goal_metric` scores predictions
  rather than the encoder — the R1/R2 probes are the thing a *featurizer* should be judged on.

### The one row that was taken past a single fold

`emb:graph+emb:ecfp` on the **full 5x2 grid** (10 models, 7m07s):

```
goal_metric 0.7964 +/- 0.0242   (min 0.7595, max 0.8284)
pooled seed0: goal 0.7897 | tail spearman 0.0194 | EF@5.0/top1% 73.0
pooled seed1: goal 0.7975 | tail spearman 0.0210 | EF@5.0/top1% 73.0
```

**Fold 0 alone read 0.8166 — about 0.8 sigma high.** That is the whole argument for the full
grid being the unit of measurement: a single-fold scan picks a *direction*, not a value, and
the six-row table above is separated by less than two of these standard deviations in places.
`sweeps/bayes_jepa_v1.yaml` pins this readout on exactly that basis — best available guess,
not a settled winner.

The pooled rows are the honest tail estimate (all 100 potent molecules, against ~20 per fold),
and this is the first time the pooling path has run on this arm.

Reproduce with:

```bash
python src/run_config.py --trainer train_jepa --fold-list 0 1 2 3 4 --seed-list 0 1 \
    --readout "emb:graph+emb:ecfp" --no-wandb
python src/run_config.py --fold-list 0 1 2 3 4 --seed-list 0 1 \
    --freeze-epochs 20 --unfrozen-epochs 0 --no-wandb          # the MiniMol baseline
```

---

## What is shared, and how

Reused **by import, unmodified**. This is what makes "identical" a fact rather than a claim —
if any of these changed, both arms would change together.

| imported by `train_jepa.py` | from | what it fixes |
|---|---|---|
| `combined_loss`, `binary_labels`, `effective_sample_size`, `PPROP_EDGE` | `losses.py` | the whole loss |
| `score_split` | `train.py` | the whole reported-metrics path |
| `scheduled_lr`, `RowDataset`, `fold_weights`, `dashed`, `sweep_int` | `train.py` | the cosine, the batch record, the weighting, the agent's flag spelling |
| `compute_norm_stats`, `normalize_pprop` | `normalization.py` | the target transform |
| `load_fold`, `load_meta` | `splits.py` | the frozen partition + provenance |
| `OBJECTIVE_VERSION` | `objective.py` | the objective's identity |
| `DualHead` | `head.py` | the head, at `in_dim=` the readout's width |

**`verify_metrics.py` still passing 8/8 is the evidence**, since it exercises the same
`losses.py`/`metrics.py` both arms now share.

Importing `train` pulls graphium in with it (`train.py` imports `MiniMolTrunk` at module
scope), costing a few seconds once per process. That is the price of the identity, and it is
worth paying: the alternative is a second copy of `score_split` that can drift.

**One phase, so one cosine.** `scheduled_lr` is reused unchanged by handing it a config whose
phase 1 is the whole run (`freeze_epochs = epochs`, `unfrozen_epochs = 0`). That keeps the
`length - 1` denominator, which is what lands the final epoch exactly on `eta_min` and makes
final-epoch selection a settled model rather than an arbitrary point on a moving trajectory.

**Why `train_jepa.py` is a separate file** rather than a frozen trunk inside `train.py`: a
zero-parameter trunk makes `model.param_groups` (`model.py:88-90`) drop the empty group, which
trips `train.py:761`'s assertion that the optimizer's groups are exactly `{trunk, head}`. That
assertion is what catches a silently-untrained trunk in the MiniMol arm — the failure mode
with a healthy-looking loss curve. Widening it would weaken a live guard on code this branch
does not change.

---

## The embedding cache — `data/embeddings/moljepa_v1/`

```bash
python src/jepa_embed.py            # ~25-90 min, 331,480 molecules, CSV row order
```

```
embeddings.npy   [331480, 13, 512] float32   8.8 GB   raw transformer output
projected.npy    [331480, 13, 512] float32   8.8 GB   cat([cls, predictions])
meta.json                                             token_names, hashes, revision
```

17.6 GB, **untracked and regenerable** (`.gitignore:33`). Written in exact CSV row order, so
`splits.py` indices index it directly with no mapping layer.

- **The Hub revision is pinned** — `jepa_embed.HF_REVISION = "4c912b45..."`.
  `trust_remote_code=True` executes Python fetched from the Hub, so an unpinned load means the
  definition of the encoder can change under us between runs.
- **The layout is re-derived bit-exactly on every run**, not assumed. `check_layout` asserts
  `cls == modality_pred[0](embeddings[:,0])` and the per-token projection relationship, and
  refuses to write a cache if either fails. The first draft of the script trusted the model
  card and was wrong; only that check caught it.
- **Reading it: `src/jepa_features.py`, never `np.load`.** It owns the provenance guard
  (re-hashes the source CSV, refuses a `--limit` cache) exactly as `features.py` does.
- **The readout grammar** is a `+`-joined list of `cls` (= `proj:cls`), `emb:<token>`,
  `proj:<token>`, `emb:*`, `proj:*`. Tokens are `cls graph ecfp uma boltz boltz_preds moe
  bioxmol chembl_targets tdc_targets pcba_targets nabla_targets xtb_targets`. Run
  `python src/jepa_features.py <root>` to print what a cache offers.

### Determinism is the reason this is not just a GPU job

Measured on this box, on 64 molecules:

| device | repeat @ same batch | re-chunk 8/16/256 vs 64 | reversed order |
|---|---|---|---|
| cuda (default) | **2.1e-06** | 2.4e-06 | 1.7e-06 |
| cpu | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| cuda + deterministic | **0.000e+00** | 0.000e+00 | 0.000e+00 |

The GPU differs from *itself* on a plain repeat, so this is kernel non-determinism — scatter
atomics in message passing, and cuBLAS split-k — not a batching artifact. ~5e-07 relative, and
it would not change a trained model; but the cache is computed **once** and every number on
this branch inherits it, so "rebuilding the cache moves the results" is a bad property to
accept for free.

**A separate property, and do not confuse the two: the model is NOT invariant to batch
composition, on either device.** Six scattered rows embedded together, against the same rows
embedded inside their natural 256-chunks:

| | cpu | cuda |
|---|---|---|
| batch **composition** (different neighbours) | **3.3e-07** | **7.5e-07** |

`to_dense_batch` pads the graph batch to its widest member and the reductions reassociate, so
a molecule's embedding depends slightly on what it was batched with. The middle column of the
table above says nothing about this: re-chunking *the same contiguous list* keeps most
compositions intact, which is why it reads 0.0. That reading is easy to over-interpret — it
was, during this branch's first verification pass.

Two consequences, both real:

- **A rebuilt cache is bit-identical only at the same `--batch`, on the same device.** It is
  identical to ~1e-06 otherwise, which is below anything that would move a result, but it is
  not bit-identical. `meta.json` records `batch` and `device` so a rebuild can match.
- **Anything checking a cached row against a fresh forward pass must re-embed it inside its
  natural chunk**, or it measures composition noise instead of what it meant to.
  `verify_jepa_embed.check_alignment` does exactly this, and is bit-exact as a result.

`jepa_embed.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` **at module scope, before torch is
imported** (cuBLAS reads it at init; set later it is ignored) and calls
`torch.use_deterministic_algorithms(True, warn_only=True)`. `warn_only` means a PyG op with no
deterministic kernel warns rather than raising — so determinism is a *request*, which is why
the run **measures a repeat** and refuses to write unless it is 0.0. Cost: 218 mol/s against
281. `--allow-nondeterministic` downgrades it to a warning; `meta.json` records the figure
either way.

---

## Verification — `src/verify_jepa_embed.py`

```bash
python src/dump_jepa_reference.py          # the frozen fixture, first
python src/verify_jepa_embed.py            # -> verification_jepa.md
```

Eight checks, in the shape of `verify_trunk.py`. The four failures they exist to catch, none
of which announces itself:

1. **The model moved.** Checked against `data/reference/moljepa_v1_ref64.npz`, dumped in its
   own process by the plain `AutoModel` path — deliberately *not* through `jepa_embed.py`, so
   the fixture cannot agree with the code it checks by construction.
2. **The rows shifted.** Cached row *i* vs fresh inference on CSV row *i*, at rows
   `0, 1, 3152, 165740, 331478, 331479` — both ends and four interior points.
3. **The columns are misnamed.** The layout is re-derived and compared to `meta.json`.
4. **It is not reproducible.** Repeat, batch size 8/16/256, and reversed order must all be
   bit-identical.

Plus: eval mode (dropout is 0.1 on all three expert encoders), loader guards shown *rejecting*
something rather than merely existing, no non-finite values and no collapsed token
(`max == min` per dimension, which is exact and needs no tolerance), and the 5 validation
folds tiling the cache exactly once.

**The alignment check is two-sided**, for the same reason `check_grad_flow` is: it is
bit-exact against the right row *and* is shown to reject the neighbouring one. Measured
2026-09-01 — right row **0.000e+00**, off-by-one **1.164**, scattered re-embed 7.5e-07. A
tolerant check that has never been seen to reject a shifted comparison tests nothing.

Measured result, `verification_jepa.md`, **OVERALL: PASS** (8/8) on the full 331,480-row cache.

---

## The sweep — `sweeps/bayes_jepa_v1.yaml`

Same machinery as `bayes_v1.yaml`: `method: bayes`, objective `final/goal_metric_mean`
maximised, no `early_terminate`, the full 5x2 grid per trial, one configuration = one wandb
run. **Run it locally first** — a trial trains a small MLP on a cached matrix rather than
fine-tuning a GNN, so the TamIA apparatus (httpproxy, 4.5 GB staging, whole-node b3 jobs) is
no longer obviously worth its overhead. Three A6000s here.

**Swept (7):** `n_layers`, `hidden_dim`, `embed_dim`, `epochs`, `head_lr`, `weight_decay`,
`dropout` — precisely the set `bayes_v1.yaml:185-204` marked *"not swept: blocked on the
architecture question"*. The MiniMol sweep was cut to five axes because bayes degrades with
dimension at ~140 trials; that argument is about the trials-to-axes ratio, and the trial count
here goes up by roughly an order of magnitude.

**Pinned:** `trainer: train_jepa`, `readout`, the four loss weights at the same values, plus
`batch_size 1200`, `weights balanced`, `pprop_norm zscore`, `eta_min`, `lr_schedule`, `w_vic 0`.

### `run_config.py` drives both arms — one dispatch point, no fork

`--trainer {train,train_jepa}` (default `train`), resolved from argv by `resolve_trainer()`
**before** the parser is built, because the trainer is what *defines* the parser.
`mirror_train_arguments` and `boolean_flags` take the module; the in-process call is
`args._trainer.main(...)`.

**On the `config_id` collision that does not happen.** The worry was that a Mol-JEPA run and a
MiniMol run over the same CSV would hash identically, land in the same bucket, and be *reused*
rather than retrained — silent, because `ID_EXCLUDED` drops `features` and the provenance
triple is held fixed by design. It dissolves once the trainers are separate programs:
`config_id` hashes `vars(args)`, and the payloads carry disjoint keys —
`freeze_epochs`/`trunk_lr`/`head_lr_unfrozen` on one side, `embeddings`/`readout`/`epochs` on
the other. **Verified: existing MiniMol ids are unchanged** (default config still `677f621f`).

`trainer` is therefore in `OWN_FLAGS` and **not hashed** — hashing it would buy nothing and
would re-stamp every existing MiniMol bucket. Belt-and-braces separation instead comes from:

- **`--outputs-root` defaults to `outputs/jepa_v1`** for this arm. Both probes `rglob` a root
  and neither gates on provenance, so directory separation is the only defence.
- **`aggregate`'s provenance tuple gained `encoder`** —
  `(objective_version, split_sha256, input_sha256, encoder)`, defaulting to `"minimol-v1"` so
  every meta.json written before the field existed still validates.

`scripts/sweep_trial.sh` gained `MINIMOL_EMBEDDINGS` -> `--embeddings`. It and
`MINIMOL_FEATURES` are **mutually exclusive**: `--features` exists only on `train.py`,
`--embeddings` only on `train_jepa.py`, so setting the wrong one for the sweep being served
makes argparse reject the whole command line — which is the loud failure.

---

## Inert on this branch

Not deleted, and not to be trusted here either:

- **`src/vn_*.py`, `src/verify_vn_taps.py`** (~1,400 lines) reach inside graphium's GNN by
  shadowing `virtual_node_layers[i].forward`. Nothing in them survives a trunk swap, and
  nothing on the training path imports them.
- **`src/export.py`, `src/export_pkg/`, `handover/`** build a MiniMol handover package.
- **`src/trunk.py`, `src/model.py`, `src/featurize.py`, `src/features.py`,
  `src/verify_trunk.py`** are the MiniMol arm. Still correct, still runnable, untouched.

---


`NOTES.md` is the reference document: background, source-level findings, and the full plan.
This file is the operational summary. When they disagree, `NOTES.md` §§1–11 is authoritative
on *why*; this file is authoritative on *what currently exists*.

`Minimol_architecture_overview.md` describes what MiniMol itself does between a SMILES string
and its 512-d output, measured against the pinned stack. Read it before touching `trunk.py`.

## The MiniMol arc this branch descends from

`frozen-moljepa` branches off **`encoder-vn`**, and everything below this line was written for
that arc — full-trunk fine-tuning of MiniMol. It is kept because the MiniMol arm is still
runnable here and its mechanics are still authoritative for `train.py`, `trunk.py`, the loss,
the metrics, the splits and the data. **Read it as the description of the other arm**, not of
this one.

Two things in it are now superseded on this branch and would mislead if followed literally:

- **"NEXT: what to run" below is `encoder-vn`'s next step**, not this branch's — it points at
  a TamIA hyperparameter tune of the MiniMol trunk followed by a probe of MiniMol's internal
  features. This branch's NEXT is at the top of this file.
- **The MiniMol-internal feature analysis (`vn_*`) and the width scan** belong to `encoder-vn`.
  The equivalent question here is natively answerable, and *without a trained checkpoint*: the
  13 tokens and their 12 projections are already on disk, so `feature_utility.py` (R1) and
  `emb_readout.py` (R2) could be pointed at any of them directly. That is the obvious follow-on
  once the readout is settled; it is deliberately not part of this branch's first pass.

```bash
git show encoder-vn:CLAUDE.md          # the MiniMol fine-tuning arc, as its own document
git log --oneline main..encoder-vn     # its 28 commits
```

---

## State

### `frozen-moljepa`, as of 2026-09-01

| Piece | Status |
|---|---|
| Branch off `encoder-vn` | **done** — `frozen-moljepa` |
| Environment (`jepa` extra) | **done, audited** — `transformers 4.57.6`, `safetensors`, `huggingface-hub`, `molfeat 0.11.0`. `uv lock` changed **no** pre-existing version; `torch 2.6.0+cu124` and `scipy 1.13.1` both held, no `cu13`/`cuda-toolkit` |
| Mol-JEPA loads and is frozen | **done** — 45,406,721 params, revision pinned to `4c912b45…` |
| Token layout | **measured, and it contradicted the model card** — `cls` is `modality_pred[0](embeddings[:,0])`, not a slice. Re-asserted bit-exactly on every run |
| Determinism | **measured and enforced** — cuda is 2.1e-06 off itself by default; `CUBLAS_WORKSPACE_CONFIG` + `use_deterministic_algorithms` bring it to 0.000e+00 at 218 vs 281 mol/s |
| Embedding extraction | **done, run** — `src/jepa_embed.py`; 331,480 molecules in **28.2 min at 196 mol/s**; 2 x `[331480, 13, 512]` float32, 17.6 GB, untracked |
| Cache loader + readout grammar | **done** — `src/jepa_features.py`; guards exercised |
| Reference fixture | **done** — `data/reference/moljepa_v1_ref64.{npz,json}`, `repeat max|Δ| = 0.000e+00` on cpu |
| Verification suite | **done, 8/8 PASS on the full cache** — `src/verify_jepa_embed.py` -> `verification_jepa.md` |
| Trainer | **done, run end to end** — `src/train_jepa.py`; **1.9 s/epoch over 265,184 rows**, 42 s for a 20-epoch model |
| `run_config.py` dual dispatch | **done, verified** — `--trainer`; existing MiniMol `config_id`s unchanged (default still `677f621f`) |
| Sweep | **written, parameters validated against the parser** — `sweeps/bayes_jepa_v1.yaml` |
| Measured trial cost | **~7 min for the full 5x2 grid** (10 x 42 s). Against ~40 min on an H100 for a MiniMol trial — which is why the sweep runs here, on three A6000s, rather than on TamIA |
| First numbers (orientation, not results) | see "A first readout scan" below |
| **The readout decision** | **OPEN — Ethan's call.** `--readout cls` is a placeholder |
| **What model sits on top** | **OPEN.** `DualHead` is the starting point, not the argument |
| Frozen-MiniMol baseline on these folds | **not yet run** — `run_config.py --fold-list 0 1 2 3 4 --seed-list 0 1 --unfrozen-epochs 0` |

### The MiniMol arc, as of 2026-08-25

| Piece | Status |
|---|---|
| Dataset subset (331,480 molecules) | **done** — `src/subset.py` |
| 5-fold cluster CV splits | **done, verified, frozen** — `src/split.py`, `src/splits.py` |
| Environment (uv, in-repo `.venv`) | **done, verified** — `pyproject.toml` + `uv.lock` |
| Trainable trunk | **done, 10/10 checks pass** — `src/trunk.py`, `src/verify_trunk.py` |
| Feature cache (Phase 2) | **done, verified** — `src/featurize.py`, `src/features.py` |
| Loss / metrics / objective ported from `pProp_MLP` | **done, 8/8 checks pass** — `src/losses.py`, `src/metrics.py`, `src/objective.py`, `src/verify_metrics.py` |
| Dual head (binary @ 3.5 + regression) | **done** — `head.DualHead` |
| Bottleneck + embedding export (`--embed-dim`) | **done, imported from the 32-d arc** — `head.py`, `train.py:173`. Width is a free variable; **the default 32 is inherited, not chosen** |
| R1 probe — held-out-cluster kNN / ridge | **done** — `src/feature_utility.py` |
| R2 probe — ECFP structural readout | **done** — `src/emb_readout.py` |
| Pooled out-of-fold tail metrics | **written, code path exercised; no real runs pooled yet** — `src/pool_oof.py` |
| Compute profile / benchmarks | **done** — `src/benchmark.py`, `reports/compute_profile.md` |
| Config-level runner (1 config = 1 wandb run) | **done, verified 2026-08-13** — `src/run_config.py` |
| TamIA sweep infrastructure | **written 2026-08-25, not yet run on TamIA** — `scripts/tamia_sweep_agent.sbatch`, `scripts/sweep_trial.sh`, `INSTRUCTIONS.md`. Guards and preflight verified locally; the cluster itself is unexercised |
| Hyperparameter tune (the branch's first experiment) | **not started** — see "NEXT" |
| MiniMol-internal feature analysis | **not started, design pending.** Post-hoc on a trained model, so it needs a `--keep-checkpoints` re-run of the sweep winner |
| Width scan | **deferred**, superseded by the above — see "Deferred: the width scan" |
| Hyperparameter sweep (wandb bayes) | **blocked on the width question** — `sweeps/bayes_v1.yaml` is stale; `bayes_v2.yaml` was deliberately not imported |
| R3 probe — GP uncertainty | **not imported.** Lives at `embedding-head-32d:src/uncertainty_probe.py`, repaired 2026-08-24; re-import if the width work needs an uncertainty reading |
| `src/export.py` | **done 2026-08-31** — builds a zippable handover package from one checkpoint. Templates in `src/export_pkg/` |
| Handover package for the downstream project | **done 2026-08-31** — `handover/minimol_ampc_encoder_v1`, built from a `--train-all` refit of the sweep's cfg1. See "The handover package" |
| Layer-wise freeze/unfreeze | **not started, deferred 2026-08-13** — see "Deferred: layer-wise freeze/unfreeze" |

The trunk reproduces frozen MiniMol embeddings **exactly** (max|Δ| = 0.000e+00 over 64×512),
gradients reach all 284 reachable trunk tensors (7,919,912 params), and an optimizer step
moves trunk weights. Two further tensors are unreachable inside graphium itself — see "The
rw_pos dead norm" below. `verification.md` reads `OVERALL: PASS`.

**Verified on this branch 2026-08-25, re-run against the final tree:** `verify_trunk.py`
`OVERALL: PASS` (10/10, plus the grad-flow meta-check that runs every time — 11 `[PASS]` lines
in total) and `verify_metrics.py` **8/8** — including the loss-parity check and the freeze
negative test, the two most at risk since `verify_metrics.py` is byte-identical to `main`'s
while `losses.py` arrived with 101 lines of change. Smoke run at `--embed-dim 64` wrote `val_embeddings.npy` at
**(5000, 64)**, which is the assertion that proves the width knob reaches the exported artifact.
Head is 1,644,866 params at width 64 against 1,611,938 at 32.

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
UV_HTTP_TIMEOUT=3600 uv sync --extra dev --extra jepa    # timeout matters: see below
```

**`--extra jepa` is this branch's addition**: `transformers>=4.50,<5` (the checkpoint's
`config.json` is stamped 4.50.3), `safetensors`, `huggingface-hub`, and `molfeat` — which is
genuinely required, not optional: `modeling_moljepa.GraphFeaturizer.__init__` imports
`AtomCalculator`, `EdgeMatCalculator` and `AdjGraphTransformer` from it, and they are what
produce the node_dim 82 / edge_dim 17 graph.

**Audited 2026-09-01, and it is clean**: resolving with the extra added 12 packages and
changed **no** pre-existing version — `torch` stayed `2.6.0+cu124`, `scipy` stayed `1.13.1`
(the `<1.14` pin graphium needs), `setuptools` stayed `80.10.2`, and there is no `cu13` or
`cuda-toolkit` anywhere in `uv.lock`. **Always run `uv lock` and grep before `uv sync`** —
resolution is metadata-only and free; the sync is not.

One documented fact this changed: molfeat drags in **`pyarrow` 25.0.1**, so the "pyarrow is
not installed anywhere here" note below is no longer true. Artifacts are still CSV + `.npy`,
which is a convention, not a constraint.

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

  jepa_embed.py                      SMILES -> Mol-JEPA -> the 17.6 GB cache, once
  jepa_features.py                   cache loader + readout grammar  <- use this
  train_jepa.py                      one fold, one seed, on FROZEN embeddings  <- this branch
  verify_jepa_embed.py               the 8-check suite -> verification_jepa.md
  dump_jepa_reference.py             the frozen fixture verify_jepa_embed.py checks against

  trunk.py / model.py / head.py      the trainable trunk, the two-group optimizer, DualHead
  train.py                           one fold, one seed  <- the entry point
  losses.py                          5-term loss (vic off by default) + the two weighting flavours
  metrics.py                         AP / correlation / error / enrichment, all suffixed
  objective.py                       goal_metric + the derived OBJECTIVE_VERSION
  normalization.py                   --pprop-norm; ported verbatim
  run_paths.py                       <outputs-root>/<sweep_id>/<run_id>  <- never hardcode a path
  pool_oof.py                        pooled out-of-fold tail metrics across the 5 folds
  run_config.py                      one hyperparameter config -> ONE wandb run  <- sweep entry
  feature_utility.py                 R1: held-out-cluster kNN + ridge on the export
  emb_readout.py                     R2: ECFP structural readout + bottleneck geometry
  verify_trunk.py / verify_metrics.py    the two verification suites
  dump_metric_reference.py           run under pProp_MLP's venv; feeds verify_metrics
  benchmark.py / concurrency.py / collect_runs.py / report_charts.py   the compute profile
sweeps/
  bayes_jepa_v1.yaml                 the frozen-Mol-JEPA sweep  <- this branch
  bayes_v1.yaml                      the wandb bayes sweep over the two-phase schedule
scripts/
  tamia_sweep_agent.sbatch           the TamIA sweep job: 4 agents, 1/GPU  <- the sweep entry
  sweep_trial.sh                     one trial; what `wandb agent` execs
  run_grid.sbatch                    the 5×2 grid as a SLURM array — LOCAL scheduler only,
                                     does NOT map to TamIA (compute_profile.md §5)
  sample_gpu.sh                      nvidia-smi telemetry sampler
reports/                             compute_profile.md + its evidence
INSTRUCTIONS.md                      the TamIA runbook  <- the operational authority
NOTES.md                             the reference document; §12 is the pProp_MLP translation
Minimol_architecture_overview.md     SMILES -> 512-d, measured; read before touching trunk.py
```

**`outputs/` is shared with the sealed 32-d arc.** This branch writes under `outputs/enc_v2/`
by default and the probes read from there by default. See "The outputs namespace" below —
it is the one mistake here that would produce a plausible number rather than an error.

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

### The deliverable is the encoder, not the predictions

The head is `512 → **embed_dim** → {Linear cls, Linear reg}` — **changed 2026-08-25**, and the
export is no longer a bottleneck. It was `512 → 1024 → 1024 → 32 → …`; the defaults are now
`--n-layers 0 --embed-dim 1024`, so the whole head is one `Linear(512→1024) → LayerNorm → GELU`
and two bare `Linear(1024→1)`. **529,410 params against 1,611,938.** The parameterisation did
not change — `--n-layers 2 --hidden-dim 1024 --embed-dim 32` restores the old shape exactly.
**The export point
is the output of `head.shared`** — post-norm, post-activation, the exact tensor the linear heads
consume — and it is written per run as `val_embeddings.npy`, `[n_val, embed_dim]`, final-epoch,
in the same row order as `val_predictions.npy` and `val_indices.npy`.

That inverts what "good" means. `goal_metric` scores the *predictions*, and the predictions are
not the product — they are a training signal for the representation. **Treat `goal_metric` as a
proxy and the R1/R2 probes as the thing being optimised.**

`--embed-dim` (`train.py`, `type=sweep_int`) sets the width. Its default was 32, inherited
from the sealed arc rather than chosen; **it is 1024 as of 2026-08-25**, which is likewise a
decision rather than a measurement — no run in this repo has yet compared two widths on the
R1/R2 probes. That is still what the deferred width scan exists to fix. Measured 2026-08-25: at width 64 the
head is 1,644,866 params against 1,611,938 at 32, and `val_embeddings.npy` comes back
`(5000, 64)` on a `--subset 5000` smoke run. **At the new default of 1024 the same smoke run
gives `(5000, 1024)` and a 529,410-param head** (verified 2026-08-25, alongside
`verify_trunk.py` 10/10 and `verify_metrics.py` 8/8).

Two properties the arc chose deliberately, both now **open rather than settled**:

- **The bottleneck is shared, not per-task.** The deliverable is one vector; a per-task
  bottleneck would be shaped by a single objective.
- **The task heads are linear** (`--cls-n-layers 0 --reg-n-layers 0`), which makes "pProp is
  linear in the exported embedding" literally true. That was chosen for an RBF/Matérn kernel.
  Under a **DKL** — which supplies its own warping — the property is worth less than it was,
  and it is not free: a target affine in the export pushes the marginal likelihood toward a very
  long lengthscale, so the kernel goes nearly flat.

### The outputs namespace — the one silent failure mode

**`--outputs-root` defaults to `outputs/enc_v2`, not `outputs/`** (`train.py:241`; one edit
covers `run_config.py` too, since `mirror_train_arguments` walks `train.py`'s parser).

The reason is measured: **61 runs from the 32-d arc carry `val_embeddings.npy` under
`outputs/`** across seven roots (`wvic_scan`, `rank_v1/2/3`, `ckpt_v4`, `_no_sweep`, `_verify`).
Both probes discover runs by `rglob` over a root, and **neither gates on the provenance
triple** — `emb_readout.py` records `objective_version` / `split_sha256` / `input_sha256` as CSV
columns but never filters on them, and `feature_utility.py` filters only on `--cells`.
Directory separation is the only defense.

It is a **default rather than a documented convention** because a probe that silently pools two
arcs returns a plausible number, not an error. Both probes' `--runs` defaults were repointed to
`outputs/enc_v2` for the same reason; they previously defaulted to `outputs/rank_v1` and
`rank_v2`, which are the old arc. `--outputs-root` is excluded from `config_id`, so none of this
moves what is trained.

**Never point a probe at bare `outputs/`.** Measured 2026-08-25: `outputs/enc_v2` holds 1 run,
bare `outputs/` finds 62.

### The export gap — CLOSED 2026-08-31

`src/export.py` now exists and `handover/minimol_ampc_encoder_v1` is a built, verified package.
See "The handover package" below. What follows is still true of any *other* run:
`run_config.py` forces `--no-save-checkpoint`, so a sweep trial is not exportable and the
configuration must be re-run with `--keep-checkpoints` first.

The head is trivially portable; **the trunk is not.** Reconstructing it needs graphium 2.4.7,
the minimol 1.3.4 wheel, and `trunk.py`'s exact construction ordering, so the downstream project
must vendor `trunk.py` / `head.py` / `model.py` or put this `src/` on `sys.path`. State this in
any handover.

### The handover package — `src/export.py`

**Built 2026-08-31 for the downstream active-learning project.**
`handover/minimol_ampc_encoder_v1` (35 MB, zips to 32 MB) is a self-contained
`SMILES -> R^512` encoder: weights, the four modules needed to rebuild the trunk, pinned
requirements, two documents, two examples, and a fixture.

```bash
python src/train.py --train-all --seed 0 --out handover/_train/cfg1_all ...   # 9.1 min
python src/export.py --checkpoint handover/_train/cfg1_all/final.pt \
                     --out handover/minimol_ampc_encoder_v1
```

`src/export.py` and the templates in `src/export_pkg/` are the source. `.gitignore` covers the
built packages and the run's 271 MB of `val_*.npy`, but **deliberately not
`handover/_train/*/final.pt`** — that checkpoint is the input a package is built from, and an
ignored `handover/` would have made the one irreplaceable file the one invisible to
`git status`. It is untracked, not ignored; committing or archiving it is a decision to take
knowingly. `handover/` deliberately is **not** under `outputs/` — both probes `rglob` a
root and neither gates on the provenance triple, so a `val_embeddings.npy` there would be
silently pooled into an analysis.

- **The shipped export is `pooled512`, not `z`** — `reports/vn_feature_analysis.md`'s
  recommendation. `z`'s effective rank is 23.5 ± 6.7 of 1024 and its `scalarness` 0.843, so
  its distances largely restate predicted pProp, which is the one thing a GP kernel must not
  be handed.
- **The shipped weights were refit on all 331,480 rows** and have no held-out set. The card
  quotes the ten cfg1 bootstrap siblings that did hold fold 0 out, and states plainly that
  every molecule in `ampc_subset_331k.csv` is in-sample — an AL benchmark scored on that
  subset reads optimistically. In-sample `goal_metric` is **1.2973** against the siblings'
  held-out **1.0186**, which is the size of the gap being warned about.
- **`export.py` refuses a checkpoint not trained with `--train-all`** (unless
  `--allow-partial-train`), because the card it generates claims the full dataset.
- **Both documents are generated**, every number interpolated from the run's `meta.json` and
  the siblings' — no figure is typed into a template.
- Verified: `verify_install.py` 5/5 from outside the repo, from an unzipped copy, on GPU
  (max|Δ| 5.6e-06) and CPU (**0.000e+00**); both examples run; the four vendored modules
  `diff` clean against `src/`; and both module-import orders leave the importer's `sys.path`
  and `sys.modules` untouched.

Measured while building it: **12,600 mol/s** to encode on an A6000, against ~1,745 mol/s to
featurize — which is why the package's `featurize.py` caches graphs and the AL example
reloads them.

### The task is joint, and binary at pProp ≥ 3.5

`head.DualHead` puts a **classification logit** and a **regression scalar** on one shared MLP
over the 512-d embedding, ending at an `--embed-dim` block — that block's output is the
exported encoder. At the default `--n-layers 0 --embed-dim 1024` that shared MLP is a single
widening layer, not a narrowing stack. 3.5 is the threshold at which binders become possible, and it is
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

`vic` is a VICReg-style variance + covariance penalty on the bottleneck, **off by default**
(`--w-vic 0.0`). It came across from the 32-d arc as an available tool; **its `w_vic=3` pin did
not.** That pin was selected against a 32-d-specific target and is not binding here — see the
sealed record if you need its evidence.

Huber is **grounded at weight 1** — the only term anchoring the absolute pProp level, so the
others are measured against it and only its `delta` is swept. Three deliberate
deviations from the port:

- **`cls` is BCE normalised by `w.sum()`, not by `N`.** `CrossEntropyLoss(weight=w)` divides
  by `Σw`; `BCEWithLogitsLoss(reduction="mean")` divides by `N`. At a 104× class ratio those
  differ by a large constant, which would move every inherited `w_cls` out of the units it
  was swept in. Verified equal to `CrossEntropyLoss(weight=…)` on 2 classes.
- **`std_match_loss` takes the sample weights.** The group-weighted target std is **1.455**
  against **0.864** unweighted, so an unweighted std term would pull prediction spread toward
  the smaller number while the weighted huber pulled toward the larger. They would fight.
- **`pair` stays unweighted**, exactly as ported. It is O(B²) — ~1.4M pairs at batch 1200.

`combined_loss` returns the term values **unscaled**, and they are logged that way: a
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

**The operational procedure lives in `INSTRUCTIONS.md`, not here.** That file is the runbook —
account setup, data transfer, the proxy, the gate, troubleshooting. This section is the *why*.

**`bayes_v1.yaml`'s parameter ranges are stale.** It predates the bottleneck entirely, so it
sweeps neither `--embed-dim` nor `--w-vic`. `bayes_v2.yaml` was deliberately **not** imported to
this branch — its ranges and its "`w_vic` is the most important axis" framing both assume the
32-d pin. Its *mechanics* below are correct and were repaired 2026-08-25; its *scope* is an open
question (see NEXT).

#### CORRECTED 2026-08-25: `wandb agent` does NOT work unaided on TamIA

This section previously read "**`wandb agent` does work there** — confirmed 2026-08-13,
superseding `reports/compute_profile.md`'s 'a wandb sweep cannot run on TamIA'". **That is
false, and it had the supersession backwards.** `reports/compute_profile.md:331` was right:
**TamIA's compute nodes have no direct internet**, and `wandb agent` must reach the wandb server
to fetch each configuration.

The failure mode is what makes this worth a correction rather than a footnote: without a route
out, **`wandb agent` does not error — it blocks**, waiting for a config that never arrives, while
the job looks healthy and burns its whole allocation.

**The route that does work is `module load httpproxy`** (Alliance support, 2026-08-25) — an
Lmod module that sets `http_proxy`/`https_proxy` itself, so there is no host:port anywhere in
this repo. Runs go under a **new wandb account on Ethan's Mila email**, so the `ethan_personal`
entity that `train.py:256` still defaults to is *not* the entity this sweep runs under;
`scripts/tamia_sweep_agent.sbatch` requires the entity explicitly and refuses to start without
it, rather than falling back into an entity that account cannot write to.

**`module` is a shell function, and that is a live footgun.** Only a function can export into
the calling shell, so `module load httpproxy` must not be piped or run in a command
substitution — doing either runs it in a subshell and the proxy variables vanish with it. This
was written as `module load httpproxy | sed ...` for prettier output and failed exactly that
way; the script now checks that `http_proxy`/`https_proxy` are non-empty **after** the load,
because `module load` of a missing module can still exit 0.

#### Two things about TamIA that change the job shape

- **Whole-node allocation** (`compute_profile.md` §5), jobs expected ≥1 h.
  `scripts/run_grid.sbatch` — ten single-GPU array tasks of ~3 min — is the shape that report
  calls one that "does not map to TamIA at all". Hence one long job running **one agent per GPU**.

  **The bynode partitions ARE the time buckets** — `b1` 3 h, `b2` 12 h, `b3` 24 h
  (`compute_profile.md:303`, `sinfo` from tamia2). So `--partition` and `--time` must move
  **together**: asking for 24 h while still naming `b2` is *rejected at `sbatch`*, not silently
  truncated. The job is `gpubase_bynode_b3` / `--time=24:00:00` as of 2026-08-26; it was `b2` /
  12 h. **24 h is the cluster maximum**, so searching longer means a *second job against the same
  sweep id* — agents attach to a sweep, so two jobs serve it together — not a longer one. The
  `INSTRUCTIONS.md` §F gate jobs stay on `b1`: they are minutes, not hours.

  **Measured 2026-08-25** (`sinfo -p gpubase_bynode_b2 -o '%c %m %G'`; the two bynode partitions
  are the same nodes under different time caps, so this holds for `b3` too), correcting this
  file's earlier "4 GPUs" as if it were universal — **the partition holds two node types**:

  | type | CPUs | memory | GPUs |
  |---|---|---|---|
  | **h100** | 48 | 500 GB | **4** |
  | h200 | 64 | 1000 GB | **8** |

  So the agent count is **derived from the GPUs actually present**, not fixed — on an h200 node a
  hardcoded 4 would idle half a whole-node allocation. The job pins `--gres=gpu:h100:4`
  deliberately: h200 is faster per GPU (~4.3× vs ~3.4× projected at batch 1024) but carries
  eight, and past roughly 8–16 concurrent trials a bayes sweep draws every trial from the same
  stale posterior and degenerates toward random search. Account is `aip-yvesbrun`.
- **Stage the feature cache to `$SLURM_TMPDIR`.** `train.py:620` calls `load_features()` inside
  `main()` and `run_config.py` calls `train.main()` per model, so the **4.5 GB cache is re-read
  every trial**. The "1.0 s to load" figure in this file was measured on warm local disk; from a
  networked home with four agents contending it would dominate the run.

```bash
# on a LOGIN node -- compute nodes cannot create a sweep
wandb sweep --project finetune_minimol --entity <mila_entity> sweeps/bayes_v1.yaml
sbatch --account=<def-xxx> scripts/tamia_sweep_agent.sbatch <sweep_id>
```

`program:` is **`scripts/sweep_trial.sh`**, a thin wrapper that execs `src/run_config.py` — so one
trial is still one configuration is one run. The wrapper exists because a sweep is created **once**
and then served by whatever agents attach to it, so the yaml must carry nothing machine-specific:
it previously hardcoded `/home/ethan2/finetune_minimol/.venv/bin/python`, which does not exist on
TamIA. The wrapper resolves the interpreter from its own location and adds the `$SLURM_TMPDIR`
data paths, which do not exist when the sweep is created. **Its flags go before `"$@"`**, because
`--wandb-tags` is `nargs="*"` and must stay last on the line.

**The objective is `final/goal_metric_mean`** — the mean over models of each model's
final-epoch `goal_metric`. Two things it is deliberately not:

- **Not the best epoch.** The per-epoch curve `agg/goal_metric_mean` is logged beside it so the
  gap stays visible; optimising *that* would be early stopping on the same validation fold the
  run reports, exactly the bias the fixed epoch budget exists to avoid. (This file used to say
  that curve carries `summary="max"`. It does not — `run_config.py:369` declares `summary="max"`
  on `final/goal_metric_mean`, which is never `run.log`'d but written once through
  `run.summary.update`, so the declaration is **inert**. Behaviour is correct either way: the
  objective is the value written at the end.)
- **Not the pooled score**, though pooling is the more honest tail estimate (100 potent
  molecules against 20 per fold). Pooling is *undefined* unless a seed holds all five folds, so
  it does not exist for any trial narrower than the full grid, whereas the mean is defined for
  any subset — which is what lets a cheap trial and a full confirmation share one sortable
  column. **Since 2026-08-26 every trial does hold all five folds**, so `pooled/*` is in fact
  populated throughout; the objective stays the mean regardless, so the column keeps its meaning
  if the cost dial is ever turned back down. Report `pooled/*` from the winner.

`fold_list` / `seed_list` are **the cost dial**, pinned as parameters: `"0"` × `"0"` is one
model per trial (~4 min on H100), `"0,1,2,3,4"` × `"0,1"` is the full grid (~40 min). **Set to
the full grid 2026-08-26** — see "NEXT" for what that buys and what it costs. Spell
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
5. **How does this interact with the deliverable?** The product **is** a pProp-shaped
   molecular encoder (settled 2026-08-17; its *width* reopened 2026-08-25). Freezing 14 of 16
   GNN blocks leaves the representation mostly pretrained, which argues for unfreezing more;
   feature distortion (LP-FT, Kumar et al. 2022) and the fact that a generative loop will query
   novel chemistry argue for less. That trade-off is a research question, not an implementation
   detail — and it should be decided on the R1/R2 probes over held-out clusters, **not** on
   `goal_metric`, because `goal_metric` scores the predictions and the predictions are not the
   product.

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
- `val_embeddings.npy` (`[n_val, embed_dim]`) is written per run beside the predictions — the
  exported artifact itself, final-epoch, in the same row order. It is what both probes read.
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
- **The structural readout returns `nan` on a `--subset` run, and that is not a bug.**
  `emb_readout.py` samples molecules from the *frozen* fold (`splits.load_fold`), not from the
  run's own `val_indices.npy`, so a subset run has almost nothing to join against. Measured: a
  `--subset 5000` run holds 5,000 of fold 0's 66,296 val rows, so a 500-molecule sample
  overlaps by ~38 and every structural column comes back `nan` while the geometry columns —
  which read `val_embeddings.npy` directly — compute normally. **Score probes on full runs
  only.**
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
- **`data/` is 1.5 GB, and since 2026-08-25 about 141 MB of it is tracked.** `.gitignore`
  excludes `data/*` wholesale and un-ignores: `data/reference/` (the frozen MiniMol embeddings
  `verify_trunk.py` checks against), `data/*.meta.json`, **`data/ampc_subset_331k.csv`** (29 MB)
  and **`data/splits/cluster_kfold_v1/`** (112 MB).

  The splits used to be excluded as "regenerable in ~2.5 min from `src/split.py`". **That
  reasoning was wrong**: rdkit shifts Morgan fingerprint details between releases, so a rebuild
  on another machine yields a different partition and a different `split_sha256` — see the
  Morgan footgun below, which this file already stated two sections away from the claim it
  contradicted. They must be copied, so they are tracked. Accepted cost: 141 MB of permanent
  history. `fingerprints.npy` is 81 MB, under GitHub's 100 MB hard limit — keep it that way.

  **`data/features/minimol_v1/` stays untracked** and has its own `.gitignore` rule: 4.5 GB, and
  genuinely regenerable, because `src/featurize.py` depends on the CSV rather than on an rdkit
  version.

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
