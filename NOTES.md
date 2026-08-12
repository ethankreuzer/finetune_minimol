# Fine-tuning the MiniMol trunk — background, findings, and plan

> Status: the trunk is **built, verified and training**. The environment, the trainable
> trunk (`src/trunk.py`, 10/10 checks), the feature cache, `src/train.py`, a 10-run grid and
> a full compute profile (`reports/compute_profile.md`) all exist. The loss, metrics and
> selection objective have been ported from `pProp_MLP` (§12) and pass 8/8 checks in
> `src/verify_metrics.py`. What remains is the hyperparameter sweep and its driver.
> §§1–11 below are the original research record and remain authoritative on *why*; where a
> statement there has been superseded by what was actually built, §12 says so.
> Last updated 2026-08-12.

---

## 1. The goal

MiniMol is a pretrained molecular foundation model (~10M parameters) that turns a SMILES
string into a **512-dimensional embedding**. The normal way people use it is as a frozen
feature extractor: run your molecules through once, get 512 numbers each, then train a
small MLP on those numbers to predict some property. The big model never changes.

**That is not what this repo is for.** The goal here is to fine-tune the **entire MiniMol
trunk** — to let gradients flow back through all ~10M parameters and update them for a
specific downstream task, rather than only fitting a decoder on top of a fixed embedding.

Why this is worth doing: a frozen embedding was optimised for MiniMol's *pretraining*
objective, not for your task. Fine-tuning lets the representation itself reshape around
your data. It usually performs better when you have enough labelled data, and it is
strictly more expressive — a frozen-embedding MLP is a special case of fine-tuning where
the trunk's learning rate is zero.

**The downstream task** (settled since this document was first written): regression on AmpC
docking score over a 331,480-molecule enriched subset of a 10M library, built by
`src/subset.py` into `data/ampc_subset_331k.csv`. Evaluation is 5-fold cross-validation on
a cluster split, each fold trained under two model seeds — 10 runs. See §11.

**Settled 2026-08-11: the regression target is `pprop`, not `score`.** The head predicts
pProp and the loss is computed against it. `score` remains in the CSV and is monotone in
`pprop`, so it stays available for reporting without retraining.

**Loss weighting is deferred, but it *will* exist.** Ethan intends to weight molecules in
the loss; the scheme is not yet chosen, and `ipw` is only one candidate — the question is
therefore broader than "does `ipw` enter the loss". Design consequence: the training step
takes a **per-sample weight vector, defaulting to uniform**, so a scheme can be dropped in
without touching the training loop. Do not hardcode an unweighted mean.

> **Settled 2026-08-12 — see §12.6.** The scheme is `balanced`, and `ipw` was rejected
> outright: it is the subsampling rate `subset.py` applied, a description of how the dataset
> was assembled rather than a modelling choice, so it weights neither the loss nor any
> reported metric. The per-sample weight vector survives exactly as designed above.

Still open: the exact head architecture and the headline metric.

---

## 2. Background you need for the rest of this document

### 2.1 How Python code gets onto your machine

A **package** is a folder of `.py` files with an `__init__.py`. MiniMol's is three items:
`__init__.py`, `model.py`, and `ckpts/` (the weights).

Packages ship in two formats:

- **sdist** (`.tar.gz`) — raw source. pip must *run* `setup.py` on your machine to install
  it. Slow, and can fail if a compiler is needed.
- **wheel** (`.whl`) — **a ZIP file with a standardised filename.** That is genuinely all
  it is; you can `unzip` one. It's pre-built, so installing is just "copy files into
  place." No build step, nothing to fail.

The filename encodes compatibility:

```
minimol-1.3.4-py3-none-any.whl
   │      │     │    │    │
   │      │     │    │    └─ any OS
   │      │     │    └────── no ABI constraint
   │      │     └─────────── any Python 3
   │      └───────────────── version
   └──────────────────────── name
```

`py3-none-any` means pure Python, works anywhere. (Contrast torch:
`cp310-cp310-manylinux_x86_64` — CPython 3.10 on Linux x86 only, because it contains
compiled CUDA code.)

`pip install minimol` downloads the wheel from PyPI, unzips it, reads its dependency list,
and repeats for each dependency.

### 2.2 Where it lands, and what `import` does

Into **`site-packages`** inside the active environment:

```
/home/ethan2/local/conda/envs/minimol_ft/lib/python3.10/site-packages/
├── minimol/
│   ├── model.py                    ← the 130 lines, readable and greppable
│   └── ckpts/minimol_v1/
│       ├── state_dict.pth          ← pretrained weights
│       ├── config.yaml
│       └── base_shape.yaml
├── graphium/                       ← equally readable; this is where the real code is
└── torch/, hydra/, ...
```

`import minimol` works by Python searching the directories in `sys.path` in order for a
folder named `minimol` containing `__init__.py`. No registry, no database — directory
search.

**So pip gives you the full source.** For a pure-Python package the wheel contains the
`.py` files verbatim — nothing compiled, stripped, or hidden.

**What pip does not give you is a tracked place to change it.** You *can* edit files in
`site-packages`, but those edits are (a) invisible to git, (b) silently erased by any
reinstall or env rebuild, and (c) undiscoverable six months later when a result won't
reproduce. That problem is what `pip install -e` ("editable install") solves: instead of
copying files in, pip drops a small text file containing the *path* to a source directory
you control, so imports follow your working copy and your edits can live in a git repo.

**Editable install is the tool for when you must modify a library.** Whether we need it is
the question section 5 answers.

### 2.3 git-LFS, and why big files break git

Git was built for source code: small text files where storing a diff per change is cheap.
It stores every version of every file forever. Put a 40 MB binary in a repo and change it
ten times, and the repo carries 400 MB that nobody can diff. Clones become enormous.

**git-LFS (Large File Storage)** fixes this by sleight of hand. The real binary goes to a
separate server. What git tracks is a **pointer file** — a ~130-byte text stub containing
a hash and a size. When you clone with LFS installed, it reads the pointers and fetches
the real files. Without LFS installed, you get the stubs, and your "weights file" is 130
bytes of text.

MiniMol does this. From `.gitattributes`:

```
minimol/ckpts/minimol_v1/state_dict.pth filter=lfs diff=lfs merge=lfs -text
```

This is the specific reason forking the repo is risky: **GitHub forks do not reliably
carry LFS objects.** You can fork, clone, and end up with a pointer stub where the model
should be — and the failure looks like a corrupt-checkpoint error, not a missing-file error.

### 2.4 The stack: graphium vs. MiniMol vs. Fingerprinter

The naming hides the real structure:

```
┌──────────────────────────────────────────────────────────────┐
│ graphium 2.4.7  (datamol-io, Apache-2.0)  ← ALL real machinery│
│   • featurization: SMILES → PyG graph                        │
│   • nn/architectures/ → FullGraphMultiTaskNetwork            │
│   • trainer/predictor.py → PredictorModule (Lightning)       │
│   • config system (hydra YAML)                               │
│   • finetuning/fingerprinting.py       → Fingerprinter       │
│   • finetuning/finetuning_architecture.py → the classes we want│
└──────────────────────────────────────────────────────────────┘
                            ▲
                            │ minimol depends on graphium (pinned ==2.4.7)
┌──────────────────────────────────────────────────────────────┐
│ minimol 1.3.4  (graphcore-research, MIT)  ← ~130 lines + weights│
│   • state_dict.pth   the pretrained 10M parameters           │
│   • config.yaml      which graphium architecture to rebuild  │
│   • model.py         glue: rebuild arch, load weights        │
└──────────────────────────────────────────────────────────────┘
```

**The key realisation: MiniMol is not a model architecture. It is a checkpoint plus a
recipe.** The layers, the featurizer, the training infrastructure are all graphium. Every
meaningful line of `model.py` is a call into graphium.

Consequence: fine-tuning MiniMol is mostly a *graphium* exercise. MiniMol just supplies
the starting weights.

**`Fingerprinter`** is a graphium utility. A model normally runs input → all layers →
prediction. Fingerprinter intercepts an **intermediate** layer and returns those
activations instead. You name the layer with a string: `'gnn:15'` = module `gnn`, layer 15.

### 2.5 Autograd: what makes learning possible, and what switches it off

When you operate on tensors, PyTorch records each operation into a **computational
graph**, retaining intermediate activations. `loss.backward()` walks that graph backwards
applying the chain rule, and writes each parameter's gradient into `parameter.grad`. The
optimizer reads `.grad` and updates the parameter.

The chain is strict:

```
graph recorded → backward() works → .grad populated → optimizer updates → learning
```

Break the first link and nothing downstream happens.

**Four mechanisms that get confused.** They are independent, and conflating them causes
real bugs:

| Mechanism | Effect | Blocks backprop? |
|---|---|---|
| `torch.no_grad()` | records no graph in the block | **Yes** |
| `torch.inference_mode()` | records no graph **and permanently stamps outputs as "inference tensors"** that autograd will never accept, even outside the block | **Yes, irreversibly** |
| `param.requires_grad = False` | that parameter receives no gradient (this is "freezing") | Yes, that param only |
| `.eval()` / `training = False` | dropout **off**, BatchNorm uses stored statistics | **No** |

**Why would an author deliberately switch off gradients?** Not sabotage — optimisation:

1. **Memory.** The graph must retain every intermediate activation so `backward()` can use
   them. For a 10M-parameter GNN over batched molecular graphs, activations typically
   dwarf the parameters. Dropping them can cut memory several-fold, allowing much larger
   batches.
2. **Speed.** Bookkeeping skipped is time saved.
3. **Safety.** During evaluation you don't want stray gradients accumulating.

For MiniMol's stated purpose — *"molecular fingerprinting using pre-trained deep nets"* —
disabling gradients is the *correct* engineering choice. It only obstructs someone doing
what the authors didn't design for. Which is us.

---

## 3. What `minimol/model.py` actually does

`__init__` — every line delegates to graphium:

```python
state_dict_path = pkg_resources.resource_filename('minimol.ckpts.minimol_v1', 'state_dict.pth')
cfg = self.load_config(...)                       # hydra.initialize + hydra.compose
self.cfg, accelerator_type = load_accelerator(cfg)
self.datamodule = load_datamodule(self.cfg, accelerator_type)
model_class, model_kwargs = load_architecture(cfg, in_dims=self.datamodule.in_dims)
predictor = load_predictor(...)
self.set_training_mode_false(predictor)
predictor.load_state_dict(torch.load(state_dict_path), strict=False)
self.predictor = Fingerprinter(predictor, 'gnn:15')
self.predictor.setup()
```

`__call__` — per batch: featurize → collate → extract → pool:

```python
input_features, _ = self.datamodule._featurize_molecules(smiles[i:(i + batch_size)])
batch = Batch.from_data_list(input_features)
batch = {"features": batch, "batch_indices": batch.batch}
node_features = self.predictor.get_fingerprints_for_batch(batch)
fingerprint_graph = global_max_pool(node_features, batch['batch_indices'])
```

So the 512-d embedding is a **global max-pool over node features taken from GNN layer 15**.
`__call__` returns a Python *list* of per-molecule tensors, not a batched tensor.

---

## 4. The decisive finding: `Fingerprinter` hard-blocks gradients

From `graphium/finetuning/fingerprinting.py` at tag `2.4.7`:

```python
def get_fingerprints_for_batch(self, batch):
    ...
    # Run the batch through the model.
    with torch.inference_mode():
        if self.predictor is not None:
            batch["features"] = self.predictor._convert_features_dtype(batch["features"])
        self.network(batch["features"])

    readout_list = []
    for module_name, layers in self._spec.items():
        readout_list.extend(
            [self.network._module_map[module_name]._readout_cache[layer].cpu() for layer in layers]
        )
```

`torch.inference_mode()` — the strongest of the four mechanisms in §2.5. The `.cpu()` on
each cached readout compounds it.

**Therefore `Minimol.__call__` cannot be used for fine-tuning.** Verified from source, not
inferred. The class docstring states the intent explicitly:

> *"This two-stage process is similar in concept to linear-probing"*

Linear probing is exactly the frozen-embedding workflow this project rejects.

> **Correction of record:** I earlier assumed Fingerprinter probably did *not* block
> gradients, on the grounds that `minimol/model.py` contains no `no_grad`. That assumption
> was wrong. `model.py` is clean; the block is one layer down in graphium.

---

## 5. graphium ships official full-trunk fine-tuning

`graphium/finetuning/finetuning_architecture.py` (348 lines) defines three plain
`nn.Module`s. Grepped for `inference_mode`, `no_grad`, `.eval()`, `requires_grad = False`
— **none present**, so they are trainable by default:

| Class | Role |
|---|---|
| `FullGraphFinetuningNetwork` | pretrained trunk + optional new layers + finetuning head |
| `PretrainedModel` | loads a pretrained net, rebuilds it with modified kwargs, copies shared weights |
| `FinetuningHead` | swappable task-specific head |

The copying mechanism:

```python
def overwrite_with_pretrained(self, pretrained_model, finetuning_module: str,
                              added_depth: int = 0, sub_module_from_pretrained: str = None):
```

Copies pretrained weights layer-by-layer up to `finetuning_module`, leaving the final
`added_depth` layers freshly initialised. That is precisely "load the trunk, attach a new
head, train everything."

Gradual freezing/unfreezing schedules are *not* in this file — see
`graphium/finetuning/finetuning.py` (likely a Lightning callback) if wanted.

graphium is **Apache-2.0**. The "Unauthorized modification... prohibited" header on those
files is boilerplate; the licence permits modification.

**This is the answer to "if we change no source code, how do we fine-tune?"** There is a
purpose-built API one layer below MiniMol, which MiniMol simply doesn't expose.

---

## 6. Decisions and open questions

### 6.1 Packaging: plain pip, no fork, no submodule

The reasoning changed twice as evidence came in, so here is the trail:

1. **Initially: fork + submodule.** On the assumption that enabling gradients would require
   patching MiniMol's source, and patches need a tracked, pushable home.
2. **Reversed to plain pip.** Because (a) `model.py` contains no gradient blocking, so the
   anticipated patch didn't exist; and (b) the checkpoint is git-LFS (§2.3), making forks
   risky — while `setup.py` declares `package_data` shipping `state_dict.pth`,
   `config.yaml`, and `base_shape.yaml` *inside the wheel*, so `pip install` obtains the
   weights with no LFS involved.
3. **Confirmed after §4 and §5.** The gradient block turned out to be in graphium, not
   MiniMol — but the fix is still not a source edit. It's either graphium's official
   fine-tuning API or a reimplementation in our own `src/`. Either way, no library source
   changes, so no fork.

```bash
source /home/ethan2/local/conda/etc/profile.d/conda.sh
mamba create -n minimol_ft python=3.10 -y && conda activate minimol_ft
pip install minimol==1.3.4      # brings graphium==2.4.7, hydra-core, and the weights
```

Do **not** install into the existing `my_conda_env` — graphium pins torch/PyG tightly
enough to break it. Reproducibility comes from pinning `minimol==1.3.4` +
`graphium==2.4.7`, recorded in this repo.

If a real need to patch either library appears later, the fork comes back — and the LFS
workaround is to take the checkpoint from the wheel rather than from LFS.

### 6.2 Open question: checkpoint format mismatch

`PretrainedModel.__init__` loads via:

```python
pretrained_model = PredictorModule.load_pretrained_model(pretrained_model, device="cpu").model
```

The docstring says the argument may be *"from a checkpoint path."* But that function looks
like it expects a **Lightning checkpoint** — weights *plus* embedded hyperparameters —
whereas MiniMol ships a **bare `state_dict.pth`** (weights only, loaded by its own code
with `torch.load` + `load_state_dict(strict=False)`). Different formats.

**This is the next thing to resolve**, in order of preference:

1. Read `PredictorModule.load_pretrained_model` and see what it actually accepts.
2. Bridge: build the predictor as `minimol/model.py` does, then re-save it as a proper
   Lightning checkpoint for `PretrainedModel` to consume.
3. Fall back to §6.3.

### 6.3 Fallback: reimplement Fingerprinter without `inference_mode`

Fingerprinter's mechanism is now fully visible, and none of it requires `inference_mode`:

1. `network._enable_readout_cache(['gnn'])`
2. `network(batch['features'])`
3. read `network._module_map['gnn']._readout_cache[15]`

Our own module can do the same **outside** `inference_mode` and **without** `.cpu()`, then
`global_max_pool(...)` → a differentiable 512-d vector. ~15 lines in `src/`, no library
modification.

Cost: `_enable_readout_cache`, `_module_map`, `_readout_cache` are private APIs (leading
underscore = not a stability promise). Acceptable pinned to 2.4.7; fragile across upgrades.
Prefer 6.2 if it works.

### 6.4 Verified vs. unverified

**Verified by reading source at pinned versions:**
- `minimol/model.py` contains no `no_grad`/`inference_mode` (all 130 lines read)
- `Fingerprinter.get_fingerprints_for_batch` uses `torch.inference_mode()`
- `finetuning_architecture.py` has no gradient-blocking constructs
- minimol is MIT, graphium Apache-2.0; minimol default branch `main`, last push 2025-05-29
- `setup.py` declares the checkpoint in `package_data`; `setup.py` exists at repo root
- `.gitattributes` marks `state_dict.pth` as git-LFS

**Not yet verified:**
- That `state_dict.pth` is really inside the published wheel at full size
  → `pip download minimol==1.3.4 --no-deps -d . && unzip -l minimol-*.whl`
- What `PredictorModule.load_pretrained_model` accepts (§6.2)
- Whether `'gnn:15'` is the final GNN layer or an intermediate one — `config.yaml` would
  say. This affects what "the whole trunk" means: if layer 15 is intermediate, layers
  beyond it contribute nothing to the embedding and shouldn't be in the optimizer.
- That gradients actually reach trunk parameters end-to-end (the §8 test)

---

## 7. What needs to be done

### Phase 0 — environment and verification
- Create `minimol_ft` env, `pip install minimol==1.3.4`
- Confirm the checkpoint is in the wheel and loads
- Confirm `Minimol()(["CCO"])` returns a 512-d vector (sanity check that the stack works)
- Resolve §6.2 by reading `PredictorModule.load_pretrained_model`

### Phase 1 — a trainable trunk (`src/trunk.py`)
The core deliverable. Either graphium's `FullGraphFinetuningNetwork` (§5) or the fallback
(§6.3). Requirements:
- Reuses MiniMol's config + `state_dict.pth`
- **Asserts** on the `load_state_dict` return value — see §9
- Does *not* call `set_training_mode_false`; trunk in `.train()` during training
- Returns a differentiable 512-d embedding
- Passes the §8 gradient test

### Phase 2 — data pipeline (`src/data.py`, `src/featurize.py`)
- Featurize the dataset **once** and cache to disk (§9) — do not re-featurize per epoch
- Use graphium's featurizer so features match what the trunk was pretrained on
- PyG `DataLoader` with the matching collate — **shuffle explicitly** (§9)
- Splits: **done**, see §11. Load with `src/splits.py`; do not re-derive them per run

### Phase 3 — training (`src/train.py`, `configs/`)
- Task head sized to the downstream task (blocked on §1)
- **Two parameter groups with different learning rates**: the pretrained trunk needs a much
  smaller LR than the freshly-initialised head. A head-sized LR applied to the trunk
  destroys the pretrained representation in the first few steps — the reason fine-tuning
  recipes use ~10–100× lower trunk LRs, and why gradual unfreezing exists.
- wandb logging (entity `ethan_personal`), checkpointing, early stopping
- Baseline to beat: **frozen trunk + MLP head.** Run this first. If full fine-tuning can't
  beat linear probing, something is wrong — and it's the comparison that justifies the repo.

### Phase 4 — cluster
- `sbatch` script following `/home/ethan2/job.sh` conventions: `--gres=mps:20`, job arrays,
  logs to `/home/ethan2/logs/`, `conda activate minimol_ft`
- Note `job.sh` sanitizes inherited venvs from `PATH` before activating conda — keep that
- wandb sweeps via `python -m wandb agent` for hyperparameter search

---

## 8. The test everything rests on

```python
emb = trunk(batch)          # 512-d, should carry a grad_fn
emb.sum().backward()
total = sum(1 for p in trunk.parameters() if p.requires_grad)
got   = sum(p.grad is not None and p.grad.abs().sum() > 0
            for p in trunk.parameters() if p.requires_grad)
print(f"{got} / {total} trunk params received gradient")
```

A large number means fine-tuning works. Zero means the chain is broken somewhere and §6.3
is needed. Run this **before** building anything on top.

---

## 9. Footguns already identified

- **`load_state_dict(..., strict=False)`** in `minimol/model.py` silently tolerates
  missing and unexpected keys. If names don't line up you get a **partly random trunk with
  no error and no warning** — which looks like bad hyperparameters, not a bug. Always
  capture and assert on the returned `_IncompatibleKeys`.
- **`hydra.initialize()` runs once per process.** Constructing the model twice in one
  process raises. Will bite in sweeps and in test suites; needs `hydra.core.global_hydra`
  cleanup or a single-instance pattern.
- **`set_training_mode_false()`** sets `module.training = False` by attribute assignment
  rather than `.eval()`. It does *not* block gradients, but it must be reversed so dropout
  and BatchNorm behave correctly during training.
- **Featurization runs inside every `Minimol.__call__`.** SMILES→graph conversion is
  CPU-bound and will dominate every epoch if repeated. Featurize once, cache, reuse.
- **`.cpu()` in the fingerprint path** moves tensors off GPU per batch. Irrelevant for
  extraction, a bottleneck in a training loop.
- **Private APIs in the §6.3 fallback** are not stability-guaranteed. Pin graphium.
- **`data/ampc_subset_331k.csv` is sorted by `(-pprop, smiles)`** — i.e. sorted by the
  regression target. A DataLoader that forgets to shuffle produces target-sorted batches
  and pathological training. Never rely on file order.
- **Morgan fingerprint details have shifted across rdkit releases.** The splits were built
  under rdkit 2024.03.5 in `my_conda_env`; `minimol_ft` will carry a different one. This is
  why `data/splits/cluster_kfold_v1/fingerprints.npy` is persisted rather than recomputed,
  and why `meta.json` records the rdkit version.

---

## 10. Environment notes (this machine)

- conda at `/home/ethan2/local/conda`; existing env `my_conda_env` (py3.10) — leave alone
- SLURM via `sbatch`; GPU requested as `--gres=mps:20` (fractional GPU), job arrays
- logs to `/home/ethan2/logs/`
- wandb entity `ethan_personal`; sweeps launched with `python -m wandb agent`
- repo `/home/ethan2/finetune_minimol`, branch `main`

---

## 11. Cross-validation splits — built, verified, frozen

Artifacts: `data/splits/cluster_kfold_v1/` — `assignments.csv`, `fingerprints.npy`,
`meta.json`, `diagnostics.md`. Generated by `src/split.py` (81s, CPU only, no GPU, in
`my_conda_env`); loaded by `src/splits.py`. Not blocked on Phase 0 — splits need only
rdkit/numpy/pandas, so they were built before `minimol_ft` exists.

### 11.1 Why cluster, not random

The subset comes from a make-on-demand library with heavy analog density. A random
validation fold leaves near-duplicates of nearly every held-out molecule in the training
set, so the metric measures interpolation, not generalisation to new chemotypes.

Method: ECFP4/2048 → sphere exclusion (rdkit `LeaderPicker`) at Tanimoto distance 0.65 →
every molecule assigned to its nearest centroid → whole clusters dealt into 5 folds.
Clustering is on ECFP, deliberately **not** on MiniMol embeddings: the partition must not
depend on the model being evaluated, and must be reproducible without loading the trunk.

Bemis–Murcko scaffold split was measured and rejected: 195,974 scaffolds over 331,480
molecules, with **51.3% of molecules in singleton scaffolds** — half the data would be split
at random. Generic (framework) scaffolds are better (54,355 groups, 10.6% singletons) and
are stored as a column for a later comparison, but are not what v1 uses.

### 11.2 The threshold was chosen from measurement, not convention

Separation measured as max Tanimoto from 5,000 sampled validation molecules to *any*
training molecule, against a random split of identical fold sizes (median NN 0.541,
5.06% ≥ 0.7):

| dist | clusters | median NN | frac ≥ 0.7 | largest cluster's share of the 100 pProp≥5.0 |
|---|---|---|---|---|
| **0.65** | **32,254** | **0.508** | **2.14%** | **9%** |
| 0.70 | 13,912 | 0.508 | 2.20% | 24% |
| 0.75 | 4,722 | 0.508 | 1.54% | 32% |
| 0.80 | 1,183 | 0.507 | 1.52% | 45% |

Raising the threshold buys essentially nothing in separation while badly concentrating the
potent tail into single clusters. 0.65 wins on both counts.

**Read the separation honestly.** Measured on all 5 folds, 5,000 sampled molecules each.
The gain is on near-duplicates: molecules with a training neighbour ≥ 0.7 drop from 5.15%
to 2.11%, a 2.4× reduction. The *median* barely moves, 0.541 → 0.508, and 55% of validation
molecules still have a training neighbour ≥ 0.5.

That small median shift is easy to misread as "the split did nothing", so it was checked
rather than explained away. The pool-size-matched control (diagnostics §1b) — nearest
neighbour *within* a molecule's own fold, measured for both the cluster and the random
split against the same ~66k-molecule pool:

| | within-fold | cross-fold |
|---|---|---|
| cluster split | **4.83%** | 2.11% |
| random split | 1.46% | 5.15% |

(fraction with a neighbour ≥ 0.7; the two columns search different pool sizes, so compare
only down a column.)

Clustering concentrates close neighbours *inside* folds by 3.3×, and removes them from
across the split by 2.4×. Both point the same way, so close analog series genuinely exist
in this library and the split moved them to one side. The residual cross-fold similarity is
background similarity between unrelated molecules in a combinatorial library — the median
molecule's nearest neighbour is unrelated either way, which is why the median hardly moves.

So: **this split removes analog leakage; it does not manufacture an out-of-distribution
test set.** Results should be described as generalisation to new clusters within this
library, not to new chemistry.

### 11.3 Fold assignment: stratified LPT

The potent tail is thin — 3,153 molecules at pProp ≥ 3.5 and only **100** at ≥ 5.0. Clusters
are packed by longest-processing-time greedy in three strata, each keyed on *the quantity
that stratum exists to balance*: the ≥5.0 stratum by its count of such molecules, the ≥3.5
stratum likewise, the bulk by cluster size (carrying the running size load across strata).

Keying the tail strata on cluster size would be the bug: a cluster holding a large share of
the 100 potent molecules can itself be small, and would be packed as if unimportant.

Result: fold sizes exactly 66,296 each, and exactly **20 molecules at pProp ≥ 5.0 per
fold**, 619–643 at ≥ 3.5.

### 11.4 How to use them

- **Splits are fixed across model seeds.** The model seed governs head init, dropout and
  shuffling; it never touches the partition. The 5×2 grid then separates fold-to-fold
  variance (data) from seed-to-seed variance (optimisation) — variation down a column is
  data, across a row is optimisation.
- **Pool out-of-fold predictions for tail metrics.** 20 potent molecules per fold is noise.
  The 5 validation folds cover all 331,480 rows exactly once, so pooling gives one honest
  estimate over all 100, and two seeds give two such estimates. Per-fold numbers stay the
  right unit for bulk metrics.
- **This is a plain K-fold** (chosen deliberately): each fold is used for both early
  stopping and reporting, so the CV score is mildly optimistic. Fixing the epoch budget
  instead of early-stopping removes the bias. The schema has room for an inner-val column
  if that becomes worth doing.
- `src/splits.py` re-hashes `data/ampc_subset_331k.csv` on load and raises if it has
  changed. This matters because `subset.py` takes `--seed`: rerunning it with different
  arguments silently produces a different 331k set and invalidates every split. Verified
  that a tampered CSV is rejected.

### 11.5 Reproducibility, checked rather than assumed

`LeaderPicker`'s order-stability under threading is undocumented, and the whole scheme
rests on it. Checked on the full set: three runs at `numThreads=64` and two at
`numThreads=1` all returned identical centroid lists (32,254 centroids; 8s vs 169s), so the
threaded default is safe. Downstream is deterministic by construction — `Pool.map` preserves
order, `argmin`/`argmax` tie toward the lowest index.

Reruns are checked by the `split_sha256` in `meta.json` — a hash over the
`(row_idx, cluster_id, fold)` triples. File byte-identity is the wrong criterion; content
identity is the right one. Current value: `3ef97e78a85d…`.

---

## 12. What was inherited from `pProp_MLP`, and what the data forced to change

`/home/ethan2/pProp_MLP` trained a dual-head MLP on **frozen** MiniMol embeddings for the
same target. It is a finished body of work — ~19,600 completed sweep runs — and most of its
loss, metric and objective design ports directly. This section records what carried, what
did not, and the measurements that decided each.

### 12.1 The result that motivates this repo

From `pProp_MLP`'s `sweeps/eval_best_model.ipynb`, run `jlwzehi6`, under a hard ≤0.65
Tanimoto val↔train ceiling:

| class | n (val) | train AP | val AP | train MAE | val MAE |
|---|---|---|---|---|---|
| 0–3.5 | 68,168 | 0.985 | 0.958 | 0.382 | 0.463 |
| 3.5–5 | 23,334 | 0.769 | 0.679 | 0.384 | 0.645 |
| **5–7.5** | **1,653** | **0.656** | **0.248** | **0.339** | **1.389** |

**The frozen embedding does not transfer to held-out chemistry in the one class that
matters** — val AP collapses to 0.248 against 0.656 on train. Closing that gap is what
full-trunk fine-tuning is for, and it is the comparison §7 Phase 3 asks for. Its data-scaling
notebook adds a second reading: val performance was still climbing monotonically at 100% of
530k training molecules, with no knee, so frozen-embedding *capacity* was not the binding
constraint either.

### 12.2 pProp means the same thing in both repos — verified, not assumed

Both compute `-log10(rank / N)`, which is a quantile, so a pProp threshold is the same
quantile regardless of library size (1.468e9 there, 1e7 here). Checked against absolute
docking score at matched pProp:

| pProp band | this repo, mean score | pProp_MLP, mean score |
|---|---|---|
| [3.4, 3.6) | −75.40 | −75.56 |
| [4.9, 5.1) | −84.41 | −84.26 |
| [5.5, 6.0) | −87.63 | −87.64 |

Agreement to ~0.2 kcal/mol confirms `ampc_unif_random_10M` is an unbiased draw from the same
library. **Consequence:** every threshold and every loss hyperparameter expressed in target
units transfers *in units*. Only densities change — and they change a lot:

| band | pProp_MLP | here |
|---|---|---|
| [0, 3.5) | 454,450 | 328,327 |
| [3.5, 5) | 155,561 | **3,053** |
| [5, 7.5) | 13,778 | **100** |
| 7.5+ | 46 | **0 — impossible**, pProp caps at 7.0 |

### 12.3 The loss hyperparameters transfer because they are in z-units

Verified at `pProp_MLP/src/sweep_train.py:460-465`: `huber`, `pair` and `std` all train
against the **normalized** target, and the best run (`zfs9n2ln`, `goal_metric` 1.4483) used
`pprop_norm=zscore`. So its `huber_delta 1.0513`, `w_pair 7.486`, `w_std 0.7911` are in
z-units and survive the change of pProp distribution (raw mean 1.367 / std 0.864 here versus
2.392 / 1.437 there). They are the defaults in `train.py`. This is the single largest saving
available: it converts 19,600 prior runs into narrow priors.

**The head-size priors do NOT transfer, and this is the trap.** That winner's architecture
(`n_layers=4, hidden_dim=1740, cls 3×468, reg 3×804`) is **13,951,442 parameters — 1.76× the
7,919,912-parameter trunk**. That was correct there, where a frozen embedding meant the MLP
did all the learning. Here the trunk is trainable and the division of labour is reversed, so
head width and depth are wide-sweep dimensions, not inherited constants. The z-unit argument
covers loss hyperparameters only — those are properties of the target distribution, whereas
head size is a statement about where the capacity should sit.

### 12.4 ECFP lost decisively — a result recorded nowhere else

Undocumented in `pProp_MLP` itself, recovered from its run artifacts. Comparing only buckets
scored under its current objective:

| arm | runs | best `goal_metric` |
|---|---|---|
| **MiniMol only** (two-tower code, `ecfp_dim=0`) | 816 | **1.4483** |
| MiniMol only, single tower | 1,431 | 1.4413 |
| MiniMol + ECFP two-tower | **6,648** | 1.4362 |

The ECFP arm never caught the MiniMol-only best despite ~8× the search budget. That settles
the whole research question of its `docs/ecfp_concat.md`. **Do not port ECFP here.** One
architectural detail is worth keeping: the winning arm ran MiniMol through an extra
`Linear(512→256) → LayerNorm → ReLU` before the shared trunk and beat the plain single-tower
arm — `MLPHead` already exposes `norm`, so that is a sweep value rather than a new module.

### 12.5 What is deliberately not ported

- **`WEIGHT_GROUPS = [0,1,2,0]`** — existed only to demote the 46 `7.5+` artifacts, a class
  that cannot occur here. Ported as the identity `[0, 1]`.
- **`wandb agent` sweeps** (§7 Phase 4, §10). `reports/compute_profile.md` identifies this as
  *the* gating item: the agent must reach the wandb server for each next configuration, and
  TamIA's compute nodes have no internet. Replace with Optuna over filesystem storage,
  `WANDB_MODE=offline`, and a `wandb sync` from a login node.
- **`--gres=mps:20` GPU packing** (§7 Phase 4, §10). Essential there — five tiny-MLP agents
  per A6000. Measured **counterproductive here**: 0.90–0.94× of a single process at both
  batch 256 and 1024. One run per GPU. `scripts/run_grid.sbatch` already departs from Phase 4
  on this point, with the reason in its header.
- **`VAL_EVAL_EVERY = 5`.** There, full-set eval was ~half an epoch. Here validation is ~9%
  of one, so throttling would save nothing and cost curve resolution. The *train*-set eval
  pass is throttled instead — final epoch only, since it scores 4× as many rows.
- **Its splits.** `pProp_MLP` guaranteed a per-molecule ≤0.65 Tanimoto ceiling by holding out
  whole connected components (`overall_max_val_to_train = 0.64999998`); the frozen cluster-CV
  splits here do not. That is a real difference in *what may be claimed*, and §11.4's framing
  stands: generalisation to new clusters within this library, not to new chemistry.

### 12.6 Corrections to earlier readings in this document

- §1 says the headline metric is open. It is now `objective.goal_metric`, version-stamped.
- §1 says loss weighting is deferred. It is settled: two-group inverse frequency at pProp 3.5
  (`--weights balanced`), with `uniform` retained as the second reporting flavour.
- **`ipw` was removed from the modelling code entirely on 2026-08-12** (it had briefly been a
  third weighting flavour and sat beside `ap_uniform` in `AP*`). Rationale: `ipw` records the
  per-bin rate at which `subset.py` sampled the 10M library, so it belongs to the data's
  provenance, not to the training or scoring methodology. It remains a CSV column and a
  `split.py` diagnostics row. Consequences, all accepted deliberately:
  - `OBJECTIVE_VERSION` re-stamped `adb3da05` → `c917327f`. This cost nothing because no run
    had been scored under the old spec — the only runs on disk were compute benchmarks with
    null provenance. The window for a free re-stamp closes once the 5×2 grid runs.
  - There is **no longer any metric that estimates full-10M-library performance.** `*_uniform`
    is a subset number and the subset is ~30× tail-enriched, so it flatters the model. Say
    "on the 331k subset" when reporting, never "on the library".
  - `verify_metrics.py`'s sign-error check was rebuilt rather than deleted: the old three-way
    `mae_ipw < mae_uniform < mae_balanced` became `mae_uniform < mae_balanced` plus two
    base-rate assertions that pin each weight vector down directly (unweighted must reproduce
    the true positive rate; balanced must be 0.5 — measured 0.499999997, the gap being float32
    in `grouped_frequency_weights` and exact in float64).
  - The objective's missing-metric negative test now takes its victim key *from*
    `OBJECTIVE_SPEC` rather than naming one. It had named `ap_ipw`; had that been deleted
    without thought, the test would have passed while testing nothing.
- §7 Phase 3 lists early stopping. Superseded by §11.4's fixed epoch budget, which is what
  removes the K-fold optimism. `pProp_MLP` measured the cost of final-epoch selection at
  0.003–0.005 of `goal_metric` (30/30 of its top-30 runs peaked earlier); that is known,
  accepted, and visible in the per-epoch history.
