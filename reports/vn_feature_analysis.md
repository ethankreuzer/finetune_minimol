# What should the encoder export? — and does MiniMol's virtual node belong in it?

**Recommendation up front: export `pooled512`, MiniMol's own 512-d output — not `z`, and not
anything involving the virtual node.** Against `z` (the head's shared output) that costs 0.017
kNN ρ on unseen chemical series, ~2% relative, and buys `scalarness` 0.40 against 0.84, 22% of
the available directions against 3%, and the best local neighbourhood structure of any candidate.
The reasoning and the full comparison are in §["What the encoder should export"](#what-the-encoder-should-export);
everything before it is the evidence.

---

**On the virtual node question: yes, there is a task where it wins.** It carries far more
*structural* information than the pooled embedding: `tanimoto_partial` **+0.298** in its favour,
paired per model, sign holding in 30/30 models and 3/3 configurations. And the pretrained control
says why: fine-tuning cost `pooled512` **54%** of its structural correlation and **28%** of its
effective rank, while the virtual node lost 5–9% and *gained* rank.

**But it does not translate.** `pooled ⊕ vn15` is worse than `pooled` alone on held-out clusters
in **30/30** models, while the dimension-matched null is *better*. The virtual node is not a
usable additional component of this encoder as it stands — though on the *pretrained* trunk a
linear probe does show a small gain, which fine-tuning removes.

**The most consequential number here is not about the virtual node at all.** `z` — the head's
shared output, and one of the candidate export points — has `scalarness` **0.843** and an
effective rank of **30 out of 1024**: a space in which "far apart" is 84% a restatement of
"different predicted pProp" — best predictor of the three candidates, worst geometry, which are
one fact rather than two. The export point is an open decision on this branch (`CLAUDE.md`,
"What this branch is, and what it is not"); this is the evidence that decides it.

Read against `reports/sweep_lyc0lh2d.md`, which produced the 30 models this scores.

> **New to this work?** [Definitions](#definitions) is self-contained and defines every object
> (`vn03…vn15`, `pooled512`, `z`), every statistic (`tanimoto_partial`, `scalarness`,
> `knn20_jaccard`, `tanimoto_spearman`, `emb_effective_rank`) and every task label (T1–T5) used
> below, including how each is computed and what its chance level is. Skip ahead if a term in
> the summary above is unfamiliar.

---

## What was already known, and why it was not the whole question

`figures/lyc0lh2d/probe_metrics.csv` scored a random forest reading pProp off each embedding.
Out-of-bag means over the 30 models — the only columns that discriminate, since in-sample AP and
AUC are **1.000 for all six embeddings** at `min_samples_leaf=1`:

| embedding | `spearman_oob` | `r2_oob` | `average_precision_oob` |
|---|---:|---:|---:|
| VN 3 | 0.695 | 0.458 | 0.042 |
| VN 6 | 0.710 | 0.479 | 0.057 |
| VN 9 | 0.720 | 0.493 | 0.059 |
| VN 12 | 0.729 | 0.507 | 0.060 |
| VN 15 | 0.738 | 0.520 | 0.059 |
| **pooled512** | **0.861** | **0.738** | **0.241** |

Two findings stand: the **depth trend is monotone** — every out-of-bag column rises VN3 → VN15,
so the virtual node accumulates pProp-relevant signal down the stack — and **pooled wins at pProp
prediction**, by 4.1× on average precision.

That second result was never in doubt and is not evidence about richness. The trunk was
fine-tuned end to end on pProp, so `pooled512` sits one `Linear(512→1024)` from the prediction.
**Any task that *is* pProp prediction favours it by construction.** The encoder's consumer is a
deep-kernel-learning GP, which uses one property `goal_metric` never scores: distance between
molecules. That is where this analysis looked.

## What was added

Three probes, all over the 30 models already on disk, plus one new baseline.

| | what | script | artifact |
|---|---|---|---|
| Probe 0 | the same tensors from an **un-fine-tuned** MiniMol | `extract_embeddings.py --pretrained` | `outputs/vn_analysis/<id>/pretrained/frozen/` |
| Probe 1 | geometry against ECFP4 Tanimoto | `src/vn_geometry.py` | `figures/<id>/geometry_metrics.csv` |
| Probe 2 | held-out-**cluster** utility + concatenation | `src/vn_cluster_probe.py` | `figures/<id>/cluster_probe.csv` |

Probe 0's trunk reproduces `data/reference/minimol_v1_ref64.npy` at **max|Δ| = 0.000e+00** on the
CPU — an exact gate, replacing the `z` round-trip it cannot do, since with no head there is
nothing to round-trip.

---

## Definitions

This section is self-contained: everything below it is readable with deep-learning background
and no prior exposure to this project.

### The objects being compared

**MiniMol** is a pretrained molecular GNN: 16 message-passing layers over the molecular graph,
ending in a 512-d graph-level vector. Alongside the atom features it maintains a **virtual
node** — a single vector per molecule, updated after each of the first 15 layers, that pools
every atom, updates residually, and broadcasts back to every atom. It is a running graph-level
summary carried in parallel with the per-atom state, 336-d here.

On this branch the **whole trunk is fine-tuned** on `pProp`, a docking-derived potency target
(higher = more potent; the positive class is `pProp ≥ 3.5`, ~0.97% of molecules). The tensors
compared are:

| name | what it is | width |
|---|---|---:|
| `vn03 … vn15` | the virtual node's state after GNN layers 3, 6, 9, 12, 15 | 336 |
| `pooled512` | MiniMol's own output — `global_max_pool` over the final layer's atom features | 512 |
| `z` | the head's shared layer on top of `pooled512` — a **candidate** export point | 1024 |

`z` feeds two bare linear heads (one classification, one regression), so **pProp is by
construction close to linear in `z`**.

**None of these is yet the designated export.** `head.py`'s docstring still calls `self.shared`
"THE DELIVERABLE", but that is inherited from the sealed 32-d contract that ended 2026-08-25;
`CLAUDE.md` supersedes it — the export point is an open design question, and `src/export.py`
does not exist. Choosing between these candidates is what this analysis is for.

**Why any of this matters:** the deliverable is not the pProp predictions. It is a frozen
`SMILES → vector` encoder, consumed by a **deep-kernel-learning GP** driving an active-learning
loop. A GP kernel consumes exactly one thing — *distance between molecules* — and its posterior
variance is what acquisition steers on. So an embedding can be an excellent pProp predictor and
a poor encoder, and the usual metrics cannot tell the difference.

### The five candidate tasks

| | task | why the virtual node could win |
|---|---|---|
| **T1** | structural fidelity — does embedding distance track chemical dissimilarity? | `pooled512` is one linear layer from the target; the VN's residual accumulation is not |
| **T2** | non-scalarness — is distance more than a restatement of the prediction? | same |
| **T3** | generalisation to **unseen clusters** (not merely unseen molecules) | pooled may be the one that overfits |
| **T4** | tail retrieval — finding `pProp ≥ 3.5` molecules in unseen clusters | same |
| **T5** | **complementarity** — does `pooled ⊕ vn` beat `pooled`? | max-pooling discards distributional information the VN retains |

T1/T2 are measured by Probe 1 (geometry), T3/T4/T5 by Probe 2 (cluster-held-out utility). **T5
is the one that would change the deliverable**; a T1/T2 win alone does not.

### The prediction statistics (the earlier probe, first table above)

Each embedding is reduced to its top **32 principal components**, and two random forests (100
trees) are fit on the model's held-out validation fold: a regressor for `pProp` and a classifier
for `pProp ≥ 3.5`. `spearman` / `r2` / `average_precision` score those forests.

The **`_oob` suffix means out-of-bag**: each tree is trained on a bootstrap resample, so ~37% of
molecules are unseen by any given tree, and a molecule's out-of-bag prediction is pooled only
from the trees that never saw it. So `spearman_oob` is the Spearman ρ between the out-of-bag
regression predictions and true `pProp`; `r2_oob` is their R²; `average_precision_oob` is the
average precision of the out-of-bag classifier probabilities for `pProp ≥ 3.5`, read against a
positive rate of 0.0097 (643 of 66,296). Without out-of-bag these numbers are meaningless — at
`min_samples_leaf = 1` a forest memorises its training set, and the in-sample AP and AUC are
**1.000 for all six embeddings**. Only the `_oob` columns discriminate.

Note what this probe does and does not hold out. The *embeddings* are out-of-sample with respect
to the encoder (fold 0 was never trained on), but out-of-bag holds out **rows**, not chemical
series — a molecule's close structural analogues can still be in-bag. That is precisely the gap
Probe 2 closes.

### The geometry statistics

All four are computed on one fixed sample of **5,000 molecules** drawn from fold 0's validation
set, over all 12,497,500 distinct pairs. Three pairwise quantities are formed per pair:

- **A — embedding distance**: Euclidean distance between the two molecules in the embedding.
- **B — prediction gap**: `|Δ predicted pProp|`, from the model's own `val_predictions.npy`.
- **C — structural dissimilarity**: `1 − Tanimoto(ECFP4)`. ECFP4 is a 2048-bit circular
  fingerprint; Tanimoto is `|a ∩ b| / |a ∪ b|` over set bits. **The model never sees ECFP4** —
  it is an independent, purely chemical description, which is what makes it a fair yardstick.

Every correlation below is **Spearman** (Pearson computed on ranks), because Euclidean-vs-Tanimoto
is monotone at best and Tanimoto is heavily right-skewed.

**`tanimoto_spearman` = ρ(A, C)** — *do molecules that sit far apart in the embedding also differ
chemically?* Range −1…1, higher is better; 0 means the geometry carries no chemistry. This is the
primary "is this a chemically meaningful metric space" reading.

**`scalarness` = ρ(A, B)** — *is embedding distance just a restatement of the model's own
prediction?* Range −1…1, **lower is better**. This names a specific failure mode: at
`scalarness → 1`, "these two molecules are far apart" and "these two molecules have different
predicted pProp" are the same statement, so GP posterior variance stops tracking genuine
ignorance about novel chemistry and instead tracks prediction spread. An active-learning loop on
such a space has nothing to explore toward. It is not a nuisance term — for this deliverable it
is arguably the single most important number here.

**`pred_tanimoto_spearman` = ρ(B, C)** — the **control**. Similar molecules dock similarly, so a
model that predicts pProp well earns *some* correlation with structure for free, without its
geometry encoding any chemistry. Measured flat at ~0.07 across everything here.

**`tanimoto_partial`** — `tanimoto_spearman` with that free correlation removed:

```
tanimoto_partial = ( ρ(A,C) − ρ(A,B)·ρ(B,C) ) / sqrt( 1 − ρ(A,B)² )
```

This is a **semi-partial (part) correlation**: it removes B from A only, not from C. Read it as
*the correlation between chemical dissimilarity and the part of embedding distance that the
model's own prediction does not already explain.* Range −1…1, higher is better. **This is the
headline T1 statistic**, because it is the one an embedding cannot inflate simply by being a good
predictor.

**`knn20_jaccard`** — the only *local* statistic here, and the only one not built from A/B/C. For
each molecule take two neighbour sets — `N_emb`, its 20 nearest by embedding distance, and
`N_chem`, its 20 nearest by Tanimoto — and average the Jaccard overlap
`|N_emb ∩ N_chem| / |N_emb ∪ N_chem|` over all 5,000 molecules. Range 0…1, higher is better.
**Chance is 0.0021** (measured: two independent random geometries over 5,000 molecules), so the
observed 0.10–0.15 is 50–70× chance.

It is reported beside `tanimoto_spearman` because they can disagree, and the disagreement is
informative: a global rank correlation can be carried by the far tail of the distance
distribution, while a GP with a length scale only ever sees the *local* neighbourhood. **It is
sample-size dependent by construction** — the 20 nearest of 5,000 is a tighter neighbourhood than
the 20 nearest of 2,500 — so it is comparable only at a fixed sample size (5,000 throughout).

**`emb_effective_rank`** — how many dimensions are actually doing work:

```
eigenvalues λ of the d×d covariance  →  p = λ / Σλ  →  exp( −Σ p log p )
```

The exponentiated Shannon entropy of the normalised eigenvalue spectrum. It equals `d` when
variance is spread evenly and 1 when a single direction dominates. **This is not matrix rank** —
a dimension carrying pure numerical noise counts toward full rank while contributing essentially
nothing here. For calibration, isotropic Gaussian noise measures 325 at d = 336 and 924 at
d = 1024, so these are the ceilings; `z`'s **30.0 out of 1024** is a collapse, not a rounding
difference.

### The utility statistics (Probe 2)

A **cluster** here is a chemical series: the 331k molecules were clustered by ECFP4 sphere
exclusion at Tanimoto distance 0.65, giving 32,254 clusters that were then dealt *whole* into the
5 cross-validation folds. So holding out a cluster holds out a whole structural neighbourhood,
not a molecule with its analogues left behind — the setting an active-learning loop actually
faces when it proposes novel chemistry.

**`knn_spearman` / `ridge_spearman`** — fit a probe on molecules from one half of fold 0's
clusters, predict `pProp` for molecules in the *other* half, and report Spearman ρ against truth.
The two halves are **cluster-disjoint**, so no molecule in the test half has its own structural
analogues in the fit half; both directions are run and averaged. kNN (k = 20, distance-weighted)
is the primary probe because a GP kernel consumes distances; ridge (α = 1) tests linear
extractability.

**`knn_ap` / `ridge_ap`** — the same fits scored as **average precision** for retrieving the
`pProp ≥ 3.5` positives. AP is read against the **positive rate** — 0.0102 here, 51 positives
per 5,000-molecule half — never against 0.

Terminology used throughout: **`pooled@scaled`** is `pooled512` after the same per-block
standardisation the concatenations receive, and it — not raw `pooled512` — is the correct
baseline for them; **`pooled+randproj`** is `pooled512` concatenated with a random linear
projection *of itself*, which adds width but no information and so isolates how much of any
concatenation gain is due to dimensionality alone.

---

## The finding: geometry

Means over the 30 fine-tuned models, at `--n-sample 5000` on fold 0.

| embedding | `tanimoto_partial` ↑ | `scalarness` ↓ | `knn20_jaccard` ↑ | `emb_effective_rank` |
|---|---:|---:|---:|---:|
| VN 3 | 0.419 | 0.090 | 0.102 | 62.7 |
| VN 6 | 0.438 | 0.114 | 0.114 | 65.8 |
| VN 9 | 0.452 | 0.129 | 0.117 | 63.7 |
| VN 12 | **0.458** | 0.143 | 0.118 | 63.5 |
| VN 15 | 0.457 | 0.154 | 0.119 | 63.9 |
| pooled512 | 0.159 | 0.403 | **0.152** | 107.3 |
| **`z` — head output, a candidate export** | 0.171 | **0.843** | 0.120 | **30.0** (of 1024) |

Paired per model, **VN15 − pooled512**:

| | mean Δ | cfg0 | cfg1 | cfg2 | sign holds |
|---|---:|---:|---:|---:|:--|
| `tanimoto_partial` | **+0.298** | +0.264 | +0.309 | +0.322 | 3/3 configs |
| `scalarness` | **−0.249** | −0.210 | −0.294 | −0.242 | 3/3 configs |
| `knn20_jaccard` | −0.033 | −0.038 | −0.031 | −0.031 | 3/3 configs |

**T1 and T2 go to the virtual node, decisively.** Embedding distance in a VN tap tracks ECFP4
dissimilarity ~2.9× better than in the pooled embedding, and it is far less a restatement of the
model's own prediction — `scalarness` 0.15 against 0.40. That second number is the one with the
direct active-learning consequence: as `scalarness` → 1, the GP's notion of "these two molecules
are far apart" becomes "these two molecules have different predicted pProp", so posterior
variance stops tracking genuine ignorance and acquisition has nothing to steer on.

### The head's output is the worst of the three on exactly this axis

`z` — what `train.py` writes as `val_embeddings.npy`, and the candidate export that
`head.py`'s docstring still assumes — scores **`scalarness` 0.843**, against pooled512's 0.403 and VN15's 0.154.
Paired per model, **VN15 − z is −0.688 on `scalarness`, holding in 30/30 models and 3/3
configurations**; `z − pooled512` is **+0.440**, so each step toward the objective costs
geometry. Its **effective rank is 30.0 — out of 1024 dimensions.**

Read plainly: in the space the DKL-GP would receive, "these two molecules are far apart" is
~84% the same statement as "these two molecules have different predicted pProp", and 994 of the
1024 exported dimensions are doing no work. `emb_readout.py` names this as the failure mode the
`vic` term exists to counter — disabled by default on this branch — and at 0.843 it is not a
risk, it is the current state. This is not a virtual-node finding; it is the most consequential
number in this report for the deliverable, and `goal_metric` cannot see it.

**`knn20_jaccard` disagrees, and the disagreement is the finding.** Pooled wins on *local*
neighbourhood agreement (0.152 vs 0.119) while losing on *global* rank correlation. Global ρ can
be carried by the far tail of the distance distribution; a kernel with a length scale only ever
sees the local part. `emb_readout.py`'s own docstring pre-registers this: treat the disagreement
as a result, not as a menu. So the VN advantage is real but it is an advantage in **global**
distance structure, and pooled retains the better **local** neighbourhoods.

### Why — the pretrained control

Fine-tuned mean minus the un-fine-tuned baseline (n = 1, deterministic):

| embedding | `tanimoto_spearman` pretrained → fine-tuned | change | `emb_effective_rank` |
|---|---|---:|---|
| **pooled512** | 0.380 → 0.174 | **−54.2%** | 148.2 → 107.3 (**−27.6%**) |
| VN 3 | 0.465 → 0.424 | −8.7% | 54.9 → 62.7 (+14.3%) |
| VN 9 | 0.496 → 0.457 | −7.9% | 61.8 → 63.7 (+3.1%) |
| VN 15 | 0.485 → 0.463 | **−4.6%** | 63.4 → 63.9 (+0.9%) |

This is the mechanism, and it settles a question the 30 fine-tuned models could not. The virtual
node does not win mainly because it is intrinsically richer — **at the start, VN15 and pooled512
were 0.485 and 0.380, a gap of 0.105.** After fine-tuning the gap is 0.288. So **about two thirds
of the VN's advantage (0.183 of 0.288, 64%) was created by fine-tuning damaging the pooled
embedding**, and about a third was there to begin with.

That is feature distortion (LP-FT, Kumar et al. 2022) measured directly on this stack, and it is
concentrated exactly where the gradient is strongest: at the readout the loss attaches to. It is
evidence on `CLAUDE.md`'s deferred question 5 — the trade-off between unfreezing more of the
trunk and preserving the pretrained representation is not hypothetical here, it is 54% of the
pooled embedding's structural correlation.

---

## The finding: no complementarity

Probe 2 fits on one cluster-disjoint half of fold 0 and scores on the other, both directions
averaged, 5,000 molecules per half. **Read the concatenations against `pooled@scaled`**, never
against raw `pooled512` — the concatenations are necessarily block-scaled, and neither kNN nor
fixed-α ridge is scale-invariant, so raw-vs-scaled would mix a change of content with a change of
preprocessing.

Paired per model, against `pooled@scaled`:

| | kNN Spearman | kNN AP | ridge Spearman | ridge AP |
|---|---:|---:|---:|---:|
| `pooled@scaled` (baseline) | 0.8404 | 0.1305 | 0.8597 | 0.1908 |
| **`+ vn15`** (test) | **−0.0183** | **−0.0458** | −0.0025 | −0.0015 |
| `+ randproj` (null) | +0.0040 | +0.0113 | −0.0014 | −0.0023 |
| test − null | **−0.0222** | **−0.0571** | −0.0011 | +0.0008 |

`+ vn15` is worse than the baseline in **30/30 models** on kNN Spearman, kNN AP and ridge
Spearman, sign holding in 3/3 configurations. The dimension-matched null — a random projection of
`pooled512` carrying no new information — is *better* than the test on every column. The single
positive entry, ridge AP at +0.0008, has a paired std of 0.009 and holds in only 1/3
configurations: noise.

**The null mostly behaved**, which is what makes this readable: `+ randproj` lands within +0.004
of the baseline on kNN Spearman. It is *not* inert on kNN AP, though — **+0.0113 against a 0.1305
baseline, 8.7% relative, better in 25/30 models.** That is extra width genuinely helping, which is
precisely what the null exists to expose. The conclusion survives it because the test−null gap on
that column is −0.0571, five times the null's own effect and in the opposite direction; but the
right reading of the AP column is "width helps a little, the virtual node hurts a lot", not
"width does nothing".

Under equal-variance blocking the concatenation is a 50/50 blend, and VN15 alone scores 0.722
against pooled's 0.846 — so a blend landing at 0.822 is close to what a weighted average
predicts. **Ridge, which learns its own weighting and could therefore ignore the VN block
entirely, gains nothing either** (−0.0025). That is the stronger version of the result: the
failure is not the fixed 50/50 weighting.

### But in the pretrained trunk, the linear probe says otherwise

The same three featurizations, scored on the **un-fine-tuned** trunk (n = 1, no spread), beside
the fine-tuned mean. `test − null` is the quantity that isolates information from width:

| `test − null` | pretrained | fine-tuned |
|---|---:|---:|
| kNN Spearman | −0.0192 | −0.0222 |
| kNN AP | −0.0014 | −0.0571 |
| **ridge Spearman** | **+0.0182** | −0.0011 |
| ridge AP | −0.0008 | +0.0008 |

**Under a linear probe, the virtual node does add to the pooled embedding before fine-tuning** —
`pooled+vn15` scores 0.7640 against a 0.7480 baseline and a 0.7459 null — **and fine-tuning
removes that gain entirely.** Under kNN it never existed, in either regime.

This is consistent with the distortion mechanism rather than an accident of it: before
fine-tuning the two tensors are much less redundant (structural gap 0.105 rather than 0.288), so
there is something left for the second block to contribute. It is a **single measurement with no
error bar**, worth +0.018 on one of four columns, and it must not be reported as a result. What
it does is change what the negative T5 finding means: *"the virtual node adds nothing"* is too
strong; *"the virtual node adds nothing to a pooled embedding that has already been fine-tuned
into near-redundancy with it"* is what was measured. Establishing whether the pretrained gain is
real needs replicates, which this design does not produce — the pretrained trunk is deterministic
and there is only one of it.

### Held-out-cluster ranking, all featurizations

| featurization | kNN Spearman | kNN AP | ridge Spearman | ridge AP |
|---|---:|---:|---:|---:|
| `z` (the current deliverable) | **0.8637** | **0.2054** | 0.8562 | 0.1924 |
| `pooled512` | 0.8463 | 0.1538 | 0.8535 | 0.1817 |
| `pooled+randproj` (null) | 0.8444 | 0.1417 | 0.8583 | 0.1886 |
| `pooled@scaled` | 0.8404 | 0.1305 | **0.8597** | 0.1908 |
| `pred1` | 0.8238 | 0.0961 | **0.8714** | **0.2394** |
| `pooled+vn15` | 0.8222 | 0.0847 | 0.8572 | 0.1893 |
| `pooled512` *(pretrained)* | 0.7569 | 0.0527 | 0.7395 | 0.0411 |
| `vn15` | 0.7216 | 0.0553 | 0.7919 | 0.0595 |
| `vn15` *(pretrained)* | 0.6751 | 0.0468 | 0.7379 | 0.0270 |
| `ecfp4` | 0.6523 | 0.0472 | 0.6869 | 0.0298 |

Two things worth noting beyond the headline. **`pred1` is not a floor here.** The sealed 32-d arc
called the model's own scalar prediction the floor, but that model never saw fold 0, so as a
1-d featurization it is strong — and on ridge it is the *best thing in the table*. Every
embedding losing to its own scalar output under a linear probe is a statement about the
embeddings. And **fine-tuning is worth ~0.09 kNN Spearman** on pooled512 (0.757 → 0.846), which
is the value it buys in exchange for the 54% of structural correlation it destroys.

---

## What the encoder should export

The export point is an open decision, and this is the evidence for making it. Every candidate,
scored on the two things that matter — predictive skill on unseen chemical series, and the
geometry a GP kernel consumes. `eff. rank` is given as a **fraction of the isotropic ceiling**
for that width — measured on isotropic Gaussian noise at n = 5,000: **324.9** at d=336,
**486.2** at d=512, **924.0** at d=1024 — because a raw count is not comparable across widths.

| candidate | width | kNN ρ ↑ | kNN AP ↑ | `scalarness` ↓ | `tanimoto_partial` ↑ | `knn20_jaccard` ↑ | eff. rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`pooled512`** | 512 | 0.846 | 0.154 | 0.403 | 0.159 | **0.152** | 107.3 = **22%** |
| `z` | 1024 | **0.864** | **0.205** | 0.843 | 0.171 | 0.120 | 30.0 = 3% |
| `vn15` | 336 | 0.722 | 0.055 | **0.154** | **0.457** | 0.119 | 63.9 = 20% |
| `pooled+vn15` | 848 | 0.822 | 0.085 | — | — | — | — |
| `ecfp4` (no model) | 2048 | 0.652 | 0.047 | — | — | — | — |

**Recommendation: export `pooled512`.** It costs **0.017 kNN ρ** against `z` (0.846 vs 0.864, a
2% relative loss) and buys, on the axis the deliverable is actually consumed on:

- **`scalarness` 0.40 against 0.84.** In `z`, distance between two molecules is 84% a restatement
  of the difference in their predicted potency. A GP on such a space reports low variance
  wherever it predicts similar values, including on chemistry it has never seen — which is the
  one thing an active-learning acquisition function must not do.
- **Effective rank 22% of ceiling against 3%.** `z` occupies 30 of 1024 available directions.
  The DKL would be handed a 1024-d vector that is a ~30-d object in disguise.
- **The best local neighbourhoods of any candidate** (`knn20_jaccard` 0.152). A kernel with a
  length scale only ever sees the local structure, and this is where `pooled512` wins outright —
  including over the virtual node, which wins the *global* structural statistics.

Three secondary reasons, none decisive alone:

- **It ends the head-shape question for the deliverable.** If the export is the trunk output,
  then `--embed-dim`, `--n-layers` and the head's norm become ordinary training hyperparameters
  rather than product decisions, and the deferred **width scan is moot** — the width is 512,
  fixed by MiniMol. That removes an open question rather than answering it.
- **`z`'s advantage is the same fact as its weakness.** It leads on kNN AP (0.205 vs 0.154)
  *because* it is nearly the prediction. Handing a DKL something already collapsed onto the
  target removes its ability to learn its own warping — the DKL's whole purpose.
- **`pooled512` is the more portable artifact**, one layer earlier in a stack the downstream
  project must already vendor.

**Do not export `vn15`, or `pooled ⊕ vn15`.** The virtual node has by far the best global
chemistry (`tanimoto_partial` 0.457 vs 0.159, `scalarness` 0.154), but it gives up 0.124 kNN ρ
and 64% of the tail-retrieval AP, its local neighbourhoods are no better than `z`'s, and the
concatenation is worse than `pooled512` alone in 30/30 models against a null that is better.

**What would change this answer.** The `pooled512`-vs-`z` choice rests on the claim that
`scalarness` 0.84 hurts a DKL. Nothing here measures a DKL — it supplies its own warping and may
recover from a collapsed input. That is one small experiment (fit a DKL-GP on each, compare
held-out-cluster calibration and posterior variance on unseen clusters), and it is the experiment
worth running before the handover. If the DKL is indifferent, export `z` and take the 0.017.

## What follows, beyond the export choice

The recommendation above is the decision this analysis was for. Four consequences that outlive
it, in the order they should be acted on.

1. **Whatever is exported, `goal_metric` cannot see whether it is good.** A representation can
   win the prediction proxy while losing the property the deliverable exists for — `z` does
   exactly that. `src/vn_geometry.py` and `src/vn_cluster_probe.py` should now run on every
   candidate model, not once. They take `--root` and no other configuration.
2. **If `pooled512` becomes the export, `--w-vic` currently regularises the wrong tensor.**
   `losses.variance_covariance_loss` is applied to `z` (`head.py:198–200`), so the one
   countermeasure in the codebase for rank collapse acts on the head's output while the product
   would be the trunk's. Repointing it is a small change and is the most direct lever on the
   measured 54% structural loss.
3. **The remaining lever is the freeze schedule, not the architecture.** The distortion is
   created by fine-tuning, and the sweep drove `freeze_epochs` to its floor (1–3) with
   `trunk_lr` pinned against its 3e-4 cap. Preserving `pooled512`'s geometry is a *training*
   question — `CLAUDE.md`'s deferred layer-wise freeze/unfreeze — and Probe 0 now supplies the
   measurement that motivates it: the trade-off between unfreezing more and preserving the
   pretrained representation is no longer hypothetical, it is 54% of the structural correlation.
4. **The virtual node line closes here.** T5 is negative in 30/30 models against a null that
   behaves, so concatenation as tested does not work. What is worth keeping from it is the
   monotone depth trend, the distortion measurement Probe 0 made possible, and the one open
   thread: on the *pretrained* trunk a linear probe does find a small VN gain (+0.018 over the
   null) that fine-tuning erases — a single unreplicated measurement, and only interesting if
   the freeze work in (3) happens.

## Reproducing under a different sweep

Every script takes `--root outputs/vn_analysis/<sweep_id>`, discovers runs by `rglob`, derives
`sweep_id` from the directory name, and writes to `figures/<sweep_id>/`. No sweep id,
configuration name or model count is hardcoded anywhere.

```bash
R=outputs/vn_analysis/<sweep_id>
# Probe 0 -- 7 s of GPU forward pass, dominated by the ~30 s feature-cache load
python src/extract_embeddings.py --root $R --pretrained
python src/vn_probe.py         --root $R --oob                      # 22.3 min, 128 cores
python src/vn_geometry.py      --root $R                            # 11.5 min, CPU
python src/vn_cluster_probe.py --root $R --n-per-half 5000 --pca    # 15.6 min, CPU

python src/vn_figures.py --metrics-csv figures/<sweep_id>/probe_metrics.csv \
    --metrics pearson spearman r2 average_precision auc --suffix _oob
python src/vn_figures.py --metrics-csv figures/<sweep_id>/geometry_metrics.csv \
    --metrics tanimoto_partial scalarness tanimoto_spearman knn20_jaccard emb_effective_rank
# --only is not optional here: cluster_probe.csv holds 23 featurizations and all of them on
# one axis is illegible. These twelve are the ones that carry the comparison.
python src/vn_figures.py --metrics-csv figures/<sweep_id>/cluster_probe.csv \
    --metrics knn_spearman knn_ap ridge_spearman ridge_ap \
    --only vn03 vn06 vn09 vn12 vn15 pooled512 z pooled@scaled pooled+vn15 \
           pooled+randproj ecfp4 pred1
```

All times measured on this box (3× RTX A6000, 128 cores) over 31 models, and recorded in each
CSV's sibling `.meta.json`.

## Caveats, measured rather than asserted

- **The extracted embeddings are not bit-reproducible, and the RF probe inherits that.** Two
  extraction passes from the *same* checkpoint differ by up to **1.4e-5** on ~90% of entries
  (CUDA scatter reductions accumulate in nondeterministic order). A random forest's splits are
  discontinuous in its inputs, so that propagates to **~1e-2 per model on `auc_oob`** and
  **~9e-3 on `average_precision_oob`**, against ~1e-4 on the regression columns. Means over the
  30 models are stable to ~1e-3, and every gap claimed above is 10–100× the floor — but a
  single-model difference below ~0.02 on a classification column is not a difference.
- **One fold.** All 30 models validate on fold 0, so nothing here separates a property of the
  representation from a property of fold 0's clusters.
- **The bootstraps are not 10 independent samples.** They share one partition and differ only in
  the resample and the seed, which is why every claim is paired per model and required to hold
  across all three configurations rather than being given a p-value.
- **In-sample AP and AUC are saturated at 1.000** for all six embeddings and carry no
  information; `vn_figures.py` now refuses to plot them without `--suffix _oob`.
- **`knn20_jaccard` is sample-size dependent** and is comparable only at a fixed `--n-sample`
  (5,000 throughout here).
- **The tail is thin at this sample size** — 51 positives per cluster-half at pProp ≥ 3.5, so
  the AP columns in Probe 2 are noisy per model. The 30-model paired means are what carry the
  claim.
