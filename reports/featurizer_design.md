# Designing the featurizer: what to build, not whether what we built works

**Hand-written design document. Created 2026-08-21.**

`reports/embedding_collapse_results.md` answers *"is the current 32-d bottleneck a valid
featurizer?"* over ~60 GPU-runs. This file asks the other question: **given that the product is
a frozen `SMILES → R³²` function feeding a deep-kernel-learning GP that is re-fit every
active-learning round, what architecture and loss decisions make the most sense to implement?**

It is a design document, not a result. Nothing here has been run. Every number quoted is
sourced, and every recommendation names the instrument that would decide it.

**Sibling files.** `embedding_collapse_experiment.md` holds the design of the completed
experiment; `embedding_collapse_results.md` holds its results and is generated. This file holds
the *next* design. When it disagrees with the results file about a number, the results file
wins.

---

## 0. The constraints this document is written under

Established with Ethan 2026-08-21. All four bind the recommendations, and two of them narrow
the space considerably.

| constraint | consequence |
|---|---|
| **AmpC is the only target.** The project is drug discovery for AmpC specifically | cross-target multi-task — normally the highest-ceiling way to fill 32 dimensions with useful chemistry — is off the table as a *data* matter, not a design one |
| **The downstream is a true DKL**: a neural feature extractor on the 32-d input, then a GP on its output | the downstream **can learn to discard** dimensions it does not need. It **cannot recover** information the bottleneck never encoded. That asymmetry is the single most important fact in this document |
| **The DKL side is a colleague's work, with little room for flexibility** | anything touching it is a *request with a fallback*, never a decision. Quarantined into §6 |
| **No code or spec for the DKL side is available to read** | every claim about the downstream is an assumption and is labelled as one |

---

## 1. What the deliverable actually has to be

Three requirements. Each is stated with the instrument that measures it, because the recurring
failure in this project has been optimising a proxy nobody validated.

### R1 — sufficiency for pProp

The 32-d must support prediction of pProp on chemistry the model has not seen.

**Instrument: `src/feature_utility.py`** — held-out-*cluster* kNN (k=20, distance-weighted) and
ridge Spearman, plus AP@3.5. Anchored on both sides:

All five figures below come from one table — `embedding_collapse_results.md` § S2.5, fold 0,
both split directions averaged — so they are comparable to each other by construction. (S2.7 and
S2.8 re-measure `A_base` on the same split at 0.8703 and 0.8714; the agreement to three decimals
is a consistency check, not a second source.)

| featurization | dim | held-out-cluster kNN Spearman |
|---|---|---|
| current 32-d export (`A_base`) | 32 | **0.8711** |
| current 32-d export (`D_w3`) | 32 | **0.8619** |
| the model's own predicted pProp | 1 | 0.8334 |
| raw frozen MiniMol | 512 | 0.7730 |
| ECFP4 | 2048 | 0.6733 |

**R1 is met and is not the binding constraint.** Fine-tuning beats the frozen-embedding
workflow by 0.089 Spearman, which is the largest single margin in the project.

### R2 — retained discriminative information

Two molecules that differ chemically must land at different points, and the difference must be
recoverable. This is what lets a GP say *"I have never seen anything like this"*.

**Instrument: none exists.** This is the gap. `emb_effective_rank` was the proxy and it has
failed three independent ways (§2). `tanimoto_partial` and `knn20_jaccard` are closer but
measure agreement with one fingerprint family, have no upper reference, and — per the results
file's own open item #2 — `tanimoto_partial` = 0.139 still has nothing to be compared against.

**Under a true DKL, R2 is the requirement that matters most.** The downstream net can learn to
ignore a dimension that carries nothing; no downstream net can reconstruct a dimension that was
never populated. Every irreversible decision this repo makes is an R2 decision.

### R3 — well-conditioned for the model that consumes it

The kernel must be able to express varying confidence: an identifiable lengthscale, and
posterior variance with real dynamic range across the pool.

**Instrument: `src/uncertainty_probe.py` — and what it reports today is 91–99% noise floor**,
so it cannot currently answer the question. See §3, finding (a).

### The regime caveat, which decides how much geometry matters

A DKL's learned warping only helps once there are labels to fit it. Early AL rounds have
10²–10³ labels; in that regime the composite behaves much closer to a plain GP on the raw
export, and raw geometry is what the acquisition function sees. Later rounds accumulate labels
and the net takes over.

**Priority rule this document commits to:** options that pay off under *both* the round-1 and
round-N readings rank above options whose value depends on which one holds.

---

## 2. What the evidence already settles

Recorded here so the rest of the document does not relitigate it. Sources are sections of
`embedding_collapse_results.md`.

| claim | evidence | status |
|---|---|---|
| The bottleneck collapses at the default | effective rank 1.32–2.85 of 32; `scalarness` = ρ(emb distance, \|Δpred\|) = 0.950 | **settled** |
| `--w-vic 3 --vic-gamma 0.5` fixes it | rank 2.4 → 28.6 mean / 23.3 worst of 5 seeds; `scalarness` 0.950 → 0.271 | **settled (S1, S2a)** |
| The fix costs a little prediction | paired Δ`pearson_uniform` = −0.00060, CI [−0.00112, −0.00007]; 8× inside the pre-registered margin. Held-out kNN Spearman −0.009 | **settled as an upper bound (S2a, S2.5)** — the −0.009 was measured under a reweighting-blind probe; see §3(b) |
| Weight decay is not the mechanism | arithmetic (4.3% head / 0.33% trunk shrink) confirmed by `K_wd0.1`: 10× the decay moves rank less than run-to-run noise | **settled (P6.3, S2b)** |
| LayerNorm is not the mechanism | `--bottleneck-norm none` changes rank 2.25 → 2.19. Exonerated | **settled (S2.7)** |
| The learned bottleneck beats PCA of the trunk | `emb32` beats `trunkpca32`, `trunkpca32_white` and `trunk512_ft` on every config and every seed, by 0.019–0.044 kNN Spearman and 2.4× on AP@3.5 | **settled (S2.8)** |
| The mechanism is **neglect** | three candidates eliminated — not decay, not normalisation, not the 512→32 squeeze — leaving the original hypothesis: the supervised signal is near rank-1 and nothing asks the other ~30 dimensions to do anything | **surviving hypothesis, not settled.** Never tested positively. D3 rides on it; if D3's descriptor cell fails to raise rank, this row is where the diagnosis restarts |
| `w_vic` *redistributes* information, it does not add any | utility falls monotonically with the dose: 0.871 → 0.869 → 0.862 → 0.857 | **settled (S2.5)** |
| Effective rank is not a quality criterion | fails three ways: utility falls as rank rises (S2.5); γ=1.0 buys rank by inflating scale, not chemistry (S1); `A_base` at rank 2.1 beats `trunkpca32_white` at rank 31.5 (S2.8) | **settled.** Keep it as a *pathology detector* only |
| `emb_trace` (operating scale) is the primary lever on rank | trace lands at ≈32γ²: 9.35 measured against 8.0 predicted at γ=0.5, 34.93 against 32.0 at γ=1.0 | **settled (S2.7)** |

**Do not spend more runs on any row above.** In particular: more `w_vic` doses, more fold-0
seeds, `w_cov`, γ, and the bottleneck norm are all closed.

---

## 3. Two findings that reorganise the priorities

Both were computed while writing this document, from artifacts already on disk. Neither appears
in any existing report, and both change what should be built next.

### (a) The uncertainty probe reports a quantity that is 91–99% noise floor

From `reports/uncertainty_probe.csv`, parsing `noise_level` out of each fitted kernel string:

| cell | mean posterior std | implied noise std | **epistemic share of posterior variance** |
|---|---|---|---|
| `A_base` (`w_vic=0`) | 0.427 | 0.426 | **0.8%** |
| `C_w1` | 0.433 | 0.421 | 5.1% |
| `D_w3` (the pin) | 0.435 | 0.415 | **8.9%** |
| `E_w10` | 0.438 | 0.418 | 9.0% |

> *Approximation stated plainly:* the conversion uses the **global** pProp sd (0.8639) in place
> of each fit-set's own sd, because `normalize_y=True` rescales internally and the per-fit value
> is not in the CSV. The exact figures will shift by a few percent; the magnitude will not.

**And the noise floor is real, not a misfit.** The model's own residual std on this fold is
**0.430** (`val/mse` = 0.1852 at `A_base`, 0.1856 at `D_w3`), against a fitted GP noise std of
0.415–0.426. The GP is correctly reporting that at n_fit = 1000 the surrogate's irreducible
error dominates its epistemic uncertainty. **So the fix is not to cap the noise — that would be
forcing the GP to lie.** The fix is to stop reporting the *total* predictive std, which is
dominated by a constant, and report the epistemic component separately. That is the table
above.

**Two consequences, pointing opposite ways.**

**V2 and V3 were reading a signal that barely exists.** If 91–99% of the posterior variance is
a constant noise floor, the part that varies across the pool is a sliver. A near-constant
posterior std cannot correlate with structural novelty (V2) and cannot steer acquisition away
from greedy (V3). Their failures are properties of the *probe's configuration*, not evidence
about the embedding. The results file's reading — "the GP's variance is dominated by its fitted
noise term rather than by distance" — is correct and, once quantified, is disqualifying for
those two rules.

**V1 is considerably stronger than the results file states.** The epistemic share rises
0.8% → 5.1% → 8.9% monotonically in the `w_vic` dose. That is an **~11× restoration of the
epistemic content of the posterior**, not merely a calibration correlation that turned
positive. A collapsed embedding does not just produce meaningless confidence — it produces
*almost no epistemic confidence at all*, and the anti-collapse term is what puts it back.
**This is the strongest single piece of evidence for the anti-collapse work in the project**,
and it was sitting in a CSV.

**And the fitted lengthscale says the same thing from the other side.** The isotropic Matérn
lengthscale comes back at 10.5–15.2 in every cell, against a typical inter-molecule distance of
`√(2·emb_trace)` = 2.95–4.42:

| cell | `emb_trace` | typical distance | fitted lengthscale | ratio |
|---|---|---|---|---|
| `A_base` | 4.35 | 2.95 | 10.52 | **3.6×** |
| `C_w1` | 7.31 | 3.82 | 14.12 | 3.7× |
| `D_w3` | 9.56 | 4.37 | 13.57 | 3.1× |
| `E_w10` | 9.75 | 4.42 | 15.18 | 3.4× |

Every molecule is well inside one lengthscale of every other, so the kernel sits in its smooth,
near-linear regime and correlates everything with everything. **The marginal likelihood is
choosing this**, which means the data support it: pProp really is close to affine in the
export — because a *linear task head* makes it so by construction.

That is not a bug on the training manifold. It is a bug off it (§4, D5).

### (b) Both of our probes are the wrong shape for the deployment model, and the bias has a direction

`src/uncertainty_probe.py` fits a plain isotropic GP. `src/feature_utility.py` scores kNN and
ridge. **None of these can reweight or discard a dimension.** The deployment model can — that
is what the "deep" in deep kernel learning does.

So both probes:

- **understate** the value of an embedding that spreads information across 32 dimensions, since
  an isotropic kernel charges full price for every pProp-irrelevant direction; and
- **overstate** the value of an embedding where pProp is already near-affine, since kNN and a
  smooth GP read that off directly.

That reframes the project's two most-quoted disappointments. S2.5's "utility falls monotonically
with `w_vic`" (−0.009 Spearman) and S2.6's V2/V3 failures were **both measured under an
instrument structurally biased toward the collapsed embedding.**

**This does not overturn them — it prices them.** Combined with the regime caveat in §1, the
honest statement is: *our probes are approximately right for round 1 and pessimistic by round
N.* The −0.009 is an upper bound on the DKL-era cost, not an estimate of it.

---

## 4. The decisions, ranked

Six, ordered by (expected value ÷ cost) with ties broken by the priority rule in §1. The first
two are instruments and cost no GPU time; they are ranked first on purpose, because this
project's characteristic failure is committing to a change before the thing that would score it
exists.

---

### D1 — Repair this repo's own evaluation GP *(0 GPU-runs, ~2 h CPU)*

**In scope without negotiation.** `uncertainty_probe.py` is our instrument, written here in
sklearn to score embeddings. It is not the colleague's model and changing it touches nothing
downstream.

**Mechanism.** Three changes, in order of impact:

1. **Report the latent std**, with the noise term excluded from the predictive variance, and
   report the epistemic *share* beside it. This single change alters what V1, V2 and V3
   measure — all three are currently computed on a quantity that is 91–99% a constant.
2. **Do not cap the `WhiteKernel`.** The obvious-looking fix is wrong: the fitted noise std
   (0.415–0.426) is already essentially the model's own residual std (0.430, from
   `val/mse` ≈ 0.1852), so a cap at the irreducible error would change nothing, and a cap
   below it would force the GP to misreport. The noise floor is a true statement about the
   surrogate at n_fit = 1000, not a fitting artifact. What was wrong was reporting the total.
3. **Add a DKL-shaped arm.** A small MLP feature extractor plus a GP on its output, fit at the
   label counts D2(b) actually uses, so the instrument has the same shape as the deployment.
   Keep the plain-GP arm — it is the honest round-1 model — and report both.

An **ARD arm** is worth adding as a *diagnostic*, not as a request to the colleague: does the
embedding contain relevant information that a reweighting model can find? It carries its own
caveat, which must be written next to the number: fitting 32 lengthscales by marginal
likelihood at n≈100 is a well-known overfitting failure and can score *worse* than isotropic in
exactly the low-label regime the deliverable is for. If it does, the fallbacks are a shared
prior over lengthscales, a structured or low-rank kernel, or simply letting the DKL net do the
reweighting.

**Instrument this decides:** R3, and by extension every subsequent uncertainty claim.

**Expected direction.** The epistemic component gains dynamic range even though the total std
does not; V1's dose response survives and sharpens; V2 and V3 become answerable questions
rather than foregone ones.

**Why first.** Until this has dynamic range, no featurizer change can be scored on the property
the whole project claims to be optimising.

---

### D2 — Build the two instruments that do not exist *(0 GPU-runs, ~1 day CPU)*

#### (a) An information-retention probe — this is R2

**Mechanism.** Decode held-out structural information from the frozen 32-d and report how much
survives. Two readouts:

- **Decodability:** train a probe (ridge / small MLP) from the 32-d to a fingerprint family the
  model has never seen, and report macro-AP or mean per-bit AUC. **MACCS (167), Avalon, or
  atom-pair — never ECFP4**, which `emb_readout.py` already consumes as its reference.
- **Collision rate:** the fraction of molecule pairs that are close in embedding space
  (below ε) while being far apart chemically (Tanimoto below τ). This measures the failure
  mode directly rather than by proxy — it is what "the GP cannot tell these two apart" means.

**Anchors are pre-specified here, not deferred.** The results file's open item #2 is that
`tanimoto_partial` = 0.139 has no upper reference; D2(a) must not rebuild that defect. Every
number is reported against: frozen MiniMol-512, a random 32-d projection of ECFP4 (structure
with no model in it — the null), `A_base`, and `D_w3`. A 32-d embedding cannot match a 2048-bit
fingerprint on decodability, so the reference points are what make the number mean anything.

#### (b) An active-learning simulation that can actually discriminate

**Why the current one cannot.** V3 seeds with n0=200 against a surrogate already at Spearman
0.87, into a pool ~1% positive at the pProp ≥ 3.5 threshold. Greedy-on-mean scores ≈34 of 51
and UCB scores ≈34 — the variance contributes nothing because exploitation has already won.
Every cell reads the same, which is the definition of a non-discriminating instrument.

**Redesign, four changes:**

- **n0 ∈ {50, 100}**, matching the regime the deliverable is actually for.
- **Recall of pProp ≥ 5.0**, not ≥ 3.5. There are 100 such molecules in the whole dataset and
  20 per fold — genuinely rare, which is what the AL loop exists for. The ≥ 3.5 class at 3,153
  molecules is too easy to be a test.
- **Batched acquisition, with within-batch redundancy reported as a first-class outcome.** This
  is the mechanism the project has claimed since day one and has never tested: *a 50-molecule
  UCB batch drawn from a rank-2 space should be 50 near-duplicates*, because the top-50 by
  `mean + 2·std` in a one-dimensional space are 50 molecules with nearly the same predicted
  score. Report mean pairwise Tanimoto within each acquired batch, against a random-acquisition
  floor and a diversity-constrained ceiling.
- **Run it on the D1-repaired GP**, or the acquisition function is scoring a constant.

**Expected direction, stated before it runs.** This is where the geometry should pay off if it
pays off anywhere. Batch redundancy is the metric on which `A_base` should look catastrophic
and `D_w3` should look fine — and if it does not, the case for the whole anti-collapse branch
rests on V1's calibration result alone.

---

### D3 — The main featurizer bet: self-derived auxiliary targets *(~12 GPU-runs)*

The results file names this itself as the best untried option, and it follows directly from
collapse-by-neglect: *if the problem is that nothing asks the other 30 dimensions to carry
anything, the direct remedy is to ask them.*

**It is also the option the DKL constraint most favours.** A downstream net that can discard
makes *adding* information nearly free on the downside; nothing downstream can recover
information the bottleneck never encoded. Under the asymmetry in §0, this is the only category
of change that operates on the right side of the ledger.

**Primary form — a linear decoder to RDKit descriptors.** `32 → Linear → ~20–40 z-scored
physicochemical descriptors`, weighted MSE, at a small weight `w_aux`. Candidates: MW, cLogP,
TPSA, HBD, HBA, rotatable bonds, ring count, aromatic rings, fsp3, formal charge, heavy atoms,
QED, and similar. rdkit 2026.03.5 is in the venv and exposes 217 descriptors; compute once,
cache beside the CSV in the same style as `data/features/minimol_v1/`.

Three design choices, each with its reason:

- **Linear decoder, deliberately** — mirroring the linear task heads. It forces the information
  into the exported geometry rather than letting a deep decoder hold it. "These descriptors are
  linearly readable from the export" then becomes a property of the artifact.
- **Descriptors before fingerprint bits.** They are dense, low-dimensional, continuous,
  chemically interpretable, and — critically — **they do not collide with the ECFP-based
  readout**, so the existing structural instruments stay valid while this is tested.
- **Small weight, anchored against huber at 1**, in the same convention as every other term in
  `losses.py`.

**Second form, higher ceiling — a 32 → 2048 ECFP-bit decoder** (masked/weighted BCE).

> **Pre-committed, before any such run is scored:** an ECFP-bit decoder — or any Tanimoto-
> distance-alignment objective — **invalidates `tanimoto_partial`, `knn20_jaccard` and the
> structural half of `emb_readout.py` simultaneously**, since the model would then be trained
> on the exact quantity the readout uses to judge it. `emb_readout.py` must be re-anchored to a
> held-out fingerprint family **first**. Skipping this is the S1 ranking defect rebuilt at
> larger scale.

**Design of the test: a 2×2, not a scan.** `w_aux ∈ {0, x}` × `w_vic ∈ {0, 3}`, three seeds,
fold 0 — 12 runs, reusing `A_base` and `D_w3` where they already exist.

**Prediction, stated in advance:** the `w_aux > 0, w_vic = 0` cell reaches high effective rank
*without* the 0.009 Spearman cost, because the dimensions are being filled with information
that is relevant rather than merely non-degenerate. If that holds, **`w_vic` becomes the
fallback rather than the design** — kept as a floor against pathological collapse, at a lower
weight, rather than as the mechanism.

**Expected cost, honestly.** Descriptors are not pProp, so some R1 cost is likely — the same
trade `w_vic` showed. The difference is that this time both sides will have numbers: D2(a)
prices the R2 gain, `feature_utility.py` prices the R1 cost. Note also (§3b) that the measured
−0.009 for `w_vic` is a plain-GP/kNN figure and therefore an **upper bound** on what the same
trade costs under a DKL.

**The failure branches, written before the runs.** A prediction with no failure branch is what
sank the S1 selection rule; these are pre-registered so the reading is not chosen after the
fact. Judged on D2(a) decodability (R2) and `feature_utility.py` (R1):

| outcome | reading | what to do |
|---|---|---|
| rank ↑, decodability ↑, R1 cost < 0.009 | the bet lands | pin the descriptor form; drop `w_vic` to a floor or to 0, and re-run the 2×2's `w_vic` arm to confirm it is no longer needed |
| **rank ↑, decodability flat** | descriptor supervision filled the dimensions with *descriptor* information that is not structural information — the `w_vic` result reproduced with extra steps | do **not** ship it. Escalate to the ECFP-decoder form (after re-anchoring the readout), which targets structure directly rather than by proxy |
| rank ↑, decodability ↑, but R1 cost > 0.009 | the trade is real and **worse than the thing it replaces** | keep `w_vic=3`. Report the descriptor arm as a measured negative and take the R1-vs-R2 exchange rate to the DKL side as a decision, not a design choice |
| **rank flat** | the neglect hypothesis is wrong — supervision was added and the dimensions still did not fill | the §2 diagnosis restarts. This is the branch that would send the project back to architecture (D5) rather than loss |

The second and fourth rows are the ones that change Tier 2, and both are plausible enough to
write down now.

---

### D4 — Representation drift and off-manifold robustness *(~8–16 GPU-runs)*

**The most active-learning-specific concern in this document, and the one nothing in this
project has measured.** A generative loop queries chemistry no fold contains. Every number in
`embedding_collapse_results.md` is held-out-*cluster* within a fixed 331k library — which is
already better than a random split, but is not the same as novel chemistry from a generator.

**This is also the supervisor's top-ranked knob, still deferred.** `CLAUDE.md`'s
"Deferred: layer-wise freeze/unfreeze" records the ranking as (1) freeze schedule, (2) *which*
layers unfreeze, (3) learning rates, (4) total epochs — placing partial unfreezing **above every
learning rate this repo has tuned** — and its item 5 already ties the decision to this
deliverable: it should be decided on `emb_*` over held-out clusters, not on `goal_metric`.

**Two mechanisms, both untested:**

- **Restrict what moves.** Gradual or top-*k* unfreezing (ULMFiT). `gnn.depth` is 16, plus
  `encoder_manager`, `pre_nn` and `pre_nn_edges` — so "unfreeze the top *k* blocks" is **one
  integer knob**, which is sweepable. The soft version is discriminative per-depth learning
  rates. The crude version is simply a lower `--trunk-lr`, which costs nothing to test and
  should be the first probe.
- **Penalize movement.** Anchor the export, or the trunk's output, to the *frozen pretrained*
  MiniMol representation with a distillation term.

> **The anchoring form is presented in its only defensible role, and the evidence against it is
> stated up front.** On *utility* it looks bad: frozen MiniMol-512 scores 0.773 against the
> export's 0.862–0.871, and S2.8's `trunkpca32` (0.847) and `trunk512_ft` (0.847) both lose to
> `emb32` on every config and every seed. Pulling the representation back toward the pretrained
> manifold is, on everything measured so far, pulling it toward something worse. It survives
> **only** as an off-manifold-robustness play whose benefit is unmeasured — never as a utility
> gain, and never as a headline.

The trade-off is real in both directions and should be written as such: feature distortion
(LP-FT, Kumar et al. 2022) argues for less movement; "at 5 frozen epochs the trunk is still
mostly pretrained, and unfreezing bought +0.049 Pearson" argues for more.

**Instrument — and it does not exist yet either.** A **distance-stratified evaluation**: score
`feature_utility.py` and the D1-repaired GP probe on the held-out clusters *furthest* from the
training fold, rather than averaging over all of them. `data/splits/cluster_kfold_v1/` already
carries `fingerprints.npy` and cluster assignments, so the stratification is cheap. Everything
in this project currently reports the mean over held-out clusters, which is the average case;
the AL loop lives in the tail.

---

### D5 — Move the export point off the last pre-head layer *(~6 GPU-runs)*

**Mechanism.** `512 → 1024 → 1024 → 32 (export) → 256 → GELU → {cls, reg}`. Today the exported
tensor is the last layer before the linear task heads. The SSL projection-head result (SimCLR
and successors) is that the layer feeding the task loss absorbs task-specific invariance and
discards everything the loss does not need, while the layer *before* it retains more — which is
why representations are exported from before the projection head, not after. **Here the
exported layer is the layer feeding the loss**, which is the sharpest possible statement of
collapse-by-neglect.

**The prior against it, stated plainly.** S2.8 measured the nearest available version of
"export earlier" — the trunk itself — and it lost on every config and every seed by 0.021–0.024
kNN Spearman and by a factor of 2.4 on AP@3.5. That is real evidence that moving the export
point backward can cost, and it should temper expectations.

**What is given up, and why it is worth less than it was.** The current design makes "pProp is
linear in the export" literally true, chosen because it is the geometry an RBF/Matérn kernel
wants. But the downstream is a **DKL**, which supplies its own warping and does not need the
target to be affine in its input. The property is not free either — see below.

**The argument for it, in its strong form.** From §3(a): the fitted lengthscale sits at 3.1–3.7×
the typical inter-molecule distance, so the kernel is nearly flat and the epistemic variance
nearly constant. A linear head guarantees this, because it makes pProp affine in the export and
the marginal likelihood then prefers a very smooth kernel.

**A flat kernel is not by itself a defect.** If pProp really is near-affine in the export, a
confident GP is a *correct* GP and small epistemic variance is honest. **The defect is that the
affinity is a property of the training manifold and has no reason to survive novel chemistry.**
The loop will be confidently wrong precisely where it is exploring — which is the same concern
as D4, in a different mechanism, and is why these two belong next to each other.

**One coherence point, so this is not misread as substitutable with D1:** reweighting and the
export point are not alternatives. ARD — or a DKL net — reweights dimensions; neither changes
that the target is near-linear in the export, so the lengthscale stays long either way.

**Instrument.** One 3-seed cell scored on `feature_utility.py` (does R1 hold up?) and the
D1-repaired probe (does the lengthscale shorten and the epistemic variance gain range?), plus
D2(a) for R2.

---

### D6 — Decide the export contract explicitly *(engineering, no experiment)*

Not a modelling decision, but the one that blocks delivery.

- **`src/export.py` does not exist**, and `run_config.py` forces `--no-save-checkpoint`
  (`src/run_config.py:239`), so **nothing on disk today is exportable.** The chosen
  configuration must be re-run with `--keep-checkpoints` before anything reaches the DKL
  project.
- **Freeze a per-dimension standardization into the export** — mean and std computed on the
  *training* fold, shipped with the weights — so the downstream never has to guess the scale
  and lengthscale initialisation is well-posed. At `D_w3` this is nearly a no-op (per-dim stds
  are all ≈0.53, which is itself a useful property of the `vic` hinge), but it must be a stated
  contract rather than an accident of the loss.
- **Do not whiten.** S2.8: `trunkpca32_white` bought effective rank 31.5 — essentially the
  theoretical maximum — and paid 0.020 kNN Spearman for it, because whitening amplifies
  low-variance noise directions. Identical under ridge and worse under kNN, which is exactly
  what an invertible linear rescaling should do.
- **The trunk is not portable.** Reconstructing it needs graphium 2.4.7, the minimol 1.3.4
  wheel and `trunk.py`'s exact construction ordering, so the downstream project must vendor
  `trunk.py` / `head.py` / `model.py` or put this `src/` on `sys.path`. State this in the
  handover.

---

## 5. Sequencing

| tier | items | GPU | gates |
|---|---|---|---|
| **0** | D1, D2(a), D2(b) | none | everything below. No featurizer change can be scored until these exist |
| **1** | D3 (descriptor form), D4 (the `--trunk-lr` and top-*k* probes) | ~20 runs | the pin, and `sweeps/bayes_v2.yaml` |
| **2** | D5, D3 (ECFP-decoder form, after re-anchoring the readout) | ~12 runs | only if Tier 1 leaves R2 unsatisfied |
| **—** | D6 | none | delivery |

**S3 (transferability) and S4 (the sweep) stay blocked**, and this document does not unblock
them. S3's corners — 40 unfrozen epochs, dropout 0.3, folds 1 and 2 — are still the right
robustness check, but they should be run against whatever Tier 1 selects, not against `D_w3` in
advance. Note that S1 already showed dropout costs two-thirds of the rank at fixed `w_vic`
(`D_w3` ~30 → `G_w3_drop` ~10), so `bayes_v2` cannot document `dropout` and `w_vic` as
independent axes regardless of what happens here.

---

## 6. What to ask the DKL side — requests, not decisions

Quarantined because this is a colleague's work and Ethan has input rather than authority. Each
carries what to do if the answer is no. Ranked by value per unit of imposition.

1. **What label-count schedule does the AL loop actually use?** — n₀, batch size, number of
   rounds. Costs nothing to state and decides whether §1's round-1 or round-N reading governs,
   which in turn decides how much the raw geometry matters versus how much the DKL net will
   absorb. It also sets D2(b)'s parameters, so **this is the highest-value question in the
   list.** *If unanswered:* assume n₀ ∈ {50, 100} and report D2(b) across a range.
2. **Is the DKL net regularized against its own feature collapse?** Deep kernel learning is
   documented to shrink distances in order to overfit the marginal likelihood (Ober, Rasmussen
   & van der Wilk, 2021, *The promises and pitfalls of deep kernel learning*), producing
   overconfident posteriors. A frozen input that is already near-rank-2 makes this worse, since
   the net starts from a degenerate space. *If unanswered:* treat it as a reason to prefer a
   high-R2 export — the more information the input carries, the less the net has to
   manufacture.
3. **Is acquisition batched, and does it have any diversity term?** If it is batched and has
   none, D2(b)'s batch-redundancy metric is measuring a live failure rather than a hypothetical
   one, and the geometry argument gets much stronger. *If unanswered:* report both.
4. **Is `embed_dim = 32` renegotiable?** Last because it is the most invasive. Raise it only
   after D2(a) has numbers — R1's k-curve already saturates by k≈4–8, so if R2 also saturates
   early, 32 is generous and the question is moot; if R2 is still climbing at 32, there is a
   concrete case to make. *If not renegotiable:* nothing above changes; 32 is a constraint the
   rest of the design already assumes.

**The fallback that covers all four.** If none can be answered, the featurizer must be built to
be robust to an unknown consumer — which is precisely the argument for ranking R2 (D2a, D3)
above geometry tuning, and for D4's off-manifold work above further loss coefficients.

---

## 7. Considered and not worth doing

| option | why not |
|---|---|
| **End-to-end GP / marginal-likelihood training of the featurizer here** | gpytorch is not installed and the link is ~0.5 MB/s; and more fundamentally, GP marginal likelihood **actively rewards shrinking distances** — the documented DKL feature-collapse hazard — so it fights the exact property being sought. Wrong tool for producing a *frozen* input |
| **Larger `w_vic`, γ = 1.0, `w_cov` > 1** | all measured, all failed for stated reasons: `E_w10` pays ~3× the utility cost for no geometric gain over `D_w3`; γ=1.0 buys rank by inflating scale and has the *lowest* `tanimoto_partial` of the clearing cells; `w_cov` > 1 is a scale shrinker with severe seed dispersion (sd 2.97 → 13.64) |
| **Removing the bottleneck LayerNorm** | measured and exonerated (S2.7): rank 2.25 → 2.19 |
| **PCA of the trunk instead of a learned bottleneck** | measured and rejected (S2.8): loses on every arm, every config, every seed |
| **Cross-target multi-task** | no second docking target exists — AmpC only. The same "give the dimensions something to do" mechanism is reachable via D3 |
| **VICReg's invariance term** | needs a second view. Molecular graph augmentation that provably preserves binding-relevant chemistry is a research project of its own, not a knob |
| **An ordinal multi-threshold classifier** (logits at 3.0/3.5/4.0/4.5/5.0 instead of one at 3.5) | cheap and tempting, but every threshold is monotone in pProp, so the supervised signal stays essentially rank-1. It spreads the gradient over the range without asking any *new* dimension to carry anything — it does not address neglect |
| **More fold-0 seeds on `w_vic`** | P5, S1 and S2a exhausted this. Seed 1 is the binding corner in 8 of 11 cells, so additional seeds mostly re-measure seed 1 |
| **Effective rank as a selection criterion** | failed three independent ways (§2). Retain as a pathology detector — it correctly flags the rank-2 case whose GP variance is uncalibrated — and never as "higher is better" |

---

## 8. Open questions this document does not settle

1. **How many of the 32 dimensions does R2 actually need?** R1's k-curve saturates by k≈4–8
   (`A_base` reaches 96% of its skill by k=4). The R2 k-curve — decodability of held-out
   structure against components retained — is unmeasured, and it is what would make the
   `embed_dim` conversation in §6 concrete rather than speculative.
2. **Does D1's repaired probe change the S2 pin?** `D_w3` was selected under rules scored on
   the saturated probe and the reweighting-blind utility probe. The selection is not obviously
   wrong — it was made on rank, guards and worst-corner stability, none of which the probes
   touch — but the *justification* rests partly on numbers §3 shows are biased. Re-score before
   pinning anything into `sweeps/bayes_v2.yaml`.
3. **Whether `w_vic` survives D3.** If descriptor supervision fills the dimensions with relevant
   information, the right value of `w_vic` is probably lower than 3 and possibly 0. That is a
   question the 2×2 answers directly, and it should be answered before the sweep is registered
   — every `train.py` flag re-hashes every `config_id`, so all of this must land pre-sweep.
