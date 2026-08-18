# The embedding geometry term: what `--w-vic` is, and what it does to the loss

**Why the 32-d export needs a loss term of its own, derived from the covariance matrix up —
and two problems the derivation exposed.**

Written 2026-08-18, after the head was reshaped around the 32-d deliverable. Every number
below is measured, not estimated; the commands are in "Reproducing" at the end. The
implementation is `losses.variance_covariance_loss` and `metrics.embedding_metrics`.

**Status of the two problems in §9:** both are real, neither is fixed yet. `--w-vic` defaults
to `0.0`, so nothing here is active in a run today.

---

## Summary

| | |
|---|---|
| What `vic` acts on | the `[B, 32]` bottleneck, **not** the predictions |
| What it is | VICReg's variance + covariance halves (Bardes, Ponce & LeCun 2022), no invariance term |
| What it minimises | the sum of squared eigenvalues of the embedding covariance |
| Why that is the right target | that is the same quantity `emb_effective_rank` measures — §5 |
| New loss | `w_cls·cls + huber + w_pair·pair + w_std·std + w_vic·vic` |
| New goal metric | **unchanged.** `OBJECTIVE_VERSION` still `v1-binary3.5-c917327f` |
| Default | `--w-vic 0.0` — inert until measured |
| Problem 1 | `--vic-gamma 1.0` is miscalibrated; measured healthy std is 0.585, so the hinge always fires |
| Problem 2 | **`w_vic` cannot be swept against `goal_metric`** — a sweep will always drive it to 0 |

---

## 1. Setup — the object everything is about

After the shared stack, one batch produces

```
Z ∈ R^(B × d),    d = 32,    B = batch size (1200)
```

Row `i` is molecule `i`'s exported embedding. Center it and form the **sample covariance**:

```
z = Z − 1·Z̄ᵀ                    (column-centered)
C = zᵀz / (B − 1)   ∈ R^(32×32)
```

- `C[j,j]` — how much dimension `j` varies across molecules
- `C[j,k]` — how much dimensions `j` and `k` move together

**Both the metric and the loss term are functions of this one matrix.** That is the whole
reason they can be reasoned about together.

---

## 2. What "collapse" means precisely

Let `λ₁ ≥ λ₂ ≥ … ≥ λ₃₂` be the eigenvalues of `C`. Two standard identities for symmetric `C`:

```
tr(C) = Σⱼ C[j,j] = Σᵢ λᵢ                    (total variance)
‖C‖_F² = Σⱼₖ C[j,k]² = Σᵢ λᵢ²
```

Normalising `pᵢ = λᵢ / Σλ`, the logged metric is

```
emb_effective_rank = exp( − Σᵢ pᵢ log pᵢ )
```

One dominant eigenvalue gives ~1; a flat spectrum gives 32. So **collapse is the eigenvalue
mass of `C` concentrating on a few directions** — nothing more exotic than that.

### Why the training signal causes it

The task heads are `Linear(32→1)`. Write their weight vectors `u_reg, u_cls ∈ R³²`. Then the
loss depends on `Z` **only** through the two scalar projections `Z·u_reg` and `Z·u_cls`.

Any component of `Z` orthogonal to `span(u_reg, u_cls)` is therefore **exactly invisible to
the loss**, and weight decay shrinks it toward zero. And because "pProp ≥ 3.5" is a threshold
of pProp, `u_cls ≈ α·u_reg` — that span is closer to one-dimensional than two.

**With linear heads this argument is exact rather than heuristic**, which is worth being
honest about: the linear heads were chosen to make pProp linear in the exported embedding
(good GP geometry), and the same choice sharpens the collapse risk. The two decisions are a
package — linear heads *plus* an explicit geometry term. With MLP branches the loss would see
`Z` through a nonlinear map that could use more of its directions, so the risk would be
milder and less predictable.

---

## 3. Why LayerNorm does not already fix this

Worth doing explicitly, because it looks like it should. **LayerNorm normalises across the 32
dimensions within a row; effective rank is a property of the columns.** It constrains each
molecule's vector, not each dimension's spread across molecules.

Concretely, suppose the pre-norm signal were rank-1, `xᵢ = aᵢ·v`:

```
LN(aᵢ·v) = (aᵢv − mean(aᵢv)) / std(aᵢv)
         = aᵢ(v − v̄) / (|aᵢ|·std(v))
         = sign(aᵢ) · (v − v̄)/std(v)
```

The magnitude `aᵢ` cancels completely, leaving two antipodal points.

So LayerNorm guarantees `tr(C)` stays off the floor — it cannot let everything shrink to zero
— but it places **no constraint whatsoever on how that variance is distributed across
eigen-directions.** That distribution is exactly what collapses. Note also that LayerNorm's
per-dimension affine scale `γⱼ` is learnable, so a dimension can still be killed by driving
`γⱼ → 0`, and the following GELU can push a dimension into its flat region.

---

## 4. The variance term

```
V = (1/d) · Σⱼ max(0, γ − σⱼ),    σⱼ = sqrt(C[j,j] + ε)
```

A hinge, per dimension, on its standard deviation across the batch. It pushes any dimension
whose spread falls below `γ` back up, and is **exactly zero** for dimensions already above it.

This is what stops a dimension going flat — the `emb_min_std → 0` failure. A hinge rather
than a penalty on `−σⱼ` so that it stops pushing once a dimension is healthy, instead of
inflating the embedding without limit.

Its effect on the covariance matrix: it keeps `C[j,j] ≳ γ²` for every `j`, hence
`tr(C) ≳ d·γ²`. **It holds the diagonal up. It does nothing about redundancy.**

---

## 5. The covariance term — and the bridge to the metric

```
K = (1/d) · Σ_{j≠k} C[j,k]²
```

Substituting the identity from §2:

```
K = (1/d)·( ‖C‖_F² − Σⱼ C[j,j]² )
  = (1/d)·( Σᵢ λᵢ²  − Σⱼ C[j,j]² )
```

**So with the diagonal held fixed, minimising `K` is exactly minimising `Σᵢ λᵢ².`**

And with the trace held fixed by the variance term, minimising `Σλᵢ²` subject to fixed `Σλᵢ`
maximises the participation ratio

```
PR = (Σλᵢ)² / Σλᵢ²
```

which is maximised precisely by the flat spectrum `λᵢ = const`, giving `PR = d` — the same
configuration that maximises `emb_effective_rank`.

> **This is the rigorous bridge.** The loss term and the metric are two different functions of
> the same eigenspectrum, and the term's minimiser is the metric's maximiser. Optimising one
> is not a proxy for improving the other; it is the same operation read two ways.

Note the diagonal subtraction: the term never penalises a dimension for being *informative*,
only for being **redundant with another dimension**.

### Measured, on controlled spectra (4096 × 32)

| true rank `k` | `emb_effective_rank` | `vic` (γ=1) | `vic` (γ=0.5) |
|---|---|---|---|
| 1 | 1.00 | 28.42 | 28.10 |
| 2 | 1.78 | 17.81 | 17.57 |
| 4 | 3.67 | 8.10 | 7.95 |
| 8 | 6.75 | 4.20 | 4.06 |
| 16 | 11.91 | 2.18 | 2.11 |
| 32 | 18.81 | 1.04 | 0.97 |

Monotone in collapse, with roughly two decades of dynamic range.

---

## 6. The two halves, divided cleanly

| half | acts on | prevents |
|---|---|---|
| variance `V` | the diagonal of `C` | a dimension going flat |
| covariance `K` | the off-diagonal of `C` | dimensions duplicating each other |

```
vic = V + K
```

**What it does not do:** it adds no target and no information. It cannot make the embedding
*mean* anything — it only forbids degeneracy, which lets the trunk's own chemical variation
occupy the dimensions pProp cannot fill. Whether that variation is *useful* to a GP is a
separate question this term does not answer.

---

## 7. The full loss, with measured magnitudes

```
L = w_cls·cls + huber + w_pair·pair + w_std·std + w_vic·vic
```

Huber stays grounded at weight 1 — it is the only term anchoring the absolute pProp level, so
every other term is measured against it.

Measured at batch 1200 on realistic z-scored data (a Pearson-0.87 model, ~5% positives):

| term | raw value | weight | contribution |
|---|---|---|---|
| `cls` | 0.466 | 0.4418 | 0.206 |
| `huber` | 0.143 | 1 (grounded) | 0.143 |
| `pair` | 0.572 | 7.486 | **4.28** |
| `std` | ~0 | 0.7911 | ~0 |
| **total without `vic`** | | | **≈ 4.6** |
| `vic` | 0.4 healthy → 28 collapsed | `w_vic` | see below |

### What this implies for `w_vic`

| `w_vic` | contribution at collapse | reading |
|---|---|---|
| 1.0 | 28 | 6× the entire rest of the loss — geometry would dominate accuracy |
| 0.3 | 8.4 | aggressive; expect a visible `goal_metric` cost |
| 0.1 | 2.8 | comparable to `pair`; a strong but not dominant pull |
| 0.03 | 0.85 | moderate |
| 0.01 | 0.28 | mild |
| 0.003 | 0.08 | barely perceptible |

So the defensible range is roughly **`w_vic ∈ [0.003, 0.3]`**. Anything at or above 1 is
optimising the wrong thing.

---

## 8. The goal metric — unchanged, deliberately

```
goal_metric = AP* + ½·(Pearson* + MAE_skill*)

AP*         = ap_uniform
Pearson*    = mean(pearson_uniform, pearson_balanced)
MAE_skill*  = mean(mae_skill_uniform, mae_skill_balanced)
```

`OBJECTIVE_VERSION` is still **`v1-binary3.5-c917327f`**. The sweep still maximises
`final/goal_metric_mean`. **Nothing about the objective moved.**

What changed is only what gets *reported*:

| new key | meaning |
|---|---|
| `val/emb_effective_rank` | `exp(entropy)` of the covariance eigenspectrum — 15–25 healthy, 2–4 the failure |
| `val/emb_top1_share` | fraction of variance on the largest eigenvalue — near 1.0 is collapse |
| `val/emb_min_std` | smallest per-dimension std — near 0 names a dead dimension |
| `val/emb_dim` | width, for provenance |
| `train/vic`, `val/vic` | the term itself, logged unscaled like the other four |

Plus `val_embeddings.npy` per run, so any of this is checkable after the fact.

**Why `emb_*` is not in `OBJECTIVE_SPEC`:** adding it would re-stamp `OBJECTIVE_VERSION`,
which makes every already-scored run non-comparable and un-poolable (`pool_oof.group_key`
keys on the provenance triple). It is a reported diagnostic, exactly like `enrichment_factor`.

---

## 9. Two problems this derivation exposed

### 9.1 The `--vic-gamma` default of 1.0 is miscalibrated

`γ = 1.0` was inherited from VICReg, which applies the term to an **unnormalised** projector
output where the natural scale is free. Here the export sits after `LayerNorm → GELU`, which
fixes the scale. Measured on a random-init bottleneck (`Linear(1024→32) → LayerNorm → GELU`,
20,000 samples):

| quantity | value |
|---|---|
| per-dimension std | **0.585** (min 0.557, max 0.613) |
| `emb_effective_rank` | 30.78 / 32 — i.e. healthy |
| `vic` at γ = 1.0 | **0.421** ← fires when nothing is wrong |
| `vic` at γ = 0.5 | **0.0057** ← correctly inert |
| `vic` at γ = 0.3 | 0.0057 |

At γ=1.0 the hinge is permanently active, applying constant pressure to inflate the LayerNorm
affine scales even on a perfectly healthy embedding. **γ ≈ 0.5 is the right default** — and it
costs essentially no sensitivity, because at true collapse the covariance half dominates
(28.10 at γ=0.5 versus 28.42 at γ=1.0, a 1% difference).

### 9.2 `w_vic` cannot be swept against `goal_metric`

This is the more serious one, and it is structural rather than a tuning error.

`vic` adds no predictive information — it constrains the representation. So raising `w_vic`
can only leave `goal_metric` flat or **lower** it. A bayes sweep maximising
`final/goal_metric_mean` will therefore drive `w_vic → 0` **regardless of how collapsed the
embedding is**, and report that as the answer.

The note currently in `sweeps/bayes_v2.yaml` — *"if that number comes back at 2-4 out of 32,
this becomes the most important axis in the file"* — is **wrong as written**. `w_vic` is not a
sweep axis at all.

This is a specific instance of the general problem stated in §8 of
`reports/pprop_mlp_transfer.md` and in CLAUDE.md: **`goal_metric` scores the predictions, and
the predictions are not the product.** Any knob that trades prediction quality for embedding
quality is invisible-or-harmful to the sweep objective by construction.

---

## 10. How to actually choose `w_vic`

Treat it as a **constrained choice**, not an optimisation: pick the smallest `w_vic` that gets
`emb_effective_rank` above target, and accept the `goal_metric` it costs.

Mechanically that is a 1-D scan at otherwise-fixed hyperparameters:

```bash
for W in 0 0.003 0.01 0.03 0.1 0.3; do
  python src/train.py --fold 0 --seed 0 --w-vic $W --vic-gamma 0.5
done
```

Then read off the trade curve — `emb_effective_rank` against `goal_metric` — and choose a
point on it. Two things make this cheap: it is six runs rather than a sweep, and it only needs
running once, because the answer is a property of the architecture rather than of the other
hyperparameters.

**Order of operations.** Run `w_vic = 0` first and read `emb_effective_rank`. If it comes back
at 15–25 the architecture is sufficient on its own and none of this is needed. Only if it
comes back low (2–4 is the prediction) does the scan matter.

---

## Reproducing

Every number in this document comes from `src/losses.py` and `src/metrics.py` directly, with
no training involved — the synthetic spectra of §5 and the random-init bottleneck of §9.1 are
constructed in-place. Local `torch` is sufficient; no graphium, no GPU, no rabelais.

The three measurement blocks are: per-dimension std at a healthy bottleneck (§9.1), `vic`
versus effective rank on controlled spectra (§5), and the existing term magnitudes (§7).

Related: `reports/pprop_mlp_transfer.md` §8 for why the objective is a proxy here, and the
"The deliverable is a 32-d embedding" section of `CLAUDE.md` for the architecture this term
guards.
