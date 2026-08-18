# Metric and loss verification: **OVERALL: PASS**

8/8 checks pass. Objective `v1-binary3.5-c917327f`.

| check | result |
|---|---|
| metric parity vs pProp_MLP | PASS |
| BCE equals CrossEntropy(weight) | PASS |
| combined loss reduces to weighted MSE | PASS |
| weighting flavours order correctly | PASS |
| objective decomposes and guards | PASS |
| effective sample size table | PASS |
| weighted std differs from unweighted | PASS |
| freeze assertion has teeth | PASS |

Ran in 6.9s under python 3.11.15, numpy 2.2.6, torch 2.6.0+cu124.

---

## PASS — metric parity vs pProp_MLP

source: /home/ethan2/pProp_MLP @ 3b88a90fca4d2212581fd8680fbddc797bf69b04
input: n=20000 seed=0 edge=3.5

| source key | ours | source | ours | abs delta |
|---|---|---|---|---|
| `mae` | `mae_uniform` | 0.555519650263 | 0.555519650263 | 0.00e+00 |
| `weighted_mae` | `mae_balanced` | 0.925223980982 | 0.925223979798 | 1.18e-09 |
| `mae_skill` | `mae_skill_uniform` | 0.283178589959 | 0.283178589959 | 0.00e+00 |
| `weighted_mae_skill` | `mae_skill_balanced` | 0.552214964348 | 0.552214964025 | 3.23e-10 |
| `pearson` | `pearson_uniform` | 0.832214245856 | 0.832214245856 | 0.00e+00 |
| `pearson_weighted` | `pearson_balanced` | 0.936913778980 | 0.936913778972 | 7.33e-12 |

positives 871 vs source 871
tolerance 1e-08 (float64 accumulation order, not a porting margin)

## PASS — BCE equals CrossEntropy(weight)

weighted_bce_loss        0.722969830036
CrossEntropyLoss(weight) 0.722970008850   delta 1.79e-07
BCEWithLogits(mean)      0.801765620708   <- the form NOT used

class weights [0.02500000037252903, 1.975000023841858] (ratio 79.0x)
The mean form differs by a large constant at this ratio, which would move every inherited `w_cls` out of the units it was swept in.

## PASS — combined loss reduces to weighted MSE

combined_loss 1.040061593056
weighted_mse  2.080123186111
ratio 0.500000000000 (expected exactly 0.5)

The factor of 2 is `F.huber_loss`'s definition -- 0.5*r^2 below the transition, not r^2 -- and is inherited unchanged from pProp_MLP, which called the same function. So the loss scale `w_cls`/`w_pair`/`w_std` were swept against is preserved.

## PASS — weighting flavours order correctly

MAE   uniform 0.487440  <  balanced 0.741530   -> True

base rate  uniform 0.00951189 (true 3153/331480 = 0.00951189, match True)
base rate  balanced 0.499999997463 (0.5 to 1e-6, match True; the 2.5e-9 gap is float32 in `grouped_frequency_weights`, exact in float64)

Direction, not magnitude: `balanced` up-weights the hard tail 104x, so it must score *worse* than unweighted on a model that fits the easy bulk better. A sign error inverts the vector while leaving each number individually plausible, so the base rates pin the vector down as well -- unweighted must reproduce the true positive rate, and balanced must be exactly one half.

This check previously fixed three flavours against each other (`MAE_ipw < MAE_uniform < MAE_balanced`). `ipw` was removed from the modelling code on 2026-08-12, so the ordering is now two-sided and the base-rate assertions carry the weight the third flavour used to.

## PASS — objective decomposes and guards

objective version `v1-binary3.5-c917327f` (derived from the spec, so an edit re-stamps it automatically)

ap_star          0.493578
pearson_star     0.872112
mae_skill_star   0.440310
goal_term_cls    0.493578
goal_term_reg    0.656211
goal_metric      1.149790

cls + reg == goal_metric: True
missing-metric guard raises when `ap_uniform` is dropped: True

pProp_MLP accumulated three incompatible `goal_metric` revisions under one name; the derived version string is what stops that recurring here.

## PASS — effective sample size table

| edge | groups | ratio | ESS | % of N |
|---|---|---|---|---|
| 3.0 | 321,502 / 9,978 | 32x | 38,711 | 11.68% |
| 3.5 | 328,327 / 3,153 | 104x | 12,492 | 3.77% |
| 4.0 | 330,484 / 996 | 332x | 3,972 | 1.20% |
| 5.0 | 331,380 / 100 | 3314x | 400 | 0.12% |

3.5 is the finest edge that leaves a usable effective sample; 5.0 would train on an effective 400 molecules.

## PASS — weighted std differs from unweighted

target std unweighted 0.8639  |  balanced-weighted 1.4545  (1.68x)

An unweighted std term paired with a weighted huber would pull prediction spread toward the smaller number while the huber pulled toward the larger. `train.py` passes weights to both.

## PASS — freeze assertion has teeth

with `set_trunk_trainable` neutered: died
  freeze violated: trunk moved by 6.038e-04 (expected exactly 0) or the head did not move (6.019e-03). The schedule did no

The trunk trains through the frozen phase when the freeze is a no-op, so `trunk_max_delta` is non-zero and the paired check fires. Confirms the assertion in `train.py` has teeth.

