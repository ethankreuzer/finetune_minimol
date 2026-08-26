# MiniMol: what actually happens between a SMILES string and a 512-d vector

A step-by-step walk through MiniMol v1's inference path, written for someone who knows deep
learning but has never worked with graph neural networks or molecular ML.

Every number here was measured on this machine against the pinned stack (graphium 2.4.7,
minimol 1.3.4, torch 2.6.0+cu124), not read off the config. The measurement batch is **8
molecules from `data/ampc_subset_331k.csv`, giving 194 atoms and 418 directed edges** — those
three numbers appear throughout, so you can re-run any claim. Source-level claims cite
`file:line` in the installed packages under `.venv/lib/python3.11/site-packages/`.

The document describes **`eval()` mode**. Two things differ in `train()` mode, both flagged in
§11, and both are live during this repo's fine-tuning because `MiniMolTrunk.__init__` ends with
`self.train()`.

---

## 1. The problem: a molecule is not a vector

Start with why none of the architectures you already know apply.

A molecule is a set of atoms plus a set of bonds between them. Two properties break the usual
tools:

**No canonical ordering.** You can number caffeine's 14 heavy atoms in 14! different ways and it is
the same molecule. If you flatten atoms into a fixed vector and feed an MLP, you have told the
network that "atom 3" means something — it does not. The network must produce the *same* answer
under any renumbering. That property is called **permutation invariance** (for the whole-molecule
output) or **permutation equivariance** (for per-atom outputs: renumber the input, the outputs
renumber the same way).

**No fixed size.** Molecules in this dataset range from a handful of atoms to dozens. A CNN wants
a grid, an MLP wants a fixed width; a molecule offers neither.

The construction that satisfies both is **message passing**. Represent the molecule as a graph:
atoms are nodes, bonds are edges. Give every atom a feature vector. Then repeat, some number of
times:

> every atom looks at its bonded neighbours, aggregates their vectors with a **permutation-invariant**
> operation (sum, mean, or max — anything that ignores the order it received them in), and updates
> its own vector from that aggregate.

Because the aggregation is order-blind, the whole thing is equivariant by construction. Because
it is defined per-atom, it works at any size. After *k* rounds, each atom's vector summarises
its *k*-bond neighbourhood — the same "growing receptive field" intuition as stacked convolutions,
with the graph replacing the grid.

At the end, one more permutation-invariant operation collapses all atom vectors into a single
molecule vector. That final step is called the **readout** or **pooling**.

That is the entire idea. Everything below is MiniMol's specific choices about features,
aggregation function, depth, and readout — plus two repairs for things plain message passing
cannot do.

---

## 2. The pipeline at a glance

```
SMILES string
   │  RDKit parse
   ▼
molecular graph  ──────────────────────────────────────────────┐
   │                                                            │
   ├─ atom features         85 per atom                         │
   ├─ bond features         13 per bond                         │
   └─ positional encodings  8 + 8 eigen-, 16 walk- per atom     │
        │                                                       │
        ▼                                                       │
   encoder_manager: PE → 32-d, concatenated in front of the 85  │
        │  117 per atom                                         │
        ▼                                                       │
   pre_nn (MLP 117→256→128)          pre_nn_edges (MLP 13→128→168)
        │  128 per atom                       │  168 per bond   │
        ▼                                     ▼                 │
   ┌────────────────────────────────────────────────────────┐   │
   │  16 × [ GINE message-passing layer + virtual node ]     │◄──┘
   │  atom vectors: 128 → 336 → … → 336 → 512                │
   └────────────────────────────────────────────────────────┘
        │  512 per atom
        ▼
   global_max_pool over the atoms of each molecule
        │
        ▼
   512-d molecule embedding   ← this is "the MiniMol fingerprint"
```

Parameter counts, measured (`sum(p.numel())` per registered child of `MiniMolTrunk`):

| module | params |
|---|---:|
| `encoder_manager` | 14,016 |
| `pre_nn` | 63,872 |
| `pre_nn_edges` | 24,056 |
| `gnn` | 7,818,000 |
| **total trunk** | **7,919,944** |

---

## 3. Stage 0 — SMILES to a graph

`"CC(=O)Oc1ccccc1C(=O)O"` is aspirin in SMILES, a linear text encoding of a molecular graph.
RDKit parses it into an atom list and a bond list. Hydrogens are implicit
(`explicit_H: false`) — carbon "knows" it has 3 hydrogens from its valence, so they are not
separate nodes. No self-loops are added (`add_self_loop: false`).

Each bond becomes **two directed edges**, one per direction. Measured on the batch: 209 bonds
across the 8 molecules, giving **418 directed edges**. Message passing needs both directions so
information can flow independently each way.

Nothing is learned in this stage. It is deterministic preprocessing, and in this repo it is
cached once for all 331,480 molecules (`src/featurize.py`).

---

## 4. Stage 1 — hand-designed features

The network needs numbers, not chemistry. MiniMol's config specifies exactly which chemical
properties become input features.

### Atom features: 85 numbers per atom

Four **one-hot** properties (`atom_property_list_onehot`) and five **scalar** ones
(`atom_property_list_float`). Measured widths, from featurizing aspirin:

| property | kind | width | what it is |
|---|---|---:|---|
| `atomic-number` | one-hot | 44 | which element. `graphium.features.nmp.ATOM_LIST` has 43 entries and the encoder adds one "unknown" slot |
| `group` | one-hot | 20 | column of the periodic table. `nmp.GROUP_SET` has 19 entries + unknown |
| `period` | one-hot | 8 | row of the periodic table. `nmp.PERIOD_SET` has 7 entries + unknown |
| `total-valence` | one-hot | 8 | how many bonds this atom makes in total. `nmp.VALENCE` is `[0..6]` + unknown |
| `degree` | float | 1 | number of *explicit* bonded neighbours |
| `formal-charge` | float | 1 | +1, 0, −1 … |
| `radical-electron` | float | 1 | unpaired electrons |
| `aromatic` | float | 1 | 0/1, is it in an aromatic ring |
| `in-ring` | float | 1 | 0/1, is it in any ring |
| | | **85** | matches `datamodule.in_dims['feat'] = 85` |

Note the redundancy: for 92 of the 118 elements, `(group, period)` determines the element, so
`atomic-number` is largely implied by the other two. (Measured: the only collisions are the 14
lanthanides and 14 actinides, which `nmp.GROUP` lumps into a catch-all group 19 — irrelevant for
this dataset's organic chemistry.) The redundancy is deliberate and normal in molecular
featurization: you hand the network the same fact in several coordinate systems and let it
choose.

### Bond features: 13 numbers per bond

`bond-type-onehot` (5: `nmp.BOND_TYPES` is single/double/triple/aromatic, + unknown), `stereo`
(7: `nmp.BOND_STEREO` is none/any/Z/E/cis/trans, + unknown), `in-ring` (1). Total 13, matching
`in_dims['edge_feat'] = 13`.

---

## 5. Stage 2 — positional encodings, and the two problems they fix

Here is where the design stops being obvious. Plain message passing has two blind spots.

**Blind spot 1: it cannot always tell different molecules apart.** Message passing of the kind
described in §1 is provably no more discriminative than the Weisfeiler–Lehman graph isomorphism
test. There are pairs of genuinely different graphs it maps to identical outputs — the textbook
example is two triangles versus one hexagon, where every node has degree 2 and every
neighbourhood looks identical at every round. In chemistry this shows up as ring systems that
message passing cannot distinguish.

**Blind spot 2: an atom does not know where it sits in the molecule.** After *k* rounds an atom
knows its *k*-hop neighbourhood, but nothing tells it "I am on a terminal side-chain" versus
"I am buried in a fused ring core."

The fix is the graph analogue of a Transformer's positional encoding: compute something about
each atom's position in the graph *before* the network runs, and feed it in as extra features.
MiniMol computes two, both from the graph structure alone (no chemistry):

### Laplacian eigenvectors — 8 + 8 numbers per atom

Build the graph Laplacian `L = D − A` (degree matrix minus adjacency matrix), and take its
first 8 eigenvectors and eigenvalues. If you have seen spectral clustering or Fourier analysis,
this is the same object: the eigenvectors of the Laplacian are the graph's natural vibration
modes, low-frequency ones splitting the molecule into large smooth regions and high-frequency
ones oscillating between neighbours. An atom's coordinates in that eigenbasis are a genuine
"where am I" descriptor. This is the direct analogue of the sinusoidal positional encoding in a
Transformer, where the sequence's Laplacian eigenvectors *are* sinusoids.

Each atom gets 8 eigenvector components (`laplacian_eigvec`) and the 8 corresponding eigenvalues
(`laplacian_eigval`, the same for every atom in a molecule — they say how "important" each mode
is). Small molecules with fewer than 8 eigenvectors are padded with NaN and masked to zero later
(`laplace_pos_encoder.py:196`).

**The sign catch.** If `v` is an eigenvector, so is `−v` — the decomposition does not fix a sign.
Two runs of the same molecule can produce opposite signs, and the network must not care. §11
covers how MiniMol handles this.

### Random-walk return probabilities — 16 numbers per atom

`rw_return_probs` with `ksteps: 16`: for `k = 1…16`, the probability that a random walk starting
at this atom is back at this atom after exactly `k` steps. It is the diagonal of `(D⁻¹A)^k`.

This is a cheap and very direct encoding of local ring structure — a walk in a 6-ring has a
strong return spike at k=6, a walk on a chain does not — which is exactly blind spot 1. It has
no sign ambiguity, which is why it needs no special handling.

---

## 6. Stage 3 — `encoder_manager`: learned encoders for the encodings

The positional encodings are raw numbers of different shapes (8, 8, 16). Two small networks turn
them into a common 32-d space. Total: **14,016 parameters**.

```
LapPENodeEncoder (la_pos)          MLPEncoder (rw_pos)
  linear_in : FCLayer(2 → 64)        first_normalization : LayerNorm(16)
  pe_encoder: MLP(64 → 64)           pe_encoder          : MLP(16 → 32)
  post_mlp  : MLP(64 → 32)
```

### `la_pos` is a DeepSet, and that matters

For each atom, its 8 (eigenvector-component, eigenvalue) pairs are treated as a **set** of 8
items, each 2-dimensional. The steps (`laplace_pos_encoder.py:182–221`):

1. `linear_in` maps each 2-d pair to 64-d — applied identically to all 8, so shape becomes
   `[n_atoms, 8, 64]`.
2. `pe_encoder`, a 2-layer MLP, transforms each item independently: still `[n_atoms, 8, 64]`.
3. `torch.sum(..., dim=1)` collapses the 8 items into one 64-d vector.
4. `post_mlp` maps 64 → 32.

Step 3 is the whole point. Summing over the set makes the output **invariant to which eigenvector
is which** — the DeepSet pattern (`φ` per element, sum, then `ρ`). Padded slots were zeroed
first (`pos_enc[empty_mask] = 0`), so they contribute nothing to the sum, which is why molecules
with fewer than 8 eigenvectors work at all.

### The two encoders are summed, then concatenated in front

Both encoders write to the output key `feat`. The manager stacks the two 32-d outputs and applies
`pe_pool: sum` — an **elementwise sum**, not a concatenation
(`encoder_manager.py:forward_simple_pooling`). So the two positional signals share one 32-d
budget.

That pooled 32-d is then concatenated with the 85 atom features (`encoder_manager.py:190`):

```python
feat = this_pe                                  # the pooled 32-d PE
if pe_key in get_keys(g):
    feat = torch.cat((feat, g[pe_key]), dim=-1)  # encoder_manager.py:190
```

**Positional encoding first, atom features second** — the intuitive guess is the reverse.
Verified by slicing: `out[:, 32:]` equals the raw 85-d atom features exactly, `out[:, :85]` does
not. Result: **117 = 32 + 85** numbers per atom.

---

## 7. Stage 4 — `pre_nn` and `pre_nn_edges`: getting to working width

Two ordinary MLPs, no graph structure involved. They lift the hand-designed features into the
widths the GNN wants.

| | shape | params |
|---|---|---:|
| `pre_nn` | 117 → 256 → 128, depth 2 | 63,872 |
| `pre_nn_edges` | 13 → 128 → 168, depth 2 | 24,056 |

Both use ReLU, dropout 0.02, LayerNorm, and `last_normalization: layer_norm` — so the tensors
entering the GNN are already normalized (which is why the GNN's own `first_normalization` is
`None`, verified).

An idiosyncrasy worth knowing because it contradicts the usual template: graphium's `FCLayer`
runs **Linear → LayerNorm → Dropout → Activation** (`base_layers.py:418–431`). Dropout *before*
the nonlinearity is unusual. Stated as observed in source, not normalized to convention.

---

## 8. Stage 5 — the GNN: 16 rounds of message passing

`gnn` is a `FeedForwardGraph` with **7,818,000 parameters**, 99% of the trunk. Measured widths:

```
full_dims       = [128, 336, 336, ... , 336, 512]    # 16 layers, 15 hidden at 336
full_dims_edges = [168, 168, ... , 168, 64]          # built, but see §8.3
```

### 8.1 What one GINE layer computes

The layer type is `pyg:gine` — **G**raph **I**somorphism **N**etwork with **E**dges (Hu et al.,
2019). One layer, for atom *i*:

```
x'_i  =  h_Θ (  (1 + ε) · x_i  +  Σ_{j ∈ N(i)}  ReLU( x_j  +  W_e · e_{j,i} )  )
```

Read it left to right:

- `Σ_{j ∈ N(i)}` — **sum** over bonded neighbours. Sum, not mean: GIN's design argument is that
  summation is the aggregator that makes message passing as discriminative as the WL test, because
  mean and max both lose multiset multiplicity ("three neighbours like this" vs "one").
- `x_j + W_e · e_{j,i}` — the message from neighbour *j* is its own vector **plus** the bond's
  feature vector, projected to matching width. This is the "E" in GINE: the bond type participates
  in the message.
- `(1 + ε) · x_i` — the atom's own vector, kept alongside the neighbour sum. **Measured: `ε = 0.0`
  and it is not a `Parameter`**, so this term is exactly `x_i`.
- `h_Θ` — an MLP applied to the result. Measured per layer: `MLP(in_dim → in_dim → out_dim,
  depth=2)` (`gin_pyg.py:231`), e.g. `MLP(336 → 336 → 336)` in the middle layers. This is the
  "MLP inside each GNN layer" — note it sits **after** aggregation, so its hidden activation is a
  per-atom quantity, not a per-message one.

Then `apply_norm_activation_dropout` (`base_graph_layer.py`) applies **LayerNorm → GELU →
Dropout(0.02)** to the output.

Per-layer parameter counts, measured: layer 0 = 82,416; layers 1–14 = 284,592 each; layer 15 =
344,256. Total across all 16 layers: **4,410,960**.

### 8.1b There are three different "layer *i* output" tensors

This matters the moment you try to read intermediate representations, and the names do not
disambiguate it. Per depth *i*, `_graph_layer_forward` and the loop around it produce three
distinct tensors in sequence (`global_architectures.py:1164`, `:1172`, `:1273–1288`):

| # | tensor | produced by | how to reach it |
|---|---|---|---|
| 1 | post-convolution | `g = layer(g)` inside `_graph_layer_forward` | wrap `gnn.layers[i].forward` |
| 2 | post-residual | `residual_layer.forward(feat, feat_prev)`, *outside* the layer | not exposed; requires instrumenting the loop |
| 3 | post-virtual-node | `feat + node_projection(vn_feat[batch])` | `_readout_cache[i]`, via `_enable_readout_cache` |

Only **#3** is what flows into layer *i*+1, and only #3 is what the readout cache stores
(`global_architectures.py:1288`). The §12 shapes table reports #1, because that is what wrapping
`layers[i].forward` captures — the widths are identical across all three, so a shape check will
not tell you which one you have.

### 8.2 Residual connections

`residual_type: simple` with `skip_steps=1` — a plain additive skip between consecutive layers.
The 15 hidden layers share a width of 336 specifically so this addition is well-defined without
projections. It is applied to every layer *except the last* (`global_architectures.py:1171`),
because layer 15 changes width 336 → 512.

The consequence matters if you ever want to read intermediate layers: 15 same-width layers chained
with additive skips produce **strongly correlated** successive representations. Consecutive layer
outputs are near-duplicates by construction.

### 8.3 Edge features never change — and why that is not a bug

Counterintuitive, and measured three ways:

1. `GINEConvPyg.layer_outputs_edges` returns `False` (`gin_pyg.py:289`) — the layer consumes edge
   features but never emits updated ones.
2. The virtual node's `use_edges` defaults to `False` and MiniMol's config never sets it.
3. Directly: `torch.equal(edge_feat_after_L0, edge_feat_after_L15)` is `True`, and both are
   bit-identical to `pre_nn_edges`'s output.

So there is exactly **one** edge tensor, 168-d, in the entire model, computed once and read 16
times.

Two precisions, because someone auditing the config will find contradicting numbers:

- `hidden_dims_edges: 168` / `out_dim_edges: 64` **are** honoured at construction — the 64 appears
  in `full_dims_edges` and shapes the built modules. What never executes is the edge *update path*.
  The config is not ignored; it is dead code downstream.
- The static edge tensor is nonetheless read through **16 different learned lenses**: PyG's
  `GINEConv(nn, edge_dim=168)` builds its own `Linear(168, in_dim)` per layer — measured as
  `Linear(168, 128)` at layer 0 and `Linear(168, 336)` at layers 1–15. Each depth learns its own
  view of the same bond descriptors.

---

## 9. The virtual node: message passing's speed limit, and the fix

### The problem

Information travels one bond per layer. Two atoms 10 bonds apart cannot influence each other
until layer 10. Push depth up to compensate and you hit **over-smoothing**: after many rounds of
neighbour aggregation, every atom's vector converges toward the same value and the network loses
the ability to tell atoms apart.

### The fix

Add an imaginary extra node connected to every real atom. It reads the whole molecule in one
step and writes back to every atom in one step, giving a global shortcut at every depth.

MiniMol uses `virtual_node: logsum` and builds **15** of these — one after each layer except the
last. Measured `VirtualNodePyg` structure:

```
layer          : scatter_logsum_pool          (atoms → one vector per molecule)
fc_layer       : FCLayer(336 → 336, GELU)
node_projection: MuReadoutGraphium(336 → 336)  (molecule vector → back to every atom)
```

The forward pass (`pooling_pyg.py:330–343`):

```python
pool     = scatter_logsum_pool(feat, batch)          # [n_molecules, 336]
vn_feat  = vn_feat + fc_layer(vn_feat + pool)        # residual accumulation
feat     = feat + node_projection(vn_feat[batch])    # broadcast back to atoms
```

`logsum` pooling is **not** a log-sum-exp, despite the name. Measured
(`pooling_pyg.py:50–60`) it is `(log N_i / N_i) · Σ_k x_k` — the mean over atoms, rescaled by
the log of the atom count. graphium's own docstring gives the reason: it keeps sum-pooling's
expressive power while growing logarithmically rather than linearly with molecule size, so a
60-atom molecule does not produce a state 10× the scale of a 6-atom one.

Three things worth taking away:

**`vn_feat` is a running graph-level state.** It starts at the scalar `0.0`, accumulates
residually through all 15 depths, and is *never normalized*. Measured mean L2 norm over the
8-molecule batch: **2.73 at depth 0, 14.18 at depth 5, 36.50 at depth 10, 45.27 at depth 14** —
it grows by ~17× across the stack.

**It is a local variable.** `vn_feat` lives only inside `FeedForwardGraph.forward`
(`global_architectures.py:1263–1285`); it is never written into the graph dict `g`, so it is not
reachable after the forward pass without instrumenting the module.

**It holds 43% of the GNN.** Measured: 227,136 params per virtual-node layer × 15 =
**3,407,040**, against 4,410,960 in the message-passing layers themselves. The component that
gets no mention in the model card is nearly half the network.

---

## 10. The readout, `graph_output_nn`, and the five heads

This is the part of the architecture where the config, the module tree and the intuition all
disagree slightly, so it is worked through one step at a time. Everything below was measured on
the usual 8-molecule / 194-atom batch by instantiating the real `FullGraphMultiTaskNetwork` and
running it.

### 10.1 First: there are two different 512-d tensors

Almost every confusion about this section comes from collapsing these two into one:

| tensor | shape | level | who consumes it |
|---|---|---|---|
| `g["feat"]` after GNN layer 15 | `[194, 512]` | **atom** | the `node` branch |
| `g["graph_feat"]` after max-pool | `[8, 512]` | **molecule** | the `graph` branch |

The second is "the MiniMol fingerprint." They are both "512-d" and they are not the same object.
The two branches of `graph_output_nn` are exactly the two consumers of these two tensors.

### 10.2 The readout itself

One permutation-invariant operation collapses atoms to molecules:

```python
fingerprint = global_max_pool(node_features, batch_indices)   # minimol/model.py:82
```

An elementwise max over the atoms of each molecule: `[194, 512] → [8, 512]`. Each of the 512
outputs answers "what is the strongest response to this learned feature anywhere in the
molecule?"

Max pooling is a real choice with a real cost: it is blind to multiplicity. A molecule with one
carboxyl group and a molecule with five pool identically on any dimension that detects carboxyls.
Sum or mean would preserve count and composition; max preserves "is this present, and how
strongly."

### 10.3 The module tree — `graph_output_nn` is inside `task_heads`, not beside it

Measured `net.named_children()`:

```
FullGraphMultiTaskNetwork
├── encoder_manager
├── pre_nn
├── pre_nn_edges
├── gnn                      ← trunk.py registers these four
└── task_heads (TaskHeads)   ← trunk.py excludes this entire subtree
    ├── graph_output_nn : ModuleDict{ 'graph', 'node' }
    └── task_heads      : ModuleDict{ 5 tasks }
```

`graph_output_nn` runs *between* the GNN and the heads in execution order, but structurally it is
a **child of `TaskHeads`**, sharing a parent with the heads themselves. That is why
`trunk.py`'s single exclusion of `network.task_heads` removes both the post-processing MLPs and
the five heads in one move — and why `.parameters()` then yields exactly the 7,919,944 trunk
params (of which 32 are inert — §11).

For scale, the excluded subtree is **2,405,753** params (939,328 in `graph_output_nn`, 1,466,425
in the heads), so the whole pretraining network is 7,919,944 + 2,405,753 = **10,325,697** and
this repo keeps 77% of it.

### 10.4 The branching rule: branches are keyed by task *level*, not by task

This is what makes "five heads, two branches" stop looking like a mismatch.

Each task declares a `task_level` in the config. Four say `graph`, one says `node`:

| head | `task_level` | out_dim | what it is |
|---|---|---:|---|
| `l1000_vcap` | graph | 2934 | gene-expression response, VCAP cell line |
| `l1000_mcf7` | graph | 2934 | gene-expression response, MCF7 cell line |
| `pcba_1328` | graph | 1328 | PubChem BioAssay activity, 1328 assays |
| `pcqm4m_g25` | graph | 25 | quantum-chemical graph properties |
| `pcqm4m_n4` | **node** | 4 | quantum-chemical per-atom properties |

`TaskHeads.__init__` (`global_architectures.py:2202`) loops over **tasks** and writes
`self.graph_output_nn[task_level] = GraphOutputNN(...)`. Two distinct levels among five tasks
means **two** surviving entries — the count comes from how many levels exist, never from how
many tasks. (It constructs five `GraphOutputNN` objects and discards three by dict overwrite.
Harmless, because the weights arrive from the state dict afterwards, but it does mean the
surviving `graph` branch is the one built during `pcqm4m_g25`.)

`TaskHeads.forward` makes the sharing explicit — the branch is evaluated **once per level** and
indexed by every head at that level:

```python
features = {lvl: self.graph_output_nn[lvl](g) for lvl in self.task_levels}   # 2 calls
task_head_outputs[task_name] = head.forward(features[task_level])            # 5 calls
```

So the four graph heads do not each get their own copy of the 1024-d vector; they read the same
tensor.

### 10.5 Branch `graph` — pool, then widen

Inside `GraphOutputNN.forward`, under `if self.task_level == "graph"`:

1. `g["graph_feat"] = self._pool_layer_forward(g, g["feat"])` — the `pooling: [max]` from the
   config. `[194, 512] → [8, 512]`. **Pooling only, no linear** — the method's docstring says
   "followed by the linear output layer", which is stale; the body just applies the pool.
2. `self.graph_output_nn.forward(h)` — `FCLayer[512 → 512 → 1024]`, ReLU then no activation,
   `last_normalization: layer_norm`. `[8, 512] → [8, 1024]`.

Measured: **791,040** params.

### 10.6 Branch `node` — no pooling at all

`task_level == "node"` falls through every special case in `__init__`, so `level_in_dim = in_dim
= 512`, and `forward` reads `g["feat"]` straight from the map `{"node": "feat"}`.

- `FCLayer[512 → 256 → 64]`, applied independently to every atom. `[194, 512] → [194, 64]`.

Measured: **148,288** params. `791,040 + 148,288 = 939,328` for `graph_output_nn` as a whole.

The two branches are siblings that never rejoin.

### 10.7 The five heads

Each head is a `FeedForwardNN` with `depth=2, hidden_dims=128`. Four consume the *same*
`[8, 1024]` tensor; one consumes `[194, 64]`:

```
                        [194, 512]   ← gnn:15 output, per ATOM
                       /           \
          max-pool    /             \    (no pooling)
                     ▼               ▼
              [8, 512]            [194, 512]
      ← THE FINGERPRINT                │
                     │  512→512→1024   │  512→256→64
                     ▼                 ▼
              [8, 1024]             [194, 64]
             /    │    │    \           │
            ▼     ▼    ▼     ▼          ▼
         vcap   mcf7  pcba  g25        n4
      [8,2934][8,2934][8,1328][8,25]  [194,4]
```

Measured by running the real `net.forward` on the batch:

```
l1000_vcap  (8, 2934)      pcqm4m_g25  (8, 25)
l1000_mcf7  (8, 2934)      pcqm4m_n4   (194, 4)
pcba_1328   (8, 1328)
```

Head parameter counts, measured: 509,942 + 509,942 + 302,768 + 134,681 + 9,092 =
**1,466,425**. `pcqm4m_n4` is 55× smaller than the others purely because its input is 64-d
rather than 1024-d — the node branch's narrowness, not the task's.

Note the two `l1000` heads set both `activation: none` and `last_activation: none`, which looks
like a linear map and is not. `normalization` resolves through the `&normalization` YAML anchor
to `layer_norm`, so measured:

```
layer0: Linear(1024→128) → LayerNorm(128) → Dropout(0.02) → (no activation)
layer1: Linear(128→2934) → (no norm, no dropout, no activation)
```

A **rank-≤128 factorization with a LayerNorm in the middle**. The only nonlinearity is
LayerNorm's per-row rescaling — everything else on the path is affine — which is what makes the
rank-≤128 framing meaningful rather than approximate. The other three heads add ReLU after
layer 0.

### 10.8 Where the 512-d embedding sits, exactly

The fingerprint is not merely "somewhere upstream of the heads." Running the graph branch's
pooling by hand and comparing against `trunk.forward`'s output on the same batch:

```
graph branch: after max-pool -> (8, 512)
   identical to trunk fingerprint?  True     max|delta| = 0.0
```

**MiniMol's fingerprint *is* `g["graph_feat"]` — the graph branch's input tensor**, bit-for-bit
(`torch.equal` is `True`).

But MiniMol does not read it from there. `Fingerprinter(predictor, 'gnn:15')`
(`minimol/model.py:59`) taps the *pre*-pool `[194, 512]`, and `minimol/model.py:82` then calls
`global_max_pool` itself, outside the network — re-doing by hand what the graph branch performs
one line later inside it. The consequences:

- The four graph heads see the fingerprint, put through a 512→512→1024 MLP.
- The node head **never sees it**. Pooling never happens anywhere on its path.

There is also a road not taken. `GraphOutputNN.forward` carries a `concat_last_layers` branch
whose comment reads *"Useful for generating fingerprints"* — graphium's own intended fingerprint
mechanism, which returns the post-NN layers concatenated. MiniMol declines it: measured
`concat_last_layers` is `None` on both branches, and the tap is placed upstream of
`graph_output_nn` entirely.

The logic is standard transfer learning: train a big network on many tasks so its internal
representation has to encode broadly useful chemistry, then throw away the task-specific top and
keep the general middle. What is worth being precise about is *how far down* "the top" starts —
here it starts at `graph_output_nn`, not at the heads.

This is also why this repo exists. `Fingerprinter.get_fingerprints_for_batch` wraps the forward
pass in `torch.inference_mode()`, which permanently marks its outputs as non-differentiable — so
MiniMol's own API can produce embeddings but cannot be fine-tuned through. `src/trunk.py`
reimplements the chain `encoder_manager → pre_nn → pre_nn_edges → gnn → max_pool` without that
wrapper, and reproduces the frozen embeddings to max|Δ| = 0.000e+00 over 64×512.

### 10.9 `node: pooling: [max]` is dead config — and a different kind of dead than §8.3's

Two config lines in this model do not do what they appear to, and they fail differently. Worth
keeping distinct:

- **`out_dim_edges: 64`** (§8.3) — *honoured at construction*. The 64 appears in
  `full_dims_edges` and shapes real modules. What never runs is the edge *update path*.
  Built, then unused.
- **`graph_output_nn.node.pooling: [max]`** — *never read at all*. `_parse_pooling_layer` is
  called only inside the `task_level == "graph"` branch of `__init__`, and the kwargs filter
  drops `pooling` before the MLP is constructed. Measured:
  `hasattr(graph_output_nn['node'], 'global_pool_layer')` is `False` on the node branch and
  `True` on the graph branch. No module is built.

Reading the config alone, you would expect the node branch to pool. It cannot — a node-level task
needs one output per atom, so pooling would destroy exactly the axis it predicts along.

### 10.10 How this maps onto this repo's own head

`head.DualHead` occupies these same two slots. MiniMol's post-GNN stack is
`[graph branch: 512 → 512 → 1024]` followed by `[head: 1024 → 128 → out]`; this repo's is
`512 → 1024 → {Linear(1024→1) cls, Linear(1024→1) reg}` — **as of 2026-08-25**. Same two stages,
with the second reduced to bare linear maps.

Until that date it was `512 → 1024 → 1024 → 32 → {Linear(32→1) cls, Linear(32→1) reg}`, a 32-d
bottleneck spliced in at the join, which is the shape most of this repo's measurements were taken
under. The defaults moved (`--n-layers 0 --embed-dim 1024`); the parameterisation did not, so
`--n-layers 2 --hidden-dim 1024 --embed-dim 32` restores the old shape exactly.

The difference that mattered for the collapse question in `CLAUDE.md`: MiniMol widens to 1024 and
lets each of its five heads compress **privately** to 128, so no single tensor carries all tasks
at low rank. This repo compressed to 32 **shared**, deliberately — the 32-d vector was the
deliverable — which is why the rank of that one tensor is the number the project tracks. **The
new shape removes the compression, not the question**: the exported tensor is still shared and
the supervised signal reaching it is still near rank-1, so `val/emb_effective_rank` at width 1024
is an open measurement, not a solved problem.

---

## 11. Details that will surprise you

**Laplacian sign flipping is training-time augmentation.** `laplace_pos_encoder.py:184–189`
multiplies each eigenvector by a random ±1 **only when `self.training` is `True`**. In `eval()`
the encodings are deterministic. This is how the sign ambiguity of §5 is handled: rather than
building a sign-invariant architecture (which is what SignNet does), MiniMol shows the network
both signs during training and lets it learn to not care. Relevant here because
`MiniMolTrunk.__init__` calls `self.train()`, so **this randomness is active during fine-tuning**
and identical inputs give slightly different embeddings between steps.

**Dropout is 0.02 everywhere** — pre_nn, pre_nn_edges, the GNN — and 0.1 in the PE encoders. Also
a `train()`/`eval()` difference.

**`ε = 0.0` and frozen**, so GINE's `(1 + ε)·x_i` self-term is just `x_i`.

**`FCLayer` order is Linear → LayerNorm → Dropout → Activation**, dropout before the nonlinearity.

**PE columns come first**: `feat[:, :32]` is positional, `feat[:, 32:117]` is chemical.

**Two dead parameters.** `encoder_manager.pe_encoders.rw_pos.first_normalization.{weight,bias}`
(32 params) is a duplicate LayerNorm graphium constructs twice and only ever runs the inner copy
of. Its `.grad` comes back `None`, so no optimizer can move it. `7,919,944 = 7,919,912 reachable
+ 32 inert`. Detail in `CLAUDE.md`, "The rw_pos dead norm".

**µP scaling is load-bearing.** `mup_load_or_save: load` with a shipped `base_shape.yaml`, and
the readout layers are `MuReadoutGraphium` rather than `nn.Linear`. Maximal-update parametrization
rescales initialization and forward multipliers as a function of width so that hyperparameters
transfer from a narrow proxy model to the full one. It changes the forward computation, so it is
not cosmetic — `trunk.py` reproduces MiniMol's exact construction ordering because whether
`load_architecture` sees `mup_base_path` at all depends on object aliasing.

**`register_forward_hook` silently does nothing on some modules.** graphium calls `.forward()`
directly rather than `__call__()` in at least three places — `trunk.py:373` (`pre_nn`),
`trunk.py:383` (`pre_nn_edges`), `global_architectures.py:1225` (virtual node layers). Hooks
registered there never fire, and never raise. To read those tensors you must wrap the `.forward`
attribute or thread the values out of the forward pass.

---

## 12. Every shape, measured

Batch: 8 molecules, 194 atoms, 418 directed edges.

| stage | tensor | shape | level |
|---|---|---|---|
| featurizer | `feat` | `[194, 85]` | atom |
| featurizer | `edge_feat` | `[418, 13]` | bond |
| featurizer | `laplacian_eigvec` / `eigval` | `[194, 8]` each | atom |
| featurizer | `rw_return_probs` | `[194, 16]` | atom |
| `la_pos` | encoder out | `[194, 32]` | atom |
| `rw_pos` | encoder out | `[194, 32]` | atom |
| `encoder_manager` | `feat` (32 PE ⊕ 85 chem) | `[194, 117]` | atom |
| `pre_nn` | `feat` | `[194, 128]` | atom |
| `pre_nn_edges` | `edge_feat` | `[418, 168]` | bond — **constant for all 16 layers** |
| `gnn.layers[0]` | `feat`, post-conv (#1 of §8.1b) | `[194, 336]` | atom |
| `gnn.layers[1..14]` | `feat`, post-conv (#1 of §8.1b) | `[194, 336]` | atom |
| `gnn.layers[15]` | `feat`, post-conv — no residual, no VN follows | `[194, 512]` | atom |
| `_readout_cache[0..14]` | `feat`, post-VN (#3 of §8.1b) | `[194, 336]` | atom |
| `_readout_cache[15]` | same tensor as `g["feat"]` — the fingerprint source | `[194, 512]` | atom |
| `virtual_node_layers[0..14]` | `vn_feat` | `[8, 336]` each | **molecule** |
| `global_max_pool` | embedding | `[8, 512]` | molecule |

Everything above is the trunk. Beyond the tap point, inside `task_heads` (§10) — not registered
by `trunk.py`, and measured by running the real `FullGraphMultiTaskNetwork.forward`:

| stage | tensor | shape | level |
|---|---|---|---|
| `graph_output_nn['graph']` pool | `graph_feat` = `global_max_pool` of `_readout_cache[15]`, i.e. **the same tensor the trunk computes outside the network** | `[8, 512]` | molecule |
| `graph_output_nn['graph']` MLP | `FCLayer[512→512→1024]` | `[8, 1024]` | molecule |
| `graph_output_nn['node']` MLP | `FCLayer[512→256→64]`, **no pooling** | `[194, 64]` | atom |
| head `l1000_vcap` / `l1000_mcf7` | logits | `[8, 2934]` each | molecule |
| head `pcba_1328` | logits | `[8, 1328]` | molecule |
| head `pcqm4m_g25` | regression | `[8, 25]` | molecule |
| head `pcqm4m_n4` | regression | `[194, 4]` | atom |

Reproduce with `.venv/bin/python`, wrapping `.forward` on the modules of interest (not hooks —
see §11).

---

## 13. One-paragraph summary

A SMILES string is parsed into a graph of atoms and bonds. Each atom gets 85 hand-designed
chemical features and each bond 13; two structural descriptors — Laplacian eigenvectors and
random-walk return probabilities — are computed to tell each atom where it sits, encoded to 32-d
by a DeepSet and an MLP, summed, and concatenated in front of the chemical features. Two MLPs
lift atoms to 128-d and bonds to 168-d. Sixteen GINE layers then run message passing: each atom
sums its neighbours' vectors plus the connecting bond's, feeds the result through a 2-layer MLP,
and adds a residual — widening 128 → 336 → 512. Between layers, fifteen virtual-node modules pool
the whole molecule, update a running global state, and broadcast it back to every atom, giving
long-range communication that 16 local hops could not provide. A final elementwise max over
atoms yields 512 numbers per molecule. That vector is not the model's prediction — it is a tap
from the middle of a network pretrained on five prediction tasks, kept precisely because it
encodes general chemistry rather than any one task. Precisely: it is the tensor the pretraining
network's `graph` output branch consumes, and everything downstream of it — a 512→1024 widening
shared by four molecule-level heads, and a separate unpooled 512→64 per-atom branch feeding the
fifth — is discarded here (§10).
