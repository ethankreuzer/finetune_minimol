"""Capture MiniMol's virtual-node state at chosen depths.

MiniMol runs 16 GNN layers with a **virtual node** updated after each of the first 15. The
virtual node is a graph-level running state: it pools every atom into one vector per molecule,
updates residually, and broadcasts back to every atom. `Minimol_architecture_overview.md` §9
measures it growing ~17x in norm across the stack, and it holds 3,407,040 params -- 43% of the
GNN. It has never been probed, which is what this module exists to make possible.

Getting at it is not a one-liner, for two reasons that both fail *silently*.

**`vn_feat` is a bare Python local.** `FeedForwardGraph.forward`
(`global_architectures.py:1263-1291`) initialises it to the scalar `0.0`, threads it through the
loop, and writes only `g["feat"]` and `g["edge_feat"]` at the end. The virtual node state is
never stored anywhere; after the forward pass it is gone.

**`register_forward_hook` on the virtual node layer does nothing.** `_virtual_node_forward`
(`global_architectures.py:1225`) calls `self.virtual_node_layers[step_idx].forward(...)` --
`.forward` directly, not `__call__` -- so the hook machinery is bypassed entirely. It does not
raise. It simply never fires.

So the route taken here is to **shadow the `.forward` attribute** on the instance. Attribute
lookup at line 1225 finds the instance attribute before the class method, the wrapper calls
through to the original, and records the `vn_feat` it returned. That value is post-residual
(`pooling_pyg.py:338`) and shaped `[n_molecules, 336]` -- exactly the state handed to the next
virtual node.

There is a second route: a forward hook on `virtual_node_layers[i].node_projection`, which *is*
invoked via `__call__` (`pooling_pyg.py:343`) and whose `input[0]` is `vn_feat[g.batch]`. It is
not used here because recovering `[M, 336]` from it requires de-gathering by first-occurrence
indices of `batch`, which would bake a PyG batch-contiguity assumption into the very measurement
this module exists to take. `verify_vn_taps.py` uses it as an independent cross-check instead,
and asserts the two routes agree exactly.
"""

from contextlib import contextmanager

import torch

# The five taps, as INTERNAL indices into `gnn.virtual_node_layers`, which holds 15 entries
# (0..14) -- one after every GNN layer except the last. Ethan named the taps 1-indexed
# ("layers 3, 6, 9, 12, 15"), and 15 is the only reading under which the last name refers to a
# virtual node that exists at all.
VN_TAPS = (2, 5, 8, 11, 14)

# The name each tap carries into `vn_embeddings.npz`. 1-indexed, matching how the taps were
# specified, so the artifact reads the way the experiment was described.
TAP_NAMES = {2: "vn03", 5: "vn06", 8: "vn09", 11: "vn12", 14: "vn15"}

# The trunk's own 512-d output -- `global_max_pool` over the final GNN layer's atom features.
# Not a virtual node, but the sixth embedding under study and stored alongside the five.
POOLED_KEY = "pooled512"

VN_DIM = 336        # width of every virtual node in MiniMol v1
POOLED_DIM = 512

# The six embeddings under study, in stack order, and how they are labelled for a reader.
# Kept here rather than in the probe or the plotting code so one edit renames them everywhere.
EMBEDDING_ORDER = ("vn03", "vn06", "vn09", "vn12", "vn15", POOLED_KEY)
DISPLAY_NAMES = {
    "vn03": "VN 3", "vn06": "VN 6", "vn09": "VN 9", "vn12": "VN 12", "vn15": "VN 15",
    POOLED_KEY: "Pooled embedding",
}


def vn_layers(module):
    """The `virtual_node_layers` ModuleList, from a trunk or from a whole regressor.

    Accepts either so callers do not have to remember which object they are holding.
    """
    gnn = getattr(getattr(module, "trunk", module), "gnn", None)
    if gnn is None:
        raise TypeError(f"{type(module).__name__} has no .gnn (nor .trunk.gnn); expected a "
                        "MiniMolTrunk or a MiniMolRegressor")
    return gnn.virtual_node_layers


class VNCapture:
    """The tensors recorded by the active taps, keyed by internal VN index.

    One forward pass fills every tap exactly once. `calls` counts invocations per index so a
    tap that fired twice (or not at all) is detectable rather than silently overwriting.
    """

    def __init__(self, indices):
        self.indices = tuple(indices)
        self.tensors = {}
        self.calls = {i: 0 for i in self.indices}

    def _record(self, idx, vn_feat):
        # detach, not clone: `vn_feat` is a fresh tensor out of `vn_feat + vn_h_temp` and is
        # never written into in place. detach is what matters -- without it every captured
        # tensor pins the whole forward graph, and this module is used over 66k molecules.
        self.tensors[idx] = vn_feat.detach()
        self.calls[idx] += 1

    def clear(self):
        """Drop the captured tensors and the call counts. Call between batches."""
        self.tensors.clear()
        self.calls = {i: 0 for i in self.indices}

    def ordered(self):
        """The captures in tap order, as `(name, tensor)` pairs."""
        return [(TAP_NAMES[i], self.tensors[i]) for i in self.indices]

    def assert_fired_once(self):
        bad = {i: n for i, n in self.calls.items() if n != 1}
        if bad:
            raise RuntimeError(
                f"virtual node taps fired {bad} times, expected 1 each. A count of 0 means "
                "the .forward shadow was bypassed; >1 means the capture was not cleared "
                "between batches.")


@contextmanager
def capture_vn(module, indices=VN_TAPS):
    """Shadow `.forward` on the chosen virtual node layers and record their `vn_feat`.

    Yields a `VNCapture`. The shadowing is undone on exit, including on exception, so a model
    is never left instrumented.

    Usage:

        with capture_vn(model) as cap:
            for batch in loader:
                cap.clear()
                emb = model.trunk(batch)
                cap.assert_fired_once()
                ...                       # cap.tensors[14] is [B, 336]
    """
    layers = vn_layers(module)
    indices = tuple(indices)
    for i in indices:
        if not 0 <= i < len(layers):
            raise IndexError(f"virtual node index {i} out of range; the stack holds "
                             f"{len(layers)} (0..{len(layers) - 1})")

    cap = VNCapture(indices)
    shadowed = []
    try:
        for i in indices:
            layer = layers[i]
            if "forward" in vars(layer):
                raise RuntimeError(f"virtual_node_layers[{i}].forward is already shadowed; "
                                   "nested capture_vn() would double-count")
            original = layer.forward          # bound method, captured before shadowing

            def wrapper(*args, _idx=i, _orig=original, **kwargs):
                feat, vn_feat, edge_feat = _orig(*args, **kwargs)
                cap._record(_idx, vn_feat)
                return feat, vn_feat, edge_feat

            # object.__setattr__ rather than plain assignment: nn.Module.__setattr__ routes
            # through its parameter/buffer/module bookkeeping, and a plain function has no
            # business passing through any of it.
            object.__setattr__(layer, "forward", wrapper)
            shadowed.append(layer)
        yield cap
    finally:
        for layer in shadowed:
            # Popping the instance attribute re-exposes the class method. Assigning the bound
            # method back would leave a permanent instance-level reference to the module,
            # which is the sort of thing that survives a `deepcopy` and confuses everyone.
            vars(layer).pop("forward", None)


def degather(broadcast, batch_index):
    """`vn_feat[g.batch]` -> `vn_feat`, by taking the first row of each molecule's block.

    Only used by the cross-check in `verify_vn_taps.py`. It is the step the primary route
    avoids: it assumes each molecule's atoms occupy one contiguous run of `batch_index`, which
    is true of `Batch.from_data_list` but is an assumption all the same.
    """
    n = batch_index.numel()
    starts = torch.cat([
        batch_index.new_zeros(1),
        (batch_index[1:] != batch_index[:-1]).nonzero(as_tuple=False).flatten() + 1,
    ])
    if starts.numel() != int(batch_index[-1].item()) + 1:
        raise RuntimeError(f"batch_index is not contiguous by molecule: found {starts.numel()} "
                           f"runs over {n} atoms for {int(batch_index[-1].item()) + 1} graphs")
    return broadcast[starts]
