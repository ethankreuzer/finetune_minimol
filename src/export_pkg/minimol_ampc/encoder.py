"""Load the shipped checkpoint and turn SMILES into vectors.

This is the whole public surface of the package:

    from minimol_ampc import MiniMolAmpcEncoder

    enc = MiniMolAmpcEncoder.load("model/final.pt", device="cuda")
    emb = enc.encode(["CCO", "c1ccccc1"])            # (2, 512) float32 -- THE EXPORT
    out = enc.encode_all(smiles, on_invalid="skip")  # + z, pProp, logit, and a kept mask

`encode` returns `pooled512`: MiniMol's own graph-level output, a `global_max_pool` over the
final (16th) GNN layer's atom features. That is the recommended export for a kernel-based
consumer -- see README.md, "Which vector to use", for the measurement behind that.

Four things this module does on your behalf, each of which is a way the model fails SILENTLY
if you build the forward pass yourself. They are enforced here rather than documented and
hoped for:

1. **The model is put in `eval()` mode at load, and `encode` asserts it.** In train mode
   MiniMol applies random Laplacian sign-flip augmentation to the positional encodings --
   measured at max|delta| ~ 2.8 on internal state between two otherwise identical passes. A
   training-mode encoder returns a *different vector every call* for the same molecule, with
   no error.

2. **The state dict is loaded with `strict=True` and the result asserted empty.** minimol's
   own loader calls `load_state_dict(..., strict=False)` and discards what it returns, so a
   key mismatch yields a partly RANDOMLY INITIALISED trunk with no error and no warning. That
   presents as disappointing accuracy, not as a crash.

3. **Every batch is collated fresh.** The positional encoders concatenate into `feat` during
   `forward`, so a collated batch's feature width GROWS as it is used and a batch is
   single-use. Reusing one is a shape error at best and wrong numbers at worst.

4. **pProp is denormalized through the checkpoint's own statistics.** The regression head was
   trained against a z-scored target; the mean and std live inside `final.pt` and are specific
   to the data that checkpoint was fit on. They are read from the file, never hardcoded.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# The vendored modules sit beside this file and import each other flatly (`model.py` does
# `from head import MLPHead`), exactly as they do in the source repository. Keeping them
# byte-identical is what lets you diff them against that repository and see nothing; the cost
# is this one path insertion instead of relative imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from head import DualHead                       # noqa: E402
from model import MiniMolRegressor              # noqa: E402
from normalization import denormalize_pprop     # noqa: E402
from featurize import load_cache                # noqa: E402
from trunk import MiniMolTrunk, cast_features   # noqa: E402

POOLED_DIM = 512


@dataclass
class EncodeResult:
    """Everything one forward pass produces, plus which inputs survived featurization.

    `kept` is a boolean mask over the SMILES you passed in. Under `on_invalid="raise"` it is
    all-True and the arrays line up with your input one-for-one. Under `on_invalid="skip"`
    the arrays hold only the kept molecules, and `kept` is the ONLY thing that ties a row
    back to the molecule it came from -- `pooled512[i]` is `smiles[kept][i]`, not
    `smiles[i]`. That is why the mask is returned rather than the failures being dropped
    quietly.
    """

    pooled512: np.ndarray      # (n_kept, 512) -- MiniMol's pooled output; the export
    z: np.ndarray              # (n_kept, 1024) -- the head's shared layer; see README
    pprop: np.ndarray          # (n_kept,) -- predicted pProp, RAW scale, higher = more potent
    logit: np.ndarray          # (n_kept,) -- raw logit for pProp >= 3.5; apply sigmoid yourself
    kept: np.ndarray           # (n_input,) bool
    invalid: list              # the SMILES that failed to featurize, in input order

    def __len__(self):
        return len(self.pooled512)


class MiniMolAmpcEncoder:
    """A frozen `SMILES -> R^512` encoder, fine-tuned on AmpC docking scores.

    Construct with `MiniMolAmpcEncoder.load(...)`, not with `__init__` -- the constructor
    takes an already-built model and exists so the loading logic has one home.
    """

    def __init__(self, model, norm_stats, config, pprop_edge=3.5):
        self.model = model
        self.norm_stats = norm_stats
        self.config = config
        self.pprop_edge = pprop_edge

    # -- loading -------------------------------------------------------------------

    @classmethod
    def load(cls, checkpoint, device=None):
        """`final.pt` -> a ready, frozen encoder in eval mode.

        The head's geometry is read back out of the config saved inside the checkpoint rather
        than assumed. That matters because the shape has moved over this project's life
        (`n_layers`, `embed_dim` and both task-head shapes are all settable), and a
        default-shaped head would fail to load loudly -- but only because of `strict=True`
        below. With minimol's own `strict=False` it would load *wrongly*.
        """
        checkpoint = Path(checkpoint)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")

        cfg = ckpt["config"]
        model = MiniMolRegressor(
            MiniMolTrunk(accelerator="gpu" if str(device).startswith("cuda") else "cpu"),
            DualHead(hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
                     embed_dim=cfg["embed_dim"],
                     cls_hidden_dim=cfg["cls_hidden_dim"], cls_n_layers=cfg["cls_n_layers"],
                     reg_hidden_dim=cfg["reg_hidden_dim"], reg_n_layers=cfg["reg_n_layers"],
                     dropout=cfg["dropout"], norm=cfg["head_norm"],
                     bottleneck_norm=cfg["bottleneck_norm"]),
        )
        incompatible = model.load_state_dict(ckpt["model_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{checkpoint} did not load cleanly: {len(incompatible.missing_keys)} missing "
                f"{incompatible.missing_keys[:3]}, {len(incompatible.unexpected_keys)} "
                f"unexpected {incompatible.unexpected_keys[:3]}. Do NOT use this model -- the "
                "unfilled tensors are at random initialisation.")

        # Not optional, and not merely hygiene -- see this module's docstring, point 1.
        model.to(device).eval()
        return cls(model, ckpt["norm_stats"], cfg, ckpt.get("pprop_edge", 3.5))

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def embed_dim(self):
        """Width of `z`. `pooled512` is always 512."""
        return self.model.head.embed_dim

    # -- featurization -------------------------------------------------------------

    def featurize(self, smiles, on_invalid="raise"):
        """`list[str] -> (list[Data], kept mask, invalid SMILES)`.

        SMILES -> graph is a pure function of the string: it depends on no weight, no fold and
        no epoch, so for anything you will encode more than once, do this ONCE and keep the
        result. `minimol_ampc.featurize` writes such a cache to disk. This method is the
        one-shot path.

        graphium returns the error *string* in place of a graph for a molecule it cannot
        featurize, rather than raising, so a bad SMILES becomes a silent hole in the batch
        that shifts every row after it. Both branches here exist to make that impossible:
        `raise` refuses the whole call, `skip` reports exactly which inputs were dropped.
        """
        if on_invalid not in ("raise", "skip"):
            raise ValueError(f"on_invalid must be 'raise' or 'skip', got {on_invalid!r}")
        if isinstance(smiles, str):
            smiles = [smiles]
        smiles = list(smiles)

        import os
        from contextlib import redirect_stderr, redirect_stdout
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            feats, _ = self.model.trunk.datamodule._featurize_molecules(smiles)

        kept = np.array([not isinstance(f, str) for f in feats], dtype=bool)
        invalid = [s for s, ok in zip(smiles, kept) if not ok]
        if invalid and on_invalid == "raise":
            raise ValueError(
                f"{len(invalid)} of {len(smiles)} SMILES could not be featurized, e.g. "
                f"{invalid[:3]}. Pass on_invalid='skip' to drop them and receive a mask -- "
                "which is what an active-learning loop over generated molecules wants.")

        graphs = cast_features([f for f, ok in zip(feats, kept) if ok])
        return graphs, kept, invalid

    # -- encoding ------------------------------------------------------------------

    def encode(self, smiles, batch_size=256, **kwargs):
        """`list[str] -> (N, 512) float32`. The export, and nothing else.

        Rows correspond one-for-one to the SMILES you passed, because the default
        `on_invalid="raise"` refuses anything else. If you need the predictions, `z`, or
        tolerance for unfeaturizable molecules, call `encode_all`.
        """
        return self.encode_all(smiles, batch_size=batch_size, **kwargs).pooled512

    def encode_all(self, smiles, batch_size=256, on_invalid="raise", grad=False):
        """`list[str] -> EncodeResult`, featurizing as it goes.

        `grad=True` keeps the autograd graph so a downstream model can backpropagate into the
        encoder. It costs memory proportional to the batch and is off by default, because the
        intended use is a frozen featurizer.
        """
        graphs, kept, invalid = self.featurize(smiles, on_invalid=on_invalid)
        result = self.encode_graphs(graphs, batch_size=batch_size, grad=grad)
        result.kept, result.invalid = kept, invalid
        return result

    def encode_graphs(self, graphs, batch_size=256, grad=False):
        """Already-featurized graphs -> `EncodeResult`. The cached path skips SMILES parsing.

        `graphs` is anything indexable yielding PyG `Data` -- a list, or the dataset returned
        by `minimol_ampc.featurize.load_cache`.
        """
        if self.model.training:
            raise RuntimeError(
                "the model is in train() mode, where MiniMol applies random Laplacian "
                "sign-flip augmentation to the positional encodings -- the same molecule "
                "would encode to a different vector on every call. Call .eval() first.")

        trunk, head = self.model.trunk, self.model.head
        pooled_parts, z_parts, pred_parts, logit_parts = [], [], [], []

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for lo in range(0, len(graphs), batch_size):
                chunk = [graphs[i] for i in range(lo, min(lo + batch_size, len(graphs)))]
                # Collated FRESH every batch. The positional encoders concatenate into
                # `feat` during forward, so its width grows and a batch cannot be reused.
                batch = trunk.collate(chunk, to_device=True)
                pooled = trunk(batch)
                logit, pred, z = head.forward_with_embedding(pooled)
                detach = (lambda t: t) if grad else (lambda t: t.detach())
                pooled_parts.append(detach(pooled).float().cpu().numpy())
                z_parts.append(detach(z).float().cpu().numpy())
                pred_parts.append(detach(pred).float().cpu().numpy())
                logit_parts.append(detach(logit).float().cpu().numpy())

        cat = lambda parts, dim: (np.concatenate(parts) if parts
                                  else np.zeros((0,) + dim, dtype=np.float32))
        pred_norm = cat(pred_parts, ())
        n = len(graphs)
        return EncodeResult(
            pooled512=cat(pooled_parts, (POOLED_DIM,)).astype(np.float32),
            z=cat(z_parts, (self.embed_dim,)).astype(np.float32),
            # Back to the raw pProp scale using THIS checkpoint's training statistics.
            pprop=np.asarray(denormalize_pprop(pred_norm.astype(np.float64)
                                               if pred_norm.size else pred_norm,
                                               self.norm_stats), dtype=np.float32),
            logit=cat(logit_parts, ()).astype(np.float32),
            kept=np.ones(n, dtype=bool),
            invalid=[],
        )

    def encode_cached(self, cache, batch_size=256, grad=False):
        """Encode a cache built by `minimol_ampc.featurize.build_cache`.

        `cache` is either the directory or the dataset `load_cache` returns. This is the path
        an active-learning loop should take after round 1: featurization is CPU-bound and
        unchanging, so paying it once and reloading in ~1 s per round is the difference
        between a loop that scales and one that does not.
        """
        if isinstance(cache, (str, Path)):
            cache, _ = load_cache(cache)
        return self.encode_graphs(cache, batch_size=batch_size, grad=grad)
