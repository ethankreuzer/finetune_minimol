"""A MiniMol trunk that gradients can actually reach.

MiniMol's own `Minimol.__call__` cannot be fine-tuned through. It routes the forward pass
via `graphium.finetuning.fingerprinting.Fingerprinter`, whose `get_fingerprints_for_batch`
wraps the network in `torch.inference_mode()` and then calls `.cpu()` on the cached
readout. `inference_mode` does not merely skip graph recording -- it stamps its outputs as
inference tensors that autograd refuses to accept ever after (NOTES.md §§2.5, 4). No
amount of care downstream recovers from that.

This module rebuilds the same computation without that wrapper (NOTES §6.3). It is a
reimplementation, not a patch: no graphium or minimol source is modified.

What was read from the pinned source to justify the details here (graphium 2.4.7,
minimol 1.3.4):

  * `FullGraphMultiTaskNetwork.forward` is a straight chain --
    `encoder_manager -> pre_nn -> pre_nn_edges -> gnn -> task_heads`. `task_heads` runs
    strictly *after* the layer we read, so stopping before it cannot change the embedding.
    That is why `task_heads` is not registered as a child module below: skipping it makes
    the five pretraining heads and `graph_output_nn` structurally unreachable rather than
    merely unused, so they cannot appear in an optimizer by accident.
  * The GNN caches readouts with a bare `self._readout_cache[ii] = feat` -- no `.detach()`,
    no `.clone()`. The cached tensor is the live one, so it carries `grad_fn` as long as no
    `inference_mode` is in scope. The block was never in the cache; it was in Fingerprinter.
  * The config declares `gnn.depth: 16`, so layer index 15 is the *final* GNN layer and
    `_readout_cache[15]` is also what the loop assigns to `g["feat"]`. `forward` therefore
    reads `g["feat"]` directly, and `verify_trunk.py` checks the two are the same tensor.
  * MiniMol's `config.yaml` has no hydra `defaults:` list and only plain OmegaConf
    interpolations, so `OmegaConf.load` replaces `hydra.initialize`/`hydra.compose`
    exactly. Avoiding hydra removes the once-per-process singleton footgun of NOTES §9,
    which otherwise makes this class impossible to construct twice (sweeps, test suites).

The private APIs relied on -- `_convert_features_dtype`, `_enable_readout_cache`,
`_module_map`, `_readout_cache` -- carry no stability promise. That is accepted, pinned to
graphium 2.4.7 (NOTES §6.3), and is why `verify_trunk.py` exists.

Usage:
    from trunk import MiniMolTrunk

    trunk = MiniMolTrunk()
    batch = trunk.featurize(["CCO", "c1ccccc1"])
    emb   = trunk(batch)              # [2, 512], with grad_fn
    emb.sum().backward()              # gradients reach all ~10M trunk parameters
"""

import os
from contextlib import ExitStack, redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool

# The layer whose readout is MiniMol's embedding, from `Fingerprinter(predictor, 'gnn:15')`
# in minimol/model.py. With `gnn.depth: 16` this is the last layer, 0-indexed.
FINGERPRINT_MODULE = "gnn"
FINGERPRINT_LAYER = 15
EMBED_DIM = 512


def minimol_paths():
    """Absolute paths to the three files MiniMol ships inside its wheel.

    Resolved through the installed package rather than hardcoded, so this follows a
    reinstall or a different environment. `pkg_resources` is deprecated but is what
    minimol itself uses; `importlib.resources` is preferred here with a fallback so the
    two agree on the same files.
    """
    try:
        from importlib.resources import files
        base = files("minimol.ckpts.minimol_v1")
        return {
            "state_dict": str(base / "state_dict.pth"),
            "config": str(base / "config.yaml"),
            "base_shape": str(base / "base_shape.yaml"),
        }
    except Exception:
        import pkg_resources
        res = lambda n: pkg_resources.resource_filename("minimol.ckpts.minimol_v1", n)
        return {"state_dict": res("state_dict.pth"), "config": res("config.yaml"),
                "base_shape": res("base_shape.yaml")}


def load_minimol_config(config_path=None, accelerator=None):
    """MiniMol's config as a plain dict, without going through hydra.

    `Minimol.load_config` calls `hydra.initialize('ckpts/minimol_v1/')`, whose path is
    relative to *minimol's own module file*; called from this repo it would look for
    `src/ckpts/minimol_v1/` and fail. `hydra.initialize` is also a per-process singleton.
    Since the config has no `defaults:` list, hydra contributes nothing that
    `OmegaConf.load(...)` + `resolve=True` does not, and `verify_trunk.py` checks that the
    two routes produce equal dicts rather than taking this on trust.

    `mup_base_path` is deliberately *not* set here. MiniMol writes it onto the config only
    after `load_accelerator` and then hands the pre-accelerator object to
    `load_architecture`, so whether the architecture sees it at all depends on aliasing.
    mup base shapes affect parameter initialisation and forward scaling, so that ordering
    is reproduced literally in `MiniMolTrunk.__init__` rather than tidied up here.
    """
    cfg = OmegaConf.to_container(
        OmegaConf.load(config_path or minimol_paths()["config"]), resolve=True
    )
    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    cfg["accelerator"]["type"] = accelerator
    return cfg


def cast_features(features):
    """Replicate `Minimol.to_fp32`: half -> float32, int32 -> int64, in place.

    Easy to leave out, and it does not fail loudly if you do. `_convert_features_dtype`
    (which `forward` calls, as Fingerprinter does) only touches *floating point* tensors,
    so the int32 -> int64 half of this has no other home. PyG indexing tensors such as
    `edge_index` are expected to be int64, and an int32 one either raises deep inside a
    scatter or, worse, silently changes behaviour. Featurization emits int32 for some
    keys, so this is not hypothetical.

    It also matters for comparability: the reference embeddings in
    `dump_reference_embeddings.py` are produced through `Minimol.to_fp32`, so skipping it
    here would make `verify_trunk.py`'s featurization check disagree for a reason that has
    nothing to do with the trunk.
    """
    for feature in features:
        if isinstance(feature, str):
            continue
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.half:
                    feature[key] = value.float()
                elif value.dtype == torch.int32:
                    feature[key] = value.long()
    return features


class StateDictMismatch(RuntimeError):
    """Raised when the pretrained checkpoint does not fully populate the trunk.

    `minimol/model.py` calls `load_state_dict(..., strict=False)` and discards the result.
    That combination turns a name mismatch into a *partly randomly initialised trunk with
    no error and no warning* -- which presents as disappointing accuracy, not as a bug, and
    can cost weeks (NOTES §9). Hence a hard failure here by default.
    """


class MiniMolTrunk(nn.Module):
    """SMILES batch -> differentiable 512-d embedding.

    Only the modules upstream of the layer-15 readout are registered as children, so
    `.parameters()` yields exactly the parameters that can receive gradient. This matters
    beyond tidiness: NOTES §8's gradient test compares `got` against `total`, and with
    `graph_output_nn` and the task heads included the correct result would read as a
    broken chain.

    Args:
        accelerator: `"cpu"`, `"gpu"`, or None to pick by `torch.cuda.is_available()`.
            Note this only sets a config field; it does not move the model. Use `.to()`.
        strict_load: raise `StateDictMismatch` if the checkpoint leaves any trunk
            parameter unfilled. Turn this off only with a recorded reason.
        use_readout_cache: read the embedding from graphium's readout cache (mirroring
            `Fingerprinter` exactly) instead of from `g["feat"]`. The two are the same
            tensor for the final layer; this exists so that claim stays testable.
        quiet: suppress graphium's construction chatter.
    """

    def __init__(self, accelerator=None, strict_load=True, use_readout_cache=False,
                 quiet=True):
        super().__init__()

        from graphium.config._loader import (
            load_accelerator,
            load_architecture,
            load_datamodule,
            load_metrics,
            load_predictor,
        )

        paths = minimol_paths()
        cfg = load_minimol_config(paths["config"], accelerator)

        with ExitStack() as silence:
            if quiet:
                fnull = silence.enter_context(open(os.devnull, "w"))
                silence.enter_context(redirect_stdout(fnull))
                silence.enter_context(redirect_stderr(fnull))
            # This sequence mirrors `Minimol.__init__` line for line, including two things
            # that look like tidying opportunities and are not. `mup_base_path` is written
            # after `load_accelerator`, and `load_architecture` receives the *pre*-accelerator
            # `cfg` -- so whether the architecture sees the mup base shapes depends on
            # whether `load_accelerator` returned the same object or a copy. Reproducing the
            # order exactly is what guarantees this trunk is the architecture the checkpoint
            # was trained for, which is in turn what makes the key check below meaningful.
            self.cfg, accelerator_type = load_accelerator(cfg)
            self.cfg["architecture"]["mup_base_path"] = paths["base_shape"]
            self.datamodule = load_datamodule(self.cfg, accelerator_type)
            model_class, model_kwargs = load_architecture(
                cfg, in_dims=self.datamodule.in_dims
            )
            metrics = load_metrics(self.cfg)
            predictor = load_predictor(
                config=self.cfg,
                model_class=model_class,
                model_kwargs=model_kwargs,
                metrics=metrics,
                task_levels=self.datamodule.get_task_levels(),
                accelerator_type=accelerator_type,
                featurization=self.datamodule.featurization,
                task_norms=self.datamodule.task_norms,
                replicas=1,
                gradient_acc=1,
                global_bs=self.datamodule.batch_size_training,
            )

        state_dict = torch.load(paths["state_dict"], map_location="cpu")
        incompatible = predictor.load_state_dict(state_dict, strict=False)
        self.load_report = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "checkpoint_keys": len(state_dict),
            "state_dict_path": paths["state_dict"],
            # Recorded because it decides whether `load_architecture` saw `mup_base_path`
            # above. A measured fact rather than an assumption about graphium's internals.
            "accelerator_returned_same_object": cfg is self.cfg,
        }

        network = predictor.model

        # Unregistered on purpose. `nn.Module.__setattr__` would register these as child
        # modules, which would put `task_heads` back into `.parameters()` -- the exact thing
        # this class exists to avoid. Writing straight into __dict__ keeps a usable handle
        # for dtype conversion, the readout-cache path, and the reachability check in
        # verify_trunk.py, without any of them being trainable here.
        self.__dict__["_predictor"] = predictor
        self.__dict__["_network"] = network
        self.__dict__["_excluded"] = {"task_heads": network.task_heads}

        # The trunk proper: everything upstream of the layer-15 readout.
        self.encoder_manager = network.encoder_manager
        self.pre_nn = network.pre_nn
        self.pre_nn_edges = network.pre_nn_edges
        self.gnn = network.gnn

        if strict_load:
            self._assert_checkpoint_covered_trunk(state_dict)

        self.use_readout_cache = use_readout_cache
        if use_readout_cache:
            network._enable_readout_cache([FINGERPRINT_MODULE])

        # MiniMol never calls `.eval()` or `.train()`; it inherits whatever Lightning's
        # PredictorModule was built in. We are training, so say so explicitly rather than
        # inheriting it -- dropout and any norm layers behave differently (NOTES §9).
        self.train()

    # -- construction checks -------------------------------------------------------

    def _assert_checkpoint_covered_trunk(self, state_dict):
        """Fail unless every trunk parameter came from the checkpoint.

        Built as a *positive* check rather than by filtering `missing_keys` for a prefix.
        Filtering has a failure mode that defeats the whole purpose: if the checkpoint
        nests its keys differently from the guess, the filter matches nothing, the missing
        list comes back empty, and a completely unloaded trunk passes silently -- exactly
        the NOTES §9 disaster this guard exists to prevent, now wearing a PASS.

        So the expectation is derived from the modules themselves: enumerate what the trunk
        actually contains, then require the checkpoint to supply every one of those names
        *and* require the count to be nonzero. An empty expectation is itself an error.

        Only trunk keys are held to this standard. The checkpoint legitimately contains
        entries for things we never build or never use (the five pretraining task heads,
        loss buffers); those surface as `unexpected` and are harmless. A *missing* trunk
        key is not harmless -- it is a silently randomly-initialised layer.
        """
        expected = {}
        for attr, module in (("encoder_manager", self.encoder_manager),
                             ("pre_nn", self.pre_nn),
                             ("pre_nn_edges", self.pre_nn_edges),
                             ("gnn", self.gnn)):
            if module is None:
                continue
            for key in module.state_dict():
                expected[f"model.{attr}.{key}"] = attr

        present = [k for k in expected if k in state_dict]
        missing_trunk = sorted(k for k in expected if k not in state_dict)

        self.load_report.update({
            "expected_trunk_keys": len(expected),
            "trunk_keys_found_in_checkpoint": len(present),
            "missing_trunk_keys": missing_trunk,
        })

        if not expected:
            raise StateDictMismatch(
                "no trunk parameters were enumerated, so the checkpoint coverage check "
                "would pass vacuously. The module layout of "
                "FullGraphMultiTaskNetwork has changed; fix this check before training."
            )
        if missing_trunk:
            by_module = {}
            for k in missing_trunk:
                by_module[expected[k]] = by_module.get(expected[k], 0) + 1
            raise StateDictMismatch(
                f"{len(missing_trunk)} of {len(expected)} trunk parameters were not found "
                f"in {self.load_report['state_dict_path']} and are therefore randomly "
                f"initialised. By module: {by_module}. First few: {missing_trunk[:10]}. "
                "This is the silent-failure mode of load_state_dict(strict=False) called "
                "out in NOTES §9; do not train through it."
            )

    # -- data ----------------------------------------------------------------------

    def featurize_raw(self, smiles):
        """SMILES list -> a list of PyG `Data` graphs, uncollated.

        Kept separate from `collate` because these are what Phase 2 will cache to disk:
        SMILES->graph conversion is CPU-bound and would otherwise dominate every epoch
        (NOTES §9), while collation is cheap and must happen per batch anyway.
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            features, _ = self.datamodule._featurize_molecules(smiles)
        # A molecule that fails to featurize is returned as the error *string* rather than
        # raising, so a silent hole in the batch is the default behaviour. Catch it here.
        bad = [s for s, f in zip(smiles, features) if isinstance(f, str)]
        if bad:
            raise ValueError(f"{len(bad)} SMILES failed to featurize, e.g. {bad[:3]}")
        return cast_features(features)

    def featurize(self, smiles, to_device=True):
        """SMILES list -> the batch dict `forward` expects."""
        return self.collate(self.featurize_raw(smiles), to_device=to_device)

    def collate(self, features, to_device=True):
        """Featurized `Data` objects -> `{"features": Batch, "batch_indices": Tensor}`."""
        batch = Batch.from_data_list(list(features))
        out = {"features": batch, "batch_indices": batch.batch}
        return self.to_device(out) if to_device else out

    def to_device(self, batch, device=None):
        """Move a batch onto the trunk's device."""
        if device is None:
            device = self.device
        features = batch["features"].to(device)
        return {"features": features, "batch_indices": batch["batch_indices"].to(device)}

    @property
    def device(self):
        return next(self.parameters()).device

    # -- forward -------------------------------------------------------------------

    def forward(self, batch, check_grad=False, return_node_features=False):
        """`batch -> [B, 512]`, differentiable into every trunk parameter.

        This is `FullGraphMultiTaskNetwork.forward` stopped after `gnn`, followed by
        MiniMol's `global_max_pool`. Everything here is a call into graphium's own modules;
        the only things removed relative to `Fingerprinter` are the `inference_mode`
        wrapper and the `.cpu()`.
        """
        g = batch["features"]
        # Cast float features to the predictor's dtype, exactly as Fingerprinter does via
        # `predictor._convert_features_dtype`. Skipping this is a silent dtype mismatch.
        g = self._predictor._convert_features_dtype(g)

        if self.encoder_manager is not None:
            g = self.encoder_manager(g)

        if self.pre_nn is not None:
            g["feat"] = self.pre_nn.forward(g["feat"])

        if self.pre_nn_edges is not None:
            e = g["edge_feat"]
            # Empty-edge guard copied from graphium: a batch with no edges at all would
            # otherwise carry the wrong last dimension into the GNN.
            if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
                e = torch.zeros(list(e.shape[:-1]) + [self.pre_nn_edges.out_dim],
                                device=e.device, dtype=e.dtype)
            else:
                e = self.pre_nn_edges.forward(e)
            g["edge_feat"] = e

        g = self.gnn.forward(g)

        if self.use_readout_cache:
            node_feats = self._network._module_map[FINGERPRINT_MODULE]._readout_cache[
                FINGERPRINT_LAYER
            ]
        else:
            node_feats = g["feat"]

        embedding = global_max_pool(node_feats, batch["batch_indices"])

        if check_grad and embedding.grad_fn is None and torch.is_grad_enabled():
            raise RuntimeError(
                "embedding has no grad_fn: the graph was not recorded. Something in the "
                "path is still detaching or running under inference_mode/no_grad."
            )
        # Returning the pre-pool tensor lets verify_trunk.py assert `cache[15] is feat`
        # within a single forward pass -- a tensor identity, which is a far stronger claim
        # than two pooled outputs comparing equal across two passes.
        return (embedding, node_feats) if return_node_features else embedding

    # -- introspection --------------------------------------------------------------

    def trunk_modules(self):
        """The registered, trainable submodules, by name."""
        return {name: module for name, module in (
            ("encoder_manager", self.encoder_manager),
            ("pre_nn", self.pre_nn),
            ("pre_nn_edges", self.pre_nn_edges),
            ("gnn", self.gnn),
        ) if module is not None}

    def excluded_modules(self):
        """Modules downstream of the readout, deliberately not part of this trunk.

        Reachable only through this accessor so `verify_trunk.py` can demonstrate they
        receive no gradient. They are not in `.parameters()` and never will be.
        """
        return dict(self._excluded)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
