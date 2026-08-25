# Running the sweep on TamIA

Follow these in order. Each step says what to run, what correct output looks like, and what to
do when it does not. **Section F is a gate** — do not pass it until it passes.

This exists because TamIA is not a machine anything here has ever run on
(`reports/compute_profile.md:18`), and because its compute nodes have **no direct internet**
(`compute_profile.md:331`) — the whole design turns on getting around that.

> An earlier version of `CLAUDE.md` claimed `wandb agent` works on TamIA unaided, "confirmed
> 2026-08-13". That was **wrong** and is corrected. The route out is a Mila HTTP proxy.

**Symbols used below:** `<...>` is something you supply. Everything else is literal.

---

## A. The new wandb account *(from your laptop, before touching TamIA)*

1. Create the wandb account under your **Mila** email. *(Done — `ethan-kreuzer-mila`.)*
2. **The only thing you still need is the API key**, from <https://wandb.ai/authorize>.

**Your entity is `ethan-kreuzer-mila`** — the username segment of your profile URL. It is
already the default in `scripts/tamia_sweep_agent.sbatch`, so you do not have to pass it.
Override with `export WANDB_ENTITY_=<other>` if that ever changes.

**Why the entity gets handled so carefully.** `src/train.py:256` still defaults
`--wandb-entity` to `ethan_personal` — the *old* account — and `run_config.py:351` passes it to
`wandb.init(entity=...)` as an explicit argument, which **overrides** the `WANDB_ENTITY`
environment variable. So setting the env var alone is not enough; `scripts/sweep_trial.sh`
passes the flag too. Without that, every trial would try to write into an entity your new key
cannot access — and it would fail *after* the preflight had already passed, deep into the job.

---

## B. Repo and data onto TamIA

Everything here runs on a **login** node. Compute nodes cannot download anything.

**Put this in project space, not `$HOME`.** The venv is ~6 GB and the feature cache 4.5 GB, and
Alliance home directories carry a **file-count** quota as well as a size one — a torch venv is
tens of thousands of small files, which is the limit you hit first.

Your personal space inside the group allocation is
`/home/e/ethankrz/links/projects/aip-yvesbrun/ethankrz` (the trailing `ethankrz` is the standard
`projects/<group>/<user>/` layout — the level above it is shared with the rest of the group, so
do not clone there).

```bash
git clone https://github.com/ethankreuzer/finetune_minimol.git \
          ~/links/projects/aip-yvesbrun/ethankrz/finetune_minimol
cd ~/links/projects/aip-yvesbrun/ethankrz/finetune_minimol
git checkout encoder-vn
```

### Getting `uv` in the first place

Alliance clusters ship **no system Python**, which is why `pip` is not found. Load one first —
this project needs **3.11 exactly** (`requires-python = "==3.11.*"`, because the PyG extension
wheels are built per `(python, torch, cuda)` triple and these are `cp311`):

```bash
module load python/3.11

# Install uv from Astral's standalone installer, NOT pip.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

**Do not use `pip install uv` here.** Alliance's wheelhouse carries no `uv` wheel for their
platform, so pip falls back to the source tarball (`uv-0.12.6.tar.gz`) — and **uv is written in
Rust**, so that path tries to bootstrap a Rust toolchain and hangs at
`Preparing metadata (pyproject.toml)`. The installer above fetches a prebuilt
`uv-x86_64-unknown-linux-gnu` binary instead, which needs no compiler. Verified 2026-08-25.

`module load python/3.11` is still needed — not for uv itself, but because this project pins
`requires-python = "==3.11.*"` and uv needs a matching interpreter to build the venv against.

### Then sync — but move uv's cache off `$HOME` first

```bash
# MANDATORY. uv caches wheels in ~/.cache/uv by default; measured 31 GB on rabelais for this
# same lock file. Left at the default it lands in your home quota and fails partway through
# the ~1 h sync -- and it silently undoes the decision to keep everything off $HOME.
export UV_CACHE_DIR=~/links/projects/aip-yvesbrun/ethankrz/.uv-cache

UV_HTTP_TIMEOUT=3600 uv sync --locked --extra dev   # ~1 h; the timeout is mandatory
```

**`--locked` is not optional.** It makes uv *assert* the lockfile is current and fail if it
would re-resolve, rather than quietly resolving something else. This repo has already been bitten
once: graphium declares no torch dependency, so an unpinned resolve picked the newest torch and
orphaned the compiled PyG extensions (CLAUDE.md, "The uv environment"). The uv on TamIA (0.12.5)
is also newer than the one that wrote this lock (0.11.7, `version = 1, revision = 3`), so
`--locked` is what turns any disagreement between them into an immediate error instead of an
hour of downloading the wrong stack.

### Make it stick across logins

`module load` and `export` last only for the current shell. uv's installer usually appends a
PATH line to `~/.bashrc` itself — check with `grep -n 'local/bin' ~/.bashrc` — but the rest does
not persist. Add this once:

```bash
cat >> ~/.bashrc <<'EOF'

# --- finetune_minimol ---
export PATH="$HOME/.local/bin:$PATH"                                   # uv
export UV_CACHE_DIR=~/links/projects/aip-yvesbrun/ethankrz/.uv-cache   # keep 31 GB off $HOME
export UV_HTTP_TIMEOUT=3600                                            # 780 MB torch wheel
EOF
```

**Deliberately not in `.bashrc`: `module load python/3.11`.** SLURM jobs source `.bashrc` too, and
a module load there applies to every job you ever run, which is a good way to break an unrelated
one later. `scripts/tamia_sweep_agent.sbatch` loads it itself (override with `PYTHON_MODULE=...`),
so the job does not depend on your shell being set up a particular way. Load it by hand when you
want to run `uv` or the venv interactively.

**Audit the lock before trusting the env** — this is a standing rule in `CLAUDE.md`, not a
formality; graphium declares no torch dependency, so a bad resolve silently installs CUDA 13 and
orphans the compiled PyG extensions:

```bash
grep -c 'cu13\|cuda-toolkit' uv.lock          # must print 0
```

### The data — most of it comes with the clone

| what | size | how |
|---|---|---|
| `data/ampc_subset_331k.csv` | 29 MB | **tracked** — arrives with `git clone` |
| `data/splits/cluster_kfold_v1/` | 112 MB | **tracked** — arrives with `git clone` |
| `data/features/minimol_v1/` | 4.5 GB | **regenerate on TamIA**, one command |

Tracked as of 2026-08-25 — there is **no `scp` step any more**. Only the feature cache has to be
built, and it depends on the CSV rather than on an rdkit version, so rebuilding it is safe:

```bash
# on TamIA, once the venv exists (~3.5 min)
.venv/bin/python src/featurize.py
```

### Verifying it worked

`featurize.py` prints a rate (~1,700 mol/s) and exits 0, but exit 0 only means it did not crash.
This checks the cache is complete, matches the tracked CSV, and actually loads:

```bash
.venv/bin/python - <<'EOF'
import sys, json, pathlib
sys.path.insert(0, "src")
from features import load_features
d = pathlib.Path("data/features/minimol_v1")
m = json.loads((d / "meta.json").read_text())
print("n_graphs     ", f"{m['n_graphs']:,}", " expect 331,480")
print("input_sha256 ", m["input_sha256"][:16], " expect b4eb8c7815e2aeed")
ds = load_features(d)                    # re-hashes the CSV and raises on mismatch
print("loaded       ", f"{len(ds):,}", "graphs")
g = ds[0]
print("graph 0      ", g.num_nodes, "atoms, feat", tuple(g.feat.shape))
assert m["n_graphs"] == len(ds) == 331480
print("\nOK -- cache complete and matches the tracked CSV")
EOF
```

Expected output, measured on rabelais:

```
n_graphs      331,480  expect 331,480
input_sha256  b4eb8c7815e2aeed  expect b4eb8c7815e2aeed
loaded        331,480 graphs
graph 0       23 atoms, feat (23, 85)

OK -- cache complete and matches the tracked CSV
```

`load_features` is the real test, not the printed numbers: it **re-hashes the source CSV and
refuses to load if it changed**, which is what catches a cache built against a different subset.
`graphs.pt` should also be about **4.8 GB** — a much smaller file means it stopped early.

Verify the splits too, before training anything:

```bash
grep split_sha256 data/splits/cluster_kfold_v1/meta.json     # must contain 3ef97e78a85d...
```

**Never regenerate the splits.** `src/split.py` would rebuild them in ~2.5 min and it is
tempting, but Morgan fingerprint details shift between rdkit releases, so a rebuild under a
different rdkit produces a *different* partition — and every number stops being comparable to
what was measured on rabelais. That is precisely why they are tracked rather than rebuilt. If
the hash above differs, stop.

---

## C. The proxy — nothing for you to configure

**There is no host or port to find.** Alliance support's answer — *"just use `module load
httpproxy`"* — is the whole mechanism: it is an Lmod module that sets `http_proxy` and
`https_proxy` for you. The sbatch loads it and then checks the variables actually got set.

**This is unrelated to your wandb account.** The account decides who owns the runs; the proxy
decides whether the compute node can reach the internet at all. Creating the Mila account will
not change anything about the proxy, and it does not need to.

Optional sanity check on a **login** node:

```bash
module spider httpproxy      # confirms the module exists on this cluster
```

You cannot usefully test the proxy from a login node — login nodes have internet regardless, so
it would pass either way. **Whether it works from a compute node is what section F tests**, and
that is the question the whole design rests on.

If the module ever turns out not to exist, the script takes `PROXY=http://<host>:<port>` as a
manual override — but you should not need it.

---

## D. Node shape — already measured, nothing to do

Measured 2026-08-25, `sinfo -p gpubase_bynode_b2 -o '%c %m %G'`:

| type | CPUs | memory | GPUs |
|---|---|---|---|
| **h100** | 48 | 500 GB | **4** |
| h200 | 64 | 1000 GB | **8** |

The directives are set from this and need no edit: `--account=aip-yvesbrun`,
`--gres=gpu:h100:4`, `--cpus-per-task=48`, `--exclusive`, `--mem=0`.

**Two things worth knowing about why.**

*The partition holds two node types with different GPU counts.* The script therefore starts **one
agent per GPU actually present**, rather than a fixed four — on an h200 node a hardcoded 4 would
leave half of a whole-node allocation idle. Worker counts follow: 11 per agent on h100
(48 cores ÷ 4), 7 on h200 (64 ÷ 8), both derived at runtime.

*h100 is pinned deliberately, despite h200 being faster per GPU* (`compute_profile.md` §5 projects
~4.3× vs ~3.4× at batch 1024). h200 nodes carry **eight** GPUs, and a bayes sweep does not use
parallelism well — past roughly 8–16 concurrent trials, every trial is drawn from the same stale
posterior and the search degenerates toward random. Four concurrent trials keeps the optimiser
meaningful. **To switch to h200 anyway:** change `--gres` to `gpu:h200:8` and `--cpus-per-task` to
`64`. Nothing else needs touching.

---

## E. Create the sweep *(login node — compute nodes cannot)*

```bash
cd ~/links/projects/aip-yvesbrun/ethankrz/finetune_minimol
export WANDB_API_KEY=<key from step A>
wandb login

wandb sweep --project finetune_minimol --entity ethan-kreuzer-mila sweeps/bayes_v1.yaml
```

The output ends with a line containing the **sweep id** — an 8-character string. It is the
argument the sbatch takes. `wandb` also prints the full `wandb agent <entity>/<project>/<id>`
command; you want only the id.

**Sanity-check the sweep page it links.** It should show **five** swept parameters —
`freeze_epochs`, `unfrozen_epochs`, `head_lr`, `head_lr_unfrozen`, `trunk_lr`. Weight decay,
dropout and the four loss weights are pinned constants (trimmed 2026-08-25). If you see ten,
you created the sweep from an older copy of the yaml.

**Do not run that agent command directly on the login node.** Login nodes have no GPUs, and
running compute there is against Alliance policy.

---

## F. The gate — prove it works before spending 12 hours

Two short jobs. **Do not submit the real one until both pass.**

```bash
cd ~/links/projects/aip-yvesbrun/ethankrz/finetune_minimol     # sbatch logs land in the cwd
export WANDB_API_KEY=<key from step A>
```

That is the whole set — the entity is already the script's default, the account is in the
script, and the proxy comes from `module load httpproxy` inside the job.

That is the whole set. `--account=aip-yvesbrun` is already in the script (override with
`sbatch --account=...`), and the proxy comes from `module load httpproxy` inside the job — there
is nothing to export for it.

### F1 — dry run: proxy, allocation, staging *(~3 min, launches nothing)*

```bash
DRY_RUN=1 sbatch --partition=gpubase_bynode_b1 --time=00:20:00 \
  scripts/tamia_sweep_agent.sbatch <sweep_id>
```

Read `sweep_<jobid>.out`. It must show **all** of:

- [ ] `module load httpproxy`, then non-empty `http_proxy` / `https_proxy` lines
- [ ] `wandb API reachable through the proxy (HTTP ...)` ← **the one that matters**
- [ ] `gpus 4`
- [ ] `staged in <n>s -> /...` — not the `$SLURM_TMPDIR unset` warning
- [ ] `DRY_RUN: preflight, allocation and staging all passed`

### F2 — one real trial *(bounded)*

```bash
AGENT_COUNT=1 AGENTS=1 sbatch --partition=gpubase_bynode_b1 --time=01:00:00 \
  scripts/tamia_sweep_agent.sbatch <sweep_id>
```

Read `logs/agent_<jobid>_0.log`. It must show a trial starting, training epochs, and finishing —
and **the run must appear in wandb under the new entity**. Check that in the browser; a log that
looks fine while nothing reaches wandb is exactly the failure mode the proxy causes.

**Also note how long the trial took.** That number sets `--count` in the next step, and it is the
only honest source for it — the "~4 min/trial" figure in `compute_profile.md` is a projection for
hardware nothing has ever run on.

---

## G. The real submission

```bash
sbatch scripts/tamia_sweep_agent.sbatch <sweep_id>          # 12 h, b2, 4 agents
```

**Leaving `AGENT_COUNT` unset is the recommended option.** The agents then run until the wall
clock, which the script handles cleanly: SLURM's TERM is forwarded to each agent and on to the
trial it is running (`--forward-signals`), so nothing is orphaned and no run is left showing as
active on wandb forever.

If you do want to bound it, two things make a precise formula misleading:

- **`--count` is per agent**, not per job (`wandb agent --help`: *"Maximum number of runs this
  agent will execute"*). Four agents at `--count 50` is 200 trials, not 50.
- **F2's timing is a floor, not an estimate.** It ran with `AGENTS=1` on an otherwise idle node.
  Four concurrent agents contend for the same staged cache, page cache and CPU loaders, so the
  real per-trial time will be *higher*. Round down generously — or just leave it unset.

**After ~15 minutes, check that all four agents are working:**

```bash
tail -n 3 logs/agent_<jobid>_*.log
```

Each of the four must show progress. One agent idle while three run means that agent failed to
get a config — per-agent logs exist precisely so this is visible, since a single merged log would
hide it.

---

## H. When it goes wrong

| symptom | cause | fix |
|---|---|---|
| Job runs, agent log empty, nothing on wandb | The proxy is not reaching wandb from the compute node. `wandb agent` does not error on this — it blocks forever. | The preflight should have caught it. If it passed and this still happens, the proxy allows the API check but not the agent's traffic — take it to support. |
| `neither http_proxy nor https_proxy is set after loading httpproxy` | The module loaded but exported nothing, or is named differently here. | `module spider httpproxy` on a login node. Worst case, set `PROXY=http://<host>:<port>` to bypass it. |
| ``no `module` command available`` | Lmod's init was not found by the job. | The script probes the usual locations; if yours differs, `export LMOD_PKG=<path>` before `sbatch`. |
| `FATAL:` in `sweep_<jobid>.out` | A guard fired before anything was spent. | The message names the missing variable. Nothing was wasted. |
| Job rejected at `sbatch` | Bad `--account`, or the wrong `--gres` form for this partition. | Section D. `sacctmgr show user $USER` lists valid accounts. |
| `gpus 1` warning in the log | `--gres=gpu:4` did not give four GPUs. | Section D — likely needs a GPU type. |
| `CUDA out of memory` | Two agents landed on one GPU, i.e. `CUDA_VISIBLE_DEVICES` was overridden. | Check nothing in your shell profile sets it. |
| Trials die instantly, `unrecognized arguments` | The agent's underscored flags are not being normalised, or `--wandb-tags` is no longer last on the line. | `CLAUDE.md` §"The sweep". Do not reorder the `command:` block in the yaml. |
| Trials far slower than F2 | Data being read from home rather than `$SLURM_TMPDIR`. | Look for the `$SLURM_TMPDIR unset` warning in the log. |
| `split_sha256` mismatch | The splits were regenerated instead of copied. | Section B. Re-copy them; do not proceed. |

---

## I. After the sweep

Sort the wandb runs table by **`final/goal_metric_mean`** — the mean over models of each model's
final-epoch `goal_metric`. One row is one configuration.

**Then the step that is easy to forget and expensive to discover late:**

```bash
.venv/bin/python src/run_config.py --keep-checkpoints \
    --fold-list 0 1 2 3 4 --seed-list 0 1 \
    <the winning hyperparameters>
```

`src/run_config.py:238` appends `--no-save-checkpoint` unless `--keep-checkpoints` is passed, so
**every trial in the sweep saved no weights.** The MiniMol-layer analysis is post-hoc on a trained
model — without this re-run you have tuned numbers and nothing to extract features from.

Re-running at the winning hyperparameters lands in the **same** output bucket (`config_id`
ignores the fold and seed lists) and reuses any model already trained there, so widening to the
full 5×2 costs only the models it adds.

Read `pooled/*` from that run for the tail metrics — it is the honest estimate (100 potent
molecules against ~20 per fold), and it only exists once one seed holds all five folds.
