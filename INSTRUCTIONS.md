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

1. Create the wandb account under your **Mila** email.
2. Note the **entity** name — wandb shows it in your profile URL, `wandb.ai/<entity>`. It is
   *not* your email, and it is what the sweep gets created under.
3. Copy the API key from <https://wandb.ai/authorize>.

**Why this is called out rather than assumed:** `src/train.py:256` still defaults
`--wandb-entity` to `ethan_personal`, the old account, and that default appears throughout
`CLAUDE.md`. The sbatch therefore **requires** the entity explicitly and refuses to start without
it — a silent fallback would file this sweep's runs under the wrong account, which is the kind of
mistake you notice a day later.

---

## B. Repo and data onto TamIA

Everything here runs on a **login** node. Compute nodes cannot download anything.

```bash
git clone <repo-url> ~/finetune_minimol
cd ~/finetune_minimol
git checkout encoder-vn

UV_HTTP_TIMEOUT=3600 uv sync --extra dev        # ~1 h; the timeout is mandatory, see CLAUDE.md
```

**Audit the lock before trusting the env** — this is a standing rule in `CLAUDE.md`, not a
formality; graphium declares no torch dependency, so a bad resolve silently installs CUDA 13 and
orphans the compiled PyG extensions:

```bash
grep -c 'cu13\|cuda-toolkit' uv.lock          # must print 0
```

### The data — `git clone` brings none of it

| what | size | how |
|---|---|---|
| `data/ampc_subset_331k.csv` | 29 MB | **copy** from rabelais |
| `data/splits/cluster_kfold_v1/` | 112 MB | **copy** from rabelais |
| `data/features/minimol_v1/` | 4.5 GB | **regenerate on TamIA** |

```bash
# from rabelais
scp     /home/ethan2/finetune_minimol/data/ampc_subset_331k.csv  <you>@tamia:~/finetune_minimol/data/
scp -r  /home/ethan2/finetune_minimol/data/splits                <you>@tamia:~/finetune_minimol/data/

# on TamIA, once the venv exists (~3.5 min)
.venv/bin/python src/featurize.py
```

**Copy the splits; do not regenerate them.** `src/split.py` would rebuild them in ~2.5 min, but
Morgan fingerprint details shift between rdkit releases, so a rebuild under a different rdkit
produces a *different* partition and every number stops being comparable to what was measured
here. Verify they survived the trip:

```bash
grep split_sha256 data/splits/cluster_kfold_v1/meta.json     # must contain 3ef97e78a85d...
```

If that hash differs, stop — you are about to train on a different partition than everything
else in this project.

---

## C. The proxy

Get the Mila HTTP proxy host and port. Then, **on a login node**:

```bash
export PROXY=http://<host>:<port>
https_proxy=$PROXY curl --max-time 25 -sS -o /dev/null -w '%{http_code}\n' https://api.wandb.ai/
```

**Expect `404`.** That is success — the bare endpoint wants a path, and any HTTP response at all
proves the route works. Do not expect `200`.

A login node has internet regardless, so this only proves your proxy *syntax* is right. Whether
the proxy works **from a compute node** is what section F tests, and it is the question the whole
design rests on.

---

## D. Node shape — confirm the GPU request form

```bash
sinfo -p gpubase_bynode_b2 -o '%c %m %G'     # cores, memory, gres per node
scontrol show node <one gpubase node>
```

The script asks for `--gres=gpu:4 --exclusive --mem=0`. Two of those need nothing from you:
`--exclusive` and `--mem=0` claim the whole node without naming its core count or memory, and the
script reads the real numbers at runtime from `$SLURM_CPUS_ON_NODE`.

**`--gres=gpu:4` is the one that might be wrong** — some Alliance clusters require a type, e.g.
`--gres=gpu:h100:4`. If the `%G` column shows a type, edit the `#SBATCH --gres=` line in
`scripts/tamia_sweep_agent.sbatch` to match. You do not have to get this right by inspection: the
script counts the GPUs it actually received and warns or fails if it did not get four.

---

## E. Create the sweep *(login node — compute nodes cannot)*

```bash
export WANDB_API_KEY=<key from step A>
wandb login

wandb sweep --project finetune_minimol --entity <entity> sweeps/bayes_v1.yaml
```

The output ends with a line containing the **sweep id** — an 8-character string. It is the
argument the sbatch takes. `wandb` also prints the full `wandb agent <entity>/<project>/<id>`
command; you want only the id.

**Do not run that agent command directly on the login node.** Login nodes have no GPUs, and
running compute there is against Alliance policy.

---

## F. The gate — prove it works before spending 12 hours

Two short jobs. **Do not submit the real one until both pass.**

```bash
export SBATCH_ACCOUNT=<def-xxx>
export WANDB_API_KEY=<key>
export WANDB_ENTITY_=<entity>
export PROXY=http://<host>:<port>
```

### F1 — dry run: proxy, allocation, staging *(~3 min, launches nothing)*

```bash
DRY_RUN=1 sbatch --partition=gpubase_bynode_b1 --time=00:20:00 \
  scripts/tamia_sweep_agent.sbatch <sweep_id>
```

Read `sweep_<jobid>.out`. It must show **all** of:

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
| Job runs, agent log empty, nothing on wandb | The proxy is not reaching wandb from the compute node. `wandb agent` does not error on this — it blocks forever. | The preflight should have caught it. If it passed and this still happens, the proxy allows the API check but not the agent's traffic — take it to Mila support. |
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
