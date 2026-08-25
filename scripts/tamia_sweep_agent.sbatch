#!/bin/bash
# Run a wandb sweep on TamIA: one whole-node job, four agents, one per GPU.
#
#   sbatch scripts/tamia_sweep_agent.sbatch <sweep_id>
#
# --account is set below from Ethan's allocation; `sbatch --account=...` overrides it.
#
# See INSTRUCTIONS.md for the full procedure. The three things this script assumes have
# already happened: the venv exists, the data is on TamIA, and the sweep has been created
# from a LOGIN node (compute nodes cannot create one).
#
# WHY FOUR AGENTS AND NOT AN ARRAY. TamIA allocates whole nodes and a job must use all four
# GPUs (reports/compute_profile.md S5). scripts/run_grid.sbatch -- ten single-GPU array tasks
# of ~3 min -- is the shape that report says "does not map to TamIA at all". So: one long job,
# four agents, each pinned to its own GPU, each pulling trials from the same sweep.
#
# WHY THE PROXY PREFLIGHT IS THE MOST IMPORTANT LINE IN THIS FILE. TamIA's compute nodes have
# no internet (compute_profile.md:331); the route out is a Mila HTTP proxy. Without it
# `wandb agent` does NOT error -- it blocks waiting for a config that never arrives, and the
# job burns its entire allocation looking healthy. The preflight converts that silent 12-hour
# stall into a ~20-second failure that names its cause.
#
# CLAUDE.md previously claimed `wandb agent` works on TamIA unaided, "confirmed 2026-08-13",
# and named compute_profile.md as superseded. That was wrong; corrected 2026-08-25.

#SBATCH --job-name=minimol_sweep
#SBATCH --account=aip-yvesbrun
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.out
#SBATCH --open-mode=append
#
# MEASURED 2026-08-25, `sinfo -p gpubase_bynode_b2 -o '%c %m %G'`:
#
#   CPUS  MEMORY   GRES
#   48    500000   gpu:h100:4(S:0-1)
#   64    1000000  gpu:h200:8(S:0-1)
#
# TWO NODE TYPES, and they differ in GPU COUNT -- which is why AGENTS is derived from the GPUs
# actually present rather than fixed at 4. On an h200 node, four agents would leave four of
# eight GPUs idle inside a whole-node allocation.
#
# --gres=gpu:h100:4 pins to the 4-GPU type deliberately. The h200 nodes are faster per GPU
#   (compute_profile.md §5 projects ~4.3x vs ~3.4x at batch 1024) but carry EIGHT, and a
#   bayes sweep does not use parallelism well: past roughly 8-16 concurrent trials every trial
#   is drawn from the same stale posterior and the search degenerates toward random
#   (compute_profile.md §5). Four concurrent trials keeps the optimiser meaningful. To use
#   h200 instead, change this to `gpu:h200:8` and --cpus-per-task to 64; everything else
#   adapts on its own.
# --cpus-per-task=48 matches the h100 nodes exactly. It is set rather than left to --exclusive
#   because with --ntasks=1 a cgroup can otherwise bind the task to a single core, which would
#   silently strangle the dataloaders. $SLURM_CPUS_ON_NODE is still what the worker count is
#   computed from at runtime, so an h200 node needs no second edit here.
# --account is set from Ethan's allocation and is still overridable: `sbatch --account=...`
#   and SBATCH_ACCOUNT both take precedence over this line.
# --output and --error are the same file on purpose: interleaved order is what makes a remote
#   diagnosis possible. Per-agent logs are separate, which is the distinction that matters.

set -uo pipefail

# ---------------------------------------------------------------------------- configuration
# Every value here is overridable from the environment, so nothing needs editing to retarget.

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$REPO/.venv/bin/python}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"   # the module the venv was BUILT against
WANDB_ENTITY_="${WANDB_ENTITY_:-ethan-kreuzer-mila}"  # the Mila-email entity
WANDB_PROJECT_="${WANDB_PROJECT_:-finetune_minimol}"

# OPTIONAL override. Normally left empty: `module load httpproxy` below sets the proxy
# variables itself, so there is no host:port to supply. Set this only to bypass the module.
PROXY="${PROXY:-}"                                  # e.g. http://proxy.example:3128
NO_PROXY_="${NO_PROXY_:-localhost,127.0.0.1,.int.ets1.calculquebec.ca,.computecanada.ca}"

AGENTS="${AGENTS:-auto}"                            # auto = one per GPU actually present
AGENT_COUNT="${AGENT_COUNT:-0}"                     # trials per agent; 0 = unbounded
DRY_RUN="${DRY_RUN:-0}"                             # 1 = preflight + stage, launch nothing
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/enc_v2}"

SWEEP_ID="${1:-}"

# ---------------------------------------------------------------------------------- helpers

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo; echo "FATAL: $*" >&2; echo "  (see INSTRUCTIONS.md)" >&2; exit 1; }

# ------------------------------------------------------------------ 1. validate before spending
# Every one of these is cheap and happens before a single GPU-second is used. A 12-hour
# allocation must never be spent discovering a typo.

# ------------------------------------------------------------------ 0. environment modules
# `module` is an Lmod shell FUNCTION, not a binary, and a non-login `#!/bin/bash` job does not
# always inherit it. Source Lmod's init once here; both this step and the proxy step need it.
ensure_lmod() {
    command -v module >/dev/null 2>&1 && return 0
    for init in /etc/profile.d/z-01-lmod.sh /etc/profile.d/modules.sh \
                "${LMOD_PKG:-/nonexistent}/init/bash"; do
        # shellcheck disable=SC1090
        [ -r "$init" ] && . "$init" && log "sourced Lmod init from $init" && return 0
    done
    return 1
}

log "=== environment ==="
if ensure_lmod; then
    # LOAD THE PYTHON MODULE THE VENV WAS BUILT AGAINST. This is not optional on Alliance
    # clusters: `.venv/bin/python` links against the module's interpreter and its shared
    # libraries, so without the module loaded it can fail to start even though the file is
    # present and executable. Checking `[ -x "$VENV" ]` alone would sail straight past that
    # and the job would die on the first trial, after the proxy preflight had already passed.
    module load "$PYTHON_MODULE" 2>/dev/null && log "module load $PYTHON_MODULE" \
        || log "WARNING: could not load $PYTHON_MODULE -- continuing, the venv may still work"
else
    log "WARNING: no Lmod. Skipping module loads; fine off-cluster, suspicious on TamIA."
fi

log "=== validating inputs ==="

[ -n "$SWEEP_ID" ] || die "no sweep id.
  Usage: sbatch $0 <sweep_id>
  Create the sweep on a LOGIN node first:
    wandb sweep --project $WANDB_PROJECT_ --entity <entity> sweeps/bayes_v1.yaml"

[ -n "${WANDB_API_KEY:-}" ] || die "WANDB_API_KEY is not set.
  Compute nodes cannot run an interactive \`wandb login\`, so the key must arrive in the
  environment. Get it from https://wandb.ai/authorize under the NEW Mila account, then:
    export WANDB_API_KEY=...     # before sbatch; it propagates into the job"

[ -n "$WANDB_ENTITY_" ] || die "WANDB_ENTITY_ is not set.
  This is the new Mila-email entity. It is deliberately required rather than defaulted:
  train.py:256 still defaults to 'ethan_personal', which is the OLD account, and a silent
  fallback would file this sweep's runs under it."

[ -d "$REPO" ]  || die "repo not found at $REPO (override with REPO=...)"
[ -x "$VENV" ]  || die "venv interpreter not found at $VENV
  Run \`uv sync --locked --extra dev\` on a LOGIN node first -- compute nodes cannot download."

# Actually RUN it. The executable bit says nothing about whether the interpreter can start:
# a venv built against a module python fails here, not at the -x test, when the module is
# missing. One subprocess now beats discovering it inside the first trial.
PYVER=$("$VENV" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>&1) \
    || die "the venv interpreter will not run: $PYVER
  Usually the python module the venv was built against is not loaded. This job loads
  \`$PYTHON_MODULE\`; override with PYTHON_MODULE=... if the venv was built against another."
[ "$PYVER" = "3.11" ] || die "venv is python $PYVER, expected 3.11.
  pyproject pins requires-python = \"==3.11.*\" -- the PyG extension wheels are cp311."
log "python    $PYVER (venv runs)"

log "repo      $REPO"
log "venv      $VENV"
log "sweep     $WANDB_ENTITY_/$WANDB_PROJECT_/$SWEEP_ID"

# ------------------------------------------------------------------------ 2. sanitize the env
# Kept from scripts/run_grid.sbatch: sbatch propagates the submitting shell's environment,
# which may carry an activated virtualenv that would shadow $VENV.

unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.venv/bin$' | tr '\n' ':' | sed 's/:$//')

# ------------------------------------------------------------------------------- 3. the proxy
# TamIA compute nodes have no direct internet. The route out is the `httpproxy` MODULE, which
# sets http_proxy/https_proxy itself -- there is no host:port to supply, and nothing to look
# up. (Confirmed with Alliance support 2026-08-25: "just use `module load httpproxy`".)

log "=== proxy ==="

ensure_lmod || true

if [ -n "$PROXY" ]; then
    # Manual override, for a cluster without the module or to test a specific proxy.
    export http_proxy="$PROXY"  HTTP_PROXY="$PROXY"
    export https_proxy="$PROXY" HTTPS_PROXY="$PROXY"
    log "proxy set manually from \$PROXY: $PROXY"
elif command -v module >/dev/null 2>&1; then
    # NO PIPE HERE, and no command substitution. `module` is a shell function whose entire
    # job is to export into the CALLING shell; a pipe would run it in a subshell and the
    # exports would die with it. This was written as `module load httpproxy | sed ...` for
    # prettier output and failed exactly that way -- the check below caught it. Its output
    # goes straight to the job log, which is good enough.
    log "module load httpproxy"
    module load httpproxy || true
else
    die "no \`module\` command available and \$PROXY is not set.
  TamIA compute nodes have no direct internet; the route out is \`module load httpproxy\`.
  If Lmod is genuinely absent here, set PROXY=http://<host>:<port> to bypass this."
fi

# Whichever path we took, something must actually be set -- `module load` of a missing module
# prints a warning and returns success in some Lmod configurations, so its exit status is not
# evidence. This checks the outcome rather than the attempt.
if [ -z "${http_proxy:-}${https_proxy:-}" ]; then
    die "neither http_proxy nor https_proxy is set after loading httpproxy.
  The module did not do what it should. Check \`module spider httpproxy\` on a login node,
  or set PROXY=http://<host>:<port> explicitly."
fi

# Only supply no_proxy if the module did not; it knows the cluster's internal domains better
# than this file does.
if [ -z "${no_proxy:-}" ]; then
    export no_proxy="$NO_PROXY_" NO_PROXY="$NO_PROXY_"
fi

log "http_proxy  ${http_proxy:-<unset>}"
log "https_proxy ${https_proxy:-<unset>}"
log "no_proxy    ${no_proxy:-<unset>}"

log "=== proxy preflight ==="

# Hard timeout, so an unreachable proxy fails in seconds rather than hanging. --max-time
# bounds the whole request, not just the connect, which is the failure mode here.
CODE=$(curl --max-time 25 -sS -o /dev/null -w '%{http_code}' https://api.wandb.ai/ 2>&1) || CODE="curl-failed: $CODE"
case "$CODE" in
  2*|3*|4*)
    # Any HTTP response at all proves the route works. Do NOT tighten this to `== 200`:
    # measured 2026-08-25, the bare endpoint returns **404**, because it wants a path
    # (/graphql) rather than because anything is wrong. A 200-only check would reject a
    # perfectly working proxy. What is being tested is reachability, not the status code.
    log "wandb API reachable through the proxy (HTTP $CODE)" ;;
  *)
    die "cannot reach https://api.wandb.ai through the proxy ($CODE).
  This is the failure the preflight exists to catch: without it, \`wandb agent\` would not
  error -- it would block for the whole allocation waiting for a config.
  The proxy variables are set (shown above) but nothing answered through them. Check that
  \`module load httpproxy\` is the right mechanism for this cluster and that your account is
  permitted to use it." ;;
esac

# ------------------------------------------------------------- 4. verify we got the allocation
# --gres=gpu:4 is the standard Alliance form but not universal. Rather than trust it, count
# what is actually visible: a job that quietly received one GPU would put four agents on it
# and die in CUDA OOM, which reads as a model problem rather than a scheduling one.

log "=== allocation ==="
GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
CORES="${SLURM_CPUS_ON_NODE:-$(nproc)}"
log "node      $(hostname)"
log "gpus      $GPUS"
log "cores     $CORES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

[ "$GPUS" -ge 1 ] || die "no GPUs visible. The --gres request did not do what it should.
  Check the right form for this partition (some clusters need --gres=gpu:h100:4):
    sinfo -p gpubase_bynode_b2 -o '%c %m %G'"

# One agent per GPU, derived rather than assumed. gpubase_bynode_b2 holds both 4-GPU h100
# nodes and 8-GPU h200 nodes, so a fixed 4 would idle half an h200 allocation.
if [ "$AGENTS" = "auto" ]; then
    AGENTS="$GPUS"
    log "agents    $AGENTS (one per GPU present)"
elif [ "$GPUS" -lt "$AGENTS" ]; then
    log "WARNING: asked for $AGENTS agents but only $GPUS GPU(s) visible -- reducing to $GPUS."
    log "         Whole-node allocation should give 4 (h100) or 8 (h200). If this says 1, the"
    log "         --gres form is wrong for this partition. See INSTRUCTIONS.md section D."
    AGENTS="$GPUS"
fi

# Dataloader workers per agent. Derived rather than hardcoded: train.py defaults to 16, and
# four agents at 16 each would oversubscribe a node whose core count we do not know here.
WORKERS=$(( CORES / AGENTS - 1 ))
[ "$WORKERS" -lt 1 ] && WORKERS=1
# Capped at train.py's own default. The division protects against oversubscription on a small
# node; the cap protects against the opposite -- a 128-core node would otherwise hand each
# agent 41 loader processes, which costs memory and start-up latency for no throughput. This
# knob only ever moves DOWNWARD from the default.
[ "$WORKERS" -gt 16 ] && WORKERS=16
log "workers   $WORKERS per agent ($AGENTS agents over $CORES cores)"

# ----------------------------------------------------------------- 5. stage data to node-local
# train.py:620 calls load_features() inside main(), and run_config.py calls train.main() once
# per model -- so the 4.5 GB cache is re-read EVERY TRIAL. CLAUDE.md's "1.0 s to load" was
# measured on warm local disk; from a networked home with four agents contending it would
# dominate the run. Copy once to node-local storage and point every trial at that.

STAGE="${SLURM_TMPDIR:-}"
if [ -n "$STAGE" ] && [ -d "$STAGE" ]; then
    log "=== staging data to \$SLURM_TMPDIR ==="
    t0=$SECONDS
    mkdir -p "$STAGE/data"
    cp -r "$REPO/data/features"                  "$STAGE/data/" || die "failed to stage features"
    cp -r "$REPO/data/splits"                    "$STAGE/data/" || die "failed to stage splits"
    cp    "$REPO/data/ampc_subset_331k.csv"      "$STAGE/data/" || die "failed to stage the csv"
    DATA_FEATURES="$STAGE/data/features/minimol_v1"
    DATA_SPLITS="$STAGE/data/splits/cluster_kfold_v1"
    DATA_CSV="$STAGE/data/ampc_subset_331k.csv"
    log "staged in $((SECONDS - t0))s -> $STAGE"
else
    log "WARNING: \$SLURM_TMPDIR unset -- reading data from $REPO."
    log "         Expect every trial to re-read the 4.5 GB feature cache over the network."
    DATA_FEATURES="$REPO/data/features/minimol_v1"
    DATA_SPLITS="$REPO/data/splits/cluster_kfold_v1"
    DATA_CSV="$REPO/data/ampc_subset_331k.csv"
fi

[ -d "$DATA_FEATURES" ] || die "feature cache missing at $DATA_FEATURES
  Regenerate it on TamIA: \`$VENV src/featurize.py\` (~3.5 min). It is 4.5 GB and untracked,
  so \`git clone\` does not bring it."
[ -d "$DATA_SPLITS" ]   || die "splits missing at $DATA_SPLITS -- scp them from rabelais.
  Do NOT regenerate: rdkit version differences shift Morgan bits and would change
  split_sha256 away from 3ef97e78a85d..., making every number incomparable."
[ -f "$DATA_CSV" ]      || die "subset csv missing at $DATA_CSV -- scp it from rabelais."

# ------------------------------------------------------------------------- 6. launch the agents

cd "$REPO" || die "cannot cd to $REPO"
mkdir -p logs

# DRY_RUN stops here, having proven the three things that are expensive to get wrong: the
# proxy route out, the shape of the allocation, and that the staged data is where it should
# be. It deliberately does NOT run a trial -- `unfrozen_epochs` is swept up to 40, so a
# sweep-sampled trial has no bounded duration and would make this check's cost unpredictable.
# Run this first, then a bounded AGENT_COUNT=1 job to prove a trial completes.
if [ "$DRY_RUN" != "0" ]; then
    log "=== DRY_RUN: preflight, allocation and staging all passed ==="
    log "nothing was launched. Re-submit with AGENT_COUNT=1 to prove one trial completes,"
    log "then unset both for the real run."
    exit 0
fi

export WANDB_ENTITY="$WANDB_ENTITY_"
# Also passed as a flag by the wrapper -- the env var alone loses to wandb.init()'s
# explicit entity= argument, whose default is the OLD account.
export MINIMOL_ENTITY="$WANDB_ENTITY_"
export WANDB_PROJECT="$WANDB_PROJECT_"
# The agent spawns run_config.py, which reads these for every trial. Passing the data paths
# through the environment rather than the sweep yaml keeps the yaml machine-independent --
# the same sweep can be served by agents on TamIA and on rabelais at once.
export MINIMOL_FEATURES="$DATA_FEATURES"
export MINIMOL_SPLITS="$DATA_SPLITS"
export MINIMOL_CSV="$DATA_CSV"
export MINIMOL_WORKERS="$WORKERS"
export MINIMOL_OUTPUTS_ROOT="$OUTPUTS_ROOT"

# `--count` is per AGENT -- verified from `wandb agent --help`: "Maximum number of runs this
# agent will execute." So four agents at --count N run 4N trials in total, not N.
COUNT_ARG=""
[ "$AGENT_COUNT" -gt 0 ] && COUNT_ARG="--count $AGENT_COUNT"

log "=== launching $AGENTS agent(s) ==="
[ -n "$COUNT_ARG" ] && log "each bounded to $AGENT_COUNT trial(s)" || log "unbounded; the wall clock ends the job"

pids=()
for i in $(seq 0 $((AGENTS - 1))); do
    AGENT_LOG="logs/agent_${SLURM_JOB_ID:-local}_${i}.log"
    log "agent $i -> GPU $i -> $AGENT_LOG"
    # --forward-signals is load-bearing, not tidiness. run_config.py trains by calling
    # train.main() IN-PROCESS, so the trial is a child of the agent. Without this flag a TERM
    # to the agent pid -- which is what the trap below sends, and what SLURM sends at the wall
    # clock -- would kill the agent and leave the trial's python running, orphaned. wandb would
    # then show that run as active forever. Verified present in this wandb:
    #   `wandb agent --help` -> "-f, --forward-signals  Forward signals ... to child runs".
    CUDA_VISIBLE_DEVICES="$i" \
    "$VENV" -m wandb agent --forward-signals $COUNT_ARG \
        "$WANDB_ENTITY_/$WANDB_PROJECT_/$SWEEP_ID" \
        > "$AGENT_LOG" 2>&1 &
    pids+=($!)
done

# SLURM sends TERM at the wall clock. Forward it to every agent, which (via
# --forward-signals above) forwards it to the trial each is running, so nothing outlives the
# job. `wait` after the kill lets them exit cleanly rather than being severed mid-write.
cleanup() {
    log "signal received -- terminating $AGENTS agent(s)"
    kill -TERM "${pids[@]}" 2>/dev/null
    wait 2>/dev/null
    log "agents stopped"
    exit 143
}
trap cleanup TERM INT

log "all agents launched; waiting"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done

log "=== done (last agent exit $rc) ==="
log "per-agent logs: logs/agent_${SLURM_JOB_ID:-local}_*.log"
exit "$rc"
