#!/bin/bash
#SBATCH --job-name=dinov2_vitl16_short_h200_8gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1200G
#SBATCH --gres=gpu:8
#SBATCH --partition=ailab
#SBATCH --constraint=h200
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cs1095@princeton.edu

set -eo pipefail

set +u
export PS1="${PS1-}"
module purge
module load anaconda3/2025.6

ENV_NAME="${ENV_NAME:-dinov2}"

# Some module/conda hook scripts reference unset vars (e.g. PS1/CONDA_BACKUP_*).
# Keep nounset disabled during shell bootstrap, then re-enable for the workload.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

REPO_DIR="${REPO_DIR:-/scratch/gpfs/BRENDEN/changho/dinov2}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/scratch/gpfs/BRENDEN/imagenet-1k}"
IMAGENET_EXTRA="${IMAGENET_EXTRA:-${REPO_DIR}/dataset_cache/imagenet1k}"

RUN_NAME="${RUN_NAME:-vitl16_short_h200_8gpu_train}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/${RUN_NAME}}"

# To preserve schedule-equivalence across restarts, keep these unchanged.
# Recommended for 8-GPU 72h chunks: OFFICIAL_EPOCH_LENGTH=1250, EPOCHS=100, BATCH_SIZE_PER_GPU=64.
EXACT_A100_RECIPE="${EXACT_A100_RECIPE:-1}"
if [[ "${EXACT_A100_RECIPE}" == "1" ]]; then
  BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
else
  BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-80}"
fi

LOCAL_CROPS_NUMBER="${LOCAL_CROPS_NUMBER:-8}"
EPOCHS="${EPOCHS:-100}"
OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH:-1250}"
EVAL_PERIOD_ITERS="${EVAL_PERIOD_ITERS:-12500}"
KNN_BATCH_SIZE="${KNN_BATCH_SIZE:-256}"
GATHER_ON_CPU="${GATHER_ON_CPU:-0}"
RUN_KNN_EVAL="${RUN_KNN_EVAL:-0}"
STRICT_RESUME_EQUIVALENT="${STRICT_RESUME_EQUIVALENT:-1}"
AUTO_REQUEUE_ON_TIMEOUT="${AUTO_REQUEUE_ON_TIMEOUT:-1}"

# W&B (enabled by default; set WANDB_ENABLED=0 to disable)
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-dinov2}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-vitl16_short_h200_8gpu}"
WANDB_MODE="${WANDB_MODE:-offline}"     # online | offline | disabled
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_TAGS="${WANDB_TAGS:-h200,vitl16_short,8gpu}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${REPO_DIR}/wandbkey}"
WANDB_ENTITY_AUTO="${WANDB_ENTITY_AUTO:-1}"           # auto-detect entity from API key when empty
WANDB_USE_PROXY_FOR_TRAIN="${WANDB_USE_PROXY_FOR_TRAIN:-1}"  # needed for WANDB_MODE=online on Della
WANDB_SYNC_ENABLED="${WANDB_SYNC_ENABLED:-1}"         # for WANDB_MODE=offline
WANDB_SYNC_USE_PROXY="${WANDB_SYNC_USE_PROXY:-1}"     # Princeton compute nodes need proxy/default
WANDB_SYNC_TIMEOUT_SEC="${WANDB_SYNC_TIMEOUT_SEC:-240}"

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}" "${IMAGENET_EXTRA}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  if [[ -f "${WANDB_API_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_API_KEY_FILE}")"
    if [[ -n "${WANDB_API_KEY}" ]]; then
      export WANDB_API_KEY
      echo "W&B API key loaded from ${WANDB_API_KEY_FILE}"
    else
      echo "[WARN] ${WANDB_API_KEY_FILE} is empty; disabling W&B." >&2
      WANDB_ENABLED=0
    fi
  elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "W&B API key found in environment."
  else
    echo "[WARN] No W&B API key available; disabling W&B." >&2
    WANDB_ENABLED=0
  fi
fi

export WANDB_ENABLED WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME WANDB_RUN_ID WANDB_GROUP WANDB_MODE WANDB_LOG_EVERY WANDB_TAGS

REQUEUE_REQUESTED=0
TRAIN_STEP_PID=""

detect_wandb_entity() {
  if [[ "${WANDB_ENABLED}" != "1" || "${WANDB_ENTITY_AUTO}" != "1" || -n "${WANDB_ENTITY}" || -z "${WANDB_API_KEY:-}" ]]; then
    return 0
  fi
  local detected_entity
  detected_entity="$(
    WANDB_API_KEY="${WANDB_API_KEY}" /home/cs1095/.conda/envs/dinov2/bin/python - <<'PY'
import wandb
try:
    api = wandb.Api(timeout=20)
    viewer = api.viewer
    print(getattr(viewer, "entity", "") or getattr(viewer, "username", ""))
except Exception:
    pass
PY
  )"
  detected_entity="$(echo "${detected_entity}" | tr -d '[:space:]')"
  if [[ -n "${detected_entity}" ]]; then
    WANDB_ENTITY="${detected_entity}"
    export WANDB_ENTITY
    echo "W&B entity auto-detected: ${WANDB_ENTITY}"
  else
    echo "[WARN] Could not auto-detect W&B entity. Set WANDB_ENTITY explicitly if sync fails with 403." >&2
  fi
}

sync_wandb_offline() {
  if [[ "${WANDB_SYNC_ENABLED}" != "1" ]]; then
    echo "Skipping W&B sync because WANDB_SYNC_ENABLED=${WANDB_SYNC_ENABLED}"
    return 0
  fi
  if [[ "${WANDB_ENABLED}" != "1" || "${WANDB_MODE}" != "offline" ]]; then
    echo "Skipping W&B sync because WANDB_ENABLED=${WANDB_ENABLED} WANDB_MODE=${WANDB_MODE}"
    return 0
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[WARN] WANDB_API_KEY is missing; cannot sync offline runs." >&2
    return 0
  fi

  if [[ "${WANDB_SYNC_USE_PROXY}" == "1" ]]; then
    module load proxy/default >/dev/null 2>&1 || echo "[WARN] Failed to load proxy/default; sync may fail." >&2
  fi

  detect_wandb_entity

  shopt -s nullglob
  local run_paths=("${OUTPUT_DIR}"/wandb/offline-run-*)
  shopt -u nullglob
  if (( ${#run_paths[@]} == 0 )); then
    echo "No offline W&B runs found under ${OUTPUT_DIR}/wandb; nothing to sync."
    return 0
  fi

  echo "Syncing ${#run_paths[@]} offline W&B run(s) from ${OUTPUT_DIR}/wandb ..."
  local sync_cmd=(wandb sync --mark-synced)
  if [[ -n "${WANDB_PROJECT}" ]]; then
    sync_cmd+=(--project "${WANDB_PROJECT}")
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    sync_cmd+=(--entity "${WANDB_ENTITY}")
  fi
  sync_cmd+=("${run_paths[@]}")

  if timeout "${WANDB_SYNC_TIMEOUT_SEC}" "${sync_cmd[@]}"; then
    echo "W&B sync completed."
  else
    echo "[WARN] W&B sync did not complete cleanly (timeout or network issue)." >&2
  fi
}

handle_usr1() {
  if [[ "${AUTO_REQUEUE_ON_TIMEOUT}" != "1" ]]; then
    echo "[WARN] USR1 received, but AUTO_REQUEUE_ON_TIMEOUT=${AUTO_REQUEUE_ON_TIMEOUT}; ignoring." >&2
    return
  fi
  echo "[WARN] USR1 received (approaching walltime). Requesting requeue for job ${SLURM_JOB_ID}." >&2
  REQUEUE_REQUESTED=1
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol requeue "${SLURM_JOB_ID}" || echo "[WARN] scontrol requeue failed." >&2
  fi
  if [[ -n "${TRAIN_STEP_PID}" ]]; then
    kill -TERM "${TRAIN_STEP_PID}" 2>/dev/null || true
  fi
}

handle_term() {
  if [[ "${REQUEUE_REQUESTED}" == "1" ]]; then
    echo "[INFO] TERM received after requeue request; exiting cleanly." >&2
    exit 0
  fi
  echo "[WARN] TERM/INT received; stopping." >&2
  if [[ -n "${TRAIN_STEP_PID}" ]]; then
    kill -TERM "${TRAIN_STEP_PID}" 2>/dev/null || true
  fi
  exit 143
}

trap handle_usr1 USR1
trap handle_term TERM INT

if [[ "${WANDB_ENABLED}" == "1" && "${WANDB_MODE}" == "online" && "${WANDB_USE_PROXY_FOR_TRAIN}" == "1" ]]; then
  module load proxy/default >/dev/null 2>&1 || echo "[WARN] Failed to load proxy/default; W&B online mode may fail." >&2
  detect_wandb_entity
fi

echo "=== CUDA COMPAT CHECK ==="
python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available in this environment.")
    sys.exit(4)

arch_list = set(torch.cuda.get_arch_list())
device_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
device_arch = f"sm_{major}{minor}"
print(f"device={device_name} capability={device_arch} build_arches={sorted(arch_list)}")

if device_arch not in arch_list:
    print("[ERROR] GPU compute capability is not supported by this PyTorch build.")
    print("[ERROR] Rebuild/update env with CUDA 12.x PyTorch including sm_90.")
    sys.exit(5)
PY

# If a WebDataset-style path is provided by habit, try the extracted sibling path.
if [[ "${IMAGENET_ROOT}" == *"-wds" ]]; then
  IMAGENET_ROOT_CANDIDATE="${IMAGENET_ROOT%-wds}"
  if [[ -d "${IMAGENET_ROOT_CANDIDATE}/train" && -d "${IMAGENET_ROOT_CANDIDATE}/val" ]]; then
    echo "[WARN] IMAGENET_ROOT=${IMAGENET_ROOT} appears to be WebDataset shards; using ${IMAGENET_ROOT_CANDIDATE}" >&2
    IMAGENET_ROOT="${IMAGENET_ROOT_CANDIDATE}"
  fi
fi

# DINOv2 ImageNet loader expects extracted ImageNet format.
if [[ ! -d "${IMAGENET_ROOT}/train" || ! -d "${IMAGENET_ROOT}/val" ]]; then
  if compgen -G "${IMAGENET_ROOT}/imagenet1k-train-*.tar" > /dev/null; then
    echo "[ERROR] ${IMAGENET_ROOT} looks like WebDataset shards (*.tar)." >&2
    echo "[ERROR] This pipeline expects extracted ImageNet layout:" >&2
    echo "        ${IMAGENET_ROOT}/train/<class>/*.JPEG" >&2
    echo "        ${IMAGENET_ROOT}/val/<class>/*.JPEG" >&2
    echo "        ${IMAGENET_ROOT}/labels.txt" >&2
    exit 2
  fi
  echo "[ERROR] Missing ${IMAGENET_ROOT}/train or ${IMAGENET_ROOT}/val" >&2
  exit 2
fi

if [[ ! -f "${IMAGENET_ROOT}/labels.txt" ]]; then
  echo "[ERROR] Missing ${IMAGENET_ROOT}/labels.txt (required by dinov2/data/datasets/image_net.py)." >&2
  exit 2
fi

TRAIN_DATASET_STR="ImageNet:split=TRAIN:root=${IMAGENET_ROOT}:extra=${IMAGENET_EXTRA}"
VAL_DATASET_STR="ImageNet:split=VAL:root=${IMAGENET_ROOT}:extra=${IMAGENET_EXTRA}"

MAX_ITERS=$((EPOCHS * OFFICIAL_EPOCH_LENGTH))
if (( EVAL_PERIOD_ITERS > MAX_ITERS )); then
  EVAL_PERIOD_ITERS="${MAX_ITERS}"
fi

# If resuming an existing run, enforce unchanged schedule-critical knobs.
if [[ "${STRICT_RESUME_EQUIVALENT}" == "1" && -f "${OUTPUT_DIR}/config.yaml" ]]; then
  export OUTPUT_DIR BATCH_SIZE_PER_GPU LOCAL_CROPS_NUMBER EPOCHS OFFICIAL_EPOCH_LENGTH
  python - <<'PY'
import os
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.path.join(os.environ["OUTPUT_DIR"], "config.yaml"))

expected = {
    "train.batch_size_per_gpu": int(os.environ["BATCH_SIZE_PER_GPU"]),
    "crops.local_crops_number": int(os.environ["LOCAL_CROPS_NUMBER"]),
    "optim.epochs": int(os.environ["EPOCHS"]),
    "train.OFFICIAL_EPOCH_LENGTH": int(os.environ["OFFICIAL_EPOCH_LENGTH"]),
}
actual = {
    "train.batch_size_per_gpu": int(cfg.train.batch_size_per_gpu),
    "crops.local_crops_number": int(cfg.crops.local_crops_number),
    "optim.epochs": int(cfg.optim.epochs),
    "train.OFFICIAL_EPOCH_LENGTH": int(cfg.train.OFFICIAL_EPOCH_LENGTH),
}

mismatches = [k for k in expected if expected[k] != actual[k]]
if mismatches:
    print("[ERROR] Resume-equivalence guard failed. Existing config differs from current settings:")
    for key in mismatches:
        print(f"  {key}: existing={actual[key]} requested={expected[key]}")
    print("[ERROR] Use the original values above, or set STRICT_RESUME_EQUIVALENT=0 intentionally.")
    sys.exit(6)
PY
fi

GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * SLURM_NTASKS))

echo "=== TRAIN (H200 single-node 8 GPU) ==="
echo "ENV_NAME=${ENV_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN_DATASET_STR=${TRAIN_DATASET_STR}"
echo "VAL_DATASET_STR=${VAL_DATASET_STR}"
echo "BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "LOCAL_CROPS_NUMBER=${LOCAL_CROPS_NUMBER}"
echo "EPOCHS=${EPOCHS}"
echo "OFFICIAL_EPOCH_LENGTH=${OFFICIAL_EPOCH_LENGTH}"
echo "MAX_ITERS=${MAX_ITERS}"
echo "EVAL_PERIOD_ITERS=${EVAL_PERIOD_ITERS}"
echo "EXACT_A100_RECIPE=${EXACT_A100_RECIPE}"
echo "STRICT_RESUME_EQUIVALENT=${STRICT_RESUME_EQUIVALENT}"
echo "AUTO_REQUEUE_ON_TIMEOUT=${AUTO_REQUEUE_ON_TIMEOUT}"
echo "WANDB_ENABLED=${WANDB_ENABLED}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_ENTITY=${WANDB_ENTITY}"
echo "WANDB_RUN_NAME=${WANDB_RUN_NAME}"
echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_ENTITY_AUTO=${WANDB_ENTITY_AUTO}"
echo "WANDB_USE_PROXY_FOR_TRAIN=${WANDB_USE_PROXY_FOR_TRAIN}"
echo "WANDB_LOG_EVERY=${WANDB_LOG_EVERY}"
echo "WANDB_TAGS=${WANDB_TAGS}"
echo "WANDB_SYNC_ENABLED=${WANDB_SYNC_ENABLED}"
echo "WANDB_SYNC_USE_PROXY=${WANDB_SYNC_USE_PROXY}"
echo "WANDB_SYNC_TIMEOUT_SEC=${WANDB_SYNC_TIMEOUT_SEC}"
echo "KNN_BATCH_SIZE=${KNN_BATCH_SIZE}"
echo "RUN_KNN_EVAL=${RUN_KNN_EVAL}"

srun --ntasks="${SLURM_NTASKS}" python -u dinov2/train/train.py \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --output-dir "${OUTPUT_DIR}" \
  train.dataset_path="${TRAIN_DATASET_STR}" \
  train.batch_size_per_gpu="${BATCH_SIZE_PER_GPU}" \
  train.num_workers="${SLURM_CPUS_PER_TASK:-8}" \
  crops.local_crops_number="${LOCAL_CROPS_NUMBER}" \
  optim.epochs="${EPOCHS}" \
  train.OFFICIAL_EPOCH_LENGTH="${OFFICIAL_EPOCH_LENGTH}" \
  evaluation.eval_period_iterations="${EVAL_PERIOD_ITERS}" &
TRAIN_STEP_PID=$!
set +e
wait "${TRAIN_STEP_PID}"
TRAIN_RC=$?
set -e
TRAIN_STEP_PID=""

if [[ "${REQUEUE_REQUESTED}" == "1" ]]; then
  echo "[INFO] Requeue requested for timeout handling; skipping post-train eval in this run."
  sync_wandb_offline
  exit 0
fi

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  echo "[ERROR] Training step failed with exit code ${TRAIN_RC}" >&2
  sync_wandb_offline
  exit "${TRAIN_RC}"
fi

echo "[INFO] Training reached configured MAX_ITERS=${MAX_ITERS}; auto-requeue is only used when walltime timeout sends USR1."

if [[ "${RUN_KNN_EVAL}" != "1" ]]; then
  echo "Skipping k-NN eval because RUN_KNN_EVAL=${RUN_KNN_EVAL}"
  sync_wandb_offline
  exit 0
fi

# Get latest teacher checkpoint exported by train.do_test.
LATEST_TRAIN_EVAL_DIR="$(ls -1d "${OUTPUT_DIR}"/eval/training_* 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -z "${LATEST_TRAIN_EVAL_DIR}" ]]; then
  echo "[ERROR] No teacher checkpoint found under ${OUTPUT_DIR}/eval/training_*" >&2
  echo "[ERROR] Reduce EVAL_PERIOD_ITERS or increase EPOCHS/OFFICIAL_EPOCH_LENGTH." >&2
  exit 3
fi
PRETRAINED_WEIGHTS="${LATEST_TRAIN_EVAL_DIR}/teacher_checkpoint.pth"
if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "[ERROR] Missing teacher checkpoint: ${PRETRAINED_WEIGHTS}" >&2
  exit 3
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
KNN_OUTPUT_DIR="${OUTPUT_DIR}/eval/knn_${TIMESTAMP}"
mkdir -p "${KNN_OUTPUT_DIR}"

KNN_CPU_FLAG=()
if [[ "${GATHER_ON_CPU}" == "1" ]]; then
  KNN_CPU_FLAG+=(--gather-on-cpu)
fi

echo "=== KNN EVAL ==="
echo "CONFIG_FILE=${OUTPUT_DIR}/config.yaml"
echo "PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS}"
echo "KNN_OUTPUT_DIR=${KNN_OUTPUT_DIR}"

srun --ntasks="${SLURM_NTASKS}" python -u dinov2/eval/knn.py \
  --config-file "${OUTPUT_DIR}/config.yaml" \
  --pretrained-weights "${PRETRAINED_WEIGHTS}" \
  --output-dir "${KNN_OUTPUT_DIR}" \
  --train-dataset "${TRAIN_DATASET_STR}" \
  --val-dataset "${VAL_DATASET_STR}" \
  --batch-size "${KNN_BATCH_SIZE}" \
  "${KNN_CPU_FLAG[@]}"

sync_wandb_offline
echo "Done. k-NN metrics: ${KNN_OUTPUT_DIR}/results_eval_knn.json"
