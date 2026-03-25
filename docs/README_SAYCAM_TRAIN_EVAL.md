# SAYCam DINOv2 Training + Final Eval Runbook

This runbook documents the current workflow for SAYCam training and evaluation in this repo.

## Scope

- Dataset splits: `S`, `A`, `Y`, `SAY`
- Training launcher: `train_vitl16_short_h200_8gpu_saycam.slurm`
- Continuation helper: `scripts/submit_saycam_continuation.sh`
- Eval launchers:
  - `eval_knn_vitl16_1gpu.slurm`
  - `eval_linear_vitl16_1gpu.slurm`

## Current behavior (important)

- Training now exports eval checkpoints every `5000` iterations (`evaluation.eval_period_iterations=5000`).
- Training now saves model checkpoints every `5000` iterations (`train.saveckp_freq_iterations=5000`).
- Continuation helper defaults to `SUBMIT_EVAL=0` (it submits training only, not kNN/linear).
- Continuation helper now supports forked replay seeding with `SOURCE_OUTPUT_ROOT`, `OUTPUT_ROOT`, and `SEED_CHECKPOINT_ITERATION`.

## Canonical paths

```bash
REPO_DIR=/scratch/gpfs/BRENDEN/changho/dinov2
OUTPUT_ROOT=${REPO_DIR}/outputs/saycam_vitl16_h200_4gpu_bs256_e600
RUN_PREFIX=saycam_vitl16_h200_4gpu_bs256_e600
```

Per-split run directories:

```bash
${OUTPUT_ROOT}/${RUN_PREFIX}_S
${OUTPUT_ROOT}/${RUN_PREFIX}_A
${OUTPUT_ROOT}/${RUN_PREFIX}_Y
${OUTPUT_ROOT}/${RUN_PREFIX}_SAY
```

## 1. Continue training until each split covers 125000 steps

`125000` steps means target max iteration `124999`.

Use the continuation helper (idempotent):

```bash
cd ${REPO_DIR}
scripts/submit_saycam_continuation.sh
```

Dry-run first (recommended):

```bash
cd ${REPO_DIR}
DRY_RUN=1 scripts/submit_saycam_continuation.sh
```

Notes:

- The helper inspects local progress (`training_metrics.json`) and active queue state.
- It only submits missing continuation jobs.
- It writes submission records to:
  - `results/dinov2_eval/saycam_continuation_submissions_<YYYYMMDD>.csv`

## 2. Monitor progress

Queue overview:

```bash
squeue -u $USER -o '%.18i %.40j %.8T %.10M %.9l %.R'
```

Track each split's latest recorded iteration:

```bash
for split in S A Y SAY; do
  run_dir=${OUTPUT_ROOT}/${RUN_PREFIX}_${split}
  printf '%-4s ' "${split}"
  tail -n 1 "${run_dir}/training_metrics.json" | sed -n 's/.*"iteration"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/iteration=\1/p'
done
```

Check whether expected eval checkpoint snapshots exist:

```bash
for split in S A Y SAY; do
  run_dir=${OUTPUT_ROOT}/${RUN_PREFIX}_${split}
  echo "=== ${split} ==="
  ls -1 "${run_dir}/eval" | rg '^training_' | sort -V | tail -n 10
done
```

## 3. Submit final eval (kNN + linear) after training is done

The recommended policy is final-only eval (no mid-run eval jobs).

Submit one kNN and one linear job per split after training reaches target:

```bash
cd ${REPO_DIR}
for split in S A Y SAY; do
  run_dir=${OUTPUT_ROOT}/${RUN_PREFIX}_${split}

  jid_knn=$(sbatch --parsable \
    --job-name="dinov2_knn_${split}_final" \
    --cpus-per-task=8 \
    --export=ALL,REPO_DIR=${REPO_DIR},TRAIN_OUTPUT_DIR=${run_dir},CONFIG_FILE=${run_dir}/config.yaml \
    eval_knn_vitl16_1gpu.slurm)

  jid_linear=$(sbatch --parsable \
    --job-name="dinov2_linear_${split}_final" \
    --cpus-per-task=8 \
    --export=ALL,REPO_DIR=${REPO_DIR},TRAIN_OUTPUT_DIR=${run_dir},CONFIG_FILE=${run_dir}/config.yaml,LINEAR_BATCH_SIZE=128,LINEAR_EPOCHS=10,LINEAR_EPOCH_LENGTH=1250,LINEAR_NUM_WORKERS=8,LINEAR_EVAL_PERIOD_ITERS=1250,LINEAR_SAVE_CKPT_FREQ=20 \
    eval_linear_vitl16_1gpu.slurm)

  echo "${split}: knn=${jid_knn} linear=${jid_linear}"
done
```

Why `--cpus-per-task=8` on eval jobs:

- AI Lab partition policy is 8 CPU cores per GPU max; requesting 12 cores with 1 GPU can be rejected.

## 3b. Optional: submit eval for every existing exported checkpoint

Use this when you want a full trajectory instead of final-only eval.

The sweep helper:

- submits `kNN`, `linear`, or both for every `eval/training_*` snapshot
- passes `PRETRAINED_WEIGHTS` explicitly, so each job targets one fixed checkpoint
- skips checkpoint/eval pairs that already have a completed result under the run's `eval/` tree

Dry-run first:

```bash
cd ${REPO_DIR}
DRY_RUN=1 ONLY_DATASETS=saycam scripts/submit_checkpoint_eval_sweep.sh
```

Submit both kNN and linear for all SAYCam checkpoints:

```bash
cd ${REPO_DIR}
ONLY_DATASETS=saycam scripts/submit_checkpoint_eval_sweep.sh
```

Useful filters:

```bash
# only one run
ONLY_RUNS=SAY scripts/submit_checkpoint_eval_sweep.sh

# only a few checkpoints
ONLY_CHECKPOINTS=24999,49999,74999 scripts/submit_checkpoint_eval_sweep.sh

# only one eval type
EVAL_TYPES=linear scripts/submit_checkpoint_eval_sweep.sh
```

Submission records are written to:

- `results/dinov2_eval/checkpoint_eval_submissions_<YYYYMMDD>.csv`

## 4. Post-eval result collection

Linear summary:

```bash
cd ${REPO_DIR}
scripts/collect_midrun_linear_summary.sh
```

This writes:

- `results/dinov2_eval/midrun_linear_summary.csv`

kNN summary:

- Existing kNN summary file used in this project:
  - `results/dinov2_eval/midrun_knn_summary.csv`

If needed, submit logs and `results_eval_knn.json`/`results_eval_linear.json` under each run's `eval/` directory are the source of truth.

Checkpoint-sweep summary:

```bash
cd ${REPO_DIR}
scripts/collect_checkpoint_eval_summary.sh
```

This writes:

- `results/dinov2_eval/checkpoint_eval_summary.csv`

## 5. Useful maintenance commands

Cancel queued final eval jobs only:

```bash
squeue -h -u $USER -o '%i %j' | \
  rg 'dinov2_(knn|linear)_[A-Z]+_(to125000|final)' | \
  awk '{print $1}' | xargs -r scancel
```

Show only SAYCam training jobs:

```bash
squeue -u $USER -o '%.18i %.40j %.8T %.R' | rg 'dinov2_vitl16_short_h200_4gpu_saycam'
```

## 6. Post-125k scheduler-reset remediation

Root cause:

- The original post-`125000` continuations rebuilt the cosine scheduler against a new `optim.epochs` horizon, which reset the LR instead of continuing it smoothly.
- The trainer now preserves scheduler horizon explicitly through `optim.schedule_total_epochs`, and replay continuations must keep that horizon fixed from the start.

Affected source root:

```bash
/scratch/gpfs/BRENDEN/changho/dinov2/outputs/saycam_vitl16_h200_4gpu_bs256_e600
```

Archive root:

```bash
/scratch/gpfs/BRENDEN/changho/dinov2/outputs/saycam_vitl16_h200_4gpu_bs256_e600_archive_post125k_reset_20260314
```

Replay root:

```bash
/scratch/gpfs/BRENDEN/changho/dinov2/outputs/saycam_vitl16_h200_4gpu_bs256_e600_replay125k_fix_20260314
```

One-shot remediation command:

```bash
cd /scratch/gpfs/BRENDEN/changho/dinov2
scripts/remediate_saycam_post125k_reset.sh
```

What the remediation does:

- cancels active bad continuation train jobs rooted in the old output tree
- cancels queued/running eval jobs that target checkpoints strictly above `124999`
- archives all post-`124999` training/eval artifacts under the dated archive root
- resets each old run's `last_checkpoint.rank_*` back to `model_0124999.rank_*`
- seeds fresh replay runs from the `124999` checkpoint family and resubmits corrected continuations to `250000` steps

Replay submission details:

- `TARGET_STEPS=250000`
- `SCHEDULE_TARGET_STEPS=250000`
- `OFFICIAL_EPOCH_LENGTH=600`
- `EPOCHS=417`
- `STRICT_RESUME_EQUIVALENT=1`

Machine-readable manifest:

```bash
/scratch/gpfs/BRENDEN/changho/dinov2/results/dinov2_eval/post125k_reset_20260314_manifest.csv
```

Future rule:

- Any eval job that targets checkpoints above `124999` must use the replay root, not the original source root.

## 7. Repro checklist

Before launching final eval:

- Confirm each split has reached at least iteration `124999`.
- Confirm each split has `eval/training_124999/teacher_checkpoint.pth`.
- Confirm no duplicate stale eval jobs are in queue.
