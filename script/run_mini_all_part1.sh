#!/bin/bash
# ============================================================================ #
# COCO mini(1/100) 主实验 Part1（52 个）/ Main experiments Part1 (52 runs)
#   detect × 7 后端 × 4 配置 = 28；seg × 前 6 后端 × 4 配置 = 24
#   detect × 7 backends × 4 configs = 28; seg × first 6 backends × 4 configs = 24
# 与 run_mini_all_part2.sh 并行执行（52 vs 54，负载均衡）/ In parallel with part2 (52 vs 54, balanced)
# ============================================================================ #

set -u

ROOT="/home/zoujiu/Desktop/projects/aLSQplus/QAT_training"
VENV_ACTIVATE="source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate"

FLOAT_EPOCHS=50
QAT_EPOCHS=10
CALIB_BATCHES=20
BATCH=8

# 前半 4 配置 / First half 4 configs
CONFIGS=(per_channel_act_unsigned per_tensor_act_unsigned per_channel_act_signed per_tensor_act_signed)
declare -A CFG_ARGS=(
  [per_channel_act_unsigned]=""
  [per_tensor_act_unsigned]="--no-per-channel"
  [per_channel_act_signed]="--no-all-positive"
  [per_tensor_act_signed]="--no-per-channel --no-all-positive"
)

# 任务 → 后端列表（Part1：detect 全 7 + seg 前 6；seg/pact 与 pose 全部在 Part2）/
# task → backend list (Part1: detect all 7 + seg first 6; seg/pact & all pose live in Part2)
TASKS=(detect seg)
declare -A TASK_QUANTS=(
  [detect]="lsqplus_v1 lsqplus_v2 lsq_v1 lsq_v2 minmax dorefa pact"
  [seg]="lsqplus_v1 lsqplus_v2 lsq_v1 lsq_v2 minmax dorefa"
)
declare -A DATA_YAML=(
  [detect]="dataset/coco_mini_detect.yaml"
  [seg]="dataset/coco_mini_seg.yaml"
  [pose]="dataset/coco_mini_pose.yaml"
)
declare -A FLOAT_BEST=(
  [detect]="yolo26n_best.pth"
  [seg]="yolo26n-seg_best.pth"
  [pose]="yolo26n-pose_best.pth"
)

mkdir -p "$ROOT/log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

echo "[WARMUP] $(date '+%F %H:%M') Warming up CUDA driver..."
if source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate && python -c '
import torch
assert torch.cuda.is_available(), "CUDA not available!"
_ = torch.zeros(1, device="cuda:0")
print(f"[WARMUP] CUDA ok: {torch.cuda.get_device_name(0)}")
'; then
  echo "[WARMUP] CUDA driver initialized successfully."
else
  echo "[WARMUP][WARN] CUDA warmup failed."
fi
deactivate 2>/dev/null || true

run_stage() {
  local task=$1 quant=$2 stage=$3 logfile=$4
  shift 4
  local extra_args="$*"
  local quant_arg=""
  [ "$quant" != "-" ] && quant_arg="--quant ${quant}"

  echo "[START] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${CFG_LABEL:-default}"

  local attempt=1 max_attempts=3 ec=0
  while [ $attempt -le $max_attempts ]; do
    (
      cd "$ROOT"
      bash -c "
        ${VENV_ACTIVATE}
        python -u networks_yolo26-${task}.py \
          --model yolo26n --stage ${stage} ${quant_arg} \
          --data ${DATA_YAML[$task]} \
          --float-epochs ${FLOAT_EPOCHS} --qat-epochs ${QAT_EPOCHS} \
          --calibration-batches ${CALIB_BATCHES} \
          --float-batch-size ${BATCH} --ptq-batch-size ${BATCH} --qat-batch-size ${QAT_BATCH_OVERRIDE:-${BATCH}} \
          --num-workers 0 \
          ${extra_args}
      " > "$logfile" 2>&1
    )
    ec=$?
    if [ $ec -ne 0 ]; then
      echo "[RETRY] attempt ${attempt}/${max_attempts} exit=${ec}"
    elif grep -q "Device: cpu" "$logfile" 2>/dev/null && [ "$stage" != "float" ]; then
      echo "[RETRY] attempt ${attempt}/${max_attempts} fell back to CPU"
      ec=999
    else
      break
    fi
    attempt=$((attempt + 1))
    sleep 5
  done

  if [ $ec -eq 0 ]; then
    echo "[OK] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${CFG_LABEL:-default}"
  else
    echo "[FAIL] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${CFG_LABEL:-default} exit=${ec}"
    FAILED+=("${task}/${quant}/${CFG_LABEL:-default}/${stage}")
  fi
  return $ec
}

FAILED=()

for task in "${TASKS[@]}"; do
  echo "=============================================================="
  echo "### TASK: ${task} (${DATA_YAML[$task]})"
  echo "=============================================================="

  float_best="$ROOT/model/yolo26-${task}/n/${FLOAT_BEST[$task]}"
  if [ -f "$float_best" ]; then
    echo "[REUSE] task=${task} float best exists"
  elif ! run_stage "$task" "-" "float" "$ROOT/log/mini_${task}_float.log"; then
    echo "[SKIP] task=${task} float failed"
    continue
  fi

  for q in ${TASK_QUANTS[$task]}; do
    for cfg in "${CONFIGS[@]}"; do
      echo "--- backend: ${q} | config: ${cfg} (${CFG_ARGS[$cfg]:-default}) ---"
      qat_override=""
      [ "$task" = "seg" ] && [ "$q" = "dorefa" ] && qat_override=4
      [ -n "$qat_override" ] && export QAT_BATCH_OVERRIDE=$qat_override || unset QAT_BATCH_OVERRIDE
      export CFG_LABEL="$cfg"

      if ! run_stage "$task" "$q" "ptq" "$ROOT/log/mini_${task}_${q}_${cfg}_ptq.log" ${CFG_ARGS[$cfg]}; then
        echo "[SKIP] PTQ failed"
        continue
      fi
      if ! run_stage "$task" "$q" "qat" "$ROOT/log/mini_${task}_${q}_${cfg}_qat.log" ${CFG_ARGS[$cfg]}; then
        echo "[SKIP] QAT failed"
        continue
      fi
      unset QAT_BATCH_OVERRIDE
      run_stage "$task" "$q" "compare" "$ROOT/log/mini_${task}_${q}_${cfg}_compare.log" ${CFG_ARGS[$cfg]}
    done
  done
done

echo ""
echo "========= PART1 DONE (failures: ${#FAILED[@]}) ========="
[ ${#FAILED[@]} -gt 0 ] && printf '  - %s\n' "${FAILED[@]}"
