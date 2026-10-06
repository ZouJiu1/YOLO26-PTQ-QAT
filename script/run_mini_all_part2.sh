#!/bin/bash
# ============================================================================ #
# COCO mini(1/100) 主实验 Part2 + 补充对照（54 个）/ Main Part2 + supplementary (54 runs)
#   主实验（32 个）/ Main runs (32):
#     seg × pact × 4 配置 = 4；pose × 7 后端 × 4 配置 = 28
#     seg × pact × 4 configs = 4; pose × 7 backends × 4 configs = 28
#   补充对照（22 个，全部 detect）/ Supplementary (22, all detect):
#     1. weight_unsigned 反例：2 后端 × 4 配置 = 8（预期 NaN/掉点 / NaN expected）
#     2. mixed_quant 对照：7 后端 × 基线配置 = 7
#     3. PTQ 校准量敏感度：calib {5,10,50}（仅 PTQ）= 3
#     4. 多种子稳定性：4 主配置 × seed=1 = 4
# 与 run_mini_all_part1.sh 并行执行（54 vs 52，负载均衡）/ In parallel with part1 (54 vs 52, balanced)
# ============================================================================ #

set -u

ROOT="/home/zoujiu/Desktop/projects/aLSQplus/QAT_training"
VENV_ACTIVATE="source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate"

FLOAT_EPOCHS=50
QAT_EPOCHS=10
CALIB_BATCHES=20
BATCH=8

# 主实验：4 配置；任务 → 后端（Part2：seg/pact + pose 全 7）/
# Main runs: 4 configs; task → backends (Part2: seg/pact + all 7 pose)
MAIN_CONFIGS=(per_channel_act_unsigned per_tensor_act_unsigned per_channel_act_signed per_tensor_act_signed)
declare -A MAIN_CFG_ARGS=(
  [per_channel_act_unsigned]=""
  [per_tensor_act_unsigned]="--no-per-channel"
  [per_channel_act_signed]="--no-all-positive"
  [per_tensor_act_signed]="--no-per-channel --no-all-positive"
)
MAIN_TASKS=(seg pose)
declare -A MAIN_TASK_QUANTS=(
  [seg]="pact"
  [pose]="lsqplus_v1 lsqplus_v2 lsq_v1 lsq_v2 minmax dorefa pact"
)

# 只留 detect + 2 个代表后端（lsqplus_v1 可学习步长 / minmax 固定范围）作为反例对照 /
# Keep only detect + 2 representative backends as counterexamples (8 experiments total)
QUANTS=(lsqplus_v1 minmax)
# weight_unsigned 对照 4 配置 / weight_unsigned ablation 4 configs
CONFIGS=(per_channel_act_unsigned_weight_unsigned per_tensor_act_unsigned_weight_unsigned per_channel_act_signed_weight_unsigned per_tensor_act_signed_weight_unsigned)
declare -A CFG_ARGS=(
  [per_channel_act_unsigned_weight_unsigned]="--w-all-positive"
  [per_tensor_act_unsigned_weight_unsigned]="--no-per-channel --w-all-positive"
  [per_channel_act_signed_weight_unsigned]="--no-all-positive --w-all-positive"
  [per_tensor_act_signed_weight_unsigned]="--no-per-channel --no-all-positive --w-all-positive"
)

TASKS=(detect)
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

# --------------------------------------------------------------------------- #
# 主实验：seg × pact × 4 配置 + pose × 7 后端 × 4 配置 = 32 个 /
# Main runs: seg × pact × 4 configs + pose × 7 backends × 4 configs = 32
# --------------------------------------------------------------------------- #
for task in "${MAIN_TASKS[@]}"; do
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

  for q in ${MAIN_TASK_QUANTS[$task]}; do
    for cfg in "${MAIN_CONFIGS[@]}"; do
      echo "--- backend: ${q} | config: ${cfg} (${MAIN_CFG_ARGS[$cfg]:-default}) ---"
      # 新版 dorefa 为线性网格（无 tanh），显存峰值与其他后端一致，无需降 batch /
      # New dorefa uses a linear grid (no tanh); memory peak matches other backends, no batch downgrade needed.
      unset QAT_BATCH_OVERRIDE
      export CFG_LABEL="$cfg"

      if ! run_stage "$task" "$q" "ptq" "$ROOT/log/mini_${task}_${q}_${cfg}_ptq.log" ${MAIN_CFG_ARGS[$cfg]}; then
        echo "[SKIP] PTQ failed"
        continue
      fi
      if ! run_stage "$task" "$q" "qat" "$ROOT/log/mini_${task}_${q}_${cfg}_qat.log" ${MAIN_CFG_ARGS[$cfg]}; then
        echo "[SKIP] QAT failed"
        continue
      fi
      unset QAT_BATCH_OVERRIDE
      run_stage "$task" "$q" "compare" "$ROOT/log/mini_${task}_${q}_${cfg}_compare.log" ${MAIN_CFG_ARGS[$cfg]}
    done
  done
done

echo ""
echo "========= main runs (seg/pact + pose) done ========="

# --------------------------------------------------------------------------- #
# 补充 1. weight_unsigned 反例：detect × 2 后端 × 4 配置 = 8 个 /
# Suppl. 1. weight_unsigned counterexamples: detect × 2 backends × 4 configs
# --------------------------------------------------------------------------- #
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

  for q in "${QUANTS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      echo "--- backend: ${q} | config: ${cfg} (${CFG_ARGS[$cfg]}) ---"
      # 新版 dorefa 为线性网格（无 tanh），显存峰值与其他后端一致，无需降 batch /
      # New dorefa uses a linear grid (no tanh); memory peak matches other backends, no batch downgrade needed.
      unset QAT_BATCH_OVERRIDE
      export CFG_LABEL="$cfg"

      if ! run_stage "$task" "$q" "ptq" "$ROOT/log/mini_${task}_${q}_${cfg}_ptq.log" ${CFG_ARGS[$cfg]}; then
        echo "[SKIP] PTQ failed"
        continue
      fi
      if ! run_stage "$task" "$q" "qat" "$ROOT/log/mini_${task}_${q}_${cfg}_qat.log" ${CFG_ARGS[$cfg]}; then
        echo "[SKIP] QAT failed"
        continue
      fi
      run_stage "$task" "$q" "compare" "$ROOT/log/mini_${task}_${q}_${cfg}_compare.log" ${CFG_ARGS[$cfg]}
    done
  done
done

echo ""
echo "========= weight_unsigned counterexamples done ========="

# --------------------------------------------------------------------------- #
# A. mixed_quant 对照：7 后端 × 基线配置（per_channel_act_unsigned）+ --mixed-quant /
#    mixed_quant ablation: 7 backends × baseline config + --mixed-quant
# --------------------------------------------------------------------------- #
ALL_QUANTS=(lsqplus_v1 lsqplus_v2 lsq_v1 lsq_v2 minmax dorefa pact)
echo ""
echo "=============================================================="
echo "### A. mixed_quant ablation (7 backends)"
echo "=============================================================="
for q in "${ALL_QUANTS[@]}"; do
  echo "--- backend: ${q} | config: per_channel_act_unsigned + --mixed-quant ---"
  export CFG_LABEL="mixed"
  if ! run_stage detect "$q" "ptq" "$ROOT/log/mini_detect_${q}_mixed_ptq.log" --mixed-quant; then
    echo "[SKIP] PTQ failed"
    continue
  fi
  if ! run_stage detect "$q" "qat" "$ROOT/log/mini_detect_${q}_mixed_qat.log" --mixed-quant; then
    echo "[SKIP] QAT failed"
    continue
  fi
  run_stage detect "$q" "compare" "$ROOT/log/mini_detect_${q}_mixed_compare.log" --mixed-quant
done

# --------------------------------------------------------------------------- #
# B. PTQ 校准量敏感度：lsqplus_v1 × calib {5,10,50}（仅 PTQ 阶段；--run-tag 隔离目录）/
#    PTQ calibration-size sensitivity: lsqplus_v1 × {5,10,50} (PTQ only; isolated dirs)
#    （calib=20 即主实验结果，无需重跑 / calib=20 is the main-run result, no rerun needed）
# --------------------------------------------------------------------------- #
echo ""
echo "=============================================================="
echo "### B. PTQ calibration-size sensitivity (lsqplus_v1)"
echo "=============================================================="
for calib in 5 10 50; do
  echo "--- calib_batches: ${calib} ---"
  export CFG_LABEL="calib${calib}"
  # extra_args 在 CLI 尾部，覆盖前面的 --calibration-batches 默认值 /
  # extra_args come last on the CLI and override the earlier --calibration-batches default
  run_stage detect lsqplus_v1 "ptq" "$ROOT/log/mini_detect_lsqplus_v1_calib${calib}_ptq.log" \
    --calibration-batches ${calib} --run-tag calib${calib}
done

# --------------------------------------------------------------------------- #
# C. 多种子稳定性：lsqplus_v1 × 4 主配置 × seed=1（目录自动加 _seed1 后缀）/
#    Multi-seed stability: lsqplus_v1 × 4 main configs × seed=1 (auto _seed1 dir suffix)
#    （MAIN_CONFIGS/MAIN_CFG_ARGS 已在顶部定义 / defined at script top）
# --------------------------------------------------------------------------- #
echo ""
echo "=============================================================="
echo "### C. Multi-seed stability (lsqplus_v1, seed=1)"
echo "=============================================================="
for cfg in "${MAIN_CONFIGS[@]}"; do
  echo "--- config: ${cfg} | seed=1 ---"
  export CFG_LABEL="seed1_${cfg}"
  if ! run_stage detect lsqplus_v1 "ptq" "$ROOT/log/mini_detect_lsqplus_v1_seed1_${cfg}_ptq.log" --seed 1 ${MAIN_CFG_ARGS[$cfg]}; then
    echo "[SKIP] PTQ failed"
    continue
  fi
  if ! run_stage detect lsqplus_v1 "qat" "$ROOT/log/mini_detect_lsqplus_v1_seed1_${cfg}_qat.log" --seed 1 ${MAIN_CFG_ARGS[$cfg]}; then
    echo "[SKIP] QAT failed"
    continue
  fi
  run_stage detect lsqplus_v1 "compare" "$ROOT/log/mini_detect_lsqplus_v1_seed1_${cfg}_compare.log" --seed 1 ${MAIN_CFG_ARGS[$cfg]}
done

echo ""
echo "========= PART2 DONE (failures: ${#FAILED[@]}) ========="
[ ${#FAILED[@]} -gt 0 ] && printf '  - %s\n' "${FAILED[@]}"
