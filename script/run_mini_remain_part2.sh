#!/bin/bash
# ============================================================================ #
# COCO mini(1/100) 剩余实验 Part2（26 主单元 + 22 补充项）/ Remaining Part2
#
# 背景 / Background:
#   clip 折叠修复（unsigned 激活 + 对称后端的干净浮点参考图输出爆炸）+ _anchors
#   设备 bug 修复后，只重跑：①日志中未完整完成的单元 ②所有 pact 单元（强制，
#   因为 pact_w_quant 默认已从 dorefa 改为 lsqplus_v1，旧产物是 dorefa 权重）。
#
# Part2 内容 / Contents:
#   主实验（26 个）/ Main units:
#     seg 12：lsq_v1/lsq_v2/dorefa 各 4（均未完成）
#     pose 14：lsq_v1/lsq_v2/minmax unsigned 各 2 = 6，dorefa 4，
#              pact 4（3 个旧 dorefa 权重强制重跑 + 1 个未完成）
#   补充对照（22 项，全部 detect）/ Supplementary:
#     weight_unsigned 反例 2 后端 × 4 = 8（预期 NaN/快速失败 / NaN expected）
#     mixed_quant 对照 7 后端 = 7
#     calib {5,10,50} 仅 PTQ = 3
#     seed=1 × 4 主配置 = 4
# 与 run_mini_remain_part1.sh 并行；dorefa QAT 一律 batch=4（8GB 显存防 OOM）/
# Parallel with part1; dorefa QAT always uses batch=4 (8GB GPU, avoid OOM).
#
# 断点续跑 / Resume: 已 [OK] 的阶段记录在 log/mini_remain_part2.log，重跑自动跳过。
# ============================================================================ #

set -u

ROOT="/home/zoujiu/Desktop/projects/aLSQplus/QAT_training"
VENV_ACTIVATE="source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate"

FLOAT_EPOCHS=50
QAT_EPOCHS=10
CALIB_BATCHES=20
BATCH=8

TAG="rem2"
SUMLOG="$ROOT/log/mini_remain_part2.log"

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

# 单元行格式 / unit line: <task> <quant> <label> <extra args... | - (none)>
# 主实验 / main units
MAIN_UNITS="
seg lsq_v1 per_channel_act_unsigned -
seg lsq_v1 per_tensor_act_unsigned --no-per-channel
seg lsq_v1 per_channel_act_signed --no-all-positive
seg lsq_v1 per_tensor_act_signed --no-per-channel --no-all-positive
seg lsq_v2 per_channel_act_unsigned -
seg lsq_v2 per_tensor_act_unsigned --no-per-channel
seg lsq_v2 per_channel_act_signed --no-all-positive
seg lsq_v2 per_tensor_act_signed --no-per-channel --no-all-positive
seg dorefa per_channel_act_unsigned -
seg dorefa per_tensor_act_unsigned --no-per-channel
seg dorefa per_channel_act_signed --no-all-positive
seg dorefa per_tensor_act_signed --no-per-channel --no-all-positive
pose lsq_v1 per_channel_act_unsigned -
pose lsq_v1 per_tensor_act_unsigned --no-per-channel
pose lsq_v2 per_channel_act_unsigned -
pose lsq_v2 per_tensor_act_unsigned --no-per-channel
pose minmax per_channel_act_unsigned -
pose minmax per_tensor_act_unsigned --no-per-channel
pose dorefa per_channel_act_unsigned -
pose dorefa per_tensor_act_unsigned --no-per-channel
pose dorefa per_channel_act_signed --no-all-positive
pose dorefa per_tensor_act_signed --no-per-channel --no-all-positive
pose pact per_channel_act_unsigned -
pose pact per_tensor_act_unsigned --no-per-channel
pose pact per_channel_act_signed --no-all-positive
pose pact per_tensor_act_signed --no-per-channel --no-all-positive
"

# 补充 1：weight_unsigned 反例（detect × lsqplus_v1/minmax × 4 配置）/
# Suppl. 1: weight_unsigned counterexamples
WU_UNITS="
detect lsqplus_v1 per_channel_act_unsigned_weight_unsigned --w-all-positive
detect lsqplus_v1 per_tensor_act_unsigned_weight_unsigned --no-per-channel --w-all-positive
detect lsqplus_v1 per_channel_act_signed_weight_unsigned --no-all-positive --w-all-positive
detect lsqplus_v1 per_tensor_act_signed_weight_unsigned --no-per-channel --no-all-positive --w-all-positive
detect minmax per_channel_act_unsigned_weight_unsigned --w-all-positive
detect minmax per_tensor_act_unsigned_weight_unsigned --no-per-channel --w-all-positive
detect minmax per_channel_act_signed_weight_unsigned --no-all-positive --w-all-positive
detect minmax per_tensor_act_signed_weight_unsigned --no-per-channel --no-all-positive --w-all-positive
"

# 补充 2：mixed_quant 对照（7 后端 × 基线配置 + --mixed-quant）/
# Suppl. 2: mixed_quant ablation (7 backends)
MIXED_UNITS="
detect lsqplus_v1 mixed --mixed-quant
detect lsqplus_v2 mixed --mixed-quant
detect lsq_v1 mixed --mixed-quant
detect lsq_v2 mixed --mixed-quant
detect minmax mixed --mixed-quant
detect dorefa mixed --mixed-quant
detect pact mixed --mixed-quant
"

# 补充 3：PTQ 校准量敏感度（仅 PTQ；label 以 calib 开头的单元只跑 ptq）/
# Suppl. 3: calibration-size sensitivity (PTQ only; calib* labels run ptq only)
CALIB_UNITS="
detect lsqplus_v1 calib5 --calibration-batches 5 --run-tag calib5
detect lsqplus_v1 calib10 --calibration-batches 10 --run-tag calib10
detect lsqplus_v1 calib50 --calibration-batches 50 --run-tag calib50
"

# 补充 4：多种子稳定性（lsqplus_v1 × 4 配置 × seed=1）/
# Suppl. 4: multi-seed stability (lsqplus_v1 × 4 configs × seed=1)
SEED_UNITS="
detect lsqplus_v1 seed1_per_channel_act_unsigned --seed 1
detect lsqplus_v1 seed1_per_tensor_act_unsigned --seed 1 --no-per-channel
detect lsqplus_v1 seed1_per_channel_act_signed --seed 1 --no-all-positive
detect lsqplus_v1 seed1_per_tensor_act_signed --seed 1 --no-per-channel --no-all-positive
"

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

# 已完成阶段检查：匹配带日期的 [OK] 行（[START] 行不含 OK 前缀，不会误跳过失败单元）/
# Completed-stage check: anchor to dated [OK] lines ([START] lines lack the OK prefix,
# so failed units are never wrongly skipped).
already_ok() {
  grep -qE "^\[OK\] [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} task=$1 stage=$4 quant=$2 cfg=$3$" "$SUMLOG" 2>/dev/null
}

run_stage() {
  local task=$1 quant=$2 stage=$3 label=$4 logfile=$5
  shift 5
  local extra_args="$*"
  local quant_arg=""
  [ "$quant" != "-" ] && quant_arg="--quant ${quant}"

  if already_ok "$task" "$quant" "$label" "$stage"; then
    echo "[SKIP-DONE] task=${task} stage=${stage} quant=${quant} cfg=${label}"
    return 0
  fi

  echo "[START] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${label}"

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
    echo "[OK] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${label}" | tee -a "$SUMLOG"
  else
    echo "[FAIL] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${label} exit=${ec}"
    FAILED+=("${task}/${quant}/${label}/${stage}")
  fi
  return $ec
}

run_unit() {
  # $1 task $2 quant $3 label; remaining words = extra CLI args
  local task=$1 quant=$2 label=$3
  shift 3
  local extra_args="$*"
  [ "$extra_args" = "-" ] && extra_args=""

  local qb="$BATCH"
  # seg 任务分辨率 640、QAT 显存需求最高：batch=8 在 7.65GB 上必 OOM（lsqplus_v1/v2、
  # minmax、dorefa 实测撞顶），一律 batch=4；dorefa（tanh 激活）其余任务也 batch=4 /
  # seg (640px) has the highest QAT memory peak: batch=8 OOMs on 7.65GB (lsqplus_v1/v2,
  # minmax, dorefa all hit the ceiling), so seg always uses batch=4; dorefa on other tasks too.
  { [ "$task" = "seg" ] || [ "$quant" = "dorefa" ]; } && qb=4
  export QAT_BATCH_OVERRIDE=$qb

  if ! run_stage "$task" "$quant" "ptq" "$label" "$ROOT/log/${TAG}_${task}_${quant}_${label}_ptq.log" $extra_args; then
    unset QAT_BATCH_OVERRIDE
    echo "[SKIP-UNIT] ${task}/${quant}/${label} PTQ failed"
    return
  fi

  # calib* 单元只做 PTQ / calib* units run PTQ only
  if [[ "$label" == calib* ]]; then
    unset QAT_BATCH_OVERRIDE
    return 0
  fi

  if ! run_stage "$task" "$quant" "qat" "$label" "$ROOT/log/${TAG}_${task}_${quant}_${label}_qat.log" $extra_args; then
    unset QAT_BATCH_OVERRIDE
    echo "[SKIP-UNIT] ${task}/${quant}/${label} QAT failed"
    return
  fi
  unset QAT_BATCH_OVERRIDE
  run_stage "$task" "$quant" "compare" "$label" "$ROOT/log/${TAG}_${task}_${quant}_${label}_compare.log" $extra_args
}

FAILED=()

for task in detect seg pose; do
  float_best="$ROOT/model/yolo26-${task}/n/${FLOAT_BEST[$task]}"
  if [ ! -f "$float_best" ]; then
    echo "[FATAL] float best missing for ${task}: ${float_best}"
    exit 1
  fi
  echo "[REUSE] task=${task} float best exists"
done

run_block() {
  local block_name=$1 block=$2
  echo ""
  echo "########## BLOCK: ${block_name} ##########"
  while IFS= read -r line; do
    [ -z "${line// }" ] && continue
    read -r -a u <<< "$line"
    echo "=============================================================="
    echo "### UNIT: ${u[0]} | ${u[1]} | ${u[2]} (${u[*]:3})"
    echo "### QAT batch: $({ [ "${u[0]}" = "seg" ] || [ "${u[1]}" = "dorefa" ]; } && echo 4 || echo $BATCH)"
    echo "=============================================================="
    run_unit "${u[0]}" "${u[1]}" "${u[2]}" "${u[@]:3}"
  done <<< "$block"
}

run_block "main (seg + pose)" "$MAIN_UNITS"
run_block "weight_unsigned counterexamples" "$WU_UNITS"
run_block "mixed_quant ablation" "$MIXED_UNITS"
run_block "calibration-size sensitivity (PTQ only)" "$CALIB_UNITS"
run_block "multi-seed stability (seed=1)" "$SEED_UNITS"

echo ""
echo "========= REMAIN PART2 DONE (failures: ${#FAILED[@]}) ========="
[ ${#FAILED[@]} -gt 0 ] && printf '  - %s\n' "${FAILED[@]}"
