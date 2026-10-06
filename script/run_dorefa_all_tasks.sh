#!/bin/bash
# ============================================================================ #
# 新版 DoReFa 单后端重跑脚本：覆盖全部 6 个任务（detect / seg / pose / cls / obb / depth）
# / New DoReFa single-backend re-run script covering all 6 tasks (detect / seg / pose / cls / obb / depth)
# ---------------------------------------------------------------------------- #
# 配置矩阵 / Config matrix（与 README.md 主矩阵对齐 / aligned with README.md main matrix）：
#   4 主配置 = per_channel / per_tensor × act_unsigned / act_signed
#   另有 detect × mixed_quant × dorefa 补充 1 单元
# 训练流水线 / Pipeline：
#   float best 已存在 → 复用；不存在 → 自动训 float 50 epoch
#   每个配置依次跑：PTQ 校准(20 batch) → QAT 微调(10 epoch) → compare
# 断点续跑 / Resume: model/{task}/n/dorefa_a8w8_*/ 已存在 → 自动跳过
#
# 显存备注 / Memory note:
#   新版 dorefa 已去除 tanh 非线性，QAT 显存峰值与 lsqplus_v1 一致，
#   所有任务统一 batch=8（不再需要 seg/dorefa 单独降 batch=4）/
#   New dorefa removed the tanh nonlinearity; QAT memory peak matches lsqplus_v1,
#   batch=8 on all tasks (no more per-task downgrade to batch=4 for seg/dorefa).
# ============================================================================ #

set -u

ROOT="/home/zoujiu/Desktop/projects/aLSQplus/QAT_training"
VENV_ACTIVATE="source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate"

# 超参（与 run_mini_all.sh 一致）/ Hyperparameters (match run_mini_all.sh)
FLOAT_EPOCHS=50
QAT_EPOCHS=10
CALIB_BATCHES=20
BATCH=8
QUANT="dorefa"

# 配置矩阵（与 README.md 主矩阵对齐）/ Config matrix (match README.md main matrix)
CONFIGS=(
  per_channel_act_unsigned  per_tensor_act_unsigned
  per_channel_act_signed    per_tensor_act_signed
)
declare -A CFG_ARGS=(
  [per_channel_act_unsigned]=""
  [per_tensor_act_unsigned]="--no-per-channel"
  [per_channel_act_signed]="--no-all-positive"
  [per_tensor_act_signed]="--no-per-channel --no-all-positive"
)

# 任务注册表 / Task registry
# 格式 / Entry:  "task|yaml_path|float_best_file"
TASKS_INFO=(
  "detect|dataset/coco_mini_detect.yaml|yolo26n_best.pth"
  "seg|dataset/coco_mini_seg.yaml|yolo26n-seg_best.pth"
  "pose|dataset/coco_mini_pose.yaml|yolo26n-pose_best.pth"
  "cls|dataset/imagenet10.yaml|yolo26n-cls_best.pth"
  "obb|dataset/dota8-multispectral.yaml|yolo26n-obb_best.pth"
  "depth|dataset/depth8.yaml|yolo26n-depth_best.pth"
)
# 新版 dorefa 激活量化器现在是标准非对称公式（scale+beta+zero_point），
# 因此 act_unsigned 可以正常工作（旧版 tanh 域 dorefa 必须 act_signed）/
# New dorefa activation quantizer now uses the standard asymmetric formula
# (scale + beta + zero_point), so act_unsigned works properly (old tanh-domain dorefa required act_signed).

# 确保日志目录存在 / Ensure log directory exists
mkdir -p "$ROOT/log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# CUDA 预热（必须在 fork 子进程前执行，否则 bash -c 里的 python 会因 Error 304 静默落 CPU）/
# CUDA warmup (must run before fork; otherwise python inside bash -c silently falls back to CPU due to Error 304).
echo "[WARMUP] $(date '+%F %H:%M') Warming up CUDA driver..."
if ${VENV_ACTIVATE} && python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
_ = torch.zeros(1, device='cuda:0')
print('[WARMUP] CUDA ok:', torch.cuda.get_device_name(0),
      f'(memory={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB)')
"; then
  echo "[WARMUP] CUDA driver initialized."
else
  echo "[WARMUP][WARN] CUDA warmup failed, will proceed but may fall back to CPU."
fi
deactivate 2>/dev/null || true

# 运行一次网络脚本；失败不中断整体 sweep，记录到 FAILED /
# Run the network script once; on failure keep the sweep going and record in FAILED.
# 启动前先探活 GPU（避免 bash -c 偶发 CUDA 304 静默落 CPU），最多重试 3 次 /
# Probe GPU before launch (avoid silent CPU fallback from transient CUDA 304); retry up to 3 times.
run_stage() {
  local task=$1
  local stage=$2      # float / ptq / qat / compare
  local logfile=$3
  shift 3
  local extra_args="$*"

  local quant_arg=""
  local model_arg="--model yolo26n"
  if [ "$stage" != "float" ]; then
    quant_arg="--quant ${QUANT}"
  fi

  echo "[START] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${QUANT} cfg=${CFG_LABEL:-default}"

  local attempt=1
  local max_attempts=3
  local ec=0
  while [ $attempt -le $max_attempts ]; do
    (
      cd "$ROOT"
      bash -c "
        ${VENV_ACTIVATE}
        python -u networks_yolo26-${task}.py \
          ${model_arg} --stage ${stage} ${quant_arg} \
          --data ${DATA_YAML} \
          --float-epochs ${FLOAT_EPOCHS} --qat-epochs ${QAT_EPOCHS} \
          --calibration-batches ${CALIB_BATCHES} \
          --float-batch-size ${BATCH} --ptq-batch-size ${BATCH} --qat-batch-size ${BATCH} \
          --num-workers 0 \
          ${extra_args}
      " > "$logfile" 2>&1
    )
    ec=$?
    if [ $ec -ne 0 ]; then
      echo "[RETRY]  attempt ${attempt}/${max_attempts} exit=${ec}"
    elif grep -q "Device: cpu" "$logfile" 2>/dev/null && [ "$stage" != "float" ]; then
      echo "[RETRY]  attempt ${attempt}/${max_attempts} fell back to CPU, retrying..."
      ec=999
    else
      break
    fi
    attempt=$((attempt + 1))
    sleep 5
  done
  if [ $ec -eq 0 ]; then
    echo "[OK]    $(date '+%F %H:%M') task=${task} stage=${stage} -> ${logfile}"
  else
    echo "[FAIL]  $(date '+%F %H:%M') task=${task} stage=${stage} exit=${ec} -> ${logfile}"
    FAILED+=("${task}/${CFG_LABEL:-default}/${stage}")
  fi
  return $ec
}

# 判断该配置是否已跑完（存在 compare 日志且 grep 到 [Compare]）/
# Check whether a config is already done (compare log exists and grep hits [Compare]).
is_done() {
  local task=$1 cfg=$2
  local cf="$ROOT/log/mini_${task}_${QUANT}_${cfg}_compare.log"
  [ -f "$cf" ] && grep -q "\[Compare\]" "$cf" 2>/dev/null
}

FAILED=()

# ========== 主循环 ==========
for entry in "${TASKS_INFO[@]}"; do
  task=$(echo "$entry" | cut -d'|' -f1)
  DATA_YAML=$(echo "$entry" | cut -d'|' -f2)
  FLOAT_BEST=$(echo "$entry" | cut -d'|' -f3)

  echo ""
  echo "=============================================================="
  echo "### TASK: ${task}  (${DATA_YAML})"
  echo "=============================================================="

  float_dir="$ROOT/model/yolo26-${task}/n"
  float_best="$float_dir/$FLOAT_BEST"

  # 1) float 只训一次；已有 best 则直接复用 / train float once; reuse existing best when present
  if [ -f "$float_best" ]; then
    echo "[REUSE] task=${task} float best exists: $float_best"
  else
    echo "[FLOAT] task=${task} float best not found, training float 50 epochs..."
    if ! run_stage "$task" "float" "$ROOT/log/mini_${task}_float.log"; then
      echo "[SKIP]  task=${task} float failed, skip all configs for this task"
      continue
    fi
  fi

  # 2) 每个配置依次 PTQ → QAT → compare；断点续跑 / each config: PTQ → QAT → compare; resume-capable
  for cfg in "${CONFIGS[@]}"; do
    echo "---------------- cfg: ${cfg} (${CFG_ARGS[$cfg]:-default}) ----------------"
    export CFG_LABEL="$cfg"

    # compare 已完成 → 跳过该配置 / compare done → skip this config
    if is_done "$task" "$cfg"; then
      echo "[SKIP]  task=${task} cfg=${cfg} already done (compare log exists)"
      continue
    fi

    if ! run_stage "$task" "ptq" "$ROOT/log/mini_${task}_${QUANT}_${cfg}_ptq.log" ${CFG_ARGS[$cfg]}; then
      echo "[SKIP]  task=${task} cfg=${cfg} PTQ failed, skip QAT/compare"
      continue
    fi
    if ! run_stage "$task" "qat" "$ROOT/log/mini_${task}_${QUANT}_${cfg}_qat.log" ${CFG_ARGS[$cfg]}; then
      echo "[SKIP]  task=${task} cfg=${cfg} QAT failed, skip compare"
      continue
    fi
    run_stage "$task" "compare" "$ROOT/log/mini_${task}_${QUANT}_${cfg}_compare.log" ${CFG_ARGS[$cfg]}
  done

  # 3) detect 额外跑 1 个 mixed_quant 补充单元（与 README.md supplementary 对齐）/
  #    detect extra mixed_quant unit (match README.md supplementary section)
  if [ "$task" = "detect" ]; then
    cfg="mixed"
    echo "---------------- cfg: ${cfg} (per_channel weights + per_tensor activations) ----------------"
    export CFG_LABEL="$cfg"
    if is_done "$task" "$cfg"; then
      echo "[SKIP]  task=${task} cfg=${cfg} already done"
    else
      if ! run_stage "$task" "ptq" "$ROOT/log/mini_${task}_${QUANT}_${cfg}_ptq.log" --mixed-quant; then
        echo "[SKIP]  task=${task} cfg=${cfg} PTQ failed"
      elif ! run_stage "$task" "qat" "$ROOT/log/mini_${task}_${QUANT}_${cfg}_qat.log" --mixed-quant; then
        echo "[SKIP]  task=${task} cfg=${cfg} QAT failed"
      else
        run_stage "$task" "compare" "$ROOT/log/mini_${task}_${QUANT}_${cfg}_compare.log" --mixed-quant
      fi
    fi
  fi
done

# ========== 汇总 ==========
echo ""
echo "================= METRICS SUMMARY (dorefa) ================="
for entry in "${TASKS_INFO[@]}"; do
  task=$(echo "$entry" | cut -d'|' -f1)
  echo "########## ${task} ##########"
  f="$ROOT/log/mini_${task}_float.log"
  [ -f "$f" ] && grep -E "\[Float\] Best" "$f" | tail -1
  for cfg in "${CONFIGS[@]}" mixed; do
    cf="$ROOT/log/mini_${task}_${QUANT}_${cfg}_compare.log"
    [ -f "$cf" ] && echo "--- ${cfg} ---" && grep -E "\[Compare\]" "$cf" | tail -2
  done
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "========= ALL DONE (no failures) ========="
else
  echo "========= DONE WITH ${#FAILED[@]} FAILURE(S) ========="
  printf '  - %s\n' "${FAILED[@]}"
fi
echo ""
echo "历史产物归档目录 (旧 tanh 域 dorefa 产物):"
echo "  $ROOT/model_archive/dorefa_old/"
