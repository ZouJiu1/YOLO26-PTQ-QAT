#!/bin/bash
# ============================================================================ #
# COCO mini(1/100) 三任务 × 全 7 个量化后端 串跑 / Three tasks × all 7 quant backends on COCO mini (1/100)
# ---------------------------------------------------------------------------- #
# 每个任务 / Per task:
#   1) float 只训练 1 次（50 epoch），产物被所有后端复用 / float trained ONCE (50 epochs), shared by all backends
#   2) 遍历 7 个量化后端，各跑 PTQ 校准 → QAT(10 epoch = 50*0.2) → float/QAT 对比 /
#      loop 7 quant backends, each: PTQ calibration → QAT (10 epochs = 50*0.2) → float/QAT compare
# 量化后端 / Backends:
#   lsqplus_v1 lsqplus_v2 lsq_v1 lsq_v2 minmax dorefa pact
# 日志 / Logs:
#   log/mini_{task}_float.log
#   log/mini_{task}_{backend}_{ptq,qat,compare}.log
# 串行原因：共享 GPU（RTX 4060 8GB） / Serialized because of one shared GPU (RTX 4060 8GB)
# ============================================================================ #

set -u

ROOT="/home/zoujiu/Desktop/projects/aLSQplus/QAT_training"
VENV_ACTIVATE="source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate"

# 超参 / Hyperparameters
FLOAT_EPOCHS=50
QAT_EPOCHS=10           # = float 50 * 0.2
CALIB_BATCHES=20
BATCH=8

# 全部 7 个量化后端 / All 7 quantization backends
QUANTS=(lsqplus_v1 lsqplus_v2 lsq_v1 lsq_v2 minmax dorefa pact)

# 任务 -> 数据集 yaml / task -> dataset yaml
TASKS=(detect seg pose)
declare -A DATA_YAML=(
  [detect]="dataset/coco_mini_detect.yaml"
  [seg]="dataset/coco_mini_seg.yaml"
  [pose]="dataset/coco_mini_pose.yaml"
)

# 确保日志目录存在 / Ensure log directory exists
mkdir -p "$ROOT/log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# 运行一次网络脚本；失败不中断整体 sweep，记录到 FAILED 列表 /
# Run the network script once; on failure keep the sweep going and record the stage in FAILED.
run_stage() {
  local task=$1
  local quant=$2      # 量化后端，float 阶段传 "-" / backend name, "-" for float stage
  local stage=$3      # float / ptq / qat / compare
  local logfile=$4
  shift 4
  local extra_args="$*"

  local quant_arg=""
  if [ "$quant" != "-" ]; then
    quant_arg="--quant ${quant}"
  fi

  echo "[START] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant}"
  (
    cd "$ROOT"
    bash -c "
      ${VENV_ACTIVATE}
      python -u networks_yolo26-${task}.py \
        --model yolo26n --stage ${stage} ${quant_arg} \
        --data ${DATA_YAML[$task]} \
        --float-epochs ${FLOAT_EPOCHS} --qat-epochs ${QAT_EPOCHS} \
        --calibration-batches ${CALIB_BATCHES} \
        --float-batch-size ${BATCH} --ptq-batch-size ${BATCH} --qat-batch-size ${BATCH} \
        --num-workers 0 \
        ${extra_args}
    " > "$logfile" 2>&1
  )
  local ec=$?
  if [ $ec -eq 0 ]; then
    echo "[OK]    $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} -> ${logfile}"
  else
    echo "[FAIL]  $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} exit=${ec} -> ${logfile}"
    FAILED+=("${task}/${quant}/${stage}")
  fi
  return $ec
}

FAILED=()

for task in "${TASKS[@]}"; do
  echo "=============================================================="
  echo "### TASK: ${task}  (${DATA_YAML[$task]})"
  echo "=============================================================="

  # 1) float 只训一次 / float training once
  if ! run_stage "$task" "-" "float" "$ROOT/log/mini_${task}_float.log"; then
    # float 失败则跳过该任务所有后端（后端都依赖 float best 权重）/
    # If float fails, skip all backends for this task (they depend on float best weights)
    echo "[SKIP]  task=${task} float 失败，跳过全部后端 / float failed, skip all backends"
    continue
  fi

  # 2) 每个量化后端 PTQ → QAT → compare；前一阶段失败则跳过后续阶段 /
  #    per backend PTQ → QAT → compare; if one stage fails, skip the remaining stages
  for q in "${QUANTS[@]}"; do
    echo "---------------- backend: ${q} ----------------"
    if ! run_stage "$task" "$q" "ptq" "$ROOT/log/mini_${task}_${q}_ptq.log"; then
      echo "[SKIP]  task=${task} backend=${q} PTQ 失败，跳过 QAT/compare / PTQ failed, skip QAT/compare"
      continue
    fi
    if ! run_stage "$task" "$q" "qat" "$ROOT/log/mini_${task}_${q}_qat.log"; then
      echo "[SKIP]  task=${task} backend=${q} QAT 失败，跳过 compare / QAT failed, skip compare"
      continue
    fi
    run_stage "$task" "$q" "compare" "$ROOT/log/mini_${task}_${q}_compare.log"
  done
done

# ============================================================================ #
# 汇总：从日志提取各阶段最终指标 / Summary: extract final metrics from logs
# ============================================================================ #
echo ""
echo "================= METRICS SUMMARY ================="
for task in "${TASKS[@]}"; do
  echo "########## ${task} ##########"
  f="$ROOT/log/mini_${task}_float.log"
  if [ -f "$f" ]; then
    echo "--- float ---"
    grep -E "\[Float\] Best" "$f" | tail -1
  fi
  for q in "${QUANTS[@]}"; do
    echo "--- ${q} ---"
    pf="$ROOT/log/mini_${task}_${q}_ptq.log"
    [ -f "$pf" ] && grep -E "\[PTQ.*重载|\[PTQ.*mAP" "$pf" | tail -2
    qf="$ROOT/log/mini_${task}_${q}_qat.log"
    [ -f "$qf" ] && grep -E "\[QAT-.*Best epoch" "$qf" | tail -1
    cf="$ROOT/log/mini_${task}_${q}_compare.log"
    [ -f "$cf" ] && grep -E "\[Compare\]" "$cf" | tail -3
  done
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "========= ALL DONE (no failures) ========="
else
  echo "========= DONE WITH ${#FAILED[@]} FAILURE(S) ========="
  printf '  - %s\n' "${FAILED[@]}"
fi
