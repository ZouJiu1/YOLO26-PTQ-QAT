#!/bin/bash
# ============================================================================ #
# COCO mini(1/100) 三任务 × 全 7 个量化后端 串跑 / Three tasks × all 7 quant backends on COCO mini (1/100)
# ---------------------------------------------------------------------------- #
# 每个任务 / Per task:
#   1) float 只训练 1 次（50 epoch），产物被所有后端复用 / float trained ONCE (50 epochs), shared by all backends
#   2) 遍历 7 个量化后端 × 8 种配置（per_channel × act_ap × w_ap），各跑 PTQ 校准 → QAT(10 epoch = 50*0.2) → float/QAT 对比 /
#      loop 7 quant backends × 8 configs, each: PTQ calibration → QAT (10 epochs = 50*0.2) → float/QAT compare
# 量化后端 / Backends:
#   lsqplus_v1 lsqplus_v2 lsq_v1 lsq_v2 minmax dorefa pact
# 日志 / Logs:
#   log/mini_{task}_float.log
#   log/mini_{task}_{backend}_{config}_{stage}.log（8 个配置全部显式命名 / all 8 configs explicitly named）
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

# 量化配置矩阵（int8；per_channel × 激活 all_positive × 权重 w_all_positive）/
# Quant config matrix (int8; per_channel × activation all_positive × weight w_all_positive)
# 推荐基线 / Recommended baseline:
#   per_channel_act_unsigned = per-channel + 激活无符号(SiLU 后非负) + 权重有符号（代码默认）/
#                              per-channel + unsigned activations + signed weights (code default)
# 对照维度 / Ablation axes:
#   per_tensor = per-tensor 权重 / per-tensor weights
#   act_signed = 激活有符号（--no-all-positive）/ signed activations
#   weight_unsigned = 权重无符号（--w-all-positive；可能破坏有符号权重的符号平衡，仅对照）/
#                     unsigned weights (may break sign balance of signed weights; ablation only)
CONFIGS=(
  per_channel_act_unsigned  per_tensor_act_unsigned
  per_channel_act_signed    per_tensor_act_signed
  per_channel_act_unsigned_weight_unsigned  per_tensor_act_unsigned_weight_unsigned
  per_channel_act_signed_weight_unsigned    per_tensor_act_signed_weight_unsigned
)
declare -A CFG_ARGS=(
  [per_channel_act_unsigned]=""
  [per_tensor_act_unsigned]="--no-per-channel"
  [per_channel_act_signed]="--no-all-positive"
  [per_tensor_act_signed]="--no-per-channel --no-all-positive"
  [per_channel_act_unsigned_weight_unsigned]="--w-all-positive"
  [per_tensor_act_unsigned_weight_unsigned]="--no-per-channel --w-all-positive"
  [per_channel_act_signed_weight_unsigned]="--no-all-positive --w-all-positive"
  [per_tensor_act_signed_weight_unsigned]="--no-per-channel --no-all-positive --w-all-positive"
)

# 任务 -> 数据集 yaml / task -> dataset yaml
TASKS=(detect seg pose)
declare -A DATA_YAML=(
  [detect]="dataset/coco_mini_detect.yaml"
  [seg]="dataset/coco_mini_seg.yaml"
  [pose]="dataset/coco_mini_pose.yaml"
)
# float best 文件名（存在则复用，不再重复 50 epoch；删除该文件即可强制重训）/
# float best checkpoint names (reused when present to skip 50 epochs; delete to force retraining).
declare -A FLOAT_BEST=(
  [detect]="yolo26n_best.pth"
  [seg]="yolo26n-seg_best.pth"
  [pose]="yolo26n-pose_best.pth"
)

# 确保日志目录存在 / Ensure log directory exists
mkdir -p "$ROOT/log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# CUDA 预热（必须在 fork 子进程前执行，否则 bash -c 里的 python 会因 Error 304 静默落 CPU）/
# CUDA warmup (must run before any fork; otherwise python inside bash -c silently falls back to CPU due to Error 304).
echo "[WARMUP] $(date '+%F %H:%M') Warming up CUDA driver..."
if source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate && python -c '
import torch
assert torch.cuda.is_available(), "CUDA not available!"
_ = torch.zeros(1, device="cuda:0")
print(f"[WARMUP] CUDA ok: {torch.cuda.get_device_name(0)} (memory={torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB)")
'; then
  echo "[WARMUP] CUDA driver initialized successfully."
else
  echo "[WARMUP][WARN] CUDA warmup failed, will proceed but may fall back to CPU."
fi
deactivate 2>/dev/null || true

# 运行一次网络脚本；失败不中断整体 sweep，记录到 FAILED 列表 /
# Run the network script once; on failure keep the sweep going and record the stage in FAILED.
# 启动前先探活 GPU（避免后台 shell 偶发 CUDA 304 静默落 CPU），最多重试 3 次 /
# Probe GPU before launch (avoid silent CPU fallback from transient CUDA 304); retry up to 3 times.
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

  echo "[START] $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${CFG_LABEL:-default}"

  local attempt=1
  local max_attempts=3
  local ec=0
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
    # 如果是 CUDA 304 导致落 CPU，重试一次 / retry if it fell back to CPU
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
    echo "[OK]    $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${CFG_LABEL:-default} -> ${logfile}"
  else
    echo "[FAIL]  $(date '+%F %H:%M') task=${task} stage=${stage} quant=${quant} cfg=${CFG_LABEL:-default} exit=${ec} -> ${logfile}"
    FAILED+=("${task}/${quant}/${CFG_LABEL:-default}/${stage}")
  fi
  return $ec
}

FAILED=()

for task in "${TASKS[@]}"; do
  echo "=============================================================="
  echo "### TASK: ${task}  (${DATA_YAML[$task]})"
  echo "=============================================================="

  # 1) float 只训一次；已有 best 则直接复用 / train float once; reuse existing best when present
  float_best="$ROOT/model/yolo26-${task}/n/${FLOAT_BEST[$task]}"
  if [ -f "$float_best" ]; then
    echo "[REUSE] task=${task} 已存在 float best，跳过 float 训练 / reuse existing float best: ${float_best}"
  elif ! run_stage "$task" "-" "float" "$ROOT/log/mini_${task}_float.log"; then
    # float 失败则跳过该任务所有后端（后端都依赖 float best 权重）/
    # If float fails, skip all backends for this task (they depend on float best weights)
    echo "[SKIP]  task=${task} float 失败，跳过全部后端 / float failed, skip all backends"
    continue
  fi

  # 2) 每个量化后端按 8 种配置各跑 PTQ → QAT → compare；前一阶段失败则跳过后续阶段 /
  #    per backend × per config: PTQ → QAT → compare; if one stage fails, skip the rest.
  #    日志名 / Log naming: 所有配置均显式命名 mini_{task}_{q}_{cfg}_{stage}.log；
  #    模型目录由网络脚本按量化参数标签隔离，如 <q>_a8w8_per_channel_act_unsigned/。
  #    新版 dorefa 已去除 tanh 非线性（线性网格），QAT 显存峰值与其他后端一致，无需降 batch /
  #    New dorefa removed the tanh nonlinearity (linear grid); its QAT memory peak matches other
  #    backends, so no batch downgrade is needed.
  #    注：pact 后端的权重量级化默认走 lsqplus_v1（pact_w_quant，见 QUANT_CFG）/
  #    Note: the pact backend quantizes weights with lsqplus_v1 by default (pact_w_quant in QUANT_CFG).
  for q in "${QUANTS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      echo "---------------- backend: ${q} | config: ${cfg} (${CFG_ARGS[$cfg]:-code default}) ----------------"
      unset QAT_BATCH_OVERRIDE
      export CFG_LABEL="$cfg"
      if ! run_stage "$task" "$q" "ptq" "$ROOT/log/mini_${task}_${q}_${cfg}_ptq.log" ${CFG_ARGS[$cfg]}; then
        echo "[SKIP]  task=${task} backend=${q} cfg=${cfg} PTQ 失败，跳过 QAT/compare / PTQ failed, skip QAT/compare"
        continue
      fi
      if ! run_stage "$task" "$q" "qat" "$ROOT/log/mini_${task}_${q}_${cfg}_qat.log" ${CFG_ARGS[$cfg]}; then
        echo "[SKIP]  task=${task} backend=${q} cfg=${cfg} QAT 失败，跳过 compare / QAT failed, skip compare"
        continue
      fi
      # compare 不用 QAT 降批 / compare keeps batch=8
      unset QAT_BATCH_OVERRIDE
      run_stage "$task" "$q" "compare" "$ROOT/log/mini_${task}_${q}_${cfg}_compare.log" ${CFG_ARGS[$cfg]}
    done
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
    for cfg in "${CONFIGS[@]}"; do
      echo "--- ${q} [${cfg}] ---"
      pf="$ROOT/log/mini_${task}_${q}_${cfg}_ptq.log"
      [ -f "$pf" ] && grep -E "\[PTQ.*重载|\[PTQ.*mAP" "$pf" | tail -2
      qf="$ROOT/log/mini_${task}_${q}_${cfg}_qat.log"
      [ -f "$qf" ] && grep -E "\[QAT-.*Best epoch" "$qf" | tail -1
      cf="$ROOT/log/mini_${task}_${q}_${cfg}_compare.log"
      [ -f "$cf" ] && grep -E "\[Compare\]" "$cf" | tail -3
    done
  done
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "========= ALL DONE (no failures) ========="
else
  echo "========= DONE WITH ${#FAILED[@]} FAILURE(S) ========="
  printf '  - %s\n' "${FAILED[@]}"
fi
