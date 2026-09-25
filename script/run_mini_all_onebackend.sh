#!/bin/bash
# 依次串行跑三个 COCO mini(1/100) 任务的 float50 → PTQ → QAT10（仅 lsqplus_v1 单后端；全后端横评用 run_mini_all.sh） /
# Run three COCO mini (1/100) tasks sequentially: float50 → PTQ → QAT10 (lsqplus_v1 only; for all backends use run_mini_all.sh)
# 每个任务独立日志文件: log/mini_{detect,seg,pose}.log / Separate log file per task in log/
# 串行原因：共享 GPU + 共享 model/ 目录（每个任务会覆盖同名 checkpoint） / Serial because shared GPU + shared model/ directory

set -e

ROOT="/home/zoujiu/Desktop/projects/aLSQplus/QAT_training"
VENV_ACTIVATE="source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate"

# 确保日志目录存在 / Ensure log directory exists
mkdir -p "$ROOT/log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

run_task() {
  local name=$1     # detect / seg / pose / obb / depth
  local mini_yaml=$2
  cd "$ROOT"
  echo "=========================================="
  echo "[START] $(date '+%F %H:%M') $name"
  echo "=========================================="
  bash -c "
    ${VENV_ACTIVATE}
    python -u networks_yolo26-${name}.py \
      --model yolo26n --stage all --quant lsqplus_v1 \
      --data ${mini_yaml} \
      --float-epochs 50 --qat-epochs 10 --calibration-batches 20 \
      --float-batch-size 8 --ptq-batch-size 8 --qat-batch-size 8 \
      --num-workers 0
  " > "log/mini_${name}.log" 2>&1
  local ec=$?
  echo "[END]   $(date '+%F %H:%M') $name exit=$ec"
  return $ec
}

# ========== detect ==========
run_task detect dataset/coco_mini_detect.yaml

# ========== seg ==========
run_task seg   dataset/coco_mini_seg.yaml

# ========== pose ==========
run_task pose  dataset/coco_mini_pose.yaml

# ========== obb ==========
# run_task obb   /home/zoujiu/Desktop/projects/aLSQplus/QAT_training/ultralytics/ultralytics/cfg/datasets/dota8-multispectral.yaml

# ========== depth ==========
# run_task depth /home/zoujiu/Desktop/projects/aLSQplus/QAT_training/ultralytics/ultralytics/cfg/datasets/depth8.yaml

echo ""
echo "========= ALL DONE ========="
for f in log/mini_{detect,seg,pose}.log; do
# for f in log/mini_{detect,seg,pose,obb,depth}.log; do
  echo "--- $f ---"
  # 提取各阶段最终指标：Float Best / PTQ / QAT Best / Compare(P、R、mAP50、mAP50-95)
  # Extract final metrics per stage: Float Best / PTQ / QAT Best / Compare (P, R, mAP50, mAP50-95)
  grep -E "\[Float\] Best|\[PTQ.*mAP|\[QAT-.*Best epoch|\[Compare\]" "$f" | tail -12
done
