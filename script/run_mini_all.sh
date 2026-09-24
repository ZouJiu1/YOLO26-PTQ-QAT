#!/bin/bash
# 依次串行跑三个任务的 float40 → PTQ → QAT8
# 每个任务独立日志文件: mini_{det,seg,pose}.log
# 串行原因：共享 GPU + 共享 model/ 目录（每个任务会覆盖同名 checkpoint）

set -e

ROOT="/home/zoujiu/Desktop/projects/aLSQplus/QAT_training"
VENV_ACTIVATE="source /home/zoujiu/Desktop/projects/zoujiu/horizon/venv/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

run_task() {
  local name=$1     # detect / seg / pose
  local mini_yaml=$2
  cd "$ROOT"
  echo "=========================================="
  echo "[START] $(date '+%H:%M:%S') $name"
  echo "=========================================="
  bash -c "
    ${VENV_ACTIVATE}
    python -u networks_yolo26-${name}.py \
      --model yolo26n --stage all --quant lsqplus_v1 \
      --data ${mini_yaml} \
      --float-epochs 40 --qat-epochs 8 --calibration-batches 20 \
      --float-batch-size 8 --ptq-batch-size 8 --qat-batch-size 8 \
      --num-workers 0
  " > "mini_${name}.log" 2>&1
  local ec=$?
  echo "[END]   $(date '+%H:%M:%S') $name exit=$ec"
  return $ec
}

# ========== detect ==========
run_task detect ultralytics/ultralytics/data/datasets/coco/mini/coco_mini_detect.yaml

# ========== seg ==========
run_task seg   ultralytics/ultralytics/data/datasets/coco/mini/coco_mini_seg.yaml

# ========== pose ==========
run_task pose  ultralytics/ultralytics/data/datasets/coco/mini/coco_mini_pose.yaml

echo ""
echo "========= ALL DONE ========="
for f in mini_{detect,seg,pose}.log; do
  echo "--- $f ---"
  grep -E "Best|重载|完成|mAP|epoch" "$f" | tail -5
done
