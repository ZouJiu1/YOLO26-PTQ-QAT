#!/usr/bin/env python3
"""
1/20 COCO Mini YOLO26 3 任务 GPU 串跑
detect → seg → pose，各 float40 → PTQ → QAT8 → Compare
GPU 必需（RTX 4060 Laptop 8GB），无需手动指定 device

用法: python run_mini_pipeline.py
结果: mini_{detect,seg,pose}.log + model/yolo26n*.pth
"""
import os, sys, time, importlib.util, traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
MINI = os.path.join(ROOT, 'ultralytics', 'ultralytics', 'data', 'datasets', 'coco', 'mini')
os.chdir(ROOT)
sys.path.insert(0, ROOT)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def run_task(task):
    file_tag = {'detect':'detect', 'seg':'seg', 'pose':'pose'}[task]
    log_file = os.path.join(ROOT, f'mini_{task}.log')
    out_f = open(log_file, 'w', buffering=1)
    t0 = time.time()
    def log(s=''):
        ts = time.strftime('%H:%M:%S')
        print(f'[{ts}] [{task.upper()}] {s}', flush=True)
        out_f.write(s + '\n'); out_f.flush()

    log(f'===== {task.upper()} START =====')
    try:
        mod = load(task, os.path.join(ROOT, f'networks_yolo26-{file_tag}.py'))
    except Exception as e:
        log(f'IMPORT FAIL: {e}'); traceback.print_exc(); return

    yaml_path = os.path.join(MINI, f'coco_mini_{task}.yaml')
    mod.COCO128_YAML = yaml_path
    try:
        _tmp = mod.check_det_dataset(yaml_path)
        mod.NUM_CLASSES = len(_tmp['names'])
    except Exception as e:
        log(f'WARN NUM_CLASSES default 80: {e}')

    # GPU 强制检查
    import torch
    if not torch.cuda.is_available():
        log(f'ERROR: CUDA 不可用! is_available={torch.cuda.is_available()}')
        log(f'请确认: nvidia-smi 有输出, torch 是 CUDA 版 (pip list | grep torch)')
        out_f.close(); return
    log(f'device={mod.device} (GPU: {torch.cuda.get_device_name(0)}) nc={mod.NUM_CLASSES}')

    steps = [
        ('FLOAT40', lambda: mod.float_train(
            batch_size=16, epochs=40, num_workers=0,
            num_classes=mod.NUM_CLASSES, scale='n')),
        ('PTQ', lambda: mod.PTQ_calibration(
            quant_method='lsqplus_v1', batch_size=8, num_workers=0,
            calibration_batches=32, num_classes=mod.NUM_CLASSES, scale='n')),
        ('QAT8', lambda: mod.QAT_training(
            quant_method='lsqplus_v1', batch_size=16, epochs=8, num_workers=0,
            num_classes=mod.NUM_CLASSES, scale='n')),
        ('COMPARE', lambda: mod.compare_precision(
            quant_method='lsqplus_v1', scale='n')),
    ]

    for stage_name, stage_fn in steps:
        s0 = time.time()
        log(f'--- {stage_name} ---')
        try:
            stage_fn()
            log(f'--- {stage_name} OK ({time.time()-s0:.0f}s) ---')
        except Exception as e:
            log(f'--- {stage_name} FAIL: {e} ---')
            traceback.print_exc()
            break

    elapsed = time.time() - t0
    log(f'===== {task.upper()} DONE ({elapsed:.0f}s = {elapsed/60:.1f}min) =====')
    out_f.close()

ALL_START = time.time()
for task in ['detect', 'seg', 'pose']:
    run_task(task)

total = time.time() - ALL_START
print(f'\n===== ALL DONE total={total:.0f}s ({total/60:.1f}min) =====', flush=True)
print('\n========= SUMMARY (从 checkpoint 读) =========', flush=True)
import torch
for t in ['detect', 'seg', 'pose']:
    suffix = '' if t == 'detect' else f'-{t}'
    for label, ckpt in [
        ('FLOAT_BEST',  f'model/yolo26n{suffix}.pth'),
        ('QAT_BEST',    f'model/qat_lsqplus_v1_yolo26n{suffix}_best.pth'),
    ]:
        p = os.path.join(ROOT, ckpt)
        if os.path.exists(p):
            sd = torch.load(p, map_location='cpu', weights_only=False)
            m = sd.get('meta', sd).get('meta', sd.get('meta', {}))
            ep = m.get('epoch', '?'); m50 = m.get('map50', m.get('pose_map50', m.get('mask_map50', '?')))
            m95 = m.get('map', m.get('pose_map', m.get('mask_map', '?')))
            print(f'  {t}/{label}: ep={ep} mAP50={m50} mAP50-95={m95}', flush=True)
