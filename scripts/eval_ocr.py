"""
OCR评估脚本：在验证集上统计PlateRecognizer的识别质量。

指标：整牌识别成功率、OCR平均置信度、检测平均置信度
按 val_split.json 中的 Base / 困难 子集分别报告。
"""
from __future__ import annotations
import os
import sys
import json
import yaml

# 将项目根目录加入路径，使 core/ 可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
from core.plate_recognizer import PlateRecognizer

# ── 读取配置 ──────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)

PRESET = _cfg['dataset']['preset']
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets', f'plate_dataset_{PRESET}')
VAL_IMG_DIR = os.path.join(DATASET_DIR, 'images', 'val')
VAL_SPLIT_JSON = os.path.join(DATASET_DIR, 'val_split.json')


def _get_weight_path() -> str:
    """从 config.yaml 中找第一个 enabled 模型的权重路径。"""
    for m in _cfg.get('models', []):
        if not m.get('enabled', True):
            continue
        path = os.path.join(
            PROJECT_ROOT, 'runs', PRESET,
            f'train_{m["name"]}_plate', 'weights', 'best.pt'
        )
        if os.path.exists(path):
            return path
    return ''


def _load_val_split() -> dict:
    """读取 val_split.json，返回 {filename: 'base'|'hard'}。"""
    if not os.path.exists(VAL_SPLIT_JSON):
        return {}
    with open(VAL_SPLIT_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def _is_failed(text: str) -> bool:
    """判断OCR输出是否为识别失败。"""
    return not text or text == '识别失败'


# ── 评估逻辑 ──────────────────────────────────────────────────────────────────
def _eval_subset(recognizer: PlateRecognizer, image_paths: list) -> dict:
    """
    对给定图片列表跑识别，返回统计结果字典。
    只统计每张图检测到的第一个车牌（最高置信度），无检测结果也计入失败。
    """
    total = len(image_paths)
    success = 0
    det_confs = []
    ocr_confs = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        _, plates = recognizer.recognize(img)

        if not plates:
            # 未检测到车牌，计为失败
            continue

        # 取检测置信度最高的一个
        best = max(plates, key=lambda p: p['det_conf'])
        det_confs.append(best['det_conf'])

        if not _is_failed(best['text']):
            success += 1
            ocr_confs.append(best['ocr_conf'])

    success_rate = success / total if total > 0 else 0.0
    avg_det_conf = sum(det_confs) / len(det_confs) if det_confs else 0.0
    avg_ocr_conf = sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.0

    return {
        'total':        total,
        'success':      success,
        'success_rate': success_rate,
        'avg_det_conf': avg_det_conf,
        'avg_ocr_conf': avg_ocr_conf,
    }


# ── 打印表格 ──────────────────────────────────────────────────────────────────
def _print_table(rows: list[tuple]):
    """
    rows: [(子集名, stats_dict), ...]
    """
    header = (
        f"{'子集':<8} {'总数':>6} {'成功数':>8} {'识别成功率':>11} "
        f"{'平均检测置信度':>15} {'平均OCR置信度':>14}"
    )
    sep = '-' * len(header)
    print('\n' + sep)
    print('  OCR 识别质量评估报告')
    print(sep)
    print(header)
    print(sep)
    for name, s in rows:
        print(
            f"{name:<8} "
            f"{s['total']:>6} "
            f"{s['success']:>8} "
            f"{s['success_rate']:>10.1%} "
            f"{s['avg_det_conf']:>15.4f} "
            f"{s['avg_ocr_conf']:>14.4f}"
        )
    print(sep + '\n')


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print('=' * 55)
    print('  OCR 识别质量评估（验证集）')
    print('=' * 55)

    weight_path = _get_weight_path()
    if not weight_path:
        print('❌ 找不到任何已训练的模型权重，请先完成训练。')
        print(f'   预期路径: runs/{PRESET}/train_<model>_plate/weights/best.pt')
        return

    print(f'  使用模型: {weight_path}')
    print(f'  数据集预设: {PRESET}')

    if not os.path.isdir(VAL_IMG_DIR):
        print(f'❌ 找不到验证集图片目录: {VAL_IMG_DIR}')
        return

    print('\n加载模型...')
    recognizer = PlateRecognizer(yolo_weight=weight_path)

    # 收集所有验证集图片
    all_images = sorted([
        os.path.join(VAL_IMG_DIR, f)
        for f in os.listdir(VAL_IMG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f'共找到 {len(all_images)} 张验证集图片\n')

    val_split = _load_val_split()

    if val_split:
        # 按 base / hard 分组
        base_imgs = [
            os.path.join(VAL_IMG_DIR, fn)
            for fn, tag in val_split.items()
            if tag == 'base' and os.path.exists(os.path.join(VAL_IMG_DIR, fn))
        ]
        hard_imgs = [
            os.path.join(VAL_IMG_DIR, fn)
            for fn, tag in val_split.items()
            if tag == 'hard' and os.path.exists(os.path.join(VAL_IMG_DIR, fn))
        ]

        print(f'  Base 子集: {len(base_imgs)} 张')
        print(f'  困难子集: {len(hard_imgs)} 张')

        rows = []
        if base_imgs:
            print('\n评估 Base 子集...')
            rows.append(('Base', _eval_subset(recognizer, base_imgs)))
        if hard_imgs:
            print('评估困难子集...')
            rows.append(('困难', _eval_subset(recognizer, hard_imgs)))

        # 合计
        if base_imgs or hard_imgs:
            combined_imgs = base_imgs + hard_imgs
            print('计算合计...')
            rows.append(('合计', _eval_subset(recognizer, combined_imgs)))

        _print_table(rows)
    else:
        print('⚠️  未找到 val_split.json，对全部验证集统一评估\n')
        print('评估全部验证集...')
        stats = _eval_subset(recognizer, all_images)
        _print_table([('全部', stats)])

    print('✅ 评估完成！')


if __name__ == '__main__':
    main()
