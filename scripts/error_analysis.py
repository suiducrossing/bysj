"""
错误案例分析脚本：收集验证集中识别失败的图片，保存典型失败样本。

失败标准：OCR输出为空/识别失败，或OCR置信度 < 0.5。
根据文件名前缀判断CCPD子集类型：
  blur → 模糊, db → 暗光, tilt → 倾斜, rotate → 旋转, base → 正常
保存前20张典型失败样本到 runs/{preset}/error_cases/。
"""
from __future__ import annotations
import os
import sys
import json
import yaml

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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'runs', PRESET, 'error_cases')

# CCPD 子集名称映射（文件名前缀 → 中文描述）
CCPD_TYPE_MAP = {
    'blur':   '模糊',
    'db':     '暗光',
    'tilt':   '倾斜',
    'rotate': '旋转',
    'base':   '正常',
}

OCR_CONF_THRESHOLD = 0.5
MAX_SAVE = 20


def _get_weight_path() -> str:
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
    if not os.path.exists(VAL_SPLIT_JSON):
        return {}
    with open(VAL_SPLIT_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_ccpd_type(filename: str) -> str:
    """从文件名推断CCPD子集类型。"""
    name = os.path.splitext(filename)[0].lower()
    for prefix, label in CCPD_TYPE_MAP.items():
        if name.startswith(prefix):
            return label
    # 按 val_split.json 标签回退
    return '未知'


def _is_failed(plates: list) -> tuple[bool, str, float, float]:
    """
    判断识别是否失败，返回 (is_fail, ocr_text, det_conf, ocr_conf)。
    无检测结果 或 OCR为空/失败 或 OCR置信度低 均视为失败。
    """
    if not plates:
        return True, '', 0.0, 0.0
    best = max(plates, key=lambda p: p['det_conf'])
    text = best['text']
    det_conf = best['det_conf']
    ocr_conf = best['ocr_conf']
    if not text or text == '识别失败' or ocr_conf < OCR_CONF_THRESHOLD:
        return True, text, det_conf, ocr_conf
    return False, text, det_conf, ocr_conf


def _draw_error_info(result_bgr, text: str, det_conf: float, ocr_conf: float,
                     fail_reason: str):
    """在图片左上角叠加失败原因标注。"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    img_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = None
    for fp in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
               'C:/Windows/Fonts/simsun.ttc']:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 18)
            except Exception:
                pass
            break
    if font is None:
        font = ImageFont.load_default()

    lines = [
        f'失败原因: {fail_reason}',
        f'OCR输出: {text or "(无)"}',
        f'检测置信度: {det_conf:.2f}',
        f'OCR置信度: {ocr_conf:.2f}',
    ]
    y = 4
    for line in lines:
        draw.text((4, y), line, font=font, fill=(255, 60, 60))
        y += 22

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print('=' * 55)
    print('  错误案例分析')
    print('=' * 55)

    weight_path = _get_weight_path()
    if not weight_path:
        print('❌ 找不到任何已训练的模型权重，请先完成训练。')
        return

    if not os.path.isdir(VAL_IMG_DIR):
        print(f'❌ 找不到验证集图片目录: {VAL_IMG_DIR}')
        return

    print(f'  使用模型: {weight_path}')
    print(f'  数据集预设: {PRESET}')
    print(f'  置信度失败阈值: < {OCR_CONF_THRESHOLD}')

    print('\n加载模型...')
    recognizer = PlateRecognizer(yolo_weight=weight_path)

    val_split = _load_val_split()
    all_images = sorted([
        f for f in os.listdir(VAL_IMG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f'共找到 {len(all_images)} 张验证集图片\n')

    # 按类型统计失败数量
    fail_counts: dict[str, int] = {v: 0 for v in CCPD_TYPE_MAP.values()}
    fail_counts['未知'] = 0
    total_fail = 0
    saved = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in all_images:
        img_path = os.path.join(VAL_IMG_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        result_bgr, plates = recognizer.recognize(img)
        is_fail, text, det_conf, ocr_conf = _is_failed(plates)

        if not is_fail:
            continue

        # 确定类型
        split_tag = val_split.get(fname, '')
        if split_tag == 'hard':
            # hard 样本需进一步从文件名判断具体子集
            ccpd_type = _get_ccpd_type(fname)
            if ccpd_type == '未知':
                ccpd_type = '困难(未知)'
        else:
            ccpd_type = _get_ccpd_type(fname)
            if ccpd_type == '未知' and split_tag == 'base':
                ccpd_type = '正常'

        # 统计
        if ccpd_type not in fail_counts:
            fail_counts[ccpd_type] = 0
        fail_counts[ccpd_type] += 1
        total_fail += 1

        # 保存前 MAX_SAVE 张典型失败样本
        if saved < MAX_SAVE:
            annotated = _draw_error_info(result_bgr, text, det_conf, ocr_conf, ccpd_type)
            save_name = f'{saved + 1:03d}_{ccpd_type}_{fname}'
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), annotated)
            saved += 1

    # ── 打印统计表格 ──────────────────────────────────────────────────────────
    total_imgs = len(all_images)
    header = f"{'类别':<12} {'失败数':>8} {'占失败比':>10} {'占总集比':>10}"
    sep = '-' * len(header)
    print('\n' + sep)
    print('  各类别失败统计')
    print(sep)
    print(header)
    print(sep)

    for type_name, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
        if count == 0:
            continue
        fail_ratio = count / total_fail if total_fail > 0 else 0.0
        total_ratio = count / total_imgs if total_imgs > 0 else 0.0
        print(
            f'{type_name:<12} '
            f'{count:>8} '
            f'{fail_ratio:>9.1%} '
            f'{total_ratio:>9.1%}'
        )

    print(sep)
    print(
        f"{'合计':<12} "
        f"{total_fail:>8} "
        f"{'100.0%':>10} "
        f"{total_fail / total_imgs:.1%}" if total_imgs > 0 else f"{'合计':<12} {total_fail:>8}"
    )
    print(sep + '\n')

    print(f'已保存 {saved} 张典型失败样本至: {OUTPUT_DIR}')
    print('✅ 错误案例分析完成！')


if __name__ == '__main__':
    main()
