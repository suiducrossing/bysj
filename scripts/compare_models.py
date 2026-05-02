"""
三模型对比评估脚本：YOLOv8 / YOLOv10 / YOLOv11
在验证集上评估各模型的 mAP、精度、召回率和推理速度，并生成对比图表。
支持分集评估（Base vs 困难样本），需要 val_split.json。
"""
from __future__ import annotations
import os
import json
import shutil
import tempfile
import time
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from ultralytics import YOLO

matplotlib.rcParams['axes.unicode_minus'] = False

# ── 中文字体配置 ──────────────────────────────────────────────────────────────
def _setup_chinese_font():
    """尝试设置中文字体，优先使用系统内置字体。"""
    candidates = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.family'] = font
            return font
    return None


# ── 模型配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)

PRESET = _cfg['dataset']['preset']
DATASET_YAML = os.path.join(PROJECT_ROOT, 'datasets', f'plate_dataset_{PRESET}', 'dataset.yaml')
VAL_SPLIT_JSON = os.path.join(PROJECT_ROOT, 'datasets', f'plate_dataset_{PRESET}', 'val_split.json')

MODEL_CONFIGS = []
for m in _cfg.get('models', []):
    if not m.get('enabled', True):
        continue
    weight_path = os.path.join(
        PROJECT_ROOT, 'runs', PRESET, f'train_{m["name"]}_plate', 'weights', 'best.pt'
    )
    MODEL_CONFIGS.append({
        'label': m['name'].upper().replace('YOLOV', 'YOLOv'),
        'weight': weight_path,
    })

OUTPUT_CHART = os.path.join(PROJECT_ROOT, 'runs', PRESET, 'comparison_chart.png')


# ── 推理速度测量 ──────────────────────────────────────────────────────────────
def _measure_fps(model: YOLO, val_images: list, n_warmup: int = 5, n_measure: int = 50) -> float:
    images = val_images[:max(n_warmup + n_measure, len(val_images))]
    if not images:
        return 0.0
    for img in images[:n_warmup]:
        model.predict(img, verbose=False)
    measure_imgs = images[n_warmup:n_warmup + n_measure]
    if not measure_imgs:
        measure_imgs = images[:n_measure]
    t0 = time.perf_counter()
    for img in measure_imgs:
        model.predict(img, verbose=False)
    elapsed = time.perf_counter() - t0
    return len(measure_imgs) / elapsed if elapsed > 0 else 0.0


def _get_val_images(dataset_yaml: str, limit: int = 60) -> list:
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    dataset_dir = os.path.dirname(dataset_yaml)
    val_path = cfg.get('val', '')
    if not os.path.isabs(val_path):
        val_path = os.path.join(dataset_dir, val_path)
    images = []
    if os.path.isdir(val_path):
        for fname in os.listdir(val_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append(os.path.join(val_path, fname))
    return images[:limit]


def _count_params(model: YOLO) -> float:
    try:
        total = sum(p.numel() for p in model.model.parameters())
        return total / 1e6
    except Exception:
        return float('nan')


# ── 单模型评估 ────────────────────────────────────────────────────────────────
def _evaluate_model(config: dict, val_images: list, data_yaml: str = None) -> dict | None:
    label = config['label']
    weight = config['weight']

    if not os.path.exists(weight):
        print(f"  ⚠️  {label} 权重文件不存在，跳过: {weight}")
        return None

    print(f"  加载 {label} 模型: {weight}")
    model = YOLO(weight)

    data = data_yaml or DATASET_YAML
    print(f"  评估 {label} 验证集指标...")
    metrics = model.val(data=data, verbose=False)

    map50     = float(metrics.box.map50)
    map50_95  = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall    = float(metrics.box.mr)

    print(f"  测量 {label} 推理速度...")
    fps = _measure_fps(model, val_images)
    params = _count_params(model)

    return {
        'label':      label,
        'map50':      map50,
        'map50_95':   map50_95,
        'precision':  precision,
        'recall':     recall,
        'fps':        fps,
        'params_m':   params,
    }


# ── 分集评估 ──────────────────────────────────────────────────────────────────
def _load_val_split():
    """读取 val_split.json，返回 {filename: 'base'|'hard'}。"""
    if not os.path.exists(VAL_SPLIT_JSON):
        return None
    with open(VAL_SPLIT_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def _create_split_yamls(val_split: dict) -> tuple:
    """创建 Base 和 Hard 的临时 data.yaml，返回 (base_yaml, hard_yaml, tmpdir)。"""
    dataset_dir = os.path.dirname(DATASET_YAML)
    val_dir = os.path.join(dataset_dir, 'images', 'val')
    labels_dir = os.path.join(dataset_dir, 'labels', 'val')

    tmpdir = tempfile.mkdtemp(prefix='bysj_split_')

    splits = {'base': [], 'hard': []}
    for fname, origin in val_split.items():
        if origin in splits:
            splits[origin].append(fname)

    yaml_paths = {}
    for subset_name, filelist in splits.items():
        if not filelist:
            print(f"  ⚠️  {subset_name} 子集没有图片，跳过")
            yaml_paths[subset_name] = None
            continue

        # 使用标准 YOLO 目录结构：subset/images/ 和 subset/labels/
        subset_dir = os.path.join(tmpdir, subset_name)
        img_dir = os.path.join(subset_dir, 'images')
        lbl_dir = os.path.join(subset_dir, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for fname in filelist:
            src_img = os.path.join(val_dir, fname)
            src_lbl = os.path.join(labels_dir, fname.replace('.jpg', '.txt'))
            if os.path.exists(src_img):
                os.symlink(src_img, os.path.join(img_dir, fname))
            if os.path.exists(src_lbl):
                os.symlink(src_lbl, os.path.join(lbl_dir, fname.replace('.jpg', '.txt')))

        # 生成临时 yaml，val: images → 标签自动找 labels/
        yaml_path = os.path.join(subset_dir, 'dataset.yaml')
        yaml_content = f"""path: {subset_dir}
train: images
val: images
names:
  0: license_plate
"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content.strip())
        yaml_paths[subset_name] = yaml_path
        print(f"  {subset_name}: {len(filelist)} 张")

    return yaml_paths.get('base'), yaml_paths.get('hard'), tmpdir


def _eval_split(model: YOLO, data_yaml: str, split_name: str) -> dict:
    """在指定子集上评估模型。"""
    print(f"    评估 {split_name} 子集...")
    metrics = model.val(data=data_yaml, verbose=False)
    return {
        'map50':     float(metrics.box.map50),
        'map50_95':  float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall':    float(metrics.box.mr),
    }


# ── 打印表格 ──────────────────────────────────────────────────────────────────
def _print_table(results: list[dict]):
    header = f"{'Model':<12} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Precision':>11} {'Recall':>9} {'FPS':>8} {'参数量(M)':>10}"
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['label']:<12} "
            f"{r['map50']:>10.4f} "
            f"{r['map50_95']:>14.4f} "
            f"{r['precision']:>11.4f} "
            f"{r['recall']:>9.4f} "
            f"{r['fps']:>8.1f} "
            f"{r['params_m']:>10.2f}"
        )
    print(sep + '\n')


def _print_split_table(split_results: dict, labels: list):
    """打印分集对比表格。
    split_results = {label: {'base': {...}, 'hard': {...}}}
    """
    if not split_results:
        return

    # 表头
    header = (
        f"{'Model':<10} "
        f"{'Base mAP@.5':>12} {'Base mAP@.5:.95':>16} "
        f"{'困难 mAP@.5':>12} {'困难 mAP@.5:.95':>16} "
        f"{'maP降幅':>8}"
    )
    sep = '-' * len(header)
    print('\n' + sep)
    print('  分集评估（Base vs 困难样本）')
    print(sep)
    print(header)
    print(sep)

    for label in labels:
        r = split_results.get(label)
        if not r:
            continue
        base = r.get('base', {})
        hard = r.get('hard', {})
        b50 = base.get('map50', 0)
        h50 = hard.get('map50', 0)
        drop = (b50 - h50) * 100 if b50 > 0 else 0  # 降幅（百分点）

        print(
            f"{label:<10} "
            f"{base.get('map50', 0):>12.4f} "
            f"{base.get('map50_95', 0):>16.4f} "
            f"{hard.get('map50', 0):>12.4f} "
            f"{hard.get('map50_95', 0):>16.4f} "
            f"{drop:>7.1f}%"
        )
    print(sep + '\n')
    print("  maP降幅越小 → 模型对困难样本越鲁棒\n")


# ── 生成对比柱状图 ────────────────────────────────────────────────────────────
def _plot_chart(results: list[dict], output_path: str, use_chinese: bool):
    labels = [r['label'] for r in results]
    metrics_def = [
        ('map50',     'mAP@0.5'       if not use_chinese else 'mAP@0.5'),
        ('map50_95',  'mAP@0.5:0.95'  if not use_chinese else 'mAP@0.5:0.95'),
        ('precision', 'Precision'     if not use_chinese else '精确率'),
        ('recall',    'Recall'        if not use_chinese else '召回率'),
    ]

    n_metrics = len(metrics_def)
    n_models  = len(labels)
    x = np.arange(n_metrics)
    width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('三模型性能对比' if use_chinese else 'Model Performance Comparison', fontsize=14)

    ax = axes[0]
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for i, (r, color, offset) in enumerate(zip(results, colors, offsets)):
        vals = [r[key] for key, _ in metrics_def]
        bars = ax.bar(x + offset, vals, width, label=r['label'], color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metrics_def])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('检测精度指标' if use_chinese else 'Detection Metrics')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    x2 = np.arange(n_models)
    fps_vals   = [r['fps']     for r in results]
    param_vals = [r['params_m'] for r in results]
    b1 = ax2.bar(x2 - 0.2, fps_vals,   0.35, label='FPS',              color='#4C72B0', alpha=0.85)
    b2 = ax2_twin.bar(x2 + 0.2, param_vals, 0.35, label='参数量(M)' if use_chinese else 'Params(M)',
                      color='#DD8452', alpha=0.85)
    for bar, val in zip(b1, fps_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(b2, param_vals):
        ax2_twin.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('FPS')
    ax2_twin.set_ylabel('参数量 (M)' if use_chinese else 'Params (M)')
    ax2.set_title('推理速度与模型大小' if use_chinese else 'Speed & Model Size')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 对比图表已保存至: {output_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  三模型对比评估：YOLOv8 / YOLOv10 / YOLOv11")
    print("=" * 60)

    font_name = _setup_chinese_font()
    use_chinese = font_name is not None
    if not use_chinese:
        print("⚠️  未找到中文字体，图表将使用英文标注")

    if not os.path.exists(DATASET_YAML):
        print(f"❌ 找不到数据集配置文件: {DATASET_YAML}")
        return

    print("\n读取验证集图片列表...")
    val_images = _get_val_images(DATASET_YAML)
    print(f"  共找到 {len(val_images)} 张验证集图片（用于 FPS 测量）")

    # ── 1. 整体评估 ──────────────────────────
    results = []
    for config in MODEL_CONFIGS:
        print(f"\n▶ 评估 {config['label']}...")
        result = _evaluate_model(config, val_images)
        if result:
            results.append(result)

    if not results:
        print("❌ 没有可用的模型权重，请先完成训练。")
        return

    _print_table(results)

    # ── 2. 分集评估（Base vs 困难）────────────
    val_split = _load_val_split()
    if val_split:
        print("=" * 60)
        print("  分集评估中...")
        print("=" * 60)
        base_yaml, hard_yaml, tmpdir = _create_split_yamls(val_split)

        if base_yaml and hard_yaml:
            split_results = {}
            labels_order = []
            for config in MODEL_CONFIGS:
                if not os.path.exists(config['weight']):
                    continue
                label = config['label']
                labels_order.append(label)
                print(f"\n▶ 分集评估 {label}...")
                model = YOLO(config['weight'])
                br = _eval_split(model, base_yaml, 'Base')
                hr = _eval_split(model, hard_yaml, '困难')
                split_results[label] = {'base': br, 'hard': hr}

            _print_split_table(split_results, labels_order)
        else:
            print("  ⚠️  无法创建分集数据，跳过")

        # 清理临时文件
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print("\n⚠️  未找到 val_split.json，跳过分集评估")
        print("  请先运行 python scripts/ccpd_to_yolo.py 生成新数据集")

    _plot_chart(results, OUTPUT_CHART, use_chinese)
    print("✅ 对比评估完成！")


if __name__ == '__main__':
    main()
