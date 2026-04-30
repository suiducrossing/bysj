"""
三模型对比评估脚本：YOLOv8 / YOLOv10 / YOLOv11
在验证集上评估各模型的 mAP、精度、召回率和推理速度，并生成对比图表。
"""
import os
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
    # 找不到中文字体时回退到英文标注
    return None


# ── 模型配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_CONFIGS = [
    {
        'label': 'YOLOv11',
        'weight': os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train_yolo11_plate(2)', 'weights', 'best.pt'),
    },
    {
        'label': 'YOLOv8',
        'weight': os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train_yolov8_plate', 'weights', 'best.pt'),
    },
    {
        'label': 'YOLOv10',
        'weight': os.path.join(PROJECT_ROOT, 'runs', 'detect', 'train_yolov10_plate', 'weights', 'best.pt'),
    },
]

DATASET_YAML = os.path.join(PROJECT_ROOT, 'datasets', 'plate_dataset', 'dataset.yaml')
OUTPUT_CHART = os.path.join(PROJECT_ROOT, 'runs', 'comparison_chart.png')


# ── 推理速度测量 ──────────────────────────────────────────────────────────────
def _measure_fps(model: YOLO, val_images: list, n_warmup: int = 5, n_measure: int = 50) -> float:
    """在验证集图片上测量平均推理 FPS。"""
    images = val_images[:max(n_warmup + n_measure, len(val_images))]
    if not images:
        return 0.0

    # 预热
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
    """从 dataset.yaml 中读取验证集图片路径列表。"""
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


# ── 参数量统计 ────────────────────────────────────────────────────────────────
def _count_params(model: YOLO) -> float:
    """返回模型参数量（单位：百万 M）。"""
    try:
        total = sum(p.numel() for p in model.model.parameters())
        return total / 1e6
    except Exception:
        return float('nan')


# ── 单模型评估 ────────────────────────────────────────────────────────────────
def _evaluate_model(config: dict, val_images: list) -> dict | None:
    label = config['label']
    weight = config['weight']

    if not os.path.exists(weight):
        print(f"  ⚠️  {label} 权重文件不存在，跳过: {weight}")
        return None

    print(f"  加载 {label} 模型: {weight}")
    model = YOLO(weight)

    print(f"  评估 {label} 验证集指标...")
    metrics = model.val(data=DATASET_YAML, verbose=False)

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


# ── 打印对比表格 ──────────────────────────────────────────────────────────────
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

    # 左图：mAP / Precision / Recall
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

    # 右图：FPS 和参数量（双 Y 轴）
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    x2 = np.arange(n_models)
    fps_vals    = [r['fps']     for r in results]
    param_vals  = [r['params_m'] for r in results]

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
    _plot_chart(results, OUTPUT_CHART, use_chinese)
    print("✅ 对比评估完成！")


if __name__ == '__main__':
    main()
