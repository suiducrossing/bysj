"""
统一训练脚本 —— 读取 config.yaml，按预设数据集顺序训练所有启用的模型。

训练结果自动保存至 runs/{dataset_preset}/train_{model}_plate/
"""
import os
import sys
import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """加载总控配置文件。"""
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    # 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config.yaml')

    print("=" * 60)
    print("  统一训练脚本")
    print("=" * 60)

    cfg = load_config(config_path)

    # 读取配置
    preset = cfg['dataset']['preset']
    train_cfg = cfg['training']
    models = [m for m in cfg['models'] if m.get('enabled', True)]

    # 数据集路径
    dataset_dir = os.path.join(project_root, 'datasets', f'plate_dataset_{preset}')
    dataset_yaml = os.path.join(dataset_dir, 'dataset.yaml')

    print(f"\n📦 数据集预设: {preset}")
    print(f"📁 数据集路径: {dataset_yaml}")

    if not os.path.exists(dataset_yaml):
        print(f"\n❌ 数据集不存在！请先运行: python scripts/ccpd_to_yolo.py")
        print(f"   （确保 config.yaml 中 dataset.preset 与脚本中 PRESET 一致）")
        sys.exit(1)

    print(f"\n🖥️  训练参数: epochs={train_cfg['epochs']}, "
          f"imgsz={train_cfg['imgsz']}, batch={train_cfg['batch']}, "
          f"device={train_cfg['device']}")

    print(f"\n📋 将要训练的模型:")
    for m in models:
        print(f"   ✅ {m['name']} ({m['weight']})")

    # 结果统一保存到 runs/{preset}/ 下
    project_dir = os.path.join(project_root, 'runs', preset)

    print(f"\n📂 结果保存目录: {project_dir}/")
    print()

    # ── 依次训练 ──────────────────────────────────────
    for i, m in enumerate(models):
        model_name = m['name']
        weight = m['weight']

        print(f"{'='*60}")
        print(f"  [{i+1}/{len(models)}] 开始训练 {model_name}")
        print(f"{'='*60}")

        print(f"🚀 加载预训练权重: {weight}")
        model = YOLO(weight)

        print(f"🎓 训练中...")
        model.train(
            data=dataset_yaml,
            epochs=train_cfg['epochs'],
            imgsz=train_cfg['imgsz'],
            batch=train_cfg['batch'],
            device=train_cfg['device'],
            workers=train_cfg['workers'],
            project=project_dir,
            name=f'train_{model_name}_plate',
        )

        print(f"✅ {model_name} 训练完成！")
        print(f"   权重保存在: {project_dir}/train_{model_name}_plate/weights/best.pt\n")

    # ── 汇总 ──────────────────────────────────────────
    print("=" * 60)
    print("  🎉 全部模型训练完成！")
    print("=" * 60)
    for m in models:
        name = m['name']
        weight_path = os.path.join(
            project_dir, f'train_{name}_plate', 'weights', 'best.pt'
        )
        print(f"  {name}: {weight_path}")
    print(f"\n  对比评估: python scripts/compare_models.py")


if __name__ == '__main__':
    main()
