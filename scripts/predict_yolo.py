"""
YOLOv11 模型预测测试脚本。
从 config.yaml 读取预设和模型配置，在验证集中取一张图片执行车牌检测，
将画好框的结果图片保存到 runs/{preset}/predict_test/。
"""
import os
import yaml
from ultralytics import YOLO


def predict_test():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 读取配置
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    preset = cfg['dataset']['preset']

    # 找第一个可用的已训练权重
    model_path = ''
    for m in cfg.get('models', []):
        if not m.get('enabled', True):
            continue
        candidate = os.path.join(
            project_root, 'runs', preset,
            f'train_{m["name"]}_plate', 'weights', 'best.pt'
        )
        if os.path.exists(candidate):
            model_path = candidate
            break

    if not model_path:
        print(f'[错误] 找不到模型权重文件，请先完成训练。')
        print(f'       预期路径: runs/{preset}/train_<model>_plate/weights/best.pt')
        return

    print(f'[加载] 成功加载训练好的模型: {model_path}')
    model = YOLO(model_path)

    # 从数据集验证集目录取第一张图片
    val_dir = os.path.join(project_root, 'datasets', f'plate_dataset_{preset}', 'images', 'val')
    if not os.path.isdir(val_dir):
        print(f'[错误] 找不到验证集目录: {val_dir}')
        return

    test_images = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not test_images:
        print('[错误] 验证集目录为空，找不到测试图片！')
        return

    test_image_path = os.path.join(val_dir, test_images[0])
    print(f'[图片] 准备测试图片: {test_image_path}')

    print('[检测] 正在进行车牌检测...')
    results = model.predict(
        source=test_image_path,
        save=True,
        project=os.path.join(project_root, 'runs', preset),
        name='predict_test'
    )

    save_dir = results[0].save_dir
    print(f'\n[完成] 预测完成！')
    print(f'[保存] 带有预测框的图片已保存在: {save_dir} 目录下。')


if __name__ == '__main__':
    predict_test()
