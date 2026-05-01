import os
from ultralytics import YOLO


def train():
    """YOLOv8 车牌检测模型训练脚本，用于与 YOLOv11 进行对比实验。"""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_yaml = os.path.join(project_root, 'datasets', 'plate_dataset_standard', 'dataset.yaml')

    if not os.path.exists(dataset_yaml):
        print(f"❌ 找不到数据集配置文件: {dataset_yaml}")
        print("请确认是否已经成功运行了 scripts/ccpd_to_yolo.py 脚本！")
        return

    print("🚀 开始加载 YOLOv8 预训练模型...")
    model = YOLO("yolov8n.pt")

    print("🎓 开始训练车牌检测模型（YOLOv8）...")
    print(f"加载数据集配置: {dataset_yaml}")

    results = model.train(
        data=dataset_yaml,
        epochs=10,
        imgsz=640,
        batch=8,
        device='cuda',
        workers=2,
        project='runs(1)',
        name='train_yolov8_plate'
    )

    print("✅ 训练完成！")
    print("训练日志和最佳模型权重(best.pt)已保存在: runs/detect/train_yolov8_plate/weights/ 目录下")


if __name__ == '__main__':
    train()
