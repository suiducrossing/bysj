import os
from ultralytics import YOLO

def train():
    """
    YOLOv11 车牌检测模型训练脚本。
    该函数将加载数据集配置，并使用预训练的 YOLOv11 模型开始微调训练。
    """
    
    # 获取项目根目录，确保路径的正确性
    # 假设当前脚本位于 scripts/ 目录下，所以我们需要向上一级获取根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据集配置文件的绝对路径
    dataset_yaml = os.path.join(project_root, 'datasets', 'plate_dataset_standard', 'dataset.yaml')
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ 找不到数据集配置文件: {dataset_yaml}")
        print("请确认是否已经成功运行了 scripts/ccpd_to_yolo.py 脚本！")
        return

    print("🚀 开始加载 YOLOv11 预训练模型...")
    # 我们选择 yolov11n.pt，其中的 'n' 代表 nano（轻量级），适合在普通电脑上快速训练和运行
    # Ultralytics 会自动从网络上下载该预训练权重文件
    model = YOLO("yolo11n.pt")  

    print("🎓 开始训练车牌检测模型...")
    print(f"加载数据集配置: {dataset_yaml}")
    
    # 启动训练
    # data: 指向包含训练集、验证集路径和类别信息的 yaml 文件
    # epochs: 训练轮数（10轮是一个体验版配置，如果效果不好可以改为 30 或 50）
    # imgsz: 图像缩放尺寸。640 是 YOLO 的默认标准尺寸，可以兼顾速度和精度
    # batch: 每批次输入模型的图片数量，根据显存/内存大小调整，一般为 8, 16 或 32
    # device: 留空（由Ultralytics自动选择，有GPU用GPU，没有用CPU）
    # workers: 数据加载的线程数，Windows上容易报错，所以这里稳妥起见设置为 0 或者低一点(1/2)
    # project: 训练结果保存的主目录名称
    # name: 本次训练结果保存的子文件夹名称
    results = model.train(
        data=dataset_yaml,
        epochs=10,          
        imgsz=640,          
        batch=8,
        device='cuda',            
        workers=2,          
        project='runs(1)',     
        name='train_yolo11_plate' 
    )
    
    print("✅ 训练完成！")
    print("训练日志和最佳模型权重(best.pt)已保存在: runs/train_yolo11_plate/weights/ 目录下")

if __name__ == '__main__':
    train()
