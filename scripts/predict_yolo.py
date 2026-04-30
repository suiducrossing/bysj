import os
from ultralytics import YOLO

def predict_test():
    """
    YOLOv11 模型预测测试脚本。
    该脚本会加载我们刚刚训练好的模型，并从验证集中找一张图片进行车牌检测，
    最终将画好框的图片保存下来供我们肉眼检查。
    """
    # 1. 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. 定位训练好的模型权重路径 (best.pt)
    # YOLO 默认保存在 runs/... 目录下，根据之前的日志，这是我们 30 轮训练的结果
    # 路径可能有两种情况，我们都尝试一下
    model_path = os.path.join(project_root, 'runs', 'detect', 'runs', 'train_yolo11_plate(2)', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, 'runs', 'train_yolo11_plate(2)', 'weights', 'best.pt')
        
    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型权重文件，请手动检查路径: {model_path}")
        return
        
    print(f"[加载] 成功加载训练好的模型: {model_path}")
    model = YOLO(model_path)
    
    # 3. 找一张测试图片
    # 这里我们直接从验证集中挑选第一张图片
    test_image_dir = os.path.join(project_root, 'datasets', 'plate_dataset', 'images', 'val')
    test_images = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
    if not test_images:
        print("[错误] 验证集目录为空，找不到测试图片！")
        return
        
    # 挑选第一张图片进行测试
    test_image_path = os.path.join(test_image_dir, test_images[0])
    print(f"[图片] 准备测试图片: {test_image_path}")
    
    # 4. 执行预测
    # save=True 表示自动将画好框的图片保存下来
    # project 和 name 用于指定预测结果保存的位置
    print("[检测] 正在进行车牌检测...")
    results = model.predict(
        source=test_image_path,
        save=True,
        project='runs',
        name='predict_test'
    )
    
    # 5. 提示用户查看结果
    save_dir = results[0].save_dir
    print(f"\n[完成] 预测完成！")
    print(f"[保存] 带有预测框的图片已保存在: {save_dir} 目录下。")
    print("您现在可以去这个目录里双击打开那张图片，看看模型有没有准确地把车牌框出来！")

if __name__ == '__main__':
    predict_test()
