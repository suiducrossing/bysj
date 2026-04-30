import os
import cv2
import shutil
import random
from tqdm import tqdm

def parse_ccpd_filename(filename):
    """
    解析 CCPD 数据集的文件名，提取边界框坐标
    文件名示例: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    """
    try:
        # 去除扩展名
        name = filename.split('-')
        # 第3部分是边界框坐标: xmin&ymin_xmax&ymax
        bbox_str = name[2]
        p1, p2 = bbox_str.split('_')
        xmin, ymin = map(int, p1.split('&'))
        xmax, ymax = map(int, p2.split('&'))
        return xmin, ymin, xmax, ymax
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    将绝对坐标转换为 YOLO 格式的相对坐标 (中心点x, 中心点y, 宽, 高)
    """
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def process_dataset(src_dir, dest_dir, split_ratio=(0.8, 0.1, 0.1), max_images=10000):
    """
    处理 CCPD 数据集，划分为 train, val, test 并生成 YOLO 标签。
    为了加快毕设进度和减少机器性能负担，默认抽取 max_images 张图片。
    """
    if not os.path.exists(src_dir):
        print(f"错误: 源目录 {src_dir} 不存在！")
        print("请先下载 CCPD 数据集并解压到该目录下。")
        return

    # 创建目标目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'labels', split), exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    if not image_files:
        print(f"错误: 在 {src_dir} 中没有找到 .jpg 图片！")
        return
        
    print(f"原数据集共有 {len(image_files)} 张图片。")
    if len(image_files) > max_images:
        print(f"为了加快进度，将随机抽取 {max_images} 张图片进行处理...")
        image_files = random.sample(image_files, max_images)
    else:
        print(f"开始打乱并划分所有 {len(image_files)} 张图片...")
        random.shuffle(image_files)

    total_files = len(image_files)
    train_end = int(total_files * split_ratio[0])
    val_end = train_end + int(total_files * split_ratio[1])

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    class_id = 0  # 只有一个类别：车牌 (license_plate)
    # CCPD 数据集的图片绝大多数都是固定的 720x1160 尺寸，直接写死可以极大提升处理速度
    fixed_img_width = 720
    fixed_img_height = 1160

    for split_name, files in splits.items():
        print(f"正在处理 {split_name} 集 ({len(files)} 张图片)...")
        for filename in tqdm(files):
            src_img_path = os.path.join(src_dir, filename)
            
            # 解析坐标
            bbox = parse_ccpd_filename(filename)
            if bbox is None:
                continue
            xmin, ymin, xmax, ymax = bbox

            # 转换为 YOLO 格式（使用固定宽高代替 cv2.imread，速度快几百倍）
            x_center, y_center, width, height = convert_to_yolo_format(
                xmin, ymin, xmax, ymax, fixed_img_width, fixed_img_height
            )

            # 目标路径
            dest_img_path = os.path.join(dest_dir, 'images', split_name, filename)
            dest_label_path = os.path.join(dest_dir, 'labels', split_name, filename.replace('.jpg', '.txt'))

            # 复制图片
            shutil.copy(src_img_path, dest_img_path)

            # 写入标签文件
            with open(dest_label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 生成 dataset.yaml
    yaml_content = f"""
path: {os.path.abspath(dest_dir)} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
test: images/test # test images (optional)

# Classes
names:
  0: license_plate
"""
    with open(os.path.join(dest_dir, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_content.strip())
    
    print(f"\n数据集处理完成！")
    print(f"YOLO 格式数据集已保存至: {os.path.abspath(dest_dir)}")
    print(f"YOLO 配置文件已生成: {os.path.join(dest_dir, 'dataset.yaml')}")

if __name__ == '__main__':
    # ==========================================
    # 请在这里修改你的 CCPD 数据集解压后的路径
    # ==========================================
    # 假设你把下载的 CCPD2019 解压到了项目的 datasets 目录下
    # 并且图片都在 datasets/CCPD2019/ccpd_base 文件夹里
    
    # 获取当前脚本所在目录的上一级目录（即项目根目录）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 源数据集路径 (你需要确保这个路径下有 .jpg 图片)
    SOURCE_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'CCPD2019', 'CCPD2019', 'ccpd_base')
    
    # 转换后的 YOLO 数据集保存路径
    DEST_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'plate_dataset')
    
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"源数据目录: {SOURCE_DIR}")
    print(f"目标数据目录: {DEST_DIR}\n")
    
    process_dataset(SOURCE_DIR, DEST_DIR)
