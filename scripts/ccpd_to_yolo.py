import os
import cv2
import shutil
import random
from tqdm import tqdm

# ==========================================
# 数据集预设配置（子集名 -> 权重/比例）
# ==========================================
DATASET_PRESETS = {
    # 纯 Base（原始配置）
    "base_only": {
        "ccpd_base": 1.0,
    },
    # 标准版：Base 70% + 四个困难子集各 7.5%
    "standard": {
        "ccpd_base": 0.70,
        "ccpd_blur": 0.075,
        "ccpd_db": 0.075,
        "ccpd_tilt": 0.075,
        "ccpd_rotate": 0.075,
    },
    # 地狱版：Base 50% + 所有子集均摊
    "hard": {
        "ccpd_base": 0.50,
        "ccpd_blur": 0.10,
        "ccpd_db": 0.10,
        "ccpd_tilt": 0.07,
        "ccpd_rotate": 0.08,
        "ccpd_fn": 0.05,
        "ccpd_weather": 0.05,
        "ccpd_challenge": 0.05,
    },
}


def parse_ccpd_filename(filename):
    """
    解析 CCPD 数据集的文件名，提取边界框坐标
    文件名示例: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    """
    try:
        name = filename.split('-')
        bbox_str = name[2]
        p1, p2 = bbox_str.split('_')
        xmin, ymin = map(int, p1.split('&'))
        xmax, ymax = map(int, p2.split('&'))
        return xmin, ymin, xmax, ymax
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None


def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    """将绝对坐标转换为 YOLO 格式的相对坐标"""
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height


def process_dataset(
    src_dir_map,          # {子集名: 源目录路径}
    ratios,               # {子集名: 权重}
    dest_dir,
    split_ratio=(0.8, 0.1, 0.1),
    max_images=10000,
    preset_name="custom",
):
    """
    从多个 CCPD 子集按比例混合，生成 YOLO 格式数据集。

    src_dir_map: 子集名 -> 该子集图片所在目录
    ratios:      子集名 -> 占总量的比例（权重）
    dest_dir:    输出目标目录
    max_images:  总共抽取的图片数量上限
    preset_name: 预设名称，用于输出目录名和日志
    """
    # 校验所有源目录
    for subset_name, src_dir in src_dir_map.items():
        if not os.path.exists(src_dir):
            print(f"错误: 子集 {subset_name} 的源目录 {src_dir} 不存在！")
            return

    # 收集各子集的图片列表
    subset_files = {}
    total_available = 0
    for subset_name, src_dir in src_dir_map.items():
        files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
        subset_files[subset_name] = files
        total_available += len(files)
        print(f"  {subset_name}: {len(files)} 张可用图片")

    print(f"\n总计可用图片: {total_available} 张，本次抽取上限: {max_images} 张")

    # 按比例计算各子集应抽取的数量
    selected_files = []
    for subset_name, ratio in ratios.items():
        if subset_name not in subset_files:
            print(f"警告: 子集 {subset_name} 不在可用列表中，跳过")
            continue
        n_sample = int(max_images * ratio)
        files = subset_files[subset_name]
        if n_sample > len(files):
            print(f"  {subset_name}: 需求 {n_sample}，实际只有 {len(files)}，全部纳入")
            selected_files.extend([(subset_name, f) for f in files])
        else:
            sampled = random.sample(files, n_sample)
            selected_files.extend([(subset_name, f) for f in sampled])
            print(f"  {subset_name}: 权重 {ratio:.1%} → 抽取 {n_sample} 张")

    # 打乱顺序
    random.shuffle(selected_files)
    total_files = len(selected_files)
    print(f"\n实际抽取图片总数: {total_files}")

    # 创建目标目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'labels', split), exist_ok=True)

    # 划分 train/val/test
    train_end = int(total_files * split_ratio[0])
    val_end = train_end + int(total_files * split_ratio[1])

    splits = {
        'train': selected_files[:train_end],
        'val': selected_files[train_end:val_end],
        'test': selected_files[val_end:],
    }

    class_id = 0  # 车牌
    fixed_img_width = 720
    fixed_img_height = 1160

    for split_name, files in splits.items():
        print(f"\n正在处理 {split_name} 集 ({len(files)} 张)...")
        for subset_name, filename in tqdm(files):
            src_img_path = os.path.join(src_dir_map[subset_name], filename)

            # 解析坐标
            bbox = parse_ccpd_filename(filename)
            if bbox is None:
                continue
            xmin, ymin, xmax, ymax = bbox

            # 转为 YOLO 格式
            x_center, y_center, width, height = convert_to_yolo_format(
                xmin, ymin, xmax, ymax, fixed_img_width, fixed_img_height
            )

            # 目标路径
            dest_img_path = os.path.join(dest_dir, 'images', split_name, filename)
            dest_label_path = os.path.join(
                dest_dir, 'labels', split_name, filename.replace('.jpg', '.txt')
            )

            # 复制图片
            shutil.copy(src_img_path, dest_img_path)

            # 写入标签
            with open(dest_label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 生成 dataset.yaml
    subset_desc = " + ".join([f"{k}({v:.0%})" for k, v in ratios.items()])
    yaml_content = f"""# CCPD 混合数据集 - {preset_name} 预设
# 构成: {subset_desc}

path: {os.path.abspath(dest_dir)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')
test: images/test    # test images (optional)

# Classes
names:
  0: license_plate
"""
    yaml_path = os.path.join(dest_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content.strip())

    print(f"\n✅ 数据集处理完成！")
    print(f"预设: {preset_name}")
    print(f"构成: {subset_desc}")
    print(f"YOLO 数据集保存至: {os.path.abspath(dest_dir)}")
    print(f"配置文件: {yaml_path}")


if __name__ == '__main__':
    # ==========================================
    # 配置区
    # ==========================================
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # CCPD 根目录（下面有 ccpd_base, ccpd_blur 等子文件夹）
    CCPD_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'CCPD2019', 'CCPD2019')

    # 选择预设: "base_only" | "standard" | "hard"
    PRESET = "standard"

    # 总共抽取图片数
    MAX_IMAGES = 10000

    # ==========================================
    # 执行
    # ==========================================
    ratios = DATASET_PRESETS[PRESET]
    src_dir_map = {}
    for subset_name in ratios:
        src_dir_map[subset_name] = os.path.join(CCPD_ROOT, subset_name)

    dest_dir = os.path.join(PROJECT_ROOT, 'datasets', f'plate_dataset_{PRESET}')

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"CCPD 根目录: {CCPD_ROOT}")
    print(f"目标数据目录: {dest_dir}")
    print(f"预设: {PRESET}\n")
    print("各子集情况:")

    process_dataset(
        src_dir_map=src_dir_map,
        ratios=ratios,
        dest_dir=dest_dir,
        max_images=MAX_IMAGES,
        preset_name=PRESET,
    )
