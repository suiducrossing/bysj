import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from paddleocr import PaddleOCR


def draw_chinese_text(img_bgr, text, position, font_size=22, color=(0, 255, 0)):
    """
    在 OpenCV 图像上绘制包含中文的文字。
    
    原理：OpenCV 的 cv2.putText 不支持中文，所以我们先把图像转换成
    PIL 格式，用 PIL 的中文字体绘制文字，再转回 OpenCV 格式。
    
    参数：
        img_bgr   - OpenCV 格式的图像（BGR numpy 数组）
        text      - 要绘制的文字（支持中文）
        position  - 文字左上角坐标 (x, y)
        font_size - 字体大小
        color     - 文字颜色，BGR 格式
    返回：
        绘制了文字的 OpenCV 图像
    """
    # OpenCV 是 BGR，PIL 是 RGB，需要互相转换
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 尝试加载 Windows 系统自带的中文字体（微软雅黑）
    # 如果找不到，则回退到 PIL 默认字体（不支持中文，但不会崩溃）
    font = None
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
            break
    if font is None:
        font = ImageFont.load_default()

    # PIL 的颜色格式是 RGB，需要把 BGR 的 color 转换一下
    color_rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=color_rgb)

    # 转回 OpenCV 的 BGR 格式
    img_bgr_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr_result

def predict_with_ocr():
    """
    YOLOv11 + PaddleOCR 完整车牌识别流程测试脚本。
    
    整体流程分三步：
      第一步：用 YOLOv11 检测图片中车牌的位置（得到边界框坐标）
      第二步：根据坐标把车牌区域从原图中裁剪出来
      第三步：把裁剪出的车牌图像送入 PaddleOCR，识别出车牌号码文字
    最终将检测框和识别结果一起画在原图上并保存。
    """

    # ── 1. 路径配置 ──────────────────────────────────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 定位训练好的 YOLOv11 模型权重（best.pt）
    model_path = os.path.join(
        project_root, 'runs', 'detect', 'runs',
        'train_yolo11_plate(2)', 'weights', 'best.pt'
    )
    if not os.path.exists(model_path):
        model_path = os.path.join(
            project_root, 'runs',
            'train_yolo11_plate(2)', 'weights', 'best.pt'
        )
    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型权重文件，请检查路径: {model_path}")
        return

    # 从验证集中取第一张图片作为测试输入
    val_dir = os.path.join(project_root, 'datasets', 'plate_dataset', 'images', 'val')
    val_images = [f for f in os.listdir(val_dir) if f.lower().endswith('.jpg')]
    if not val_images:
        print("[错误] 验证集目录为空，找不到测试图片！")
        return
    test_image_path = os.path.join(val_dir, val_images[0])

    # 结果保存目录
    save_dir = os.path.join(project_root, 'runs', 'ocr_test')
    os.makedirs(save_dir, exist_ok=True)

    print(f"[加载] YOLOv11 模型: {model_path}")
    print(f"[图片] 测试图片: {test_image_path}")

    # ── 2. 初始化模型 ─────────────────────────────────────────────────────────
    # 加载 YOLOv11 检测模型
    yolo_model = YOLO(model_path)

    # 初始化 PaddleOCR
    # use_angle_cls=True：开启文字方向分类，防止倒置车牌识别错误
    # lang='ch'：使用中文识别模型，支持汉字+字母+数字（适合中国车牌）
    # show_log=False：关闭 PaddleOCR 的冗长日志，让输出更清爽
    # 注意：显式指定模型路径（纯英文路径），避免 PaddlePaddle 在 Windows 上
    #       无法处理含中文字符的用户目录路径（已知 Bug）
    models_base = os.path.join(project_root, 'models', 'paddleocr')
    det_model_dir = os.path.join(models_base, 'det', 'ch_PP-OCRv4_det_infer')
    rec_model_dir = os.path.join(models_base, 'rec', 'ch_PP-OCRv4_rec_infer')
    cls_model_dir = os.path.join(models_base, 'cls', 'ch_ppocr_mobile_v2.0_cls_infer')

    print("[初始化] PaddleOCR 模型加载中...")
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        show_log=False,
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        cls_model_dir=cls_model_dir
    )

    # ── 3. 第一步：YOLOv11 检测车牌位置 ──────────────────────────────────────
    print("\n[第一步] YOLOv11 正在检测车牌位置...")
    # verbose=False 关闭 YOLO 自身的打印输出，由我们自己控制提示信息
    results = yolo_model.predict(source=test_image_path, verbose=False)

    # 用 OpenCV 读取原图，后续在上面画框和文字
    original_img = cv2.imread(test_image_path)
    if original_img is None:
        print(f"[错误] OpenCV 无法读取图片: {test_image_path}")
        return

    # 取第一张图片的检测结果
    detection = results[0]
    boxes = detection.boxes  # 所有检测到的边界框

    if boxes is None or len(boxes) == 0:
        print("[警告] 本张图片中未检测到车牌，请换一张图片重试。")
        return

    print(f"[完成] 检测到 {len(boxes)} 个车牌区域。")

    # ── 4. 第二步 & 第三步：逐个裁剪车牌并送入 OCR 识别 ─────────────────────
    print("\n[第二步+第三步] 裁剪车牌并进行 OCR 文字识别...")

    for i, box in enumerate(boxes):
        # 获取边界框的像素坐标（xyxy 格式：左上角x, 左上角y, 右下角x, 右下角y）
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])  # 检测置信度

        print(f"\n  车牌 #{i+1}：坐标=({x1},{y1})-({x2},{y2})，检测置信度={confidence:.2f}")

        # 第二步：从原图中裁剪出车牌区域
        # 注意：OpenCV 图像的索引顺序是 [行(y), 列(x)]
        plate_crop = original_img[y1:y2, x1:x2]

        if plate_crop.size == 0:
            print(f"  [警告] 裁剪区域为空，跳过此框。")
            continue

        # 第三步：将裁剪出的车牌图像送入 PaddleOCR 进行文字识别
        # PaddleOCR 接受 numpy 数组（BGR 格式的图像）作为输入
        ocr_result = ocr_engine.ocr(plate_crop, cls=True)

        # 解析 OCR 返回结果
        # ocr_result 的结构：[ [ [坐标框], (识别文字, 置信度) ], ... ]
        plate_text = ""
        ocr_confidence = 0.0

        if ocr_result and ocr_result[0]:
            # 把所有识别到的文字片段拼接成完整车牌号
            text_parts = []
            confidences = []
            for line in ocr_result[0]:
                text = line[1][0]       # 识别出的文字
                conf = line[1][1]       # 该文字的置信度
                text_parts.append(text)
                confidences.append(conf)
            plate_text = "".join(text_parts)
            ocr_confidence = sum(confidences) / len(confidences)  # 平均置信度
            print(f"  [OCR结果] {plate_text}（置信度={ocr_confidence:.2f}）")
        else:
            plate_text = "识别失败"
            print(f"  [警告] OCR 未能识别出文字。")

        # ── 5. 在原图上绘制检测框和识别结果 ──────────────────────────────────
        # 画绿色矩形框标出车牌位置
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # 在框的上方写出识别的车牌号（使用 PIL 支持中文显示）
        label = f"{plate_text}  ({confidence:.2f})"

        # 计算文字显示位置（在框的上方，如果框在图片顶部则显示在框内）
        text_y = y1 - 28 if y1 - 28 > 0 else y1 + 5
        original_img = draw_chinese_text(
            original_img, label, (x1, text_y),
            font_size=22, color=(0, 255, 0)
        )

    # ── 6. 保存最终结果图片 ───────────────────────────────────────────────────
    save_path = os.path.join(save_dir, 'result_' + os.path.basename(test_image_path))
    cv2.imwrite(save_path, original_img)

    print(f"\n[完成] 全部识别完成！")
    print(f"[保存] 带有检测框和车牌号的结果图片已保存至: {save_path}")
    print("您可以打开该图片，查看 YOLOv11 检测框和 PaddleOCR 识别的车牌号是否正确。")


if __name__ == '__main__':
    predict_with_ocr()
