import os
import cv2
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from paddleocr import PaddleOCR


# 项目根目录（core/ 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def draw_chinese_text(img_bgr, text, position, font_size=22, color=(0, 255, 0)):
    """
    在 OpenCV 图像上绘制含中文的文字。

    OpenCV 的 cv2.putText 不支持中文，这里借助 PIL 库加载系统中文字体来绘制，
    绘制完成后再转回 OpenCV 的 BGR 格式。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = None
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
            break
    if font is None:
        font = ImageFont.load_default()

    # PIL 颜色是 RGB，需要把传入的 BGR color 做一次翻转
    color_rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=color_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class PlateRecognizer:
    """
    车牌检测与识别的核心类，封装了完整的两阶段流水线：
      阶段一：YOLOv11 负责检测车牌在图片中的位置（输出边界框）
      阶段二：PaddleOCR 负责读取裁剪出的车牌图像中的文字（输出车牌号）

    使用方式：
      recognizer = PlateRecognizer()
      results = recognizer.recognize(image_bgr)
    """

    def __init__(self, yolo_weight: Optional[str] = None):
        """
        初始化时加载 YOLO 和 PaddleOCR 两个模型。
        yolo_weight: 可选，指定 best.pt 的绝对路径；不传则自动查找默认路径。
        """
        if yolo_weight and os.path.exists(yolo_weight):
            model_path = yolo_weight
        else:
            # 默认路径：兼容两种目录层级
            model_path = os.path.join(
                PROJECT_ROOT, 'runs', 'detect', 'runs',
                'train_yolo11_plate(2)', 'weights', 'best.pt'
            )
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    PROJECT_ROOT, 'runs',
                    'train_yolo11_plate(2)', 'weights', 'best.pt'
                )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 YOLO 模型文件: {model_path}")

        self.yolo_model = YOLO(model_path)

        # 加载 PaddleOCR 模型（PaddleOCR 2.x API）
        # 显式指定本地模型路径，避免自动下载
        models_base = os.path.join(PROJECT_ROOT, 'models', 'paddleocr')
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            det_model_dir=os.path.join(models_base, 'det', 'ch_PP-OCRv4_det_infer'),
            rec_model_dir=os.path.join(models_base, 'rec', 'ch_PP-OCRv4_rec_infer'),
            cls_model_dir=os.path.join(models_base, 'cls', 'ch_ppocr_mobile_v2.0_cls_infer'),
        )

    def recognize(self, image_bgr):
        """
        对输入图像执行车牌检测和文字识别，返回识别结果列表和标注后的图像。

        参数：
            image_bgr - OpenCV 格式的图像（BGR numpy 数组）

        返回：
            result_image - 画了检测框和车牌号的图像（BGR numpy 数组）
            plates       - 识别结果列表，每个元素是一个字典：
                           {
                             'text': '皖AK581R',      # 识别出的车牌号
                             'det_conf': 0.68,        # YOLO 检测置信度
                             'ocr_conf': 0.93,        # OCR 识别置信度
                             'box': (x1, y1, x2, y2) # 车牌在原图中的坐标
                           }
        """
        result_image = image_bgr.copy()
        plates = []

        # 第一阶段：YOLOv11 检测车牌位置
        yolo_results = self.yolo_model.predict(source=image_bgr, verbose=False)
        detection = yolo_results[0]
        boxes = detection.boxes

        if boxes is None or len(boxes) == 0:
            # 没有检测到车牌，直接返回原图和空列表
            return result_image, plates

        # 第二阶段：逐个裁剪车牌区域，送入 PaddleOCR 识别文字
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])

            # 裁剪车牌区域
            plate_crop = image_bgr[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # OCR 识别
            ocr_result = self.ocr_engine.ocr(plate_crop, cls=True)

            plate_text = ""
            ocr_conf = 0.0
            if ocr_result and ocr_result[0]:
                text_parts = []
                conf_list = []
                for line in ocr_result[0]:
                    text_parts.append(line[1][0])
                    conf_list.append(line[1][1])
                plate_text = "".join(text_parts)
                ocr_conf = sum(conf_list) / len(conf_list)
            else:
                plate_text = "识别失败"

            # 记录本次识别结果
            plates.append({
                'text': plate_text,
                'det_conf': det_conf,
                'ocr_conf': ocr_conf,
                'box': (x1, y1, x2, y2)
            })

            # 在结果图像上绘制绿色检测框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # 在框上方用 PIL 绘制中文车牌号
            label = f"{plate_text}  ({det_conf:.2f})"
            text_y = y1 - 28 if y1 - 28 > 0 else y1 + 5
            result_image = draw_chinese_text(
                result_image, label, (x1, text_y),
                font_size=22, color=(0, 255, 0)
            )

        return result_image, plates
