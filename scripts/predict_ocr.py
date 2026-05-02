"""
YOLOv11 + PaddleOCR 完整车牌识别流程测试脚本。
从 config.yaml 读取预设，在验证集中取一张图片，执行检测+OCR，
将画好框和车牌号的结果图保存到 runs/{preset}/ocr_test/。
"""
import os
import sys
import yaml
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from paddleocr import PaddleOCR


def draw_chinese_text(img_bgr, text, position, font_size=22, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = None
    for font_path in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
                      'C:/Windows/Fonts/simsun.ttc']:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
            break
    if font is None:
        font = ImageFont.load_default()

    color_rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def predict_with_ocr():
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

    # 验证集目录
    val_dir = os.path.join(project_root, 'datasets', f'plate_dataset_{preset}', 'images', 'val')
    if not os.path.isdir(val_dir):
        print(f'[错误] 找不到验证集目录: {val_dir}')
        return

    val_images = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not val_images:
        print('[错误] 验证集目录为空，找不到测试图片！')
        return
    test_image_path = os.path.join(val_dir, val_images[0])

    save_dir = os.path.join(project_root, 'runs', preset, 'ocr_test')
    os.makedirs(save_dir, exist_ok=True)

    print(f'[加载] YOLOv11 模型: {model_path}')
    print(f'[图片] 测试图片: {test_image_path}')

    # 初始化模型
    yolo_model = YOLO(model_path)

    models_base = os.path.join(project_root, 'models', 'paddleocr')
    print('[初始化] PaddleOCR 模型加载中...')
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        show_log=False,
        det_model_dir=os.path.join(models_base, 'det', 'ch_PP-OCRv4_det_infer'),
        rec_model_dir=os.path.join(models_base, 'rec', 'ch_PP-OCRv4_rec_infer'),
        cls_model_dir=os.path.join(models_base, 'cls', 'ch_ppocr_mobile_v2.0_cls_infer'),
    )

    # 第一步：YOLO检测
    print('\n[第一步] YOLOv11 正在检测车牌位置...')
    results = yolo_model.predict(source=test_image_path, verbose=False)

    original_img = cv2.imread(test_image_path)
    if original_img is None:
        print(f'[错误] OpenCV 无法读取图片: {test_image_path}')
        return

    detection = results[0]
    boxes = detection.boxes

    if boxes is None or len(boxes) == 0:
        print('[警告] 本张图片中未检测到车牌，请换一张图片重试。')
        return

    print(f'[完成] 检测到 {len(boxes)} 个车牌区域。')

    # 第二步+第三步：裁剪并OCR
    print('\n[第二步+第三步] 裁剪车牌并进行 OCR 文字识别...')

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])

        print(f'\n  车牌 #{i + 1}：坐标=({x1},{y1})-({x2},{y2})，检测置信度={confidence:.2f}')

        plate_crop = original_img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            print('  [警告] 裁剪区域为空，跳过此框。')
            continue

        ocr_result = ocr_engine.ocr(plate_crop, cls=True)

        plate_text = ''
        ocr_confidence = 0.0
        if ocr_result and ocr_result[0]:
            text_parts = []
            confidences = []
            for line in ocr_result[0]:
                text_parts.append(line[1][0])
                confidences.append(line[1][1])
            plate_text = ''.join(text_parts)
            ocr_confidence = sum(confidences) / len(confidences)
            print(f'  [OCR结果] {plate_text}（置信度={ocr_confidence:.2f}）')
        else:
            plate_text = '识别失败'
            print('  [警告] OCR 未能识别出文字。')

        cv2.rectangle(original_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        label = f'{plate_text}  ({confidence:.2f})'
        text_y = y1 - 28 if y1 - 28 > 0 else y1 + 5
        original_img = draw_chinese_text(
            original_img, label, (x1, text_y),
            font_size=22, color=(0, 255, 0)
        )

    save_path = os.path.join(save_dir, 'result_' + os.path.basename(test_image_path))
    cv2.imwrite(save_path, original_img)

    print(f'\n[完成] 全部识别完成！')
    print(f'[保存] 结果图片已保存至: {save_path}')


if __name__ == '__main__':
    predict_with_ocr()
