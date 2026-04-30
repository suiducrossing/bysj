"""辅助函数：中文绘制、图像处理等，供 app.py 调用。"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CACHE: dict = {}

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            _FONT_CACHE[size] = font
            return font
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_chinese_text(img_bgr: np.ndarray, text: str, position: tuple,
                      font_size: int = 22, color: tuple = (0, 255, 0)) -> np.ndarray:
    """在 OpenCV BGR 图像上绘制含中文的文字，返回 BGR 图像。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    color_rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """PIL RGB 图像转 OpenCV BGR numpy 数组。"""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR 转 RGB（用于 Streamlit st.image 显示）。"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_for_display(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """等比缩放，使最长边不超过 max_side，避免超大图拖慢界面。"""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))
