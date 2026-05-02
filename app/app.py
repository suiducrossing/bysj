import os
import sys
import tempfile
from typing import List, Optional, Set

import cv2
import streamlit as st
import yaml
from PIL import Image

# 确保项目根目录在 sys.path 中，使 core/ 可被导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.plate_recognizer import PlateRecognizer
from ui_utils import pil_to_bgr, bgr_to_rgb, resize_for_display

# ── 从 config.yaml 动态生成模型选项 ─────────────────────────────────────────
def _load_model_options() -> dict:
    """读取 config.yaml，返回 {显示名称: 权重绝对路径}。"""
    config_path = os.path.join(_PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    preset = cfg['dataset']['preset']
    options = {}
    for m in cfg.get('models', []):
        if not m.get('enabled', True):
            continue
        # 显示名称：yolov8 → YOLOv8，yolov11 → YOLOv11
        display = m['name'].upper().replace('YOLOV', 'YOLOv')
        weight_abs = os.path.join(
            _PROJECT_ROOT, 'runs', preset,
            f'train_{m["name"]}_plate', 'weights', 'best.pt'
        )
        options[display] = weight_abs
    return options

_MODEL_OPTIONS = _load_model_options()

def _available_models() -> List[str]:
    return [name for name, path in _MODEL_OPTIONS.items()
            if os.path.exists(path)]


# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="智能车牌检测与识别系统",
    page_icon="car",
    layout="wide"
)


@st.cache_resource
def load_recognizer(model_name: str) -> Optional[PlateRecognizer]:
    weight_abs = _MODEL_OPTIONS.get(model_name, "")
    if not os.path.exists(weight_abs):
        return None
    return PlateRecognizer(yolo_weight=weight_abs)


# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 参数设置")

    available = _available_models()
    if not available:
        st.warning("未找到任何已训练的模型权重，请先运行训练脚本。")
        selected_model = None
    else:
        selected_model = st.selectbox("选择检测模型", available)

    conf_threshold = st.slider("置信度阈值", min_value=0.1, max_value=0.9,
                               value=0.5, step=0.05)

    st.divider()
    st.caption("模型权重路径")
    for name, path in _MODEL_OPTIONS.items():
        exists = os.path.exists(path)
        icon = "✅" if exists else "❌"
        rel = os.path.relpath(path, _PROJECT_ROOT)
        st.caption(f"{icon} {name}: {rel}")


# ── 主界面标题 ────────────────────────────────────────────────────────────────
st.title("智能车牌检测与识别系统")
st.markdown(
    "基于 **YOLO** 目标检测 + **PaddleOCR** 文字识别，"
    "实现对图片/视频中车牌的自动定位与号码识别。"
)
st.divider()

# ── 加载模型 ──────────────────────────────────────────────────────────────────
recognizer = None
if selected_model:
    with st.spinner(f"正在加载 {selected_model} 模型，请稍候..."):
        recognizer = load_recognizer(selected_model)
    if recognizer is None:
        st.error(f"模型 {selected_model} 权重文件不存在，请先完成训练。")

# ── 输入方式选择 ──────────────────────────────────────────────────────────────
tab_image, tab_video = st.tabs(["📷 图片识别", "🎬 视频识别"])


def _run_recognition(image_bgr, conf: float):
    """调用识别器，返回 (result_bgr, plates)。"""
    if recognizer is None:
        return image_bgr, []
    # 临时覆盖置信度阈值
    recognizer.yolo_model.overrides['conf'] = conf
    return recognizer.recognize(image_bgr)


# ── 图片 Tab ──────────────────────────────────────────────────────────────────
with tab_image:
    uploaded_file = st.file_uploader(
        "请选择一张包含车牌的图片（支持 JPG、PNG 格式）",
        type=["jpg", "jpeg", "png"],
        key="img_uploader"
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_bgr = pil_to_bgr(pil_image)

        if st.session_state.get('img_last_file') != uploaded_file.name:
            st.session_state['img_last_file'] = uploaded_file.name
            st.session_state.pop('img_result', None)
            st.session_state.pop('img_plates', None)

        if st.button("开始识别", type="primary", use_container_width=True, key="img_btn"):
            if recognizer is None:
                st.error("请先在侧边栏选择一个可用的模型。")
            else:
                with st.spinner("正在检测车牌并识别文字..."):
                    result_bgr, plates = _run_recognition(image_bgr, conf_threshold)
                st.session_state['img_result'] = bgr_to_rgb(result_bgr)
                st.session_state['img_plates'] = plates

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("原始图片")
            st.image(pil_image, use_container_width=True)
        with col_right:
            st.subheader("识别结果")
            if 'img_result' in st.session_state:
                st.image(st.session_state['img_result'], use_container_width=True)
            else:
                st.info("点击上方「开始识别」按钮查看结果")

        if 'img_plates' in st.session_state:
            plates = st.session_state['img_plates']
            st.divider()
            if not plates:
                st.warning("未在图片中检测到车牌，请尝试换一张图片。")
            else:
                st.markdown("**识别详情：**")
                for i, plate in enumerate(plates):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(label=f"车牌 #{i + 1}", value=plate['text'])
                    with c2:
                        st.metric(label="检测置信度", value=f"{plate['det_conf']:.0%}")
                    with c3:
                        st.metric(label="识别置信度", value=f"{plate['ocr_conf']:.0%}")
    else:
        st.info("请在上方上传一张包含车牌的图片，然后点击「开始识别」按钮。")


# ── 视频 Tab ──────────────────────────────────────────────────────────────────
with tab_video:
    video_file = st.file_uploader(
        "请选择一段包含车牌的视频（支持 MP4、AVI、MOV 格式）",
        type=["mp4", "avi", "mov"],
        key="vid_uploader"
    )

    if video_file is not None:
        if st.button("开始逐帧识别", type="primary", use_container_width=True, key="vid_btn"):
            if recognizer is None:
                st.error("请先在侧边栏选择一个可用的模型。")
            else:
                # 将上传的视频写入临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name

                cap = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_video = cap.get(cv2.CAP_PROP_FPS) or 25

                progress_bar = st.progress(0, text="正在处理视频帧...")
                frame_display = st.empty()
                plates_set: Set[str] = set()

                frame_idx = 0
                # 每隔 N 帧处理一次，避免处理过慢
                step = max(1, int(fps_video // 5))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % step == 0:
                        result_bgr, plates = _run_recognition(frame, conf_threshold)
                        for p in plates:
                            if p['text'] and p['text'] != "识别失败":
                                plates_set.add(p['text'])
                        display = resize_for_display(bgr_to_rgb(result_bgr), max_side=960)
                        frame_display.image(display, use_container_width=True)

                    progress = min((frame_idx + 1) / max(total_frames, 1), 1.0)
                    progress_bar.progress(progress, text=f"处理中... {frame_idx + 1}/{total_frames} 帧")
                    frame_idx += 1

                cap.release()
                os.unlink(tmp_path)
                progress_bar.progress(1.0, text="处理完成！")

                st.divider()
                if plates_set:
                    st.markdown("**视频中识别到的车牌号：**")
                    for plate_text in sorted(plates_set):
                        st.success(plate_text)
                else:
                    st.warning("视频中未识别到车牌。")
    else:
        st.info("请上传一段视频文件，然后点击「开始逐帧识别」按钮。")


# ── 页脚 ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("毕业设计项目 | YOLO + PaddleOCR | Streamlit")


def main():
    pass  # Streamlit 脚本本身即入口，main() 供外部 import 调用
