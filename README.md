# 基于 YOLOv11 的智能车牌检测与识别系统

本项目为本科毕业设计项目，主要研究并实现一个具备较高准确率和实时性的智能车牌自动检测与识别系统。系统以最新的 YOLOv11 目标检测算法为核心，结合 PaddleOCR 技术，实现复杂场景下的车牌定位与字符提取，并提供基于 Streamlit 的可视化交互界面。

## 1. 项目结构

```text
bysj/
├── datasets/               # 数据集目录 (需自行下载CCPD并转换)
├── models/                 # 模型权重目录 (存放YOLO和OCR权重)
├── core/                   # 核心算法代码 (检测、识别、流水线)
├── scripts/                # 辅助脚本 (数据处理、训练、评估)
├── app/                    # 可视化系统界面代码 (Streamlit)
├── assets/                 # 静态资源 (测试图片/视频)
├── runs/                   # 训练日志目录
├── requirements.txt        # 项目依赖包列表
└── Project_Plan.md         # 项目详细规划文档
```

## 2. 环境配置

建议使用 Anaconda 创建虚拟环境：

```bash
conda create -n plate_env python=3.8
conda activate plate_env
pip install -r requirements.txt
```

*注意：如果需要使用 GPU 加速，请根据你的 CUDA 版本安装对应的 PyTorch 和 PaddlePaddle 版本。*

## 3. 快速开始

### 3.1 数据准备
1. 下载 CCPD 数据集并解压到 `datasets/CCPD2019/`。
2. 运行数据转换脚本：
   ```bash
   python scripts/ccpd_to_yolo.py
   ```

### 3.2 模型训练
运行 YOLO 训练脚本：
```bash
python scripts/train_yolo.py
```

### 3.3 启动可视化系统
```bash
streamlit run app/app.py
```

## 4. 核心技术栈
*   **目标检测**: YOLOv11 (Ultralytics)
*   **字符识别**: PaddleOCR
*   **Web 界面**: Streamlit
*   **图像处理**: OpenCV, PIL
