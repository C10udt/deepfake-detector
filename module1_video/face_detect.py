# -*- coding: utf-8 -*-
"""
人脸检测模块（任务1 的前置步骤）

作用：
    从一帧图像中找出所有人脸的位置，供后续"裁剪 -> 送入 MesoNet"使用。
    换脸检测必须"先裁脸再判断"：MesoNet 是按人脸区域训练的，直接喂整帧会掉精度。

方案：
    OpenCV 官方 YuNet 人脸检测模型（face_detection_yunet_2023mar.onnx，约 227KB）
    - OpenCV 4.5.4+ 自带接口，无需额外 pip 包
    - 模型文件在 face_detector/ 目录下
    - 开源地址：https://github.com/opencv/opencv_zoo

降级：
    如果 onnx 模型文件缺失，则把整帧当作一个"人脸区域"处理，保证程序不崩溃（精度下降）。

用法：
    from face_detect import detect_faces
    boxes = detect_faces(frame_bgr)   # -> [(x, y, w, h), ...]
"""
import os

import cv2

DETECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detector")
YUNET_PATH = os.path.join(DETECTOR_DIR, "face_detection_yunet_2023mar.onnx")

_detector = None


def _get_detector(w, h):
    """懒加载 YuNet 检测器；模型缺失时返回 None"""
    global _detector
    if _detector is None:
        if not os.path.exists(YUNET_PATH):
            print(f"[face_detect] 警告: 未找到 {YUNET_PATH}，将退化为整帧检测（精度下降）")
            return None
        _detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            (w, h),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
    # YuNet 要求每次按当前帧尺寸设置输入大小
    _detector.setInputSize((w, h))
    return _detector


def detect_faces(frame):
    """
    输入：一帧 BGR 图像
    输出：人脸框列表 [(x, y, w, h), ...]，坐标为像素值
          - 无 YuNet 模型时：返回整帧框（降级）
          - 真没检测到人脸时：返回空列表
    """
    h, w = frame.shape[:2]
    det = _get_detector(w, h)
    if det is None:
        return [(0, 0, w, h)]

    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        return []

    boxes = []
    for f in faces:
        x, y, ww, hh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        # 越界保护 + 过滤过小的框
        x = max(0, x)
        y = max(0, y)
        ww = min(ww, w - x)
        hh = min(hh, h - y)
        if ww > 10 and hh > 10:
            boxes.append((x, y, ww, hh))
    return boxes
