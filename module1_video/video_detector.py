# -*- coding: utf-8 -*-
"""
视频深度伪造检测模块 —— 任务1 主入口（双模型合判版）

模块定位：检测"假货"——不仅限于 AI 换脸，也包括全 AI 生成的视频。
（申请书写的是"换脸检测"，但本模块实际覆盖范围更广，答辩时可如实说明）

接口约定（与 git 仓库《接口契约》一致，其他模块直接调用）：
    detect_video(video_path: str) -> tuple[bool, float]
        - bool  : True = 检测到伪造（假），False = 真实（真）
        - float : 风险分 0~100，越高越可能是伪造

双模型合判逻辑（为什么这么做，见 README.md）：
    MesoNet   ：专长是"AI 换脸"（人脸区域被替换）
    XceptionNet：在 FaceForensics++ 上微调，对"伪造 vs 真实"整体更敏感，
                能响应全 AI 生成视频（GAN/扩散模型造出来的人脸）
    同一张人脸帧，两个模型各出一个伪造概率，取"最高分"作为该帧结论——
    任何一个模型认为可疑，就按可疑算（宁可误报，不可漏过诈骗视频）。

整体流程：
    视频 -> 每 SAMPLE_INTERVAL 帧采样一帧 -> 人脸检测(YuNet) -> 裁剪人脸 ->
    双模型预测 -> 每帧取最高伪造概率 -> 全部帧取平均 -> 转 0~100 风险分 -> 判定真/假

顶部参数可调：
    SAMPLE_INTERVAL : 采样间隔（越小越准越慢）
    RISK_THRESHOLD  : 风险分阈值（Celeb-DF 实测，60 比 50 误报更低、检出率相同）
    MODEL_MODE      : "meso"=只用 MesoNet（默认，微调后已实测可用）；"dual"=双模型合判（暂不推荐，Xception 权重质量差会误报真实视频）
"""
import os

import cv2
import numpy as np

from face_detect import detect_faces
from meso_detect import predict_face

SAMPLE_INTERVAL = 10   # 每隔多少帧检测一次（可调：越小越准越慢）
RISK_THRESHOLD = 60    # 风险分阈值：>=60 判定为伪造（Celeb-DF 实测调优值）
MODEL_MODE = "meso"    # meso（默认，微调后实测）/ dual / xception

# XceptionNet 是可选增强（文件较大，缺失时自动降级只用 MesoNet）
_xception_predictor = None
_xception_available = None


def _get_xception_predictor():
    """懒加载 XceptionNet；权重文件缺失时返回 None（自动降级，不阻断主流程）"""
    global _xception_predictor, _xception_available
    if _xception_available is not None:
        return _xception_predictor
    try:
        from xception_detect import predict_face_xception
        _xception_predictor = predict_face_xception
        _xception_available = True
    except Exception as e:
        print(f"[video_detector] XceptionNet 未启用（{e}），仅用 MesoNet 检测")
        _xception_predictor = None
        _xception_available = False
    return _xception_predictor


def _crop_square_face(frame, box):
    """
    把人脸框扩展成正方形再裁剪（MesoNet 按正方形人脸训练，直接裁剪会拉伸变形）
    :param frame: BGR 帧
    :param box: (x, y, w, h)
    :return: 正方形人脸区域（BGR）
    """
    x, y, w, h = box
    side = max(w, h)                      # 取较长边作为正方形边长
    cx, cy = x + w // 2, y + h // 2       # 人脸中心
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(frame.shape[1], x0 + side)
    y1 = min(frame.shape[0], y0 + side)
    return frame[y0:y1, x0:x1]


def _frame_fake_proba(face):
    """
    单张人脸 -> 伪造概率（0~1）
    双模型合判：MesoNet 与 XceptionNet 各出一个概率，取最高分。
    """
    p_meso = predict_face(face)

    if MODEL_MODE == "meso":
        return p_meso

    xcep = _get_xception_predictor()
    if xcep is None:  # Xception 权重缺失/加载失败：降级只用 MesoNet
        return p_meso

    if MODEL_MODE == "xception":
        return xcep(face)

    # dual：取两个模型中的最高分
    try:
        p_xcep = xcep(face)
    except Exception as e:
        print(f"[video_detector] Xception 单帧预测失败（{e}），该帧用 Meso 分数")
        return p_meso
    return max(p_meso, p_xcep)


def detect_video(video_path: str):
    """
    深度伪造检测接口（供其他模块/界面调用）
    :param video_path: 视频文件路径字符串
    :return: (is_fake: bool, score: float)，score 范围 0~100，越高风险越大
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video_detector] 无法打开视频: {video_path}")
        return False, 0.0

    scores = []
    face_count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 SAMPLE_INTERVAL 帧采样一次
        if frame_count % SAMPLE_INTERVAL == 0:
            boxes = detect_faces(frame)
            for (x, y, w, h) in boxes:
                face = _crop_square_face(frame, (x, y, w, h))
                if face.size == 0:
                    continue
                proba = _frame_fake_proba(face)   # 伪造概率 0~1
                scores.append(proba)
                face_count += 1

        frame_count += 1

    cap.release()

    if not scores:
        # 整个视频没检测到人脸：无法判断，按"未检测到人脸"处理
        print("[video_detector] 未在视频中检测到人脸，无法判断")
        return False, 0.0

    fake_proba = float(np.mean(scores))          # 所有帧的平均伪造概率
    risk_score = round(fake_proba * 100, 2)      # 转成 0~100 风险分
    is_fake = risk_score >= RISK_THRESHOLD
    return is_fake, risk_score


def detect_deepfake_video(video_path: str):
    """
    兼容原来 2.py 的调用方式：返回中文结果 + 风险分
    :return: ("假（深度伪造）" 或 "真", 风险分 0~100)
    """
    is_fake, risk = detect_video(video_path)
    result = "假（深度伪造）" if is_fake else "真"
    return result, risk


if __name__ == "__main__":
    # 自测入口：python video_detector.py
    video = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.mp4")
    if not os.path.exists(video):
        video = "test.mp4"
    if not os.path.exists(video):
        print("请先放一个 test.mp4 到本文件夹做测试")
    else:
        print("当前模型模式:", MODEL_MODE)
        is_fake, risk = detect_video(video)
        print("是否伪造:", is_fake)
        print("风险分:", risk)
