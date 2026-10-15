# -*- coding: utf-8 -*-
"""
MesoNet 换脸检测模型封装（任务1 的核心模型）

作用：
    加载开源预训练 MesoNet 权重（mesonet_model.h5），对单张"人脸裁剪图"输出伪造概率。

关键说明（为什么能直接用、不用自己训练）：
    - mesonet_model.h5 来自开源项目 hamster-deepfake-detector（MesoNet 架构，Apache 2.0 开源）
    - 该文件只保存了权重（没有模型结构），所以需要在这里手动重建 Meso-4 网络再加载权重
    - 已探测确认：这份权重对应的输入尺寸是 128x128（不是 README 里写的 256）
    - 模型输出：伪造概率 0~1，越大越可能是 AI 换脸

用法：
    from meso_detect import predict_face
    proba = predict_face(face_bgr_image)   # -> float 0~1
"""
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D,
)
from tensorflow.keras.models import Model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesonet_model.h5")
# 若存在微调权重（用 Celeb-DF 数据微调过，见 train_finetune.py），优先加载它
FINETUNED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results", "mesonet_finetuned.h5")
INPUT_SIZE = 128  # 与这份权重匹配的输入尺寸（已探测确认）

_model = None  # 全局只加载一次


def _resolve_model_path():
    if os.path.exists(FINETUNED_PATH):
        return FINETUNED_PATH
    return MODEL_PATH


def _build_mesonet():
    """重建 Meso-4 网络结构（层名必须与 mesonet_model.h5 里的权重名一一对应）"""
    x = Input((INPUT_SIZE, INPUT_SIZE, 3))
    c = Conv2D(8, (3, 3), padding="same", activation="relu", name="conv2d")(x)
    c = BatchNormalization(name="batch_normalization")(c)
    c = MaxPool2D((2, 2), padding="same")(c)

    c = Conv2D(8, (5, 5), padding="same", activation="relu", name="conv2d_1")(c)
    c = BatchNormalization(name="batch_normalization_1")(c)
    c = MaxPool2D((2, 2), padding="same")(c)

    c = Conv2D(16, (5, 5), padding="same", activation="relu", name="conv2d_2")(c)
    c = BatchNormalization(name="batch_normalization_2")(c)
    c = MaxPool2D((2, 2), padding="same")(c)

    c = Conv2D(16, (5, 5), padding="same", activation="relu", name="conv2d_3")(c)
    c = BatchNormalization(name="batch_normalization_3")(c)
    c = MaxPool2D((2, 2), padding="same")(c)

    y = Flatten()(c)
    y = Dense(16, activation="relu", name="dense")(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation="sigmoid", name="dense_1")(y)
    return Model(x, y)


def get_model():
    """获取全局 MesoNet 模型（懒加载，只加载一次）"""
    global _model
    if _model is None:
        path = _resolve_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"模型文件不存在: {path}\n"
                f"请把 mesonet_model.h5 放在本文件夹下（来自 hamster-deepfake-detector 开源项目）"
            )
        _model = _build_mesonet()
        _model.load_weights(path)  # 加载开源预训练权重（微调权重优先），无需训练
        if path.endswith("mesonet_finetuned.h5"):
            print("[meso_detect] 已加载 Celeb-DF 微调权重 mesonet_finetuned.h5")
    return _model


def predict_face(face_bgr):
    """
    输入：一张 BGR 人脸图像（任意尺寸，通常是从视频帧裁剪出来的人脸）
    输出：伪造概率 float（0~1，越大越可能是 AI 换脸）
    """
    img = cv2.resize(face_bgr, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    model = get_model()
    proba = model.predict(img, verbose=0)[0][0]
    return float(proba)
