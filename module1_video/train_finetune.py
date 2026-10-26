# -*- coding: utf-8 -*-
"""
MesoNet 微调脚本（独立工具）——用 Celeb-DF-v2 人脸数据做领域适配

背景：
    开源预训练 MesoNet（hamster 权重）在 Celeb-DF-v2 上区分度很差
    （真实44.7分 vs 伪造48.9分，几乎无法区分）。用本地 Celeb-DF 数据
    微调（迁移学习），让模型适应该数据集的换脸风格。

做法：
    从预训练权重出发，全网络小学习率微调（不是从零训练）。
    MesoNet 极轻量（128x128 输入），CPU 也能快速训练。

用法：
    python train_finetune.py [--epochs 15] [--lr 0.0001] [--batch 16]

输出：
    results/mesonet_finetuned.h5  （微调后权重，meso_detect.py 会自动优先加载）
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meso_detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/celeba_faces.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/mesonet_finetuned.h5")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, args.data)
    if not os.path.exists(data_path):
        print(f"找不到数据: {data_path}，先运行 prepare_faces.py")
        return

    d = np.load(data_path)
    X, y = d["X"], d["y"]
    print(f"加载 {len(X)} 张人脸图（伪造 {y.sum()} / 真实 {len(y)-y.sum()}）")

    # 数据增强：轻度翻转（换脸伪造在左右翻转后仍应判假，增强泛化）
    Xf, yf = [], []
    for img, lab in zip(X, y):
        Xf.append(img)
        yf.append(lab)
        Xf.append(cv2_flip(img))
        yf.append(lab)
    X = np.stack(Xf).astype("float32") / 255.0
    y = np.array(yf, dtype="float32")

    # 随机打乱 + 8:2 划分
    idx = np.random.RandomState(42).permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    # 从预训练权重构建模型
    model = meso_detect._build_mesonet()
    pretrained = os.path.join(base, meso_detect.MODEL_PATH)
    if not os.path.exists(pretrained):
        print(f"找不到预训练权重: {pretrained}")
        return
    model.load_weights(pretrained)
    print("已加载预训练权重:", pretrained)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss="binary_crossentropy", metrics=["accuracy"])

    print(f"开始微调：epochs={args.epochs} lr={args.lr} batch={args.batch} "
          f"训练{len(X_tr)} 验证{len(X_va)}")
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=args.epochs, batch_size=args.batch, verbose=1)

    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save_weights(out_path)
    print(f"微调权重已保存: {out_path}")


def cv2_flip(img):
    import cv2
    return cv2.flip(img, 1)


if __name__ == "__main__":
    main()
