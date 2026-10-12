# -*- coding: utf-8 -*-
"""
独立工具：从 test.mp4（国产AI生成风格代表）提取人脸样本
====================================================
作用：AEGIS（kling/sora）与 test.mp4 生成风格不同，把 test.mp4 的人脸
     加入训练集，确保模型覆盖"国产文生视频"风格。
产物：results/testmp4_faces.npz (faces, labels=1)
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "test.mp4")
OUT = os.path.join(HERE, "results", "testmp4_faces.npz")

SAMPLE_EVERY = 5     # 每5帧抽1帧（视频通常30fps -> 每秒6帧）
MIN_FACE = 48
EXPAND = 1.3
SIZE = 128
MAX_TOTAL = 40       # 最多取40张，避免过拟合单视频

sys.path.insert(0, HERE)
from face_detect import detect_faces

def main():
    if not os.path.exists(SRC):
        print("找不到 test.mp4:", SRC); return
    cap = cv2.VideoCapture(SRC)
    faces = []
    fc = 0
    while cap.isOpened() and len(faces) < MAX_TOTAL:
        ret, frame = cap.read()
        if not ret:
            break
        if fc % SAMPLE_EVERY == 0:
            for b in detect_faces(frame):
                x, y, w, h = b
                if max(w, h) < MIN_FACE:
                    continue
                cx, cy = x + w/2, y + h/2
                s = max(w, h) * EXPAND
                x0 = int(max(0, cx - s/2)); y0 = int(max(0, cy - s/2))
                x1 = int(min(frame.shape[1], cx + s/2)); y1 = int(min(frame.shape[0], cy + s/2))
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0 or min(crop.shape[:2]) < 20:
                    continue
                faces.append(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
                if len(faces) >= MAX_TOTAL:
                    break
        fc += 1
    cap.release()
    if not faces:
        print("test.mp4 未提取到人脸！"); return
    arr = np.stack(faces).astype(np.uint8)
    labels = np.ones(len(arr), dtype=np.int32)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, faces=arr, labels=labels)
    print("test.mp4 提取 %d 张人脸, 已保存: %s" % (len(arr), OUT))

if __name__ == "__main__":
    main()
