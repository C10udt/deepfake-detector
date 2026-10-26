# -*- coding: utf-8 -*-
"""
独立工具：构建混合训练集（真实 + 换脸假 + AI生成假）
====================================================
1. 加载 celeba_faces.npz（真200/换脸假200）
2. 加载 aegis_ai_faces.npz（AI生成假64）
3. 从 Celeb-real/YouTube-real 额外抽真脸，把真实侧补齐到与假侧相等
4. 合并保存 results/mixed_faces.npz（X, y, names）

用法（Python3.8 环境）：
    C:\\Python\\Python38\\python.exe build_mixed_dataset.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "mixed_faces.npz")

REAL_DIR = r"D:\大创\Celeb-DF-v2\Celeb-real"
YT_DIR = r"D:\大创\Celeb-DF-v2\YouTube-real"

sys.path.insert(0, HERE)
from face_detect import detect_faces
from video_detector import _crop_square_face

def extra_real_faces(n_target, out_list):
    """从真实视频目录补抽真脸，直到 out_list 数量达标"""
    names = sorted(os.listdir(REAL_DIR) + os.listdir(YT_DIR))
    names = [n for n in names if n.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    for name in names:
        if len(out_list) >= n_target:
            break
        path = os.path.join(REAL_DIR, name) if os.path.exists(os.path.join(REAL_DIR, name)) \
            else os.path.join(YT_DIR, name)
        cap = cv2.VideoCapture(path)
        fc = 0
        while cap.isOpened() and len(out_list) < n_target:
            ret, frame = cap.read()
            if not ret:
                break
            if fc % 30 == 0:
                for b in detect_faces(frame):
                    face = _crop_square_face(frame, b)
                    if face.size == 0 or face.shape[0] < 40:
                        continue
                    out_list.append((cv2.resize(face, (128, 128)), 0, name))
                    if len(out_list) >= n_target:
                        break
            fc += 1
        cap.release()
    return out_list

def main():
    # 1. 现有训练集
    d = np.load(os.path.join(RES, "celeba_faces.npz"))
    X, y, names = d["X"], d["y"], d["names"]
    print("celeba: %d 张 (真%d/假%d)" % (len(X), (y == 0).sum(), y.sum()))

    # 2. AEGIS AI 假脸 + test.mp4 假脸
    a = np.load(os.path.join(RES, "aegis_ai_faces.npz"))
    Xa = a["faces"]
    print("aegis_ai: %d 张 (全部为假)" % len(Xa))
    tp = np.load(os.path.join(RES, "testmp4_faces.npz"))
    Xt = tp["faces"]
    print("testmp4: %d 张 (全部为假)" % len(Xt))

    # 3. 补齐真脸到平衡
    n_fake = y.sum() + len(Xa) + len(Xt)
    n_real_now = (y == 0).sum()
    need = n_fake - n_real_now
    extra = []
    if need > 0:
        extra_real_faces(need, extra)
        print("补抽真脸: %d 张" % len(extra))
    else:
        print("真脸已多于假脸，无需补充")

    Xe = np.stack([s[0] for s in extra]).astype("uint8") if extra else np.zeros((0, 128, 128, 3), dtype="uint8")
    ye = np.array([s[1] for s in extra], dtype=np.int32) if extra else np.zeros(0, dtype=np.int32)
    ne = [s[2] for s in extra] if extra else []

    # 4. 合并
    X_all = np.concatenate([X, Xa, Xt, Xe], axis=0)
    y_all = np.concatenate([y, np.ones(len(Xa), dtype=np.int32),
                            np.ones(len(Xt), dtype=np.int32), ye], axis=0)
    names_all = list(names) + ["aegis_ai_%d" % i for i in range(len(Xa))] + \
                ["testmp4_%d" % i for i in range(len(Xt))] + ne
    print("\n混合训练集: %d 张 (真%d / 换脸假%d / AI假%d[含test.mp4 %d])" %
          (len(X_all), (y_all == 0).sum(), y.sum(), len(Xa) + len(Xt), len(Xt)))
    np.savez(OUT, X=X_all, y=y_all, names=names_all)
    print("已保存:", OUT, "(%.1f MB)" % (os.path.getsize(OUT) / 1e6))

if __name__ == "__main__":
    main()
