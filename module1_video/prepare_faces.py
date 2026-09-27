# -*- coding: utf-8 -*-
"""
Celeb-DF-v2 人脸数据集准备（独立工具，供微调训练使用）

从真实/伪造视频中抽帧、检测人脸、裁剪正方形，保存为 .npz 训练数据。

用法：
    python prepare_faces.py --out results/celeba_faces.npz \
        --real_num 150 --fake_num 150 --frames_per_video 2

输出：
    results/celeba_faces.npz：含 X(人脸图 uint8) 和 y(0=real, 1=fake)
    （另存 results/celeba_faces/ 目录便于人工抽查）
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_detect import detect_faces
from video_detector import _crop_square_face


def sample_faces_from_dir(d, label, max_videos, frames_per_video, out_list):
    names = sorted(n for n in os.listdir(d)
                   if n.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
    if max_videos > 0:
        names = names[:max_videos]
    got = 0
    for name in names:
        cap = cv2.VideoCapture(os.path.join(d, name))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:  # 每秒抽1帧（30fps）
                boxes = detect_faces(frame)
                for b in boxes:
                    face = _crop_square_face(frame, b)
                    if face.size == 0 or face.shape[0] < 40:
                        continue
                    face = cv2.resize(face, (128, 128))
                    out_list.append((face, label, name))
                    got += 1
                    if got >= frames_per_video:
                        break
            if got >= frames_per_video:
                break
            frame_count += 1
        cap.release()
        if got < frames_per_video and got > 0:
            pass
        # 一个视频最多贡献 frames_per_video 张
        # 重新计数不限制总视频数内，循环继续
        got = 0
    return out_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default=r"D:\大创\Celeb-DF-v2\Celeb-real")
    ap.add_argument("--fake_dir", default=r"D:\大创\Celeb-DF-v2\Celeb-synthesis")
    ap.add_argument("--youtube_dir", default=r"D:\大创\Celeb-DF-v2\YouTube-real")
    ap.add_argument("--out", default="results/celeba_faces.npz")
    ap.add_argument("--real_num", type=int, default=150, help="用多少真实视频")
    ap.add_argument("--fake_num", type=int, default=150, help="用多少伪造视频")
    ap.add_argument("--frames_per_video", type=int, default=2)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    samples = []
    sample_faces_from_dir(args.real_dir, 0, args.real_num // 2,
                          args.frames_per_video, samples)
    sample_faces_from_dir(args.youtube_dir, 0, args.real_num - args.real_num // 2,
                          args.frames_per_video, samples)
    sample_faces_from_dir(args.fake_dir, 1, args.fake_num,
                          args.frames_per_video, samples)

    if not samples:
        print("没有抽到任何样本，检查目录路径")
        return
    X = np.stack([s[0] for s in samples]).astype("uint8")
    y = np.array([s[1] for s in samples])
    print(f"样本数: {len(samples)}  其中伪造 {y.sum()} / 真实 {len(y)-y.sum()}")
    print(f"人脸图尺寸: {X.shape[1:]}")
    np.savez(out_path, X=X, y=y, names=[s[2] for s in samples])
    print(f"已保存: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
