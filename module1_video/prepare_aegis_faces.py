# -*- coding: utf-8 -*-
"""
独立工具（v2）：从 AEGIS AI 生成视频提取人脸训练样本 —— 带跨帧一致性过滤
=====================================================================
v1 的问题：YuNet 对 AI 生成画面误检严重（月亮/火车/猫/花被判成人脸）。
v2 修复：
  1. 提高检测阈值 score_threshold 0.6 -> 0.8
  2. 跨帧一致性：同一张脸必须出现在 >=2 个不同抽帧（位置相近）才保留
     —— 真脸稳定出现，单帧误检会被滤掉
  3. 输出前保存抽样网格图，供人工目检确认

产物：
  results/aegis_ai_faces.npz     (faces, labels=1, 均为 AI 生成假脸)
  results/aegis_ai_manifest.csv  (来源明细)
  results/aegis_ai_sample_grid.png (抽样图供检查)
"""
import os
import csv

import cv2
import numpy as np

BASE = r"D:\大创\hard_test_set_video\test_data\ai_gen"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = os.path.join(HERE, "results", "aegis_ai_faces.npz")
OUT_CSV = os.path.join(HERE, "results", "aegis_ai_manifest.csv")
OUT_GRID = os.path.join(HERE, "results", "aegis_ai_sample_grid.png")

FRAMES_PER_VIDEO = 24      # 每视频抽帧数
MIN_FACE = 48              # 人脸最小边长(px)（v4: 56->48 继续扩大样本量）
MAX_PER_VIDEO = 4          # 每视频最多保留人脸数（v3: 2->4 扩大样本量）
EXPAND = 1.3               # 外扩系数
SIZE = 128

_det = None

def get_detector(w, h):
    global _det
    if _det is None:
        p = os.path.join(HERE, "face_detector", "face_detection_yunet_2023mar.onnx")
        _det = cv2.FaceDetectorYN.create(p, "", (w, h),
                                         score_threshold=0.8, nms_threshold=0.3, top_k=5000)
    _det.setInputSize((w, h))
    return _det

def cluster_faces(dets):
    """把不同帧的检测框按位置聚簇；返回出现>=2帧的簇（代表框=最大框）"""
    clusters = []
    for d in dets:  # d = (x, y, w, h)
        cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
        s = max(d[2], d[3])
        for c in clusters:
            rep = c[0]
            cx2, cy2 = rep[0] + rep[2] / 2, rep[1] + rep[3] / 2
            s2 = max(rep[2], rep[3])
            if abs(cx - cx2) < 0.4 * max(s, s2) and abs(cy - cy2) < 0.4 * max(s, s2):
                c.append(d)
                break
        else:
            clusters.append([d])
    out = []
    for c in clusters:
        if len(c) >= 2:  # 跨帧一致性：至少2帧检出
            out.append(max(c, key=lambda t: max(t[2], t[3])))
    return out

def crop_resize(frame, box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h) * EXPAND
    x0 = int(max(0, cx - s / 2)); y0 = int(max(0, cy - s / 2))
    x1 = int(min(frame.shape[1], cx + s / 2)); y1 = int(min(frame.shape[0], cy + s / 2))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return cv2.resize(frame[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)

def main():
    faces, meta, frames_pool = [], [], []
    dirs = [os.path.join(BASE, "kling"), os.path.join(BASE, "sora")]
    total_vids = 0
    for d in dirs:
        vids = sorted(f for f in os.listdir(d) if f.lower().endswith(".mp4"))
        for v in vids:
            path = os.path.join(d, v)
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release(); continue
            det = get_detector(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            dets = []       # 跨帧收集 (x,y,w,h)
            frame_store = {}  # 帧号 -> frame
            idxs = sorted(set(int(i * total / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)))
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    continue
                _, fs = det.detect(frame)
                if fs is None:
                    continue
                for f in fs:
                    x, y, w, h = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
                    if max(w, h) >= MIN_FACE:
                        dets.append((x, y, w, h))
                        frame_store[len(dets) - 1] = frame
            cap.release()
            keep = cluster_faces(dets)
            if not keep:
                continue
            keep.sort(key=lambda b: max(b[2], b[3]), reverse=True)
            # 取对应帧裁剪
            for b in keep[:MAX_PER_VIDEO]:
                # 找一帧（任意出现帧）裁剪
                for idx, dd in enumerate(dets):
                    if dd == b:
                        frame = frame_store[idx]
                        break
                crop = crop_resize(frame, b)
                if crop is None:
                    continue
                faces.append(crop)
                meta.append({"sub": os.path.basename(d), "video": v, "face_px": max(b[2], b[3])})
                frames_pool.append(crop)
            if keep:
                total_vids += 1
            print("[%s] %s -> %d 张 (簇数 %d)" % (os.path.basename(d), v, min(len(keep), MAX_PER_VIDEO), len(keep)), flush=True)

    if not faces:
        print("没有提取到任何样本！")
        return
    arr = np.stack(faces).astype(np.uint8)
    labels = np.ones(len(arr), dtype=np.int32)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, faces=arr, labels=labels)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.DictWriter(f, fieldnames=["sub", "video", "face_px"])
        wtr.writeheader(); wtr.writerows(meta)

    # 抽样网格图（最多16张）
    n = min(16, len(arr))
    sel = np.linspace(0, len(arr) - 1, n).astype(int)
    cols, rows = 4, 4
    canvas = np.full((rows * 132, cols * 132, 3), 200, dtype=np.uint8)
    for i, idx in enumerate(sel):
        r, c = i // cols, i % cols
        canvas[r*132+2:r*132+130, c*132+2:c*132+130] = arr[idx]
    cv2.imwrite(OUT_GRID, canvas)

    print("\n======== 完成 ========")
    print("有效视频数: %d" % total_vids)
    print("提取人脸总数: %d 张" % len(arr))
    print("保存:", OUT_NPZ)
    print("明细:", OUT_CSV)
    print("抽样图:", OUT_GRID)

if __name__ == "__main__":
    main()
