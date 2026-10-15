# -*- coding: utf-8 -*-
"""
Celeb-DF-v2 批量测试工具（独立脚本，不影响核心检测流程）

作用：
    用真实/伪造视频批量跑任务1的 detect_video，收集每个视频的风险分，
    统计检测效果，并扫描不同判定阈值下的表现，给出阈值建议。

用法示例：
    # 按官方测试列表跑（从 List_of_testing_videos.txt 读取）
    python batch_test.py --test_list "D:\大创\Celeb-DF-v2\List_of_testing_videos.txt" --limit 30

    # 直接指定真实/伪造目录，每类取 10 个
    python batch_test.py --real_dir "D:\大创\Celeb-DF-v2\Celeb-real" \
                         --fake_dir "D:\大创\Celeb-DF-v2\Celeb-synthesis" --limit 10

    # 调节参数
    --interval 15    # 采样间隔（越大越快越粗，默认 10）
    --mode dual      # dual=双模型(默认) / meso / xception
    --limit 50       # 每类最多测多少个（默认 0=全部）
    --threshold 50   # 判定阈值（默认取 video_detector.RISK_THRESHOLD）

输出：
    results/celebdf_batch_results.csv   每个视频一行：文件、真实标签、风险分、判定
    控制台统计报告：各类平均分、准确率、阈值扫描表
"""
import argparse
import csv
import os
import sys

# 允许从本文件所在目录 import 核心模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import video_detector as vd


def collect_videos(real_dir, fake_dir, youtube_dir, test_list, limit):
    """收集 (视频路径, 真实标签) 列表；标签: real / fake"""
    items = []  # (path, label)

    def add_dir(d, label):
        if not d or not os.path.isdir(d):
            return
        names = sorted(os.listdir(d))
        names = [n for n in names if n.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if limit > 0:
            names = names[:limit]
        for n in names:
            items.append((os.path.join(d, n), label))

    if test_list and os.path.isfile(test_list):
        # 官方测试列表格式：每行 "目录/文件名.mp4"，目录相对数据根目录
        root = os.path.dirname(os.path.abspath(test_list))
        lines = [l.strip() for l in open(test_list, encoding="utf-8", errors="ignore") if l.strip()]
        if limit > 0:
            lines = lines[:limit]
        for line in lines:
            # 兼容两种格式：纯路径 / 带行号前缀（"1 YouTube-real/00170.mp4"）
            parts = line.split()
            if not parts:
                continue
            rel = parts[-1].replace("/", os.sep)
            p = os.path.join(root, rel)
            label = "fake" if "synthesis" in line else "real"
            if os.path.isfile(p):
                items.append((p, label))
    else:
        add_dir(real_dir, "real")
        add_dir(fake_dir, "fake")
        add_dir(youtube_dir, "real")

    return items


def scan_thresholds(results):
    """在 30~70 之间扫描阈值，输出 准确率/检出率/误报率"""
    lines = []
    for thr in range(30, 71, 5):
        tp = tn = fp = fn = 0
        for label, score in results:
            pred_fake = score >= thr
            if label == "fake":
                if pred_fake: tp += 1
                else: fn += 1
            else:
                if pred_fake: fp += 1
                else: tn += 1
        acc = (tp + tn) / max(1, len(results))
        tpr = tp / max(1, tp + fn)      # 伪造检出率（越高越好）
        fpr = fp / max(1, fp + tn)      # 真实误报率（越低越好）
        lines.append((thr, acc, tpr, fpr, tp, fn, fp, tn))
    return lines


def main():
    ap = argparse.ArgumentParser(description="Celeb-DF-v2 批量测试")
    ap.add_argument("--test_list", default=None,
                    help=r"官方测试列表txt路径（如 D:\大创\Celeb-DF-v2\List_of_testing_videos.txt）")
    ap.add_argument("--real_dir", default=r"D:\大创\Celeb-DF-v2\Celeb-real")
    ap.add_argument("--fake_dir", default=r"D:\大创\Celeb-DF-v2\Celeb-synthesis")
    ap.add_argument("--youtube_dir", default=r"D:\大创\Celeb-DF-v2\YouTube-real")
    ap.add_argument("--limit", type=int, default=0, help="每类最多测多少个（0=全部）")
    ap.add_argument("--interval", type=int, default=None, help="覆盖采样间隔")
    ap.add_argument("--mode", default=None, help="覆盖模型模式 dual/meso/xception")
    ap.add_argument("--threshold", type=int, default=None, help="覆盖判定阈值")
    args = ap.parse_args()

    if args.interval:
        vd.SAMPLE_INTERVAL = args.interval
    if args.mode:
        vd.MODEL_MODE = args.mode
    if args.threshold:
        vd.RISK_THRESHOLD = args.threshold

    print(f"[batch_test] 模式={vd.MODEL_MODE} 采样间隔={vd.SAMPLE_INTERVAL} "
          f"判定阈值={vd.RISK_THRESHOLD}")

    items = collect_videos(args.real_dir, args.fake_dir, args.youtube_dir,
                           args.test_list, args.limit)
    print(f"[batch_test] 共收集 {len(items)} 个视频 "
          f"(真实 {sum(1 for _, l in items if l=='real')} / "
          f"伪造 {sum(1 for _, l in items if l=='fake')})")

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                exist_ok=True)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "celebdf_batch_results.csv")
    results = []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "risk_score", "prediction"])
        for i, (path, label) in enumerate(items, 1):
            try:
                is_fake, score = vd.detect_video(path)
            except Exception as e:
                print(f"[batch_test] 出错 {path}: {e}")
                continue
            pred = "fake" if is_fake else "real"
            results.append((label, score))
            writer.writerow([os.path.basename(path), label, score, pred])
            print(f"  [{i}/{len(items)}] {label:>4} | 风险分 {score:6.2f} | "
                  f"{'假' if is_fake else '真'} | {os.path.basename(path)}")

    # ---- 统计报告 ----
    reals = [s for l, s in results if l == "real"]
    fakes = [s for l, s in results if l == "fake"]
    print("\n========== 统计报告 ==========")
    print(f"真实视频 {len(reals)} 个：平均风险分 {sum(reals)/max(1,len(reals)):.2f}"
          f"（期望接近 0）")
    print(f"伪造视频 {len(fakes)} 个：平均风险分 {sum(fakes)/max(1,len(fakes)):.2f}"
          f"（期望接近 100）")

    thr_now = vd.RISK_THRESHOLD
    correct = sum(1 for l, s in results if (s >= thr_now) == (l == "fake"))
    print(f"\n当前阈值 {thr_now}：准确率 {correct}/{len(results)} "
          f"({correct/max(1,len(results))*100:.1f}%)")

    print("\n阈值扫描（acc=准确率 tpr=伪造检出率 fpr=真实误报率）：")
    print(f"{'阈值':>4} {'准确率':>8} {'检出率':>8} {'误报率':>8}  明细(TP/FN/FP/TN)")
    for thr, acc, tpr, fpr, tp, fn, fp, tn in scan_thresholds(results):
        flag = " <-- 当前" if thr == thr_now else ""
        print(f"{thr:>4} {acc*100:>7.1f}% {tpr*100:>7.1f}% {fpr*100:>7.1f}%   "
              f"{tp}/{fn}/{fp}/{tn}{flag}")

    print(f"\n明细已保存: {csv_path}")


if __name__ == "__main__":
    main()
