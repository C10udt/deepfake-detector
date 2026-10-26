# -*- coding: utf-8 -*-
"""
module2_audio 域外(OOD)泛化测试
用途: 用训练阶段从未出现过的说话人/语种/合成引擎的音频, 评估模型泛化能力。
运行: 在 module2_audio 目录下执行  python run_ood_test.py
输出: 终端打印 + reports/ood_test_log.txt

约定: 标签 0=伪造(FAKE), 1=真实(REAL), 与训练一致; 风险分>50 判为 AI 合成。
"""
import os
import sys
import glob
import subprocess
from datetime import datetime

import joblib
import numpy as np
import librosa
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "src", "fake_audio_detection"))
from extract_features import extract_features_from_array  # noqa: E402

MODEL_PATH = os.path.join(CURRENT_DIR, "models", "my_model.joblib")
OOD_DIR = os.path.join(CURRENT_DIR, "dataset", "ood_test", "data")
TRAIN_DIR = os.path.join(CURRENT_DIR, "dataset", "train", "data")
REPORT_DIR = os.path.join(CURRENT_DIR, "reports")


def predict_file(pipeline, path):
    """对单个音频文件返回 (is_fake, risk_score)"""
    y, sr = librosa.load(path, sr=22050)
    feats = extract_features_from_array(y, sr).reshape(1, -1)
    proba = pipeline.predict_proba(feats)[0]
    risk = float(proba[0]) * 100  # 类别0=伪造
    return risk > 50, round(risk, 2)


def evaluate(pipeline, data_dir, log, title):
    """遍历目录下 REAL/FAKE 两类音频并输出逐条结果与指标"""
    rows = []
    for sub, label in [("REAL", 1), ("FAKE", 0)]:
        for f in sorted(glob.glob(os.path.join(data_dir, sub, "*"))):
            if not f.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                continue
            is_fake, risk = predict_file(pipeline, f)
            ok = (is_fake == (label == 0))
            tag = "真实" if label == 1 else "合成"
            verdict = "AI合成" if is_fake else "真实人声"
            line = f"  [{tag}] {os.path.basename(f):<22} 风险分: {risk:>6.2f}/100  判定: {verdict}  {'√' if ok else '×'}"
            log(line)
            rows.append((label, 0 if is_fake else 1, risk))

    y_true = [r[0] for r in rows]
    y_pred = [r[1] for r in rows]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "样本数": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision(伪造类)": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall(伪造类)": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1(伪造类)": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "混淆矩阵[伪造/真实]": cm.tolist(),
        "真实类平均风险分": round(float(np.mean([r[2] for r in rows if r[0] == 1])), 2) if any(r[0] == 1 for r in rows) else None,
        "伪造类平均风险分": round(float(np.mean([r[2] for r in rows if r[0] == 0])), 2) if any(r[0] == 0 for r in rows) else None,
    }
    log(f"  -- {title} 汇总: " + ", ".join(f"{k}={v}" for k, v in metrics.items() if k != "混淆矩阵[伪造/真实]"))
    log(f"     混淆矩阵 (行=真实标签[伪造,真实], 列=预测[伪造,真实]): {metrics['混淆矩阵[伪造/真实]']}")
    return metrics


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    log_path = os.path.join(REPORT_DIR, "ood_test_log.txt")
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=CURRENT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"

    model = joblib.load(MODEL_PATH)
    pipeline = model["pipeline"]

    log("=" * 70)
    log("module2_audio 域外(OOD)泛化测试")
    log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"代码版本: {commit}  模型: {os.path.basename(MODEL_PATH)}")
    log("判定规则: 风险分 > 50 判为 AI 合成 (标签: 0=伪造, 1=真实)")
    log("=" * 70)

    log("")
    log("[A] 域内测试集(训练分布, 供对照):")
    in_metrics = evaluate(pipeline, TRAIN_DIR, log, "域内")

    log("")
    log("[B] 域外(OOD)测试集(全新说话人/语种/合成引擎, 训练未见过):")
    ood_metrics = evaluate(pipeline, OOD_DIR, log, "域外")

    log("")
    log("[C] 泛化差距:")
    log(f"  accuracy: 域内 {in_metrics['accuracy']:.4f} -> 域外 {ood_metrics['accuracy']:.4f} "
        f"(下降 {in_metrics['accuracy'] - ood_metrics['accuracy']:.4f})")
    log("=" * 70)

    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"\n日志已写入: {log_path}")


if __name__ == "__main__":
    main()
