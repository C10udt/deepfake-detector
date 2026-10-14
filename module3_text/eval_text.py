# module3_text/eval_text.py
import os
from text_detector import TextDetector


def load_dataset(path):
    """加载数据集，支持带类别标签的版本"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                text, label = parts[0], int(parts[1])
                data.append((text, label, "未知"))
            elif len(parts) == 3:
                text, label, ftype = parts[0], int(parts[1]), parts[2]
                data.append((text, label, ftype))
    return data


def evaluate(dataset_path="dataset/test_dataset_with_type.txt", threshold=50):
    detector = TextDetector()
    data = load_dataset(dataset_path)

    tp = fp = tn = fn = 0
    type_stats = {}

    for text, label, ftype in data:
        score = detector.predict(text)
        pred = 1 if score >= threshold else 0

        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1
        
        if label == 0 and pred == 1:
            print(f"[误报] score={score:.1f} | {text[:60]}")
        if label == 1 and pred == 0:
            print(f"[漏报] score={score:.1f} | {text[:60]}")
        

        if ftype not in type_stats:
            type_stats[ftype] = {"total": 0, "correct": 0}
        type_stats[ftype]["total"] += 1
        if pred == label:
            type_stats[ftype]["correct"] += 1

    accuracy = (tp + tn) / len(data)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0   # 漏报率
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0   # 误报率

    print("=" * 55)
    print(f"评测数据集：{dataset_path}")
    print(f"决策阈值  ：{threshold} 分")
    print("=" * 55)
    print(f"总样本数   ：{len(data)}")
    print(f"准确率 Acc ：{accuracy:.4f}")
    print(f"精确率 Pre ：{precision:.4f}")
    print(f"召回率 Rec ：{recall:.4f}")
    print(f"F1 分数    ：{f1:.4f}")
    print(f"漏报率 FNR ：{fnr:.4f}  ← 诈骗被判正常")
    print(f"误报率 FPR ：{fpr:.4f}  ← 正常被判诈骗")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 55)
    print("分层准确率：")
    for ftype, stat in type_stats.items():
        acc = stat["correct"] / stat["total"]
        print(f"  {ftype:14s}: {acc:.2%}  ({stat['correct']}/{stat['total']})")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/test_dataset_with_type.txt"
    evaluate(dataset_path=path, threshold=50)