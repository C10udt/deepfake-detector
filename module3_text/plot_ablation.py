import numpy as np
import matplotlib.pyplot as plt
from text_detector import TextDetector

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_at_threshold(detector, dataset_path, threshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                text = parts[0]
                label = int(parts[1])
                score = detector.predict(text)
                pred = 1 if score >= threshold else 0
                if pred == 1 and label == 1:
                    TP += 1
                elif pred == 1 and label == 0:
                    FP += 1
                elif pred == 0 and label == 0:
                    TN += 1
                elif pred == 0 and label == 1:
                    FN += 1
    pre = TP/(TP+FP) if (TP+FP) > 0 else 0
    rec = TP/(TP+FN) if (TP+FN) > 0 else 0
    fpr = FP/(FP+TN) if (FP+TN) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre+rec) > 0 else 0
    return pre, rec, fpr, f1


def draw_curve(detector, dataset_path, title_name):
    thresholds = [30, 35, 40, 45, 50, 55, 60]
    pre_list = []
    rec_list = []
    fpr_list = []
    f1_list = []
    for th in thresholds:
        p, r, fpr, f1 = evaluate_at_threshold(detector, dataset_path, th)
        pre_list.append(p)
        rec_list.append(r)
        fpr_list.append(fpr)
        f1_list.append(f1)
        print(f"【{title_name}】阈值={th:2d} | Pre={p:.3f} | Rec={r:.3f} | FPR={fpr:.3f} | F1={f1:.3f}")

    plt.figure(figsize=(12,7), dpi=100)
    plt.plot(thresholds, pre_list, marker='o', linewidth=2, label="精确率 Pre")
    plt.plot(thresholds, rec_list, marker='s', linewidth=2, label="机器召回 Rec")
    plt.plot(thresholds, fpr_list, marker='x', linewidth=2, label="误报率 FPR")
    plt.plot(thresholds, f1_list, marker='^', linewidth=2, color="#9932cc", label="F1-score")

    best_idx = np.argmax(f1_list)
    best_th = thresholds[best_idx]
    best_f1 = f1_list[best_idx]
    plt.scatter(best_th, best_f1, c="red", s=100, zorder=10)
    plt.annotate(f"F1最大:{best_f1:.3f}\n阈值={best_th}",
                 xy=(best_th, best_f1),
                 xytext=(best_th + 1.2, best_f1 + 0.04),
                 fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

    plt.xlabel("阈值", fontsize=12)
    plt.ylabel("指标数值", fontsize=12)
    plt.title(f"阈值消融曲线 - {title_name}", fontsize=14, pad=12)
    plt.xticks(thresholds)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    detector = TextDetector()
    # 在这里填你的两个数据集路径
    simu_path = "dataset/test_dataset_with_type.txt"    # 仿真数据集
    external_path = "dataset/external_dataset.txt"      # 外部天池数据集

    draw_curve(detector, simu_path, "仿真数据集")
    draw_curve(detector, external_path, "外部天池数据集")
