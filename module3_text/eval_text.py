# module3_text/eval_text.py
import os
import time
from text_detector import TextDetector
# ========== 配置开关 ==========
SHOW_ONE_PER_TYPE = True   # 仿真集打开；外部数据集ftype="未知"自动忽略
SHOW_WRONG_CASE = False    # 是否控制台逐条打印误报漏报，默认关闭
GRAY_LOW = 20
GRAY_HIGH = 50
LOG_DIR = "log"
FN_DIR = "fn_cases"
# ==================================
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

def _ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FN_DIR, exist_ok=True)

def save_log(stem, threshold, metrics, gray_stats, coop_recall, subdir=""):
    _ensure_dirs()
    if subdir:
        target_dir = os.path.join(LOG_DIR, subdir)
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = LOG_DIR
    log_path = os.path.join(target_dir, f"log_{stem}_t{threshold}.txt")
    lines = []
    lines.append("=" * 50)
    lines.append(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"数据集：{stem} | 阈值：{threshold}")
    lines.append("=" * 50)
    for k, v in metrics.items():
        lines.append(f"{k}: {v}")
    lines.append(f"灰度样本总数: {gray_stats['total']} | 诈骗:{gray_stats['fraud']} | 正常:{gray_stats['normal']}")
    lines.append(f"人机协同检出率(机器告警+灰度复核): {coop_recall:.4f}")
    lines.append("-" * 50)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    return log_path

def save_fn_cases(stem, fn_cases):
    _ensure_dirs()
    fn_path = os.path.join(FN_DIR, f"fn_{stem}.txt")
    with open(fn_path, "w", encoding="utf-8") as f:
        f.write(f"# 漏报样本（诈骗label=1被判正常）共{len(fn_cases)}条\n")
        f.write("# score\ttext\n")
        for score, text in fn_cases:
            f.write(f"{score:.2f}\t{text}\n")
    return fn_path

def evaluate(dataset_path="dataset/test_dataset_with_type.txt", threshold=50, log_subdir=""):
    detector = TextDetector()
    data = load_dataset(dataset_path)
    tp = fp = tn = fn = 0
    type_stats = {}
    printed_types = set()   # 记录已经打印过的类别
    gray_total = 0
    gray_fraud = 0
    gray_normal = 0
    fn_cases = []
    for text, label, ftype in data:
        score = detector.predict(text)
        pred = 1 if score >= threshold else 0
        # ========== 每类打印一条演示样例（你原版正确逻辑保留不动） ==========
        if SHOW_ONE_PER_TYPE and ftype != "未知" and ftype not in printed_types:
            printed_types.add(ftype)
            print(f"\n————【{ftype}】————")
            print(f"话术：{text}")
            print(f"得分：{score:.2f} | 真实标签：{label} | 判定：{pred}")
        # ========== 灰度统计 ==========
        if GRAY_LOW <= score < GRAY_HIGH:
            gray_total += 1
            if label == 1:
                gray_fraud += 1
            else:
                gray_normal += 1
        # ========== 二分类计数 ==========
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1
            fn_cases.append((score, text))
        
        # ========== 误报漏报打印，由开关控制 ==========
        if SHOW_WRONG_CASE:
            if label == 0 and pred == 1:
                print(f"[误报] score={score:.1f} | {text[:60]}")
            if label == 1 and pred == 0:
                print(f"[漏报] score={score:.1f} | {text[:60]}")
        
        # ========== 分层统计 ==========
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

    # ============ 新增：人机协同检出率 ============
    total_real_fraud = tp + fn
    if total_real_fraud > 0:
        coop_recall = (tp + gray_fraud) / total_real_fraud
    else:
        coop_recall = 0.0

    print("\n" + "=" * 55)
    print(f"评测数据集：{dataset_path}")
    print(f"决策阈值  ：{threshold} 分")
    print("=" * 55)
    print(f"总样本数   ：{len(data)}")
    print(f"准确率 Acc ：{accuracy:.4f}")
    print(f"精确率 Pre ：{precision:.4f}")
    print(f"召回率 Rec ：{recall:.4f}  ← 纯机器自动告警召回")
    print(f"F1 分数    ：{f1:.4f}")
    print(f"漏报率 FNR ：{fnr:.4f}  ← 诈骗被判正常（不进灰度）")
    print(f"误报率 FPR ：{fpr:.4f}  ← 正常被判诈骗")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 55)
    print(f"灰度样本统计（{GRAY_LOW} ≤ score < {GRAY_HIGH}）：")
    print(f"  总数={gray_total} | 诈骗={gray_fraud} | 正常={gray_normal}")
    print(f"✅ 人机协同检出率（机器告警+灰度人工复核）：{coop_recall:.4f}")
    print("=" * 55)
    print("分层准确率：")
    for ftype, stat in type_stats.items():
        acc = stat["correct"] / stat["total"]
        print(f"  {ftype:14s}: {acc:.2%}  ({stat['correct']}/{stat['total']})")
    
    # 保存日志和漏报
    metrics = {
        "总样本": len(data),
        "Acc": round(accuracy,4),
        "Pre": round(precision,4),
        "Rec": round(recall,4),
        "F1": round(f1,4),
        "FNR": round(fnr,4),
        "FPR": round(fpr,4),
        "TP/FP/TN/FN": f"{tp}/{fp}/{tn}/{fn}"
    }
    gray_stats = {"total":gray_total,"fraud":gray_fraud,"normal":gray_normal}
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    save_log(stem, threshold, metrics, gray_stats, coop_recall, subdir=log_subdir)
    if fn_cases:
        save_fn_cases(stem, fn_cases)
        print(f"\n[导出完成] 漏报样本：{FN_DIR}/fn_{stem}.txt")
    print(f"[日志保存] log/log_{stem}_t{threshold}.txt")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/test_dataset_with_type.txt"
    evaluate(dataset_path=path, threshold=50)
