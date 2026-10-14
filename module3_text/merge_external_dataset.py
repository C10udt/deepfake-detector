# module3_text/merge_external_dataset.py
import os
import random
import pandas as pd

# 数据文件所在目录
DATA_DIR = "external_data"
OUTPUT_PATH = "dataset/external_dataset.txt"

# 标签映射 & 抽样数量
DATA_CONFIG = [
    ("label00-last.csv", 0, 250),  # 正常文本抽 250 条
    ("label01-last.csv", 1, 63),   # 公共安全诈骗
    ("label02-last.csv", 1, 63),   # 贷款诈骗
    ("label03-last.csv", 1, 62),   # 冒充客服诈骗
    ("label04-last.csv", 1, 62),   # 冒充领导/熟人诈骗
]


def merge_and_sample():
    random.seed(42)
    all_rows = []

    for filename, label, n_sample in DATA_CONFIG:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[Warning] 文件不存在，跳过: {filepath}")
            continue

        # 尝试读取，兼容不同编码
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="gbk")

        texts = df["content"].astype(str).tolist()
        texts = [t.strip() for t in texts if len(t.strip()) >= 5]

        if len(texts) < n_sample:
            print(f"[Warning] {filename} 样本不足，实际 {len(texts)} 条")
            sampled = texts
        else:
            sampled = random.sample(texts, n_sample)

        print(f"读取 {filename}: 原始 {len(df)} 条 → 抽样 {len(sampled)} 条")
        for t in sampled:
            all_rows.append((t, label))

    random.shuffle(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for text, label in all_rows:
            text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{text}\t{label}\n")

    n_pos = sum(1 for _, l in all_rows if l == 1)
    n_neg = sum(1 for _, l in all_rows if l == 0)
    print(f"\n✅ 合并完成：{len(all_rows)} 条")
    print(f"   诈骗样本（1）：{n_pos} 条")
    print(f"   正常样本（0）：{n_neg} 条")
    print(f"   正负比例：{n_pos}:{n_neg}")
    print(f"   输出文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    merge_and_sample()