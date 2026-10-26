# ablation.py
from eval_text import evaluate

if __name__ == "__main__":
    th_list = [30, 35, 40, 45, 50, 55, 60]
    sim_path = "dataset/test_dataset_with_type.txt"
    ext_path = "dataset/external_dataset.txt"
    for th in th_list:
        print("\n" + "#"*60)
        print(f"===== 阈值 = {th} =====")
        print("【仿真数据集】")
        evaluate(sim_path, threshold=th, log_subdir="ablation")
        print("\n【外部天池数据集】")
        evaluate(ext_path, threshold=th, log_subdir="ablation")