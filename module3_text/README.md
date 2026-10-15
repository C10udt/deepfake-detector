## 评测说明

- 主评测集：自建仿真集，192 条（8 类骗局 + 正常），覆盖空巢老人高发场景。
- 辅助评测集：天池电信诈骗五分类数据集抽样 500 条。
- 核心指标：仿真集 F1=1.0；外部集 F1=0.79，FPR=0.05。
- 说明：天池数据集为广义电信诈骗，与空巢老人场景不重叠，F1 仅作泛化参考。

# module3_text：空巢老人诈骗文本检测模块

本模块是"空巢老人诈骗检测系统"的**文本检测部分**，接收音频模块（任务4）
ASR 转写出的文本文件，输出 0~100 的诈骗风险分数。

---

## 一、 模块定位

- **输入**：ASR 转写出的文本文件（`.txt`，纯文本）
- **输出**：风险分数（`float`，0~100，越高越危险）
- **方法**：基于三级关键词规则（亲属词、紧急词、高危词、保密词）加权打分

---

## 二、 目录结构
module3_text/
├── config/
│ └── rules.json # 关键词库 + 权重配置
├── dataset/
│ ├── test_dataset.txt # 仿真评测集（两列：文本\t标签）
│ ├── test_dataset_with_type.txt # 仿真评测集（三列：文本\t标签\t类别）
│ └── external_dataset.txt # 外部真实集抽样（两列）
├── external_data/ # 原始公开数据集（天池电信诈骗五分类）
├── text_detector.py # 核心检测器 + detect_text 接口
├── eval_text.py # 评测脚本
├── gen_dataset_v2.py # 仿真数据集生成脚本
├── merge_external_dataset.py # 外部数据集合并脚本
├── run_detect.py # 命令行检测工具
├── requirements.txt # 本模块依赖
└── README.md # 本文件

## 三、 接口说明

### 3.1 核心接口：`detect_text(text_path: str) -> float`

**签名固定，与任务4对接**。

```python
from text_detector import detect_text

score = detect_text("input.txt")  # 返回 float，0~100
print(f"风险分：{score}")