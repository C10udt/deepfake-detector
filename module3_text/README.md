## 评测说明

- 主评测集：自建仿真集，共192条（8类骗局 + 正常对话），专门面向空巢老人高发诈骗场景，用于效果展示、分类样例演示。
- 辅助评测集：天池电信诈骗五分类数据集抽样500条，作为外部真实数据集，用于泛化能力测试。
- 核心指标：仿真数据集最优阈值50，F1=0.995；外部天池数据集最优阈值30，F1=0.793。
- 说明：天池数据集覆盖广义电信诈骗，场景和空巢老人定向诈骗存在分布差异，该F1仅作为泛化能力参考；规则方法存在天然泛化短板，亲属借钱类灰度诈骗容易产生漏报。
- 阈值策略：分场景选用阈值；仿真演示场景使用阈值50（均衡精确率、误报）；面向真实陌生话术时选用阈值30，低分灰度样本送入人工复核，优先降低漏报。

---
## 一、 模块定位
- **输入**：ASR 转写出的文本文件（`.txt`，纯文本）
- **输出**：风险分数（`float`，0~100，越高代表诈骗风险越高）
- **方法**：基于分级关键词加权打分规则体系
  词库分为：亲属 kin、低风险 low_risk、中风险 mid_risk、高风险 high_risk、紧急 emergency、保密 secret；支持多词联动加分、豁免衰减，避免正常闲聊误报。

---
## 二、 目录结构
```

module3_text/
├── config/
│   └── rules.json               # 关键词库、分级权重配置
├── dataset/
│   ├── test_dataset.txt          # 仿真评测集（两列：文本 \t 标签）
│   ├── test_dataset_with_type.txt# 仿真评测集（三列：文本 \t 标签 \t 诈骗类别，用于按类抽样展示）
│   └── external_dataset.txt      # 外部天池抽样评测集（两列：文本 \t 标签）
├── external_data/                # 天池原始公开数据集存放目录
├── text_detector.py              # 核心检测器 + detect_text 对外接口
├── eval_text.py                  # 评测脚本：指标计算、漏报样本导出、灰度样本统计
├── gen_dataset_v2.py            # 仿真数据集生成脚本，固定随机种子保证复现
├── merge_external_dataset.py     # 外部数据集抽样、清洗合并脚本
├── plot_ablation.py              # 阈值消融绘图脚本，分别输出仿真集、外部集消融曲线
├── run_detect.py                 # 独立命令行检测工具，单文本快速打分
├── requirements.txt              # 本模块 Python 依赖
└── README.md                     # 模块说明文档（本文档）

## 三、 接口说明

### 3.1 核心接口：`detect_text(text_path: str) -> float`

**签名固定，与任务4对接**。

```python
from text_detector import detect_text

score = detect_text("input.txt")  # 返回 float，0~100
print(f"风险分：{score}")