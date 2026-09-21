# module2_audio 音频伪造检测模块

检测音频是否为 AI 合成（deepfake），返回真假判定与 0-100 风险分，供 module4 风险融合模块调用。

## 使用方法

```python
from module2_audio.audio_detector import detect_audio, detect_audio_with_details

is_fake, risk_score = detect_audio("测试音频.wav")
details = detect_audio_with_details("测试音频.wav")
# details = {"is_fake": True, "risk_score": 87.5, "result_text": "伪造音频", "risk_level": "高"}
```

命令行自测（在 module2_audio 目录下）：

```bash
python audio_detector.py
```

## 目录结构

```
module2_audio/
├── audio_detector.py            # 对外接口：detect_audio / detect_audio_with_details
├── requirements.txt             # 本模块依赖
├── src/
│   └── fake_audio_detection/
│       ├── extract_features.py  # 特征提取（MFCC / IMFCC / CQCC / 谱特征等，共 41 维）
│       ├── model.py             # SVM 训练管线（train_svm）与分段预测
│       └── normalization.py     # L2 归一化（sklearn Pipeline 兼容）
├── models/                      # 训练好的模型（不入库，见下）
└── dataset/                     # 本地数据集（不入库，见下）
```

## 模型文件与数据（团队规范）

按团队 Git 规范，模型与特征数据不入库：

- `models/my_model.joblib`：训练好的 SVM 模型（当前版本约 5KB，可联系模块负责人获取，或自行训练）
- `features.npy` / `labels.npy`：训练用特征与标签，可用下面的命令重新生成

重新训练 / 重新提取特征：

```bash
python -c "from src.fake_audio_detection.extract_features import extract_all_dataset; extract_all_dataset('dataset/train/data')"
python -c "from src.fake_audio_detection.model import train_svm; print(train_svm('dataset/train/data', 'models'))"
```

数据集目录约定：`dataset/train/data/REAL/`（真实音频）与 `dataset/train/data/FAKE/`（合成音频）。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 当前状态

- [x] 特征提取（chroma / RMS / 谱质心 / 带宽 / rolloff / 过零率 / tonnetz / 谱对比度 / MFCC20 / IMFCC13）
- [x] SVM 训练管线（L2 归一化 + RBF 核，输出概率）
- [x] 对外检测接口（含异常处理与详细版返回）
- [ ] 扩充训练数据，替换当前小样本 demo 模型
- [ ] 与 module4_pipeline/risk_fusion.py 对接联调
