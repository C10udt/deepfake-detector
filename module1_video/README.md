# module1_video — 视频深度伪造检测（任务1，双模型合判版）

> 模块定位：**检测"假货"**——不只是 AI 换脸，还包括全 AI 生成的视频。
> 申请书写的是"换脸检测"，本模块实际覆盖更广（MesoNet 抓换脸 + XceptionNet 抓伪造整体），
> 答辩时可如实说明这是对申请书的**增强实现**。

## 接口（其他模块直接调用）
```python
from video_detector import detect_video
is_fake, score = detect_video("test.mp4")
# is_fake: True=深度伪造(假) / False=真实(真)
# score:   风险分 0~100，越高越危险
```

## 文件说明
| 文件 | 作用 | 来源 |
|---|---|---|
| `video_detector.py` | 主入口：抽帧→检测人脸→裁剪→双模型合判→返回 (真/假, 0~100分) | **自己写的"流程拼接"** |
| `meso_detect.py` | 加载开源 MesoNet 权重，对单张人脸输出伪造概率（专长：换脸） | 模型开源，只写了加载代码 |
| `xception_net.py` | XceptionNet 网络结构定义（FaceForensics++ 官方实现） | **开源直接使用** |
| `xception_detect.py` | 加载开源 XceptionNet 权重，对单张人脸输出伪造概率（专长：伪造整体） | 模型开源，只写了加载代码 |
| `face_detect.py` | YuNet 人脸检测（找出视频帧里的人脸位置） | 检测器开源，只写了调用代码 |
| `mesonet_model.h5` | MesoNet 预训练权重 | **开源直接使用，不训练**（hamster 项目） |
| `x-model23.p` | XceptionNet 预训练权重（FF++ 微调，83.5MB） | **开源直接使用，不训练**（i3p9 项目） |
| `face_detector/face_detection_yunet_2023mar.onnx` | YuNet 人脸检测模型 | **开源直接使用**（OpenCV opencv_zoo） |
| `requirements.txt` | 依赖清单（TF + PyTorch CPU） | — |

参考文件（不参与运行，留作溯源，均在 `reference/` 子目录）：
- `reference/xception_def.py`：FaceForensics 官方 XceptionNet 原版定义
- `reference/ff_models_full.py`：FaceForensics 官方 TransferModel 包装类（模型加载兼容参考）
- `reference/transform_ref.py`：官方预处理（Resize 299 + Normalize 0.5）

## 运行
```bash
# 1. 装依赖（Python 3.8 实测环境：TF 2.13 + PyTorch 2.4.1）
pip install -r requirements.txt
pip install pretrainedmodels                                           # 模型加载兼容需要
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # 无GPU机器
# 有 NVIDIA GPU（实测 RTX 4060）会自动用 GPU：PyTorch 请装 cu121 版，速度提升明显

# 2. 自测（自动检测同目录 test.mp4）
python video_detector.py
```

> 计算设备说明：XceptionNet（PyTorch）会自动检测 GPU 并使用；MesoNet（TensorFlow）
> 为 CPU 版，速度较慢，是当前主要耗时点（可考虑 `pip install tensorflow[cuda]` 升级）。
> 双模型模式下建议把 `SAMPLE_INTERVAL` 保持在 10 以上，避免等待过长。

## 实测记录（2026-09-17，RTX 4060 本机）
| 测试视频 | 单 MesoNet | 双模型合判 | 结论 |
|---|---|---|---|
| `test.mp4`（AI 全生成人像） | 47.9 分 → 漏检（判真） | **90.6 分 → 检出（判假）** | 双模型明显更强 |

说明：`test.mp4` 是 AI 直接生成的视频（非换脸），MesoNet 只擅长换脸所以漏检；
XceptionNet 对伪造整体更敏感，多帧平均后把风险分拉高到 90.6。这正好证明
"换脸 + 全 AI 生成"都交给双模型合判的必要性。

## 顶部可调参数（video_detector.py 第 26~28 行）
| 参数 | 默认值 | 含义 | 调法 |
|---|---|---|---|
| `SAMPLE_INTERVAL` | 10 | 每 10 帧采样 1 帧 | 越小越准越慢；演示用 5，批量测试用 10+ |
| `RISK_THRESHOLD` | 50 | 风险分 ≥ 50 判"假" | 防诈骗建议 40~45（宁可误报不放过） |
| `MODEL_MODE` | `"dual"` | `dual`=双模型取最高；`meso`=只用 MesoNet；`xception`=只用 XceptionNet | 对比实验时切换 |

## 双模型合判逻辑（为什么取最高分）
```
同一张人脸帧
  ├── MesoNet      → 伪造概率 p1（专长：AI 换脸——人脸区域被替换）
  └── XceptionNet  → 伪造概率 p2（专长：伪造 vs 真实整体，能响应全 AI 生成视频）
该帧结论 = max(p1, p2)      ← 任一模型觉得可疑，就按可疑算（防漏过诈骗视频）
视频结论 = 所有帧结论取平均 ×100 → 风险分 → 与 RISK_THRESHOLD 比较 → 真/假
```
> 注意：XceptionNet 权重缺失时自动降级为只用 MesoNet（控制台会打印提示），不影响主流程。

## 整体流程
```
视频文件
  │ 1. OpenCV 读视频，每隔 SAMPLE_INTERVAL 帧取 1 帧
  ▼
取出的帧
  │ 2. YuNet 人脸检测：找出帧里所有人脸框 (x,y,w,h)
  ▼
人脸区域
  │ 3. 裁剪人脸 → MesoNet 缩放到 128x128 / XceptionNet 缩放到 299x299
  ▼
双模型预测
  │ 4. 每帧取两模型最高伪造概率
  ▼
所有帧的分数
  │ 5. 取平均 → ×100 转成 0~100 风险分 → ≥RISK_THRESHOLD 判"假"
  ▼
(是否伪造, 风险分)
```

## 关键知识点（为什么之前 2.py 是错的）
1. **2.py 用了"没训练过的 MobileNetV2"**：`weights="imagenet"` 只让卷积层认识自然图片，
   最后的 Dense 层是随机初始化的，没有任何伪造数据训练过 → 输出≈0.5，等于抛硬币。
2. **必须"先裁脸再判断"**：MesoNet/XceptionNet 都是按人脸区域训练的，整帧直接喂会引入大量
   背景干扰，精度大降。
3. **不要自己从头训练**：MesoNet（换脸）和 XceptionNet（FF++ 伪造）都有开源预训练权重，
   直接下载即可；自己训练需要 FaceForensics++ 数据集（几十 GB）和 GPU，不划算。
4. **换脸 ≠ 全 AI 生成**：MesoNet 只擅长换脸；全 AI 生成的视频（Sora/即梦等）它只能碰运气，
   这正是引入 XceptionNet 的原因。但两者都是"人脸伪造"检测器，无法覆盖声音/话术伪造——
   那是任务 2/3 的分工，整套系统靠"脸+声+话术"三路合判兜底。

## 模型来源（开源致谢）
- MesoNet 权重：https://github.com/muffin-123/hamster （hamster-deepfake-detector，Apache-2.0）
- XceptionNet 权重与结构：https://github.com/i3p9/deepfake-detection-with-xception
  （结构即 FaceForensics++ 官方 XceptionNet；若日后拿到官方权重
   `xception-b5690688.pth`，替换 `xception_detect.py` 里的 `MODEL_PATH` 即可，接口不变）
- YuNet 人脸检测：https://github.com/opencv/opencv_zoo （Apache-2.0）
