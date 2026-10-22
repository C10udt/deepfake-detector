# 域外(OOD)泛化测试数据集说明

## 目的

训练集只覆盖有限的说话人、录音条件与合成系统。为评估模型泛化能力，本测试集全部由**训练阶段从未出现过的说话人、语种、录音条件与合成引擎**构成，与训练数据零重叠。

## 数据构成（共 16 条）

### 真实人声 REAL（8 条）

| 文件 | 内容 | 与训练集的差异点 | 来源 |
|---|---|---|---|
| ood_real_000.wav | LJSpeech LJ050-0131，英语女声朗读 | 全新说话人；录音棚级干净录音 | speechbrain/speechbrain `tests/samples/TTS`（Apache-2.0；LJSpeech 语料为公有领域） |
| ood_real_001.wav | 现场带噪语音（VAD train） | 全新说话人+带环境噪声 | speechbrain `tests/samples/VAD`（Apache-2.0） |
| ood_real_002.wav | 现场带噪语音（VAD valid） | 同上 | 同上 |
| ood_real_003.wav | 单人语音（分离任务源信号） | 全新说话人 | speechbrain `tests/samples/separation/source1_0.wav`（Apache-2.0） |
| ood_real_004.flac | 单麦克风录音 | 全新录音条件 | speechbrain `tests/samples/single-mic/example2.flac`（Apache-2.0） |
| ood_real_005.wav | 单麦克风录音 | 全新录音条件 | speechbrain `tests/samples/single-mic/example6.wav`（Apache-2.0） |
| ood_real_006.wav | 双说话人混合语音 | 多人重叠说话（困难样本） | speechbrain `tests/samples/separation/mixture_0.wav`（Apache-2.0） |
| ood_real_007.wav | 双说话人混合语音 | 多人重叠说话（困难样本） | speechbrain `tests/samples/separation/mixture_1.wav`（Apache-2.0） |

### AI 合成语音 FAKE（8 条）

| 文件 | 合成引擎 | 声线/语种 | 与训练集的差异点 |
|---|---|---|---|
| ood_espeak_000.wav | espeak-ng 1.7（共振峰合成器，离线） | cmn 普通话 | **训练未使用过的合成引擎**；合成机制与 SAPI/神经网络完全不同 |
| ood_espeak_001.wav | espeak-ng | cmn 普通话 | 同上 |
| ood_espeak_002.wav | espeak-ng | en 英语 | 同上 |
| ood_espeak_003.wav | espeak-ng | en 英语 | 同上 |
| ood_espeak_004.wav | espeak-ng | yue 粤语 | 引擎+语种双重域外 |
| ood_edge_000.mp3 | 微软 Edge 神经网络 TTS | zh-CN-YunyangNeural（云扬） | 训练未用过的声线 |
| ood_edge_002.mp3 | 微软 Edge 神经网络 TTS | en-GB-SoniaNeural | 训练未用过的声线+英语口音（英式） |
| ood_edge_003.mp3 | 微软 Edge 神经网络 TTS | ja-JP-NanamiNeural | 训练未用过的语种（日语） |

对照：训练集合成音仅来自 Windows SAPI（Huihui/Zira 两声线）与 Edge 神经网络（晓晓/云希/云健/晓伊/Aria/Guy 六声线）。

## 协议

- OOD 测试集在**模型训练完成后**收集/生成，训练过程零接触（保持严格 held-out）。
- 全部音频统一 22050Hz 单声道加载（librosa 重采样），检测阈值沿用线上规则：风险分 > 50 判为 AI 合成。
- 音频文件不入库（符合团队 Git 规范），可通过 [数据来源说明](../dataset/数据来源说明.md) 与上表来源自行复现。
