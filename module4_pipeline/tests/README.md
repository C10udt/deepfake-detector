# module4 视频预处理 - 边界场景测试记录

测试日期：2026-10-25
测试环境：Windows 10/11 + Python 3.8 + faster-whisper(base) + CPU推理

## 测试用例与结果

| 用例 | 视频 | 结果 | 耗时 | 说明 |
|------|------|------|------|------|
| 正常视频-有音轨 | 视频检测/0/0.mp4 (3.1MB) | ✅ 成功 | 4.43s | 识别文字："3、2、1開始了" |
| 无音轨视频 | Trae CN/.../1_solo.mp4 (628KB) | ✅ 成功（兜底） | 1.18s | ffmpeg提取失败，自动生成1秒静音wav，whisper转写为空 |
| 损坏视频文件 | tests/corrupt_test.mp4 (100字节垃圾) | ✅ 成功（兜底） | 1.25s | ffmpeg识别为无效文件，走静音兜底，不抛500 |
| 大视频-约11MB | 视频检测/6/6.mp4 (11.3MB) | ✅ 成功 | 6.13s | 识别文字："It's a call show" |
| 文件不存在 | 不存在的路径 | ✅ 正确报错 | 0.0s | 抛出 FileNotFoundError，调用方可捕获 |

## 关键验证点

### 1. 无音轨/损坏视频兜底
负责人要求：无音轨视频、损坏视频文件不能让整个请求500。
**验证通过**：ffmpeg 提取音频失败时，自动生成 1 秒静音 wav，后续 whisper 正常跑通，转写为空文本。

### 2. whisper 模型离线下载
代码内置 `HF_ENDPOINT=https://hf-mirror.com` 国内镜像，首次运行自动从镜像下载 base 模型（约140MB）。
模型缓存到用户目录 `~/.cache/huggingface/`，不进 git。

### 3. Windows 环境 PyAV DLL 拦截
部分 Windows 环境下，PyAV 的原生 DLL 会被系统应用控制策略拦截（ImportError: DLL load failed）。
代码做了兼容处理：mock 空的 av 模块，自己用标准库 wave 读取 wav 文件转 numpy 数组，绕过 PyAV 依赖。
不影响正常功能，仅在 DLL 被拦截的环境下生效。

## 运行方式

```bash
# 激活虚拟环境
C:\Users\hhjgh\Desktop\deepfake-env38\Scripts\activate

# 跑测试
python -m module4_pipeline.tests.test_boundary
```

## 测试产物
- `test_boundary.py`：测试脚本
- `test_run_log.txt`：完整运行日志
- `corrupt_test.mp4`：损坏测试文件
