from __future__ import annotations

import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

# 国内网络环境：模型默认从 HuggingFace 下载会失败，改用镜像源
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 某些 Windows 环境下 PyAV 的 DLL 会被应用控制策略拦截（ImportError），
# 我们已经用 ffmpeg 转好了 wav，可以绕过 faster_whisper 对 av 的依赖。
# 先 mock 一个空的 av 模块，让 faster_whisper 能正常 import。
if "av" not in sys.modules:
    import types
    sys.modules["av"] = types.ModuleType("av")

import imageio_ffmpeg
import numpy as np
from faster_whisper import WhisperModel

# 输出音频规格：whisper 标准输入为 16kHz 单声道 wav
SAMPLE_RATE = 16000
# 模型大小：base 轻量（约 140MB，首次运行自动下载）；可选 tiny/base/small/medium/large-v3
WHISPER_MODEL_SIZE = "base"


def _get_ffmpeg_exe() -> str:
    """获取 imageio-ffmpeg 自带的 ffmpeg 可执行文件路径（免安装系统 ffmpeg）"""
    return imageio_ffmpeg.get_ffmpeg_exe()


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """用 ffmpeg 从视频提取音频，转成 16kHz 单声道 wav（whisper 标准输入格式）"""
    ffmpeg = _get_ffmpeg_exe()
    result = subprocess.run(
        [
            ffmpeg, "-i", str(video_path), "-vn",
            "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-y", str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # 边界情况：视频无音轨/损坏时 ffmpeg 会失败。
        # 生成 1 秒静音 wav 兜底，保证后续检测链路可继续（whisper 转写为空、音频判低风险），
        # 而不是让整个请求 500。
        silent = subprocess.run(
            [
                ffmpeg, "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-t", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if silent.returncode == 0:
            return
        raise RuntimeError(f"ffmpeg 提取音频失败: {result.stderr[-300:]}")


def _load_model() -> WhisperModel:
    """加载 faster-whisper 模型，使用 CPU 推理。

    说明：ctranslate2 在 Windows 上即使设置 CUDA_VISIBLE_DEVICES 为空，
    显式 device="cuda" 仍会尝试加载 cuDNN9 并崩溃（本机 RTX4060 + TF2.13 环境），
    故联调阶段强制 CPU（int8 量化，精度损失可接受）。
    """
    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def _read_wav_as_np(wav_path: Path) -> np.ndarray:
    """用标准库 wave 读 16kHz 单声道 wav，转成 float32 numpy 数组（whisper 标准输入）。"""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # 只支持 16-bit PCM（我们 ffmpeg 输出就是 pcm_s16le）
    if sampwidth != 2:
        raise ValueError(f"不支持的 wav 位宽: {sampwidth} 字节")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples


def extract_audio_and_text(video_path: str) -> tuple[str, str]:
    """
    视频预处理：从视频提取音频（wav），语音转文字保存到文本文件。

    :param video_path: 原始视频路径
    :return: (audio_path, text_path) 生成的音频文件路径、文本文件路径
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 输出路径：与原视频同目录
    audio_path = video_path.with_suffix(".wav")
    text_path = video_path.with_suffix(".txt")

    # 1. 提取音频（wav）
    _extract_audio(video_path, audio_path)

    # 2. 语音转文字（自己读 wav 成 numpy 数组，绕过 PyAV 依赖）
    model = _load_model()
    audio_np = _read_wav_as_np(audio_path)
    segments, _info = model.transcribe(audio_np, beam_size=5)
    text = "".join(segment.text for segment in segments).strip()

    # 3. 保存文本
    text_path.write_text(text if text else "未识别到语音内容", encoding="utf-8")

    return str(audio_path), str(text_path)
