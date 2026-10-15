from __future__ import annotations

import os
import subprocess
from pathlib import Path

# 国内网络环境：模型默认从 HuggingFace 下载会失败，改用镜像源
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import imageio_ffmpeg
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
        raise RuntimeError(f"ffmpeg 提取音频失败: {result.stderr[-500:]}")


def _load_model() -> WhisperModel:
    """加载 faster-whisper 模型，优先 GPU 加速，无 GPU 自动回退 CPU。

    模型自动下载到用户缓存目录（~/.cache/huggingface），不在项目目录内，不会进 git。
    """
    try:
        return WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
    except Exception:
        return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


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

    # 2. 语音转文字
    model = _load_model()
    segments, _info = model.transcribe(str(audio_path), beam_size=5)
    text = "".join(segment.text for segment in segments).strip()

    # 3. 保存文本
    text_path.write_text(text if text else "未识别到语音内容", encoding="utf-8")

    return str(audio_path), str(text_path)
