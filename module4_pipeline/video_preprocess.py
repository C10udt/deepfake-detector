import os
import base64
import urllib.parse
import subprocess
from pathlib import Path
from difflib import SequenceMatcher

import requests

# 从环境变量读取百度 OCR 密钥（不要硬编码在代码里）
API_KEY = os.getenv("BAIDU_API_KEY")
SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")


def is_similar(text1, text2, threshold=0.8):
    """判断两段文字是否相似（默认 80% 相似度视为重复）"""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1, text2).ratio() > threshold


def get_access_token():
    """获取百度 OCR 的 access_token"""
    url = (
        f"https://aip.baidubce.com/oauth/2.0/token"
        f"?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}"
    )
    response = requests.post(url)
    return response.json().get("access_token")


def ocr_image_bytes(image_bytes, access_token, mode="high"):
    """调用百度 OCR 识别单张图片中的文字"""
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    img_encoded = urllib.parse.quote_plus(img_base64)

    if mode == "standard":
        url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={access_token}"
    else:
        url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token={access_token}"

    payload = f"image={img_encoded}&language_type=CHN_ENG"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers)
    result = response.json()

    words = []
    if "words_result" in result:
        for item in result["words_result"]:
            words.append(item["words"])
    return words


def _extract_audio(video_path, audio_path):
    """用 ffmpeg 从视频中抽出音频（mp3）"""
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path), "-vn",
            "-acodec", "libmp3lame", "-b:a", "128k",
            "-y", str(audio_path),
        ],
        capture_output=True,
    )


def extract_audio_and_text(video_path: str) -> tuple[str, str]:
    """
    视频预处理：从视频提取音频文件、识别画面字幕文本，保存到文本文件。

    :param video_path: 原始视频路径
    :return: (audio_path, text_path) 生成的音频文件路径、文本文件路径
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 1. 输出路径（与原视频同目录）
    audio_path = video_path.parent / f"{video_path.stem}.mp3"
    text_path = video_path.parent / f"{video_path.stem}_文本.txt"

    # 2. 用 ffmpeg 抽音频
    _extract_audio(video_path, audio_path)

    # 3. 获取百度 OCR token
    access_token = get_access_token()
    if not access_token:
        raise RuntimeError("百度 OCR 授权失败，请检查环境变量 BAIDU_API_KEY / BAIDU_SECRET_KEY")

    # 4. ffmpeg 抽帧（每秒 2 帧）
    temp_dir = video_path.parent / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    output_pattern = str(temp_dir / "frame_%06d.jpg")
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=2",
            "-q:v", "2", output_pattern, "-y",
        ],
        capture_output=True,
    )

    frames = sorted(temp_dir.glob("*.jpg"))

    # 5. 逐帧 OCR，并用相似度去重
    all_texts = []
    last_text = ""
    for frame_path in frames:
        with open(frame_path, "rb") as f:
            img_bytes = f.read()

        words = ocr_image_bytes(img_bytes, access_token)
        if words:
            current_text = " ".join(words)
            if current_text.strip() and not is_similar(current_text, last_text):
                all_texts.append(current_text)
                last_text = current_text

    # 6. 清理临时帧
    for f in frames:
        f.unlink()
    temp_dir.rmdir()

    # 7. 保存文本
    final_text = "\n".join(all_texts) if all_texts else "未识别到字幕内容"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    return str(audio_path), str(text_path)
