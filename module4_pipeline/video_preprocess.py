def extract_audio_and_text(video_path: str) -> tuple[str, str]:
    """
    视频预处理：从视频提取音频，语音转文字保存到文本文件
    :param video_path: 原始视频路径
    :return: (audio_path, text_path) 生成的音频文件、文本文件路径
    """
    # Mock占位，后续ffmpeg+whisper实现
    audio_path = "temp/temp_audio.wav"
    text_path = "temp/temp_text.txt"
    return audio_path, text_path
