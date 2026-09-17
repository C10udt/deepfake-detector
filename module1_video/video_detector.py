def detect_video(video_path: str) -> tuple[bool, float]:
    """
    AI换脸检测接口
    :param video_path: 视频文件路径字符串
    :return: (is_fake: bool, score: float) 分数范围0~100，越高风险越大
    """
    # Mock占位函数，后期替换真实Mesonet模型
    return False, 10.0
