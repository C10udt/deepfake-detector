def fusion_final(video_score: float, audio_score: float, text_score: float) -> tuple[bool, float]:
    """
    多模块风险分数加权融合，输出最终诈骗判定
    权重：视频0.4，音频0.35，文本0.25
    final_score >=60 →判定存在诈骗风险
    """
    final_score = 0.4 * video_score + 0.35 * audio_score + 0.25 * text_score
    is_fraud = final_score >= 60
    return is_fraud, final_score
