"""
音频伪造检测模块
作者：Liber
功能：检测音频是否为AI合成，返回真/假 + 风险分

使用方法：
    from audio_detector import detect_audio

    is_fake, score = detect_audio("音频文件.wav")
    print(f"是否伪造: {is_fake}, 风险分: {score}")
"""

import os
import sys
import joblib
import librosa
import numpy as np

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 添加模块路径
sys.path.insert(0, os.path.join(CURRENT_DIR, 'src', 'fake_audio_detection'))

# 导入特征提取函数
try:
    from extract_features import extract_features_from_array
except ImportError:
    print("警告: 无法导入 extract_features，使用简化版")


    def extract_features_from_array(y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)

# 模型路径
MODEL_PATH = os.path.join(CURRENT_DIR, 'models', 'my_model.joblib')


def detect_audio(audio_path):
    """
    检测音频是否为AI合成

    参数:
        audio_path: 音频文件路径 (支持 .wav, .mp3, .flac 等)

    返回:
        (is_fake: bool, risk_score: float)
        - is_fake: True=伪造, False=真实
        - risk_score: 0-100 的风险分数，越高越可能是伪造

    使用示例:
        is_fake, score = detect_audio("test.wav")
        print(f"是否伪造: {is_fake}, 风险分: {score}")
    """
    try:
        # 1. 检查模型是否存在
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

        # 2. 检查音频文件是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 3. 加载音频
        y, sr = librosa.load(audio_path, sr=22050)

        # 4. 提取特征
        features = extract_features_from_array(y, sr)
        features = features.reshape(1, -1)

        # 5. 加载模型并预测
        model_data = joblib.load(MODEL_PATH)
        pipeline = model_data["pipeline"]
        proba = pipeline.predict_proba(features)[0]

        # 6. 计算风险分
        fake_probability = proba[0]
        risk_score = fake_probability * 100
        is_fake = risk_score > 50

        return (is_fake, round(risk_score, 2))

    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
        return (False, 0.0)
    except Exception as e:
        print(f"❌ 检测出错: {e}")
        return (False, 0.0)


def detect_audio_with_details(audio_path):
    """
    检测音频并返回详细信息

    参数:
        audio_path: 音频文件路径

    返回:
        dict: 包含检测结果的详细信息
    """
    is_fake, risk_score = detect_audio(audio_path)

    return {
        "is_fake": is_fake,
        "risk_score": risk_score,
        "result_text": "伪造音频" if is_fake else "真实音频",
        "risk_level": "高" if risk_score > 70 else ("中" if risk_score > 50 else ("低" if risk_score > 30 else "极低"))
    }


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("音频伪造检测模块")
    print("=" * 50)
    print("\n使用方式:")
    print("  from audio_detector import detect_audio")
    print("  is_fake, score = detect_audio('音频文件.wav')")
    print("\n" + "=" * 50)

    # 如果有测试文件，自动测试
    test_real = os.path.join(CURRENT_DIR, 'dataset', 'train', 'data', 'REAL', 'real_000.wav')
    test_fake = os.path.join(CURRENT_DIR, 'dataset', 'train', 'data', 'FAKE', 'fake_000.wav')

    if os.path.exists(test_real):
        print("\n【测试】真实音频:")
        is_fake, score = detect_audio(test_real)
        print(f"  结果: {'伪造' if is_fake else '真实'}")
        print(f"  风险分: {score}/100")

    if os.path.exists(test_fake):
        print("\n【测试】伪造音频:")
        is_fake, score = detect_audio(test_fake)
        print(f"  结果: {'伪造' if is_fake else '真实'}")
        print(f"  风险分: {score}/100")