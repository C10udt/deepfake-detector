from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile

import cv2

from module1_video.video_detector import detect_video
from module1_video.face_detect import detect_faces
from module2_audio.audio_detector import detect_audio
from module3_text.text_detector import detect_text
from module4_pipeline.video_preprocess import extract_audio_and_text
from module4_pipeline.risk_fusion import fusion_final

app = FastAPI(title="AI音视频诈骗检测系统")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.py 所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 模板目录：module5_ui/templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def has_face(video_path: str) -> bool:
    """抽样检测视频中是否存在人脸：任何一帧检测到人脸即返回 True。

    注意：detect_video 在“没检测到人脸”和“正常但风险低”时都返回 (False, 0.0)，
    二者无法区分，所以这里单独做人脸存在性判定，无人脸时提前返回 no_face。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 与 video_detector 的采样间隔保持一致
            if frame_count % 10 == 0 and detect_faces(frame):
                return True
            frame_count += 1
    finally:
        cap.release()
    return False


# 首页路由
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 检测接口
@app.post("/detect")
async def detect(video: UploadFile):
    # 1. 保存上传文件到临时目录（带原始扩展名，方便 cv2 / ffmpeg 识别格式）
    suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    base_path = os.path.splitext(video_path)[0]  # 供清理派生文件（wav/txt）用

    try:
        print(f"===收到文件：{video.filename}===")
        # 2. 无人脸判定：视频里没检测到任何人脸，无法进行换脸判定
        if not has_face(video_path):
            return {"status": "no_face", "msg": "未检测到人脸，无法判定"}

        # 3. 真实检测：视频换脸 / 提取音频+文字 / 声音合成 / 文本诈骗
        video_is_fake, video_score = detect_video(video_path)
        audio_path, text_path = extract_audio_and_text(video_path)
        audio_is_fake, audio_score = detect_audio(audio_path)
        text_score = detect_text(text_path)

        # 4. 多模块风险融合，输出最终判定
        is_fraud, final_score = fusion_final(video_score, audio_score, text_score)

        # 5. numpy 标量转 Python 原生类型（FastAPI 无法序列化 numpy.bool_/numpy.float64）
        video_is_fake = bool(video_is_fake)
        video_score = float(video_score)
        audio_is_fake = bool(audio_is_fake)
        audio_score = float(audio_score)
        text_score = float(text_score)
        is_fraud = bool(is_fraud)
        final_score = float(final_score)

        return {
            "video": {
                "isFake": video_is_fake,
                "score": video_score,
                "desc": f"{'假' if video_is_fake else '真'}（风险分：{video_score}）",
            },
            "audio": {
                "isFake": audio_is_fake,
                "score": audio_score,
                "desc": f"{'假' if audio_is_fake else '真'}（风险分：{audio_score}）",
            },
            "text": {"score": text_score, "desc": f"风险分：{text_score}"},
            "final": {
                "isFraud": is_fraud,
                "desc": (f"高风险，疑似诈骗（综合风险分：{final_score:.1f}）"
                         if is_fraud
                         else f"风险较低，暂未发现诈骗特征（综合风险分：{final_score:.1f}）"),
            },
        }

    finally:
        # 5. 清理临时文件及派生文件（wav / txt），避免残留
        for p in [video_path, base_path + ".wav", base_path + ".txt"]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)