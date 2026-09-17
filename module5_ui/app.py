from fastapi import FastAPI, UploadFile
from module1_video.video_detector import detect_video
from module2_audio.audio_detector import detect_audio
from module3_text.text_detector import detect_text
from module4_pipeline.video_preprocess import extract_audio_and_text
from module4_pipeline.risk_fusion import fusion_final

app = FastAPI(title="AI音视频诈骗检测系统")

# 界面与接口代码，由第5位同学后续实现
@app.get("/")
def home():
    return {"msg": "Deepfake Detector Backend Ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
