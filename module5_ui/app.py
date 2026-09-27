from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os

from module1_video.video_detector import detect_video
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


# 首页路由
@app.get("/")
def home(request: Request):
    # 新版 FastAPI 正确写法：第一个参数传 request，第二个传模板名
    return templates.TemplateResponse(request, "index.html")


# 检测接口
@app.post("/detect")
async def detect(video: UploadFile):
    return {
        "video": {"isFake": True, "score": 85, "desc": "假（风险分：85）"},
        "audio": {"isFake": False, "score": 20, "desc": "真（风险分：20）"},
        "text": {"score": 70, "desc": "风险分：70"},
        "final": {"isFraud": True, "desc": "高风险，疑似诈骗"}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
