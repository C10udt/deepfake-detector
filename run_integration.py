# -*- coding: utf-8 -*-
"""联调启动脚本：强制 CPU 运行，规避 TF 2.13 在本机 GPU/cuDNN 崩溃问题"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import uvicorn

if __name__ == "__main__":
    uvicorn.run("module5_ui.app:app", host="127.0.0.1", port=8000)
