# -*- coding: utf-8 -*-
"""
XceptionNet 深度伪造检测封装（任务1 双模型合判的第二个模型）

作用：
    加载开源预训练 XceptionNet 权重（FaceForensics++ 微调模型），
    对单张"人脸裁剪图"输出伪造概率 0~1。

来源说明（开源复用，非自研、非自训练）：
    - 权重：x-model23.p（83.5MB），来自 GitHub 开源项目
      i3p9/deepfake-detection-with-xception
      （模型结构即 FaceForensics++ 官方 XceptionNet，在 FF++ 上微调为 2 分类：
       类别 0 = 真实，类别 1 = 伪造）
    - 若日后拿到官方权重 xception-b5690688.pth（FF++ 官方训练版），
      放到本目录替换 XCEPTION_MODEL 路径即可，接口不变。

用法：
    from xception_detect import predict_face_xception
    proba = predict_face_xception(face_bgr_image)   # -> float 0~1（越大越可能是伪造）
"""
import os
import sys
import types
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from xception_net import xception

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "x-model23.p")
INPUT_SIZE = 299  # XceptionNet 官方输入尺寸

_model = None  # 全局只加载一次

# 自动选择计算设备：有 NVIDIA GPU 就用 GPU（快数倍），否则 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _register_pickle_helpers():
    """
    权重文件是 torch.save 的完整模型（pickle 序列化），pickle 里记录的
    类路径可能是多种写法（不同训练项目保存的），统一在这里注册同名模块，
    让 torch.load 反序列化时能找到类定义：
        - helper_codes.xception（i3p9 仓库自己的路径）
        - network.xception   （FaceForensics 官方仓库的路径）
    """
    from xception_net import Block, SeparableConv2d, Xception

    candidates = [
        ("helper_codes", "helper_codes.xception"),
        ("network", "network.xception"),
    ]
    for pkg_name, mod_name in candidates:
        pkg = sys.modules.get(pkg_name)
        if pkg is None:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []
            sys.modules[pkg_name] = pkg
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
        mod.SeparableConv2d = SeparableConv2d
        mod.Block = Block
        mod.Xception = Xception

    # pickle 里还引用了 network.models（FaceForensics 官方包），
    # 里面需要 TransferModel 包装类——定义最小占位类并挂上去
    net_models = sys.modules.get("network.models")
    if net_models is None:
        net_models = types.ModuleType("network.models")
        sys.modules["network.models"] = net_models

    class _TransferModel(torch.nn.Module):
        """官方 network.models.TransferModel 的占位定义（仅反序列化用）"""
        def __init__(self, modelchoice="xception", num_out_classes=2,
                     dropout=0.0):
            super().__init__()
            self.modelchoice = modelchoice
            self.num_out_classes = num_out_classes

        def forward(self, x):
            return self.model(x)

    net_models.TransferModel = _TransferModel

    for extra in ("helper_codes.transform",):
        if sys.modules.get(extra) is None:
            sys.modules[extra] = types.ModuleType(extra)


def get_model():
    """加载全局 XceptionNet 模型（懒加载，只加载一次）"""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"模型文件不存在: {MODEL_PATH}\n"
            f"请把 x-model23.p 放在本文件夹下"
            f"（来自 GitHub 开源项目 i3p9/deepfake-detection-with-xception）"
        )

    _register_pickle_helpers()
    with warnings.catch_warnings():
        # 权重 pickle 里的类名与本地占位类不完全一致，torch 会打 SourceChangeWarning，
        # 属正常现象（权重与模型结构匹配已验证），这里静默掉避免刷屏
        warnings.simplefilter("ignore")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu",
                                weights_only=False)

    # 兼容两种保存格式：完整模型对象（可能是 TransferModel 包装） / 纯 state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint

    # TransferModel 包装类会把内部网络挂在 "model." 下，剥掉这层前缀
    state_dict = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }

    model = xception(num_classes=2, pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[xception_detect] 警告：权重中缺失这些层: {missing}")
    if unexpected:
        print(f"[xception_detect] 警告：权重中多余的键(已忽略): {unexpected}")

    model = model.to(DEVICE)   # 搬到 GPU（若可用）
    model.eval()
    _model = model
    print(f"[xception_detect] XceptionNet 已加载，运行设备: {DEVICE}")
    return _model


def _preprocess(face_bgr):
    """
    BGR 人脸图 -> XceptionNet 输入张量 (1, 3, 299, 299)，像素范围 [-1, 1]
    与官方预处理一致：Resize(299) -> ToTensor -> Normalize(0.5, 0.5)
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype("float32") / 255.0            # [0, 1]
    img = (img - 0.5) / 0.5                        # [-1, 1]
    img = np.transpose(img, (2, 0, 1))             # HWC -> CHW
    img = np.expand_dims(img, axis=0)              # -> NCHW
    return torch.from_numpy(img)


def predict_face_xception(face_bgr):
    """
    输入：一张 BGR 人脸图像（任意尺寸）
    输出：伪造概率 float（0~1，越大越可能是伪造）
    """
    model = get_model()
    tensor = _preprocess(face_bgr).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        proba = F.softmax(logits, dim=1)[0]
    # 类别 1 = 伪造
    return float(proba[1].item())


if __name__ == "__main__":
    # 自测：读一张本地图片（拖进来的或文件夹里的 jpg/png）看输出
    test_img = None
    for f in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and "ref" not in f:
            test_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
            break
    if test_img:
        img = cv2.imread(test_img)
        print(f"[xception_detect] 测试图片: {test_img}")
        print("伪造概率:", round(predict_face_xception(img), 4))
    else:
        print("[xception_detect] 本目录没有测试图片，跳过自测")
