import os
import json

# ==========================================
# 1. 内部实现：规则打分器（增强子串匹配版本）
# ==========================================
class TextDetector:
    def __init__(self, rules_path=None):
        if rules_path is None:
            # 默认读取同级目录下的 config/rules.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rules_path = os.path.join(current_dir, 'config', 'rules.json')

        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)

        self.kin = self.rules.get('kin', [])
        self.emergency = self.rules.get('emergency', [])
        self.high_risk = self.rules.get('high_risk', [])
        self.mid_risk = self.rules.get('mid_risk', [])
        self.secret = self.rules.get('secret', [])

        # 权重配置
        self.W_KIN = self.rules.get('kin_weight', 10)
        self.W_EMERGENCY = self.rules.get('emergency_weight', 10)
        self.W_HIGH = self.rules.get('high_risk_weight', 30)
        self.W_MID = self.rules.get('mid_risk_weight', 20)
        self.W_SECRET = self.rules.get('secret_weight', 15)
        self.BONUS_KIN_EMERGENCY = self.rules.get('kin_emergency_bonus', 25)
        self.MAX_SCORE = self.rules.get('max_rule_score', 100)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """预处理：清除全部空白字符（空格、换行、tab），解决ASR插入空格导致漏匹配"""
        return "".join(text.split())

    def predict(self, text: str) -> float:
        """内部预测函数，返回 0~100 的 float 分数"""
        if not text or not isinstance(text, str):
            return 0.0

        # 全部空白清除，之后再做子串匹配
        clean_text = self.preprocess_text(text)
        if not clean_text:
            return 0.0

        score = 0
        hit_details = {"kin": [], "emergency": [], "high_risk": [], "mid_risk": [], "secret": []}

        # 1. 亲属词 (Kin) - 只计一次
        kin_hit = False
        for w in self.kin:
            if w in clean_text:
                score += self.W_KIN
                hit_details["kin"].append(w)
                kin_hit = True
                break

        # 2. 紧急词 (Emergency) - 最多计两次
        emergency_hit = False
        for w in self.emergency:
            if w in clean_text:
                score += self.W_EMERGENCY
                hit_details["emergency"].append(w)
                emergency_hit = True
                if len(hit_details["emergency"]) >= 2:
                    break
        # 3. 高危词 (High Risk) - 全部累加
        for w in self.high_risk:
            if w in clean_text:
                score += self.W_HIGH
                hit_details["high_risk"].append(w)
        # 4. 中危场景词 (Mid Risk) - 只有和高危词同时出现才加分
        has_high_risk = len(hit_details["high_risk"]) > 0
        if has_high_risk:
            for w in self.mid_risk:
                if w in clean_text:
                    score += self.W_MID
                    hit_details["mid_risk"].append(w)
        # 5. 保密词 (Secret)
        for w in self.secret:
            if w in clean_text:
                score += self.W_SECRET
                hit_details["secret"].append(w)
        # 6. 亲属 + 紧急 联动加分
        if kin_hit and emergency_hit:
            score += self.BONUS_KIN_EMERGENCY
        # 7. 封顶，并转为 float 返回
        final_score = float(min(score, self.MAX_SCORE))
        return final_score

# ==========================================
# 2. 对外接口：保持原签名不变
# ==========================================
_DETECTOR = None
def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = TextDetector()
    return _DETECTOR

def detect_text(text_path: str) -> float:
    """
    文本诈骗风险检测接口
    :param text_path: 文本文件路径
    :return: score: float，0~100，越高诈骗风险越大
    """
    # 1. 参数校验
    if not text_path or not isinstance(text_path, str):
        print(f"[Warning] detect_text 收到非法路径参数: {text_path}")
        return 0.0
    # 2. 读取文本文件
    if not os.path.exists(text_path):
        print(f"[Error] detect_text 找不到文件: {text_path}")
        return 0.0

    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        print(f"[Error] detect_text 读取文件失败 {text_path}: {e}")
        return 0.0
    # 3. 调用内部检测器打分，预处理逻辑放到predict内部
    try:
        detector = _get_detector()
        score = detector.predict(text_content)
        return score
    except Exception as e:
        print(f"[Error] detect_text 推理失败: {e}")
        return 0.0
