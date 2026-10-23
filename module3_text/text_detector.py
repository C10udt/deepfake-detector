import os
import json
# ==========================================
# 1. 内部实现：规则打分器（新版分级加权 + 联动折扣，对外接口不变）
# ==========================================
class TextDetector:
    def __init__(self, rules_path=None):
        if rules_path is None:
            # 默认读取同级目录下的 config/rules.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rules_path = os.path.join(current_dir, 'config', 'rules.json')
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)

        # 加载词库
        self.kin = self.rules.get('kin', [])
        self.emergency = self.rules.get('emergency', [])
        self.high_risk = self.rules.get('high_risk', [])
        self.mid_risk = self.rules.get('mid_risk', [])
        self.secret = self.rules.get('secret', [])
        self.low_risk = self.rules.get('low_risk', [])
        self.exclude_words = self.rules.get('exclude_words', [])

        # 读取打分配置
        self.score_cfg = self.rules.get("score_cfg", {})
        self.MAX_SCORE = self.rules.get('max_rule_score', 100)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """预处理：清除全部空白字符（空格、换行、tab），解决ASR插入空格导致漏匹配"""
        return "".join(text.split())

    def predict(self, text: str) -> float:
        """内部预测函数，返回 0~100 的 float 分数，对外行为保持兼容"""
        if not text or not isinstance(text, str):
            return 0.0
        clean_text = self.preprocess_text(text)
        if not clean_text:
            return 0.0

        score = 0
        hit_details = {
            "kin": [],
            "emergency": [],
            "high_risk": [],
            "mid_risk": [],
            "secret": [],
            "low_risk": [],
            "exclude_words": []
        }

        # 标记flag
        hit_high = False
        hit_mid = False
        hit_low = False
        hit_kin = False
        hit_secret = False
        hit_emergency = False
        hit_exclude = False

        # 1. kin亲属词：命中即标记，只计一次
        for w in self.kin:
            if w in clean_text:
                score += self.score_cfg["kin"]
                hit_details["kin"].append(w)
                hit_kin = True
                break

        # 2. emergency紧急词：最多命中1个（用来判断联动flag，不再重复累加）
        for w in self.emergency:
            if w in clean_text:
                hit_details["emergency"].append(w)
                hit_emergency = True
                break

        # 3. high_risk 高危词
        for w in self.high_risk:
            if w in clean_text:
                score += self.score_cfg["high"]
                hit_details["high_risk"].append(w)
                hit_high = True

        # 4. mid_risk 中危词：独立加分（新方案）
        for w in self.mid_risk:
            if w in clean_text:
                score += self.score_cfg["mid"]
                hit_details["mid_risk"].append(w)
                hit_mid = True

        # 5. low_risk 弱风险借钱词
        for w in self.low_risk:
            if w in clean_text:
                score += self.score_cfg["low"]
                hit_details["low_risk"].append(w)
                hit_low = True
                break

        # 6. secret保密词（仅标记，不加基础分！只参与联动bonus）
        for w in self.secret:
            if w in clean_text:
                hit_details["secret"].append(w)
                hit_secret = True
                break

        # 7. exclude豁免词（仅标记，不加基础分，衰减因子）
        for w in self.exclude_words:
            if w in clean_text:
                hit_details["exclude_words"].append(w)
                hit_exclude = True
                break

        # ========== 联动bonus逻辑（核心） ==========
        raw_bonus = 0
        if hit_kin and hit_low and hit_secret:
            if hit_emergency:
                raw_bonus = self.score_cfg["secret_bonus_full"]
            else:
                raw_bonus = self.score_cfg["secret_bonus_light"]

        # 豁免折扣：命中豁免 并且 没有高危词，才打折
        if hit_exclude and not hit_high:
            final_bonus = raw_bonus * self.score_cfg["exclude_discount"]
        else:
            final_bonus = raw_bonus
        score += final_bonus

        # kin + emergency 原有联动加分，high存在时不叠加这个bonus
        if hit_kin and hit_emergency and not hit_high:
            score += self.score_cfg["kin_emergency_bonus"]

        # ==== 新增 kin + secret 联动加分 ====
        if hit_kin and hit_secret:
            score += self.rules.get("kin_secret_bonus", 18)

        # 保底不能负数 + 封顶
        score = max(score, 0)
        final_score = float(min(score, self.MAX_SCORE))
        return final_score

# ==========================================
# 2. 对外接口：保持原签名不变，完全不动
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
    # 3. 调用内部检测器打分
    try:
        detector = _get_detector()
        score = detector.predict(text_content)
        return score
    except Exception as e:
        print(f"[Error] detect_text 推理失败: {e}")
        return 0.0
