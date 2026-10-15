# module3_text/run_detect.py
"""
命令行工具：对 ASR 转写出的文本文件进行诈骗风险检测


示例：
    python run_detect.py input.txt
    python run_detect.py dataset/asr_result.txt
"""
import sys
import os
from text_detector import detect_text


def main():
    # 1. 参数校验
    if len(sys.argv) < 2:
        print("用法: python run_detect.py <文本文件路径>")
        print("示例: python run_detect.py input.txt")
        sys.exit(1)

    text_path = sys.argv[1]

    # 2. 文件存在性检查
    if not os.path.exists(text_path):
        print(f"[Error] 文件不存在: {text_path}")
        sys.exit(1)

    # 3. 调用检测接口
    score = detect_text(text_path)

    # 4. 输出结果
    print("=" * 50)
    print(f"文件路径：{text_path}")
    print(f"风险分数：{score:.1f}  (范围 0~100)")
    print("-" * 50)

    if score >= 50:
        print("⚠️  高风险：疑似诈骗，建议立即报警或联系家人核实")
    elif score >= 30:
        print("⚠️  中风险：存在可疑特征，请谨慎对待")
    else:
        print("✅  低风险：正常文本")
    print("=" * 50)


if __name__ == "__main__":
    main()