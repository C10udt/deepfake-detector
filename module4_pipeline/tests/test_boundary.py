"""
module4 视频预处理 - 边界场景测试脚本
运行：python -m module4_pipeline.tests.test_boundary
"""
import sys
import time
import traceback
from pathlib import Path

# 把项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from module4_pipeline.video_preprocess import extract_audio_and_text


def run_case(name: str, video_path: str) -> dict:
    """跑一个测试用例，返回结果记录"""
    result = {
        "case": name,
        "video": video_path,
        "success": False,
        "audio_path": None,
        "text_path": None,
        "text_content": None,
        "elapsed_sec": None,
        "error": None,
    }
    print(f"\n{'='*60}")
    print(f"测试用例: {name}")
    print(f"视频路径: {video_path}")
    print(f"文件存在: {Path(video_path).exists()}")

    t0 = time.time()
    try:
        audio, text = extract_audio_and_text(video_path)
        result["success"] = True
        result["audio_path"] = audio
        result["text_path"] = text
        # 读取文字内容
        if Path(text).exists():
            result["text_content"] = Path(text).read_text(encoding="utf-8")[:200]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 2)

    print(f"结果: {'成功' if result['success'] else '失败'}")
    print(f"耗时: {result['elapsed_sec']}s")
    if result["error"]:
        print(f"错误: {result['error']}")
    if result["text_content"] is not None:
        print(f"识别文字: {result['text_content']}")
    return result


def main():
    cases = [
        ("正常视频-有音轨", r"C:\Users\hhjgh\Desktop\视频检测\0\0.mp4"),
        ("无音轨视频", r"C:\Users\hhjgh\Desktop\Trae CN\resources\app\node_modules\@byted-icube\desktop-modules\dist\media\1_solo.mp4"),
        ("损坏视频文件", str(Path(__file__).parent / "corrupt_test.mp4")),
        ("大视频-约11MB", r"C:\Users\hhjgh\Desktop\视频检测\6\6.mp4"),
        ("文件不存在", r"C:\Users\hhjgh\Desktop\不存在的视频.mp4"),
    ]

    results = []
    for name, path in cases:
        results.append(run_case(name, path))

    # 汇总
    print(f"\n\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")
    print(f"{'用例':<20} {'结果':<6} {'耗时(s)':<10} {'说明'}")
    print("-" * 60)
    for r in results:
        status = "成功" if r["success"] else "失败/兜底"
        note = r["error"] if r["error"] else (r["text_content"] or "")[:30]
        print(f"{r['case']:<20} {status:<6} {r['elapsed_sec']:<10} {note}")


if __name__ == "__main__":
    main()
