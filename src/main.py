import argparse
import os
import json
from core import FaceEngine
import processor


def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        # 尝试在上级目录找 (兼容从 src 运行和从根目录运行)
        if os.path.exists(os.path.join("..", config_path)):
            config_path = os.path.join("..", config_path)
        else:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_pipeline(input_path, config_path="config.json"):
    # 1. 加载配置
    cfg = load_config(config_path)

    # 2. 初始化 AI 引擎
    print("[System] 初始化模型...")
    engine = FaceEngine(
        model_name=cfg['model_params']['model_name'],
        det_size=tuple(cfg['model_params']['det_size'])
    )

    # 3. 分发任务
    if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        return processor.process_video(engine, input_path, cfg)
    else:
        return processor.process_image(engine, input_path, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频/图片人脸分析服务")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--config", "-c", default="config.json", help="配置文件路径")

    args = parser.parse_args()

    try:
        final_output = run_pipeline(args.input, args.config)
        print(f"\n[SUCCESS] 任务完成，输出路径: {final_output}")
    except Exception as e:
        print(f"\n[ERROR] 任务失败: {e}")