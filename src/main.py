import argparse
import os
from core import FaceEngine
import processor
from utils import load_config # <--- 导入 Utils

def run_pipeline(input_path, project_name, config_path="config.json"):
    # 使用统一的配置加载
    cfg = load_config(config_path)

    # 2. 初始化 AI 引擎 (只初始化一次，批量复用)
    print("[System] 初始化模型...")
    engine = FaceEngine(
        model_name=cfg['model_params']['model_name'],
    )

    # 3. 扫描任务 (识别文件还是文件夹)
    tasks = []
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    if os.path.isdir(input_path):
        print(f"[System] 检测到文件夹，正在扫描: {input_path}")
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(video_exts) or file.lower().endswith(image_exts):
                    tasks.append(os.path.join(root, file))
    elif os.path.isfile(input_path):
        tasks.append(input_path)
    else:
        print(f"[Error] 输入路径不存在: {input_path}")
        return

    print(f"[System] 扫描完成，共找到 {len(tasks)} 个待处理文件。项目: {project_name}")

    # 4. 批量执行
    success_count = 0
    for i, file_path in enumerate(tasks):
        print(f"\n>>> 正在处理 [{i + 1}/{len(tasks)}]: {os.path.basename(file_path)}")
        try:
            output_path = ""
            # 根据后缀名分流
            if file_path.lower().endswith(video_exts):
                output_path = processor.process_video(engine, file_path, cfg, project_name)
            elif file_path.lower().endswith(image_exts):
                output_path = processor.process_image(engine, file_path, cfg, project_name)

            if output_path:
                print(f"[Success] 输出: {output_path}")
                success_count += 1
        except Exception as e:
            print(f"[ERROR] 处理失败 {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[Done] 全部任务完成。成功: {success_count}/{len(tasks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频/图片人脸分析服务 (批量版)")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径 或 文件夹路径")
    parser.add_argument("--project", "-p", default="default_project", help="项目名称(用于隔离数据库和输出目录)")
    parser.add_argument("--config", "-c", default="config.json", help="配置文件路径")

    args = parser.parse_args()

    run_pipeline(args.input, args.project, args.config)