import cv2
import os
import argparse
import ast  # 用于把字符串 "[100, 100, 200, 200]" 安全地转回列表
from database import VectorDB
from core import FaceEngine
from main import load_config


def visualize_search(image_path, limit=3, output_dir="store/visualized"):
    # 1. 加载配置
    cfg = load_config()
    print(f"[System] 正在初始化模型与数据库...")

    # 初始化 AI 引擎 (用于提取目标图片的特征)
    engine = FaceEngine(model_name=cfg['model_params']['model_name'])

    # 初始化数据库
    db_path = cfg['project_settings'].get('vector_db_path', 'store/vector_db')
    db = VectorDB(db_path=db_path)

    # 2. 读取目标图片并搜索
    print(f"[Search] 读取目标图片: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 无法读取图片: {image_path}")
        return

    faces = engine.extract(img)
    if not faces:
        print("[Error] 图片中未检测到人脸，无法搜索。")
        return

    # 默认取第一张脸
    target_emb = faces[0]['embedding']
    print(f"[Search] 开始在数据库中检索 Top {limit}...")

    results = db.search(target_emb, limit=limit)

    if not results:
        print("[Result] 未找到匹配结果。")
        return

    # 3. 截取视频帧并画图
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Visual] 开始生成可视化结果，保存至: {output_dir}")

    # 缓存打开的视频对象，避免重复 open/close
    video_caps = {}

    for i, item in enumerate(results):
        meta = item['meta']
        video_name = meta['video_name']
        frame_id = meta['frame_id']
        dist = item['distance']

        # 解析 bbox (数据库里存的是字符串，需要转回 list)
        try:
            bbox = ast.literal_eval(meta['bbox'])
            bbox = [int(b) for b in bbox]  # 转整数
        except:
            print(f"[Warn] BBox 解析失败: {meta['bbox']}")
            continue

        # 打开视频 (假设视频都在 data/video 下，如果不是请修改这里)
        video_path = os.path.join("data/video", video_name)

        if video_path not in video_caps:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[Error] 无法打开视频: {video_path}")
                continue
            video_caps[video_path] = cap

        cap = video_caps[video_path]

        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if ret:
            # 画红框 (BGR: 0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)

            # 写上相似度
            label = f"Rank {i + 1} | Dist: {dist:.4f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 保存图片
            save_name = f"rank_{i + 1}_{video_name}_f{frame_id}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, frame)
            print(f" -> 已保存: {save_name} (时间: {meta['timestamp_ms']}ms)")
        else:
            print(f"[Warn] 无法读取视频帧: {video_name} 第 {frame_id} 帧")

    # 释放所有视频资源
    for cap in video_caps.values():
        cap.release()
    print("[Done] 可视化完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="目标图片路径")
    parser.add_argument("--limit", "-n", type=int, default=3, help="生成几张结果图")
    args = parser.parse_args()

    visualize_search(args.input, limit=args.limit)