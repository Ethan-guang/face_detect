import argparse
import cv2
import os
from service import Searcher


def visualize(image_path, limit, level, threshold, output_dir):
    # 1. 调用 Service
    print(f"[Visual] Searching for: {image_path} (Threshold: {threshold})")
    searcher = Searcher()
    # 这里的 searcher 会自动加载 config.json
    results = searcher.search(image_path, limit=limit, level=level, threshold=threshold)

    if not results:
        print("[Visual] No results found matching the criteria.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"[Visual] Found {len(results)} results. Saving to {output_dir}...")

    # 2. 遍历结果画图
    for i, item in enumerate(results):
        video_name = item['file_name']
        frame_id = item['frame_id']
        bbox = item['bbox']  # utils 已将其解析为列表 [x1, y1, x2, y2]

        # ⚠️ 注意：这里假设视频都在 data/video 下。
        # 如果你的视频分散在不同目录，建议在入库时的 metadata 里存绝对路径
        video_path = os.path.join("data/video", video_name)

        # 如果找不到视频文件，尝试去 config 配置的 input 目录找（可选优化）
        if not os.path.exists(video_path):
            print(f"[Warn] Video file not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # 画框 (BGR 红色, 线宽 4)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 4)

            # 加上文字标签 (可选)
            label = f"Score: {item['score']}"
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # 保存
            save_name = f"rank_{i + 1}_{item['data_level']}_{video_name}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, frame)
            print(f" -> Saved: {save_name}")
        else:
            print(f"[Warn] Could not read frame {frame_id} from {video_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--limit", "-n", type=int, default=3, help="Max results")
    parser.add_argument("--level", "-l", default="auto", choices=["auto", "track", "frame"])
    parser.add_argument("--threshold", "-t", type=float, default=0.6, help="Similarity threshold")
    parser.add_argument("--output", "-o", default="store/visualized", help="Output directory for result images")

    args = parser.parse_args()

    # ✅ 修复：正确的函数名和参数传递
    visualize(
        image_path=args.input,
        limit=args.limit,
        level=args.level,
        threshold=args.threshold,
        output_dir=args.output
    )