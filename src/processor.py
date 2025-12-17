import os
import cv2
import json
import numpy as np
from utils import round_list
from database import VectorDB
from tracker import SmartTracker


def process_image(engine, img_path, config):
    """处理单张图片"""
    out_root = config['project_settings']['output_root']
    out_dir = os.path.join(out_root, "img_result")
    os.makedirs(out_dir, exist_ok=True)

    print(f"处理图片: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图片 {img_path}")
        return None

    h, w = img.shape[:2]
    faces = engine.extract(img)
    face_items = []

    for idx, f in enumerate(faces):
        # 存单人特征文件（可选）
        np.save(os.path.join(out_dir, f"face_{idx}.npy"), f["embedding"])
        face_items.append({
            "face_id": idx,
            "bbox": round_list(f["bbox"].tolist()),
            "score": round(f["score"], 4)
        })

    output_data = {
        "meta": {"file_name": os.path.basename(img_path), "resolution": [w, h], "count": len(faces)},
        "faces": face_items
    }

    json_path = os.path.join(out_dir, "image_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return json_path


def process_video(engine, video_path, config):
    """处理视频主流程"""
    # 1. 准备配置和路径
    out_root = config['project_settings']['output_root']
    db_path = config['project_settings'].get('vector_db_path', 'store/vector_db')

    video_conf = config['video_config']
    save_mode = config['run_mode']['save_mode']
    is_save_all = (save_mode == 1)

    out_dir = os.path.join(out_root, "video_result")
    os.makedirs(out_dir, exist_ok=True)

    # 2. 根据模式初始化 数据库 或 追踪器
    db = None
    tracker = None

    if is_save_all:
        print(f"模式: [全量采集] -> 数据将存入向量库")
        db = VectorDB(db_path=db_path)
    else:
        print(f"模式: [智能追踪] -> 生成去重轨迹报告")
        tracker = SmartTracker(
            sim_threshold=video_conf['similarity_threshold'],
            miss_tolerance=video_conf['miss_tolerance']
        )

    # 3. 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0

    frame_id = 0
    stride = video_conf['stride']
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_id % stride == 0:
            timestamp = int((frame_id / fps) * 1000) if fps else 0
            # 调用引擎提取特征
            current_faces = engine.extract(frame)
            processed_count += len(current_faces)

            # === 分支逻辑 ===
            if is_save_all:
                # Mode 1: 存入数据库缓冲区
                for i, f in enumerate(current_faces):
                    unique_id = f"{os.path.basename(video_path)}_{frame_id}_{i}"
                    meta = {
                        "video_name": os.path.basename(video_path),
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp,
                        "score": float(f["score"]),
                        "bbox": str(f["bbox"].tolist())
                    }
                    db.buffer_add(unique_id, f['embedding'], meta)
            else:
                # Mode 0: 更新追踪器
                tracker.update(current_faces, frame_id, timestamp)

        frame_id += 1

    cap.release()

    # 4. 扫尾和保存结果
    result_content = {}

    if is_save_all:
        db.flush()  # 把剩下的写入
        result_content = {
            "info": "Data saved to VectorDB",
            "db_path": db_path,
            "total_faces": db.count()
        }
    else:
        tracks = tracker.get_results()
        result_content = {"tracks": tracks}

    # 5. 生成 JSON 报告
    output_data = {
        "meta": {
            "file_name": os.path.basename(video_path),
            "mode": "raw_db" if is_save_all else "smart_track",
            "resolution": [width, height],
            "fps": round(fps, 3),
            "duration": round(duration, 2),
            "processed_faces": processed_count
        }
    }
    output_data.update(result_content)

    json_path = os.path.join(out_dir, "process_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"处理完成 -> 报告: {json_path}")
    return json_path