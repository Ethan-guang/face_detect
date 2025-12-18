import os
import cv2
import json
import time
import numpy as np
from utils import round_list
from database import VectorDB
from tracker import SmartTracker


def get_output_dir(config, project_name, file_path):
    """统一生成输出目录: store/{日期}/{项目名}/{文件名}/"""
    out_root = config['project_settings']['output_root']
    date_str = time.strftime("%Y%m%d")
    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(out_root, date_str, project_name, file_name_no_ext)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def process_image(engine, img_path, config, project_name="default_project"):
    """处理单张图片 (支持单图多人脸)"""
    # 1. 准备路径和数据库
    out_dir = get_output_dir(config, project_name, img_path)

    # 图片通常直接入库，视为微观数据(Frame)
    db_path = config['project_settings'].get('vector_db_path', 'store/vector_db')
    db = VectorDB(db_path=db_path, collection_name=project_name)

    print(f" -> 读取图片: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图片 {img_path}")
        return None

    h, w = img.shape[:2]

    # 2. 核心：提取人脸 (返回列表，天然支持多人脸)
    faces = engine.extract(img)
    print(f" -> 检测到 {len(faces)} 张人脸")

    face_items = []

    # 3. 遍历每一张人脸进行处理
    for idx, f in enumerate(faces):
        # 构造唯一ID: 文件名_face_索引
        file_name = os.path.basename(img_path)
        unique_id = f"{file_name}_face_{idx}"

        # 构造 Metadata
        meta = {
            "video_name": file_name,  # 这里复用 video_name 字段存文件名
            "data_level": "frame",  # 图片本质上是单帧
            "frame_id": 0,  # 图片默认为第0帧
            "timestamp_ms": 0,
            "score": float(f["score"]),
            "bbox": str(f["bbox"].tolist())
        }

        # 入库
        db.buffer_add(unique_id, f['embedding'], meta)

        # 保存 NPY 特征文件 (可选)
        # np.save(os.path.join(out_dir, f"face_{idx}.npy"), f["embedding"])

        face_items.append({
            "face_id": idx,
            "bbox": round_list(f["bbox"].tolist()),
            "score": round(f["score"], 4)
        })

    # 提交入库
    db.flush()

    # 4. 生成报告
    output_data = {
        "meta": {
            "file_name": os.path.basename(img_path),
            "project": project_name,
            "type": "image",
            "resolution": [w, h],
            "count": len(faces)
        },
        "faces": face_items,
        "info": "Data saved to VectorDB"
    }

    json_path = os.path.join(out_dir, "process_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return json_path


def process_video(engine, video_path, config, project_name="default_project"):
    """处理视频主流程 (支持动态路径)"""
    # 1. 准备路径
    out_dir = get_output_dir(config, project_name, video_path)

    # 2. 初始化数据库 (传入项目名)
    db_path = config['project_settings'].get('vector_db_path', 'store/vector_db')
    db = VectorDB(db_path=db_path, collection_name=project_name)  # <--- 关键修改

    # 配置参数读取
    video_conf = config['video_config']
    save_mode = config['run_mode']['save_mode']
    is_save_all = (save_mode == 1)

    print(f" -> 模式: {'[全量/微观]' if is_save_all else '[追踪/宏观]'} | 集合: {project_name}")

    tracker = None
    if not is_save_all:
        tracker = SmartTracker(
            sim_threshold=video_conf['similarity_threshold'],
            miss_tolerance=video_conf['miss_tolerance']
        )

    # 3. 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

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

            # 提取人脸 (这里本身就支持返回多张人脸)
            current_faces = engine.extract(frame)
            processed_count += len(current_faces)

            if is_save_all:
                # Mode 1: 存每一帧里的每一个人
                for i, f in enumerate(current_faces):
                    unique_id = f"{os.path.basename(video_path)}_{frame_id}_{i}"
                    meta = {
                        "video_name": os.path.basename(video_path),
                        "data_level": "frame",
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp,
                        "score": float(f["score"]),
                        "bbox": str(f["bbox"].tolist())
                    }
                    db.buffer_add(unique_id, f['embedding'], meta)
            else:
                # Mode 0: 追踪 (Tracker 内部逻辑会处理多个人脸的分配)
                tracker.update(current_faces, frame_id, timestamp)

        frame_id += 1
        if frame_id % 100 == 0:
            print(f" -> 进度: {frame_id}/{total_frames}", end="\r")

    cap.release()
    print("")

    # 4. 扫尾和保存结果
    result_content = {}

    if is_save_all:
        db.flush()
        result_content = {"info": "Saved Frame-Level Data", "total_faces": db.count()}
    else:
        # Mode 0: 宏观模式入库
        all_tracks = tracker.final_tracks + tracker.active_tracks
        print(f" -> [Smart] 提取到 {len(all_tracks)} 条人物轨迹")

        for track in all_tracks:
            unique_id = f"{os.path.basename(video_path)}_track_{id(track)}"
            meta = {
                "video_name": os.path.basename(video_path),
                "data_level": "track",
                "start_time_ms": track.start_time,
                "end_time_ms": track.end_time,
                "frame_id": track.best_frame,
                "best_score": float(track.best_score),
                "duration": track.end_time - track.start_time,
                "bbox": str(track.best_bbox.tolist())
            }
            db.buffer_add(unique_id, track.best_embedding, meta)

        db.flush()
        tracks = tracker.get_results()
        result_content = {"info": "Saved Track-Level Data", "tracks": tracks}

    # 5. 生成报告
    output_data = {
        "meta": {
            "file_name": os.path.basename(video_path),
            "project": project_name,
            "mode": "micro" if is_save_all else "macro",
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

    return json_path