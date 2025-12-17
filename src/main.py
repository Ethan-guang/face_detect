import os
import argparse
import json
import cv2
import numpy as np
from core import FaceEngine, compute_sim
from utils import round_list


# ================= 0. 配置加载器 =================
def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ================= 1. 辅助类 =================
class FaceTrack:
    """用于在内存中暂存一个人脸轨迹"""

    def __init__(self, face_data, frame_id, timestamp):
        self.start_frame = frame_id
        self.end_frame = frame_id
        self.start_time = timestamp
        self.end_time = timestamp
        self.best_score = face_data['score']
        self.best_embedding = face_data['embedding']
        self.best_bbox = face_data['bbox']
        self.miss_count = 0


# ================= 2. 核心处理逻辑 =================

def process_image(engine, img_path, config):
    """处理单张图片"""
    # 从配置中读取输出路径
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
        # 保存 embedding
        np.save(os.path.join(out_dir, f"face_{idx}.npy"), f["embedding"])
        face_items.append({
            "face_id": idx,
            "bbox": round_list(f["bbox"].tolist()),
            "score": round(f["score"], 4)
        })

    output_data = {
        "meta": {
            "file_name": os.path.basename(img_path),
            "type": "image",
            "resolution": [w, h],
            "count": len(faces)
        },
        "faces": face_items
    }

    json_path = os.path.join(out_dir, "image_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # === 返回输出文件的路径 ===
    return json_path


def process_video(engine, video_path, config):
    """处理视频 (根据 config['run_mode']['save_mode'] 决定模式)"""

    # 1. 解析配置
    out_root = config['project_settings']['output_root']
    vector_db_path = config['project_settings']['vector_db_path']
    stride = config['video_config']['stride']
    sim_threshold = config['video_config']['similarity_threshold']
    miss_tolerance = config['video_config']['miss_tolerance']

    # 获取开关状态：1=全量(Raw), 0=智能(Smart)
    save_mode = config['run_mode']['save_mode']
    is_save_all = (save_mode == 1)

    # 2. 准备路径
    out_dir = os.path.join(out_root, "video_result")
    os.makedirs(out_dir, exist_ok=True)

    # 这里的 vector_db_path 如果你需要单独存向量库，可以利用起来
    # 目前我们暂时还是存在 result 下，或者你可以改成存到 vector_db_path

    mode_str = "全量采集 (Mode=1)" if is_save_all else "智能追踪 (Mode=0)"
    print(f"正在处理视频: {video_path}")
    print(f" -> 当前模式: {mode_str}")
    print(f" -> 阈值: {sim_threshold}, 跳帧: {stride}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0

    # 数据容器
    all_embeddings = []
    final_tracks = []
    active_tracks = []
    raw_detections = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_id % stride == 0:
            timestamp = int((frame_id / fps) * 1000) if fps else 0
            current_faces = engine.extract(frame)

            # === 分支 1：全量采集 (Mode 1) ===
            if is_save_all:
                for f in current_faces:
                    emb_idx = len(all_embeddings)
                    all_embeddings.append(f['embedding'])
                    raw_detections.append({
                        "global_id": emb_idx,
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp,
                        "bbox": round_list(f["bbox"].tolist()),
                        "score": round(f["score"], 4)
                    })

            # === 分支 2：智能追踪 (Mode 0) ===
            else:
                matched_indices = set()
                for curr_face in current_faces:
                    best_sim = 0
                    best_idx = -1
                    for idx, track in enumerate(active_tracks):
                        sim = compute_sim(curr_face['embedding'], track.best_embedding)
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = idx

                    # 使用配置文件的阈值
                    if best_sim > sim_threshold and best_idx != -1:
                        track = active_tracks[best_idx]
                        track.end_frame = frame_id
                        track.end_time = timestamp
                        track.miss_count = 0
                        matched_indices.add(best_idx)
                        if curr_face['score'] > track.best_score:
                            track.best_score = curr_face['score']
                            track.best_embedding = curr_face['embedding']
                            track.best_bbox = curr_face['bbox']
                    else:
                        active_tracks.append(FaceTrack(curr_face, frame_id, timestamp))

                for i in range(len(active_tracks) - 1, -1, -1):
                    if i not in matched_indices:
                        track = active_tracks[i]
                        track.miss_count += 1
                        if track.miss_count > miss_tolerance:
                            final_tracks.append(track)
                            active_tracks.pop(i)

        frame_id += 1
    cap.release()

    # === 结果生成 ===
    file_prefix = "raw" if is_save_all else "smart"

    if not is_save_all:
        final_tracks.extend(active_tracks)
        tracks_data = []
        for idx, track in enumerate(final_tracks):
            all_embeddings.append(track.best_embedding)
            tracks_data.append({
                "track_id": idx,
                "time_range_ms": [track.start_time, track.end_time],
                "duration_ms": track.end_time - track.start_time,
                "best_score": round(track.best_score, 4),
                "best_bbox": round_list(track.best_bbox.tolist()),
                "embedding_idx": idx
            })
        result_content = {"tracks": tracks_data}
    else:
        result_content = {"detections": raw_detections}

    output_data = {
        "meta": {
            "file_name": os.path.basename(video_path),
            "config_mode": save_mode,
            "resolution": [width, height],
            "fps": round(fps, 3),
            "duration_sec": round(duration, 2),
            "total_faces": len(all_embeddings)
        }
    }
    output_data.update(result_content)

    json_path = os.path.join(out_dir, f"{file_prefix}_result.json")
    emb_path = os.path.join(out_dir, f"{file_prefix}_embeddings.npy")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    if all_embeddings:
        np.save(emb_path, np.stack(all_embeddings))

    print(f"处理完成 -> {json_path}")

    # === 返回输出文件的路径 (Output) ===
    return json_path


# ================= 3. 主入口封装 =================
def run_pipeline(input_path, config_path="config.json"):
    """
    统一入口函数：
    Input: 文件路径
    Output: 结果JSON路径
    """
    # 1. 加载配置
    cfg = load_config(config_path)

    # 2. 初始化引擎
    engine = FaceEngine(
        model_name=cfg['model_params']['model_name'],
        det_size=tuple(cfg['model_params']['det_size'])
    )

    # 3. 分发任务
    if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        result_file = process_video(engine, input_path, cfg)
    else:
        result_file = process_image(engine, input_path, cfg)

    return result_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="输入路径")
    parser.add_argument("--config", "-c", default="config.json", help="配置文件路径")
    args = parser.parse_args()

    # 执行并获取返回结果
    final_output = run_pipeline(args.input, args.config)

    # 打印最终结果供外部调用捕获
    print(f"\n[PIPELINE_SUCCESS] Output saved to: {final_output}")