import numpy as np
from core import compute_sim
from utils import round_list


class FaceTrack:
    """单个人的轨迹对象"""

    def __init__(self, face_data, frame_id, timestamp):
        self.start_frame = frame_id
        self.end_frame = frame_id
        self.start_time = timestamp
        self.end_time = timestamp
        self.best_score = face_data['score']
        self.best_embedding = face_data['embedding']
        self.best_bbox = face_data['bbox']
        # [新增] 记录最高分出现的帧号，用于可视化截图
        self.best_frame = frame_id
        self.miss_count = 0


class SmartTracker:
    def __init__(self, sim_threshold=0.65, miss_tolerance=3):
        self.sim_threshold = sim_threshold
        self.miss_tolerance = miss_tolerance
        self.active_tracks = []  # 正在画面里的人
        self.final_tracks = []  # 已经离开的人

    def update(self, current_faces, frame_id, timestamp):
        """处理当前帧的人脸，更新轨迹"""
        matched_indices = set()

        # 1. 尝试匹配现有轨迹
        for curr_face in current_faces:
            best_sim = 0
            best_idx = -1
            for idx, track in enumerate(self.active_tracks):
                sim = compute_sim(curr_face['embedding'], track.best_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            # 判定匹配成功
            if best_sim > self.sim_threshold and best_idx != -1:
                track = self.active_tracks[best_idx]
                track.end_frame = frame_id
                track.end_time = timestamp
                track.miss_count = 0
                matched_indices.add(best_idx)

                # 如果这张脸更清晰（分更高），更新最佳照
                if curr_face['score'] > track.best_score:
                    track.best_score = curr_face['score']
                    track.best_embedding = curr_face['embedding']
                    track.best_bbox = curr_face['bbox']
                    # [新增] 同步更新最佳帧
                    track.best_frame = frame_id
            else:
                # 没匹配上，视为新出现的人
                self.active_tracks.append(FaceTrack(curr_face, frame_id, timestamp))

        # 2. 清理消失的人 (Miss Tolerance)
        for i in range(len(self.active_tracks) - 1, -1, -1):
            if i not in matched_indices:
                track = self.active_tracks[i]
                track.miss_count += 1
                if track.miss_count > self.miss_tolerance:
                    self.final_tracks.append(track)
                    self.active_tracks.pop(i)

    def get_results(self):
        """返回最终序列化结果"""
        all_tracks = self.final_tracks + self.active_tracks
        tracks_data = []
        for idx, track in enumerate(all_tracks):
            tracks_data.append({
                "track_id": idx,
                "time_range_ms": [track.start_time, track.end_time],
                "duration_ms": track.end_time - track.start_time,
                "best_score": round(track.best_score, 4),
                "best_frame": track.best_frame,  # 报告里也带上最佳帧
                "bbox": round_list(track.best_bbox.tolist())
            })
        return tracks_data