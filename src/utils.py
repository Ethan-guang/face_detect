import json
import os
import numpy as np


def round_list(data, decimals=4):
    """将列表中的浮点数保留指定小数位"""
    if isinstance(data, list):
        return [round(x, decimals) for x in data]
    return data


def l2_normalize(x):
    """特征向量归一化"""
    return x / np.linalg.norm(x)


def load_config(config_path="config.json"):
    """
    统一加载配置文件，支持从 src 目录或根目录运行
    """
    # 尝试当前目录
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 尝试上一级目录 (兼容 src/ 运行环境)
    parent_path = os.path.join("..", config_path)
    if os.path.exists(parent_path):
        with open(parent_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    raise FileNotFoundError(f"配置文件未找到: {config_path}")


def parse_metadata(item_data):
    """
    统一解析数据库返回的元数据，标准化输出格式
    输入: DB返回的原始 item (包含 'meta', 'distance' 等)
    输出: 标准化的字典结构
    """
    meta = item_data.get('meta', {})
    distance = item_data.get('distance', 0.0)
    score = 1 / (1 + distance)

    file_name = meta.get('video_name', 'Unknown')
    is_video = file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    # 基础信息
    result = {
        "file_name": file_name,
        "type": "video" if is_video else "image",
        "score": round(score, 4),
        "score_in_db": meta.get('best_score', meta.get('score', 0)),
        "data_level": meta.get('data_level', 'unknown'),
        "frame_id": meta.get('frame_id', 0),
        "bbox": eval(meta.get('bbox', '[0,0,0,0]'))  # 转换字符串为列表
    }

    # 时间信息标准化
    if 'start_time_ms' in meta:  # 宏观轨迹
        result["time_info"] = {
            "mode": "range",
            "start_ms": meta['start_time_ms'],
            "end_ms": meta['end_time_ms'],
            "duration_ms": meta.get('duration', 0),
            "display": f"{meta['start_time_ms']}ms ~ {meta['end_time_ms']}ms"
        }
    elif is_video:  # 微观单帧
        result["time_info"] = {
            "mode": "point",
            "timestamp_ms": meta.get('timestamp_ms', 0),
            "display": f"{meta.get('timestamp_ms', 0)} ms"
        }
    else:  # 纯图片
        result["time_info"] = {
            "mode": "static",
            "timestamp_ms": 0,
            "display": "Static Image"
        }

    return result