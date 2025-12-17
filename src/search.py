# src/search.py

import argparse
import os
import cv2
import json
from core import FaceEngine
from database import VectorDB


# 复用 main.py 里的配置加载逻辑，或者直接复制过来
def load_config(config_path="config.json"):
    # 兼容从 src 目录或根目录运行的情况
    if not os.path.exists(config_path):
        if os.path.exists(os.path.join("..", config_path)):
            config_path = os.path.join("..", config_path)
        else:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_search(image_path, limit=5, config_path="config.json"):
    # 1. 加载配置
    cfg = load_config(config_path)

    # 2. 初始化 AI 引擎 (用于提取那张照片的特征)
    print(f"[System] 初始化模型...")
    engine = FaceEngine(
        model_name=cfg['model_params']['model_name'],
        det_size=tuple(cfg['model_params']['det_size'])
    )

    # 3. 读取并处理图片
    print(f"[Search] 读取目标图片: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 无法读取图片")
        return

    # 提取特征
    faces = engine.extract(img)
    if len(faces) == 0:
        print(f"[Result] 图片中未检测到人脸，无法搜索。")
        return

    # 默认取画面中最大/置信度最高的那张脸
    target_face = faces[0]
    print(f"[Search] 检测到人脸，置信度: {target_face['score']:.4f}，开始搜索数据库...")

    # 4. 连接数据库并查询
    db_path = cfg['project_settings'].get('vector_db_path', 'store/vector_db')
    db = VectorDB(db_path=db_path)

    # === 新增 Debug 代码开始 ===
    count = db.count()
    print(f"[Debug] 当前数据库集合 '{db.collection.name}' 中的数据总量: {count}")

    if count == 0:
        print("[Error] 数据库连接成功，但集合是空的！")
        print("可能原因：")
        print("1. main.py 入库时用的集合名称和这里不一样？")
        print("2. main.py 写入到了另一个路径？")
        print("3. 数据没 flush 进去？")
        return
    # === 新增 Debug 代码结束 ===

    results = db.search(target_face['embedding'], limit=limit)

    # 5. 展示结果
    print(f"\n=== 搜索结果 (Top {limit}) ===")
    if not results:
        print("数据库为空或未找到匹配项。")
        return

    for idx, item in enumerate(results):
        meta = item['meta']
        score = 1 / (1 + item['distance'])  # (可选) 简单的距离转分数可视化

        print(f"[{idx + 1}] 相似度(L2距离): {item['distance']:.4f}")
        print(f"    - 视频来源: {meta.get('video_name', 'Unknown')}")
        print(f"    - 出现时间: {meta.get('timestamp_ms', 0)} ms")
        print(f"    - 视频帧号: {meta.get('frame_id', 0)}")
        print(f"    - 原始置信度: {meta.get('score', 0)}")
        print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人脸以图搜图工具")
    parser.add_argument("--input", "-i", required=True, help="要搜索的目标人脸图片路径")
    parser.add_argument("--limit", "-n", type=int, default=5, help="返回结果数量")
    parser.add_argument("--config", "-c", default="config.json", help="配置文件路径")

    args = parser.parse_args()

    run_search(args.input, limit=args.limit, config_path=args.config)