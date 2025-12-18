import cv2
import os
from core import FaceEngine
from database import VectorDB
from utils import load_config, parse_metadata


class Searcher:
    def __init__(self, config_path="config.json", project_name="default_project"):
        """
        初始化搜索服务
        :param project_name: 默认连接的项目，也可以在 search 时动态指定
        """
        self.cfg = load_config(config_path)

        # 初始化 AI 引擎
        print(f"[Service] Loading FaceEngine...")
        self.engine = FaceEngine(
            model_name=self.cfg['model_params']['model_name'],
            # 建议这里把 providers 也做成配置项，目前默认
        )

        # 初始化数据库连接 (持有一个客户端实例)
        self.db_path = self.cfg['project_settings'].get('vector_db_path', 'store/vector_db')
        # 预加载默认库，search 时可切换
        self.default_db = VectorDB(db_path=self.db_path, collection_name=project_name)
        print(f"[Service] Ready. DB Path: {self.db_path}")

    def search(self, image_data, limit=5, level="auto", threshold=0.6, project=None):
        """
        核心搜索方法
        :param image_data: 图片路径(str) 或 图片矩阵(numpy array)
        :param project: 指定搜索的项目集合，None则使用默认
        :return: 标准化的结果列表
        """
        # 1. 图片预处理
        img = None
        if isinstance(image_data, str):
            if not os.path.exists(image_data):
                print(f"[Error] File not found: {image_data}")
                return []
            img = cv2.imread(image_data)
        else:
            img = image_data  # 假设是 numpy array

        if img is None:
            return []

        # 2. 提取特征
        faces = self.engine.extract(img)
        if not faces:
            print("[Service] No face detected.")
            return []

        # 取最大的人脸进行搜索
        target_emb = faces[0]['embedding']

        # 3. 确定数据库集合
        db = self.default_db
        if project and project != "default_project":
            # 动态连接其他项目
            db = VectorDB(db_path=self.db_path, collection_name=project)

        # 4. 构造过滤条件
        where_filter = None
        if level == "track": where_filter = {"data_level": "track"}
        if level == "frame": where_filter = {"data_level": "frame"}

        # 5. 执行搜索
        raw_results = db.search(target_emb, limit=limit, where=where_filter)

        # 6. 解析与过滤
        parsed_results = []
        for item in raw_results:
            info = parse_metadata(item)

            # 阈值过滤
            if info['score'] < threshold:
                continue

            parsed_results.append(info)

        return parsed_results