import numpy as np
from insightface.app import FaceAnalysis
from utils import l2_normalize

def compute_sim(feat1, feat2):
    """计算两个特征向量的余弦相似度"""
    return np.dot(feat1, feat2)

class FaceEngine:
    def __init__(self, model_name="buffalo_1", ctx_id=0, det_size=(640, 640)):
        """
        初始化模型
        """
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print(f"[{model_name}] 模型加载完毕 (ctx_id={ctx_id})")

    def extract(self, img_array):
        """
        输入: 图片矩阵 (cv2 read result)
        输出: 结构化的人脸数据列表
        """
        faces = self.app.get(img_array)
        results = []

        for f in faces:
            # 统一在这里做归一化
            emb = np.asarray(f.embedding, dtype=np.float32)
            emb = l2_normalize(emb)

            results.append({
                "bbox": f.bbox,          # [x1, y1, x2, y2]
                "score": float(f.det_score),
                "embedding": emb,        # 512维向量
                "kps": f.kps             # 关键点
            })
        return results