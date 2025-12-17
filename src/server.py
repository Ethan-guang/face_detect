import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from core import FaceEngine
from database import VectorDB
from main import load_config

# === 全局变量 (用于保持模型常驻内存) ===
models = {}


# === 生命周期管理 (启动时加载，关闭时释放) ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 启动前：加载配置和模型
    print("[Server] 正在启动... 加载模型中...")
    cfg = load_config("config.json")

    models["engine"] = FaceEngine(model_name=cfg['model_params']['model_name'])

    db_path = cfg['project_settings'].get('vector_db_path', 'store/vector_db')
    models["db"] = VectorDB(db_path=db_path)

    print("[Server] 模型与数据库加载完毕，服务就绪！")
    yield
    # 2. 关闭后：清理资源 (如果有的话)
    print("[Server] 服务关闭。")


app = FastAPI(title="Video Face Search API", lifespan=lifespan)


# === 定义 API 接口 ===

@app.get("/")
def read_root():
    return {"status": "running", "message": "Welcome to Video Face Search API"}


@app.post("/search")
async def search_face(file: UploadFile = File(...), limit: int = 5):
    """
    上传一张图片，返回视频中出现的时间点
    """
    # 1. 读取上传的图片 (直接在内存中处理，不存磁盘)
    try:
        file_bytes = await file.read()
        # 将字节流转换为 numpy 数组
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # 2. 提取特征
    engine = models["engine"]
    faces = engine.extract(img)

    if len(faces) == 0:
        return {"status": "failed", "message": "No face detected in the uploaded image"}

    # 取最大的人脸
    target_emb = faces[0]['embedding']

    # 3. 数据库检索
    db = models["db"]
    results = db.search(target_emb, limit=limit)

    if not results:
        return {"status": "success", "data": []}

    # 4. 格式化返回结果
    response_data = []
    for item in results:
        meta = item['meta']
        response_data.append({
            "video": meta.get('video_name'),
            "timestamp_ms": meta.get('timestamp_ms'),
            "frame_id": meta.get('frame_id'),
            "similarity_score": round(1 / (1 + item['distance']), 4),  # 简单的分数转换
            "l2_distance": round(item['distance'], 4)
        })

    return {
        "status": "success",
        "count": len(response_data),
        "data": response_data
    }


# === 本地调试入口 ===
if __name__ == "__main__":
    # 运行服务: host="0.0.0.0" 表示允许局域网访问
    uvicorn.run(app, host="0.0.0.0", port=8000)