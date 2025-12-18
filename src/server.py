import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Query
from contextlib import asynccontextmanager
from service import Searcher # <--- 核心依赖

# 全局服务实例
search_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_service
    print("[Server] Init Search Service...")
    search_service = Searcher() # 初始化一次，常驻内存
    yield
    print("[Server] Shutting down.")

app = FastAPI(lifespan=lifespan)

@app.post("/search")
def search_face(
        file: UploadFile = File(...),
        limit: int = 5,
        level: str = "auto",
        threshold: float = 0.6,
        project: str = "default_project"
):
    # 1. 读取图片流
    file_bytes = file.file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. 调用 Service (传入 numpy array)
    results = search_service.search(
        img,
        limit=limit,
        level=level,
        threshold=threshold,
        project=project
    )

    # 3. 返回 (Service 已经返回了标准化的 dict，直接吐给前端即可)
    return {
        "status": "success",
        "count": len(results),
        "data": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)