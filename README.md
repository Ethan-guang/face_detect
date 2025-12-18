# Video-OCR: 智能视频人脸追踪与全量检索系统

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![InsightFace](https://img.shields.io/badge/InsightFace-Buffalo__1-green) ![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange) ![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)

这是一个基于 **InsightFace** 和 **ChromaDB** 构建的智能视频分析系统。它能够对视频中的人脸进行检测、特征提取，支持人脸轨迹追踪，并实现了“以图搜视频”的毫秒级检索功能。

## ✨ 主要功能

* **🎥 视频全量采集 (Ingestion)**：自动处理长视频，按帧提取人脸特征并存入向量数据库 (ChromaDB)。
* **🔍 以图搜图 (Search)**：上传一张人脸照片，快速检索出该人物在视频中出现的所有时间点。
* **📊 可视化验证 (Visualization)**：自动截取搜索结果对应的视频帧，并绘制人脸框和置信度，直观验证搜索结果。
* **🌐 RESTful API 服务**：内置 FastAPI 服务，提供标准的 HTTP 接口，方便前端或第三方系统调用。
* **🧠 智能去重 (Smart Tracking)**：支持生成去重后的人员轨迹报告 (JSON格式)。

## 📂 项目结构

```text
vedio-ocr/
├── config.json           # 项目核心配置文件
├── requirements.txt      # 依赖库列表
├── data/                 # 输入数据目录 (视频/图片)
├── store/                # 输出结果目录 (数据库/JSON/可视化图)
│   └── vector_db/        # ChromaDB 向量数据库文件
└── src/                  # 源代码
    ├── core.py           # AI 引擎 (InsightFace 封装)
    ├── database.py       # 数据库管理模块 (ChromaDB)
    ├── tracker.py        # 人脸追踪算法模块
    ├── processor.py      # 视频流处理业务逻辑
    ├── main.py           # 命令行入口脚本 (处理视频/图片)
    ├── search.py         # 命令行搜索脚本
    ├── visualize.py      # 结果可视化脚本
    └── server.py         # Web 服务入口
